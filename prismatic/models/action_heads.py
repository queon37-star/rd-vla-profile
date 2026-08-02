import math
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from configs.rdvla_precheck import (
    canonicalize_recurrence_strategy,
    validate_latent_only_configuration,
    validate_latent_precheck_configuration,
)
from prismatic.models.latent_only_stopping import run_latent_only_adaptive
from prismatic.models.action_head_workload import build_action_head_workload
from prismatic.models.origin_aware_scheduler import (
    NonFiniteOriginAwareInferenceError,
    run_origin_aware_adaptive,
)
from prismatic.models.numerical_retry import run_cold_full_coda_retry
from prismatic.models.shadow_trace import build_shadow_trace_record, run_shadow_tail
from prismatic.vla.constants import ACTION_DIM, NUM_ACTIONS_CHUNK
from prismatic.utils.rdvla_profiler import rdvla_range
from dataclasses import dataclass


def apply_rope(q, k, cos, sin):
    cos = cos.unsqueeze(0).unsqueeze(0)
    sin = sin.unsqueeze(0).unsqueeze(0)

    def rotate_half(x):
        x1 = x[..., ::2]
        x2 = x[..., 1::2]
        return torch.stack((-x2, x1), dim=-1).reshape_as(x)

    q_rot = (q * cos) + (rotate_half(q) * sin)
    k_rot = (k * cos) + (rotate_half(k) * sin)
    return q_rot, k_rot


class RotaryPositionEmbedding(nn.Module):
    def __init__(self, dim, base=10000):
        super().__init__()
        assert dim % 2 == 0
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(self, seq_len, device, dtype):
        t = torch.arange(seq_len, device=device, dtype=self.inv_freq.dtype)
        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        emb = torch.cat([freqs, freqs], dim=-1)
        return emb.cos().to(dtype), emb.sin().to(dtype)


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        dtype = x.dtype
        x = x.float()
        norm = x.pow(2).mean(-1, keepdim=True).add(self.eps).rsqrt()
        return (x * norm * self.weight).to(dtype)


@dataclass
class RecurrentConfigInternal:
    hidden_dim: int = 896
    num_heads: int = 8
    prelude_vlm_layers: tuple = ()
    recurrent_vlm_layers: tuple = (6, 23)
    coda_vlm_layers: tuple = ()
    action_chunk_len: int = 8
    action_dim: int = 7
    mean_recurrence: int = 12
    backprop_depth: int = 8
    random_iterations: bool = True
    init_std: float = 0.632
    rms_norm_eps: float = 1e-6
    rope_base: float = 10000.0

    @property
    def weight_std(self) -> float:
        return math.sqrt(2.0 / (5.0 * self.hidden_dim))

    @property
    def output_std(self) -> float:
        return self.weight_std / math.sqrt(self.mean_recurrence * len(self.recurrent_vlm_layers))


def select_warm_start_candidate(states, actual_iter, source):
    if not states:
        return None, None, None

    if source == "s1":
        source_index = 0
    elif source == "midpoint":
        source_index = max(0, actual_iter // 2 - 1)
    elif source == "final":
        source_index = actual_iter - 1
    else:
        raise ValueError(f"Unsupported warm_start_source: {source}")

    source_index = min(source_index, len(states) - 1)
    return states[source_index], source_index, source_index + 1


class RecurrentLayer(nn.Module):
    """Recurrent layer: self-attention -> cross-attention (with gating) -> SwiGLU FFN."""

    def __init__(self, hidden_dim: int, num_heads: int = 8, eps: float = 1e-6, rope_base: float = 10000.0):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads

        self.norm1 = RMSNorm(hidden_dim, eps)
        self.q_self = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.k_self = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.v_self = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.o_self = nn.Linear(hidden_dim, hidden_dim, bias=False)

        self.rope = RotaryPositionEmbedding(self.head_dim, base=rope_base)

        self.norm2 = RMSNorm(hidden_dim, eps)
        self.q_cross = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.k_latents = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.v_latents = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.k_vision = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.v_vision = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.o_cross = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.gate = nn.Parameter(torch.zeros(1))

        self.norm3 = RMSNorm(hidden_dim, eps)
        self.ffn_gate = nn.Linear(hidden_dim, hidden_dim * 4, bias=False)
        self.ffn_up = nn.Linear(hidden_dim, hidden_dim * 4, bias=False)
        self.ffn_down = nn.Linear(hidden_dim * 4, hidden_dim, bias=False)

    def forward(self, x: torch.Tensor, latent_tokens: torch.Tensor, vision_tokens: torch.Tensor, p: torch.Tensor) -> torch.Tensor:
        B, T, D = x.shape

        def reshape(t, seq_len):
            return t.view(B, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        # 1. Self-Attention
        residual = x
        x_n = self.norm1(x)
        q_s = reshape(self.q_self(x_n), T)
        k_s = reshape(self.k_self(x_n), T)
        v_s = reshape(self.v_self(x_n), T)
        cos, sin = self.rope(T, x.device, x.dtype)
        q_s, k_s = apply_rope(q_s, k_s, cos, sin)
        scale = self.head_dim ** -0.5
        attn_s = torch.matmul(q_s, k_s.transpose(-2, -1)) * scale
        attn_s = F.softmax(attn_s, dim=-1)
        out_s = torch.matmul(attn_s, v_s)
        out_s = out_s.transpose(1, 2).contiguous().view(B, T, D)
        x = residual + self.o_self(out_s)

        # 2. Cross-Attention
        context_latents = torch.cat([latent_tokens, p], dim=1)
        K_latents_len = context_latents.size(1)
        K_vision_len = vision_tokens.size(1)
        residual = x
        x_n = self.norm2(x)
        q_c = reshape(self.q_cross(x_n), T)
        k_lat = reshape(self.k_latents(context_latents), K_latents_len)
        v_lat = reshape(self.v_latents(context_latents), K_latents_len)
        k_vis = reshape(self.k_vision(vision_tokens), K_vision_len)
        v_vis = reshape(self.v_vision(vision_tokens), K_vision_len)
        attn_latents = torch.matmul(q_c, k_lat.transpose(-2, -1))
        attn_vision = torch.matmul(q_c, k_vis.transpose(-2, -1)) * torch.tanh(self.gate)
        attn_weights = F.softmax(torch.cat([attn_latents, attn_vision], dim=-1) * scale, dim=-1)
        v_combined = torch.cat([v_lat, v_vis], dim=2)
        out_c = torch.matmul(attn_weights, v_combined)
        out_c = out_c.transpose(1, 2).contiguous().view(B, T, D)
        x = residual + self.o_cross(out_c)

        # 3. SwiGLU FFN
        residual = x
        x_n = self.norm3(x)
        x = residual + self.ffn_down(F.silu(self.ffn_gate(x_n)) * self.ffn_up(x_n))

        return x


class VLARecurrent(nn.Module):
    """Prelude -> recurrent (iterated) -> coda."""

    def __init__(self, cfg):
        super().__init__()
        if isinstance(cfg, dict):
            cfg = RecurrentConfigInternal(**cfg)
        self.cfg = cfg
        dim = cfg.hidden_dim

        self.prelude_vlm_layers = list(cfg.prelude_vlm_layers)
        self.recurrent_vlm_layers = list(cfg.recurrent_vlm_layers)
        self.coda_vlm_layers = list(cfg.coda_vlm_layers)

        self.action_queries = nn.Parameter(torch.randn(cfg.action_chunk_len, dim) * cfg.init_std)

        if self.prelude_vlm_layers:
            self.prelude = nn.ModuleList([
                RecurrentLayer(dim, cfg.num_heads, cfg.rms_norm_eps, cfg.rope_base)
                for _ in self.prelude_vlm_layers
            ])

        self.adapter = nn.Linear(dim * 2, dim, bias=False)
        self.adapter_norm = RMSNorm(dim, cfg.rms_norm_eps)
        self.gamma_adapt = nn.Parameter(torch.ones(1))
        self.recurrent = nn.ModuleList([
            RecurrentLayer(dim, cfg.num_heads, cfg.rms_norm_eps, cfg.rope_base)
            for _ in self.recurrent_vlm_layers
        ])
        self.recurrent_norm = RMSNorm(dim, cfg.rms_norm_eps)

        if self.coda_vlm_layers:
            self.coda = nn.ModuleList([
                RecurrentLayer(dim, cfg.num_heads, cfg.rms_norm_eps, cfg.rope_base)
                for _ in self.coda_vlm_layers
            ])

        self.output_norm = RMSNorm(dim, cfg.rms_norm_eps)
        self.output_proj = nn.Linear(dim, cfg.action_dim)
        self.gamma_init = nn.Parameter(torch.ones(1))
        self._last_get_output_timing = None
        self.last_inference_metadata = None
        self._init_weights()

    def _init_weights(self):
        cfg = self.cfg
        weight_std = cfg.weight_std
        output_std = cfg.output_std
        for name, module in self.named_modules():
            if isinstance(module, nn.Linear):
                is_output = any(x in name for x in ['o_proj', 'ffn_down', 'output_proj'])
                std = output_std if is_output else weight_std
                nn.init.trunc_normal_(module.weight, mean=0.0, std=std, a=-3*std, b=3*std)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
        nn.init.trunc_normal_(self.action_queries, mean=0.0, std=cfg.init_std, a=-3*cfg.init_std, b=3*cfg.init_std)

    def init_state(self, B: int, device, dtype) -> torch.Tensor:
        with rdvla_range("RDVLA/action_head/init_state"):
            with rdvla_range("RDVLA/action_head/init_state_total"):
                with rdvla_range("RDVLA/action_head/init_state/gamma_item"):
                    std = (self.gamma_init * self.cfg.init_std).item()
                with rdvla_range("RDVLA/action_head/init_state/random_noise"):
                    state = torch.empty(B, self.cfg.action_chunk_len, self.cfg.hidden_dim, device=device, dtype=dtype)
                with rdvla_range("RDVLA/action_head/init_state/trunc_normal"):
                    nn.init.trunc_normal_(state, mean=0.0, std=std, a=-3*std, b=3*std)
                with rdvla_range("RDVLA/action_head/init_state/device_dtype_cast"):
                    return state

    def _select_initial_state(
        self,
        warm_start_state,
        B: int,
        device,
        dtype,
        validate_warm_start_finite: bool = False,
        validate_warm_start_dtype: bool = False,
    ):
        expected_shape = (B, self.cfg.action_chunk_len, self.cfg.hidden_dim)
        metadata = {
            "state_provided": warm_start_state is not None,
            "state_used": False,
            "initial_state_origin": "random",
            "reset": False,
            "reset_reason": None,
        }

        if warm_start_state is None:
            return self.init_state(B, device, dtype), metadata

        reset_reason = None
        if not torch.is_tensor(warm_start_state):
            reset_reason = "warm_start_state_not_tensor"
        elif validate_warm_start_dtype and warm_start_state.dtype != dtype:
            reset_reason = (
                f"warm_start_dtype_mismatch:expected={dtype},"
                f"actual={warm_start_state.dtype}"
            )
        elif tuple(warm_start_state.shape) != expected_shape:
            reset_reason = (
                f"warm_start_shape_mismatch:expected={expected_shape},"
                f"actual={tuple(warm_start_state.shape)}"
            )
        else:
            try:
                state = warm_start_state.detach().clone().to(device=device, dtype=dtype)
                is_finite = (
                    bool(torch.isfinite(state).all().item())
                    if validate_warm_start_finite
                    else True
                )
            except (RuntimeError, TypeError) as exc:
                reset_reason = f"warm_start_conversion_failed:{type(exc).__name__}"
            else:
                if is_finite:
                    metadata["state_used"] = True
                    metadata["initial_state_origin"] = "cached"
                    return state, metadata
                reset_reason = "warm_start_state_non_finite"

        metadata["reset"] = True
        metadata["reset_reason"] = reset_reason
        return self.init_state(B, device, dtype), metadata

    def _store_warm_start_candidate(self, states, actual_iter: int, source: str):
        selected_state, source_index, source_iteration = select_warm_start_candidate(
            states, actual_iter, source
        )
        if selected_state is None:
            return

        self.last_inference_metadata["next_warm_start_state"] = selected_state.detach().clone()
        self.last_inference_metadata["warm_start"].update(
            {
                "source": source,
                "source_index": source_index,
                "source_iteration": source_iteration,
                "source_K": actual_iter,
                "candidate_state_count": len(states),
            }
        )

    def sample_iterations(self) -> int:
        r_mean = self.cfg.mean_recurrence
        tau = torch.normal(mean=math.log(r_mean) - 0.125, std=0.5, size=(1,))
        lam = torch.exp(tau).clamp(max=100)
        return max(1, min(torch.poisson(lam).int().item() + 1, 64))

    def _run_one_iteration(self, state, prelude_out, h_a, h_t, p):
        with rdvla_range("RDVLA/action_head/recurrent_one_iteration"):
            with rdvla_range("RDVLA/action_head/iter_adapter"):
                x = self.adapter(torch.cat([state, prelude_out], dim=-1))
                x = self.adapter_norm(self.gamma_adapt * x)
            with rdvla_range("RDVLA/action_head/iter_recurrent_layers_total"):
                for i, layer in enumerate(self.recurrent):
                    with rdvla_range(f"RDVLA/action_head/iter_recurrent_layer_{i}"):
                        x = layer(x, h_a[:, self.recurrent_vlm_layers[i]], h_t[:, self.recurrent_vlm_layers[i]], p)
            with rdvla_range("RDVLA/action_head/iter_recurrent_norm"):
                return self.recurrent_norm(x)

    @staticmethod
    def _sync_time():
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        return time.perf_counter()

    def _get_output(self, state, h_a, h_t, p, profile=False):
        if not profile:
            x = state
            with rdvla_range("RDVLA/action_head/coda_total"):
                if self.coda_vlm_layers:
                    for i, layer in enumerate(self.coda):
                        with rdvla_range(f"RDVLA/action_head/coda_layer_{i}"):
                            x = layer(x, h_a[:, self.coda_vlm_layers[i]], h_t[:, self.coda_vlm_layers[i]], p)
            with rdvla_range("RDVLA/action_head/output_norm_proj"):
                return self.output_proj(self.output_norm(x))

        get_output_start = self._sync_time()
        x = state
        with rdvla_range("RDVLA/action_head/coda_total"):
            if self.coda_vlm_layers:
                coda_start = get_output_start
                for i, layer in enumerate(self.coda):
                    with rdvla_range(f"RDVLA/action_head/coda_layer_{i}"):
                        x = layer(x, h_a[:, self.coda_vlm_layers[i]], h_t[:, self.coda_vlm_layers[i]], p)
                coda_end = self._sync_time()
                coda_ms = (coda_end - coda_start) * 1000.0
            else:
                coda_end = get_output_start
                coda_ms = 0.0

        with rdvla_range("RDVLA/action_head/output_norm_proj"):
            output = self.output_proj(self.output_norm(x))
        output_proj_end = self._sync_time()
        self._last_get_output_timing = {
            "get_output_ms": (output_proj_end - get_output_start) * 1000.0,
            "coda_ms": coda_ms,
            "output_proj_ms": (output_proj_end - coda_end) * 1000.0,
        }
        return output

    def forward(self, h_a: torch.Tensor, h_t: torch.Tensor, p: torch.Tensor,
                num_iter: int = None, convergence_strategy: str = None,
                warm_start_state: torch.Tensor = None,
                enable_warm_start: bool = False,
                warm_start_source: str = "s1",
                warm_start_min_iter: int = 2,
                validate_warm_start_finite: bool = False,
                kl_thresh: float = 0.001, cos_thresh: float = 0.999,
                max_iter: int = 32, profile_coda_cost: bool = False,
                use_cached_final_output: bool = False,
                use_latent_precheck: bool = False,
                latent_precheck_thresh: float = 0.12,
                latent_precheck_min_iter: int = 2,
                latent_precheck_force_interval: int = 0,
                latent_precheck_mode: str = "legacy",
                latent_precheck_trace_level: str = "off",
                latent_precheck_warm_thresh: float = None,
                latent_precheck_max_skip_iters: int = 0,
                latent_precheck_confirmation_mode: str = "next_iter",
                nonfinite_policy: str = "legacy",
                shadow_full_depth: bool = False,
                capture_action_head_workload: bool = False,
                latent_only_metric: str = "raw_mse",
                latent_only_cold_threshold: float = 0.0,
                latent_only_warm_threshold: float = 0.0,
                latent_only_min_iter: int = 2,
                latent_only_eps: float = 1e-8,
                **kwargs) -> torch.Tensor:
        requested_recurrence_strategy = convergence_strategy
        canonical_recurrence_strategy = canonicalize_recurrence_strategy(convergence_strategy)
        validate_latent_only_configuration(
            convergence_strategy,
            metric=latent_only_metric,
            cold_threshold=latent_only_cold_threshold,
            warm_threshold=latent_only_warm_threshold,
            min_iter=latent_only_min_iter,
            eps=latent_only_eps,
            use_latent_precheck=use_latent_precheck,
            latent_precheck_mode=latent_precheck_mode,
            shadow_full_depth=shadow_full_depth,
            use_cached_final_output=use_cached_final_output,
        )
        latent_precheck_mode = validate_latent_precheck_configuration(
            latent_precheck_mode,
            latent_precheck_trace_level,
            use_latent_precheck,
            origin_aware_implemented=True,
            warm_threshold=latent_precheck_warm_thresh,
            max_skip_iters=latent_precheck_max_skip_iters,
            confirmation_mode=latent_precheck_confirmation_mode,
            warm_start_source=warm_start_source,
            recurrence_strategy=convergence_strategy,
            use_warm_start=enable_warm_start,
            min_iter=latent_precheck_min_iter,
            nonfinite_policy=nonfinite_policy,
            shadow_full_depth=shadow_full_depth,
        )
        if latent_precheck_mode == "origin_aware" and self.training:
            raise ValueError("latent_precheck_mode='origin_aware' is inference-only")
        if shadow_full_depth and self.training:
            raise ValueError("shadow_full_depth is inference-only")
        if canonical_recurrence_strategy == "latent_only" and self.training:
            raise ValueError("recurrence_strategy='latent_only' is inference-only")

        B = h_a.size(0)
        device, dtype = h_a.device, h_a.dtype

        with rdvla_range("RDVLA/action_head/action_queries"):
            x = self.action_queries.unsqueeze(0).expand(B, -1, -1).to(dtype=dtype)

        with rdvla_range("RDVLA/action_head/prelude_total"):
            if self.prelude_vlm_layers:
                for i, layer in enumerate(self.prelude):
                    with rdvla_range(f"RDVLA/action_head/prelude_layer_{i}"):
                        x = layer(x, h_a[:, self.prelude_vlm_layers[i]], h_t[:, self.prelude_vlm_layers[i]], p)
        prelude_out = x

        capture_warm_start_candidates = bool(enable_warm_start and not self.training)
        if capture_warm_start_candidates:
            state, warm_start_metadata = self._select_initial_state(
                warm_start_state,
                B,
                device,
                dtype,
                validate_warm_start_finite=(
                    validate_warm_start_finite
                    or latent_precheck_mode == "origin_aware"
                    or canonical_recurrence_strategy == "latent_only"
                ),
                validate_warm_start_dtype=latent_precheck_mode == "origin_aware",
            )
        else:
            state = self.init_state(B, device, dtype)
            warm_start_metadata = {
                "state_provided": False,
                "state_used": False,
                "initial_state_origin": "random",
                "reset": False,
                "reset_reason": None,
            }
        warm_start_metadata.update(
            {
                "source": None,
                "source_index": None,
                "source_iteration": None,
                "source_K": None,
                "candidate_state_count": None,
            }
        )
        warm_start_candidate_states = [] if capture_warm_start_candidates else None
        self.last_inference_metadata = {
            "next_warm_start_state": None,
            "warm_start": warm_start_metadata,
        }
        if capture_action_head_workload:
            self.last_inference_metadata["_workload_selected_initial_state"] = (
                state.detach().clone()
            )
        cached_state_used = bool(warm_start_metadata.get("state_used", False))
        warm_start_min_iter_configured = int(warm_start_min_iter)
        effective_min_iter = (
            max(2, warm_start_min_iter_configured)
            if cached_state_used
            else 2
        )
        self.last_recurrence_debug = None
        self._last_get_output_timing = None

        # Convergence-based stopping
        # 이게 원본 adaptive branch

        # if convergence_strategy in ("kl_divergence", "cosine_similarity") and not self.training:
        #     prev_output = None
        #     actual_iter = 0
        #     final_kl = None
        #     with torch.no_grad():
        #         for it in range(max_iter):
        #             state = self._run_one_iteration(state, prelude_out, h_a, h_t, p)
        #             actual_iter = it + 1
        #             curr_output = self._get_output(state, h_a, h_t, p)

        #             if prev_output is not None:
        #                 if convergence_strategy == "cosine_similarity":
        #                     cos_sim = F.cosine_similarity(
        #                         prev_output.flatten(), curr_output.flatten(), dim=0
        #                     ).item()
        #                     final_kl = 1 - cos_sim
        #                     if cos_sim > cos_thresh:
        #                         break
        #                 elif convergence_strategy == "kl_divergence":
        #                     mse = torch.mean((curr_output - prev_output) ** 2).item()
        #                     final_kl = mse
        #                     if mse < kl_thresh:
        #                         break
        #             prev_output = curr_output

        #     return self._get_output(state, h_a, h_t, p), actual_iter, final_kl

        # 아래는 측정용 metric 추가를 위해 수정한 adaptive branch


        if canonical_recurrence_strategy == "latent_only" and not self.training:
            actual_origin = "ACTUAL_WARM" if cached_state_used else "COLD"
            return run_latent_only_adaptive(
                self,
                state,
                prelude_out,
                h_a,
                h_t,
                p,
                max_iter=max_iter,
                metric_name=latent_only_metric,
                cold_threshold=latent_only_cold_threshold,
                warm_threshold=latent_only_warm_threshold,
                min_iter=latent_only_min_iter,
                eps=latent_only_eps,
                actual_origin=actual_origin,
                requested_recurrence_strategy=requested_recurrence_strategy,
                profile_coda_cost=profile_coda_cost,
                capture_warm_start_candidates=capture_warm_start_candidates,
                warm_start_candidate_states=warm_start_candidate_states,
                warm_start_source=warm_start_source,
                warm_start_min_iter_configured=warm_start_min_iter_configured,
            )


        if (
            canonical_recurrence_strategy == "adjacent_action_mse"
            and latent_precheck_mode == "origin_aware"
            and not self.training
        ):
            actual_origin = "ACTUAL_WARM" if cached_state_used else "COLD"
            first_warm_metadata = dict(warm_start_metadata)
            try:
                return run_origin_aware_adaptive(
                    self,
                    state,
                    prelude_out,
                    h_a,
                    h_t,
                    p,
                    max_iter=max_iter,
                    action_mse_threshold=kl_thresh,
                    effective_min_iter=effective_min_iter,
                    warm_start_min_iter_configured=warm_start_min_iter_configured,
                    warm_threshold=float(latent_precheck_warm_thresh),
                    latent_precheck_min_iter=latent_precheck_min_iter,
                    max_skip_iters=latent_precheck_max_skip_iters,
                    confirmation_mode=latent_precheck_confirmation_mode,
                    trace_level=latent_precheck_trace_level,
                    actual_origin=actual_origin,
                    requested_recurrence_strategy=requested_recurrence_strategy,
                    profile_coda_cost=profile_coda_cost,
                    use_cached_final_output_requested=use_cached_final_output,
                    capture_warm_start_candidates=capture_warm_start_candidates,
                    warm_start_candidate_states=warm_start_candidate_states,
                    warm_start_source=warm_start_source,
                )
            except NonFiniteOriginAwareInferenceError as first_error:
                return run_cold_full_coda_retry(
                    self,
                    prelude_out,
                    h_a,
                    h_t,
                    p,
                    max_iter=max_iter,
                    action_mse_threshold=kl_thresh,
                    requested_recurrence_strategy=requested_recurrence_strategy,
                    profile_coda_cost=profile_coda_cost,
                    trace_level=latent_precheck_trace_level,
                    warm_threshold=float(latent_precheck_warm_thresh),
                    latent_precheck_min_iter=latent_precheck_min_iter,
                    max_skip_iters=latent_precheck_max_skip_iters,
                    confirmation_mode=latent_precheck_confirmation_mode,
                    first_error=first_error,
                    first_attempt_origin=actual_origin,
                    first_warm_metadata=first_warm_metadata,
                )

        if canonical_recurrence_strategy in ("adjacent_action_mse", "cosine_similarity") and not self.training:
            prev_state = None
            prev_output = None
            actual_iter = 0
            final_kl = None
            conv_score_list = []
            action_delta_list = []
            latent_mse_list = []
            latent_l2_list = []
            latent_action_mse_pairs = []
            latent_precheck_coda_call_mask = []
            latent_precheck_skipped_iters = []
            latent_precheck_called_iters = []
            latent_precheck_decisions = []
            first_converged_k_1e_4 = None
            first_converged_k_5e_4 = None
            adaptive_stop = False
            stop_reason = None
            curr_output = None
            min_iter_gate_block_count = 0
            first_threshold_satisfied_k = None
            shadow_trace_records = []
            shadow_error = None

            run_one_iteration_ms_list = []
            # Output timing lists include the final return-path call unless the
            # cached-final-output option reuses the last adaptive-loop result.
            get_output_ms_list = []
            coda_ms_list = []
            output_proj_ms_list = []
            convergence_check_ms_list = []

            def append_get_output_timing():
                timing = self._last_get_output_timing
                get_output_ms_list.append(timing["get_output_ms"])
                coda_ms_list.append(timing["coda_ms"])
                output_proj_ms_list.append(timing["output_proj_ms"])

            with rdvla_range("RDVLA/action_head/recurrent_loop_total"):
                with torch.no_grad():
                    for it in range(max_iter):
                        shadow_previous_state = state.detach().clone() if shadow_full_depth else None
                        if profile_coda_cost:
                            run_one_iteration_start = self._sync_time()
                        state = self._run_one_iteration(state, prelude_out, h_a, h_t, p)
                        if profile_coda_cost:
                            run_one_iteration_end = self._sync_time()
                            run_one_iteration_ms_list.append((run_one_iteration_end - run_one_iteration_start) * 1000.0)
                        if capture_warm_start_candidates:
                            warm_start_candidate_states.append(state.detach())

                        actual_iter = it + 1
                        latent_mse = None
                        latent_l2 = None
                        if latent_precheck_mode == "legacy":
                            with rdvla_range("RDVLA/action_head/latent_precheck_total"):
                                if prev_state is not None:
                                    with rdvla_range("RDVLA/action_head/latent_precheck/mse_compute"):
                                        latent_diff = state.float() - prev_state.float()
                                        latent_mse_tensor = torch.mean(latent_diff ** 2)
                                        latent_l2_tensor = torch.norm(latent_diff.flatten())
                                    with rdvla_range("RDVLA/action_head/latent_precheck/item_sync"):
                                        latent_mse = latent_mse_tensor.item()
                                        latent_l2 = latent_l2_tensor.item()
                                    latent_mse_list.append(latent_mse)
                                    latent_l2_list.append(latent_l2)

                                if not use_latent_precheck:
                                    should_call_coda = True
                                    precheck_reason = "disabled"
                                elif actual_iter == 1:
                                    should_call_coda = True
                                    precheck_reason = "first_iter"
                                elif actual_iter == max_iter:
                                    should_call_coda = True
                                    precheck_reason = "max_iter"
                                elif latent_precheck_force_interval > 0 and actual_iter % latent_precheck_force_interval == 0:
                                    should_call_coda = True
                                    precheck_reason = "force_interval"
                                elif (
                                    latent_mse is not None
                                    and actual_iter >= latent_precheck_min_iter
                                    and latent_mse <= latent_precheck_thresh
                                ):
                                    should_call_coda = True
                                    precheck_reason = "latent_below_thresh"
                                else:
                                    should_call_coda = False
                                    precheck_reason = "skip_latent_above_thresh"

                            latent_precheck_coda_call_mask.append(bool(should_call_coda))
                            if should_call_coda:
                                latent_precheck_called_iters.append(int(actual_iter))
                            else:
                                latent_precheck_skipped_iters.append(int(actual_iter))
                        else:
                            should_call_coda = True

                        action_mse = None
                        action_l2 = None
                        with rdvla_range("RDVLA/action_head/coda_stop_check_total"):
                            if should_call_coda:
                                with rdvla_range("RDVLA/action_head/coda_stop_get_output"):
                                    with rdvla_range("RDVLA/action_head/get_output_each_iter"):
                                        curr_output = self._get_output(state, h_a, h_t, p, profile=profile_coda_cost)
                                if profile_coda_cost:
                                    append_get_output_timing()
                                    convergence_check_start = self._sync_time()

                                if prev_output is not None:
                                    with rdvla_range("RDVLA/action_head/stop_check/mse_compute"):
                                        diff = curr_output - prev_output
                                        action_mse_tensor = torch.mean(diff ** 2)
                                    with rdvla_range("RDVLA/action_head/stop_check/item_sync"):
                                        action_mse = action_mse_tensor.item()
                                    with rdvla_range("RDVLA/action_head/stop_check/mse_compute"):
                                        action_l2_tensor = torch.norm(diff.float())
                                    with rdvla_range("RDVLA/action_head/stop_check/item_sync"):
                                        action_l2 = action_l2_tensor.item()

                                    conv_score_list.append(action_mse)
                                    action_delta_list.append(action_l2)
                                    if latent_mse is not None:
                                        latent_action_mse_pairs.append({
                                            "k": int(actual_iter),
                                            "latent_mse": float(latent_mse),
                                            "latent_l2": float(latent_l2),
                                            "action_mse": float(action_mse),
                                            "action_l2": float(action_l2),
                                        })

                                    with rdvla_range("RDVLA/action_head/stop_check/condition"):
                                        if first_converged_k_1e_4 is None and action_mse < 1e-4:
                                            first_converged_k_1e_4 = actual_iter
                                        if first_converged_k_5e_4 is None and action_mse < 5e-4:
                                            first_converged_k_5e_4 = actual_iter

                                        if canonical_recurrence_strategy == "cosine_similarity":
                                            cos_sim_tensor = F.cosine_similarity(
                                                prev_output.flatten(), curr_output.flatten(), dim=0
                                            )
                                            with rdvla_range("RDVLA/action_head/stop_check/item_sync"):
                                                cos_sim = cos_sim_tensor.item()
                                            final_kl = 1 - cos_sim
                                            if cos_sim > cos_thresh:
                                                if first_threshold_satisfied_k is None:
                                                    first_threshold_satisfied_k = actual_iter
                                                if actual_iter >= effective_min_iter:
                                                    with rdvla_range("RDVLA/action_head/stop_reason_update"):
                                                        adaptive_stop = True
                                                        stop_reason = "cosine_similarity"
                                                else:
                                                    min_iter_gate_block_count += 1
                                        elif canonical_recurrence_strategy == "adjacent_action_mse":
                                            final_kl = action_mse
                                            if action_mse < kl_thresh:
                                                if first_threshold_satisfied_k is None:
                                                    first_threshold_satisfied_k = actual_iter
                                                if actual_iter >= effective_min_iter:
                                                    with rdvla_range("RDVLA/action_head/stop_reason_update"):
                                                        adaptive_stop = True
                                                        stop_reason = requested_recurrence_strategy
                                                else:
                                                    min_iter_gate_block_count += 1

                                if shadow_full_depth:
                                    shadow_trace_records.append(
                                        build_shadow_trace_record(
                                            iteration=actual_iter,
                                            phase="production",
                                            previous_state=shadow_previous_state,
                                            current_state=state,
                                            previous_output=prev_output,
                                            current_output=curr_output,
                                            eps=latent_only_eps,
                                        )
                                    )
                                    shadow_record = shadow_trace_records[-1]
                                    if not shadow_record["state_finite"]:
                                        shadow_error = {
                                            "iteration": int(actual_iter),
                                            "stage": "production_state",
                                            "reason": "non_finite",
                                        }
                                    elif not shadow_record["output_finite"]:
                                        shadow_error = {
                                            "iteration": int(actual_iter),
                                            "stage": "production_output",
                                            "reason": "non_finite",
                                        }

                                prev_output = curr_output.detach()
                                if profile_coda_cost:
                                    convergence_check_end = self._sync_time()
                                    convergence_check_ms_list.append((convergence_check_end - convergence_check_start) * 1000.0)

                        if latent_precheck_mode == "legacy":
                            latent_precheck_decisions.append({
                                "k": int(actual_iter),
                                "latent_mse": float(latent_mse) if latent_mse is not None else None,
                                "call_coda": bool(should_call_coda),
                                "reason": precheck_reason,
                                "action_mse": float(action_mse) if action_mse is not None else None,
                            })
                            prev_state = state.detach()
                        if adaptive_stop:
                            break

            if stop_reason is None and actual_iter >= max_iter:
                with rdvla_range("RDVLA/action_head/stop_reason_update"):
                    stop_reason = "max_iter"

            with rdvla_range("RDVLA/action_head/final_get_output"):
                if use_cached_final_output and curr_output is not None:
                    final_output = curr_output
                else:
                    final_output = self._get_output(state, h_a, h_t, p, profile=profile_coda_cost)
                    if profile_coda_cost:
                        append_get_output_timing()

            # Freeze every production-visible value before diagnostic recurrence.
            self._store_warm_start_candidate(
                warm_start_candidate_states, actual_iter, warm_start_source
            )
            if shadow_full_depth:
                final_output = final_output.detach().clone()
                production_snapshot = {
                    "K_t": int(actual_iter),
                    "terminal_iteration": int(actual_iter),
                    "stop_reason": stop_reason,
                    "midpoint_source_iteration": self.last_inference_metadata[
                        "warm_start"
                    ].get("source_iteration"),
                    "cached_final_output_reused": bool(
                        use_cached_final_output and curr_output is not None
                    ),
                }
            else:
                production_snapshot = None

            self.last_recurrence_debug = {
                "strategy": convergence_strategy,
                "requested_recurrence_strategy": requested_recurrence_strategy,
                "canonical_recurrence_strategy": canonical_recurrence_strategy,
                "canonical_metric_name": canonical_recurrence_strategy,
                "action_mse_threshold": float(kl_thresh) if canonical_recurrence_strategy == "adjacent_action_mse" else None,
                "threshold": float(kl_thresh) if canonical_recurrence_strategy == "adjacent_action_mse" else float(cos_thresh),
                "fixed_K": None,
                "K_t": int(actual_iter),
                "max_iter": int(max_iter),
                "adaptive_stop": adaptive_stop,
                "metric_name": "mse_between_action_outputs",
                "iteration_mse": conv_score_list,
                "iteration_metric_values": conv_score_list,
                "conv_score_list": conv_score_list,
                "action_delta_list": action_delta_list,
                "latent_mse_list": latent_mse_list,
                "latent_l2_list": latent_l2_list,
                "latent_action_mse_pairs": latent_action_mse_pairs,
                "latent_action_pair_count": len(latent_action_mse_pairs),
                "use_latent_precheck": bool(use_latent_precheck),
                "latent_precheck_mode": latent_precheck_mode,
                "latent_precheck_trace_level_requested": latent_precheck_trace_level,
                "latent_precheck_trace_level_applied": None if latent_precheck_mode == "legacy" else latent_precheck_trace_level,
                "latent_precheck_trace_collected": latent_precheck_mode == "legacy",
                "latent_precheck_thresh": float(latent_precheck_thresh),
                "latent_precheck_min_iter": int(latent_precheck_min_iter),
                "latent_precheck_force_interval": int(latent_precheck_force_interval),
                "latent_precheck_coda_call_mask": latent_precheck_coda_call_mask,
                "latent_precheck_skipped_iters": latent_precheck_skipped_iters,
                "latent_precheck_called_iters": latent_precheck_called_iters,
                "latent_precheck_skip_count": len(latent_precheck_skipped_iters) if latent_precheck_mode == "legacy" else None,
                "latent_precheck_call_count": len(latent_precheck_called_iters) if latent_precheck_mode == "legacy" else None,
                "latent_precheck_skip_ratio": (
                    len(latent_precheck_skipped_iters) / len(latent_precheck_coda_call_mask)
                    if latent_precheck_coda_call_mask
                    else (0.0 if latent_precheck_mode == "legacy" else None)
                ),
                "latent_precheck_decisions": latent_precheck_decisions,
                "first_converged_k_1e_4": first_converged_k_1e_4,
                "first_converged_k_5e_4": first_converged_k_5e_4,
                "final_mse": conv_score_list[-1] if conv_score_list else None,
                "final_conv_score": final_kl,
                "stop_reason": stop_reason,
                "canonical_stop_reason": canonical_recurrence_strategy if adaptive_stop else stop_reason,
                "profiling_enabled": bool(profile_coda_cost),
                "use_cached_final_output": bool(use_cached_final_output),
                "warm_start_min_iter_configured": warm_start_min_iter_configured,
                "effective_min_iter": int(effective_min_iter),
                "warm_start_state_used": cached_state_used,
                "min_iter_gate_block_count": int(min_iter_gate_block_count),
                "first_threshold_satisfied_k": first_threshold_satisfied_k,
                "shadow_full_depth_enabled": bool(shadow_full_depth),
                "latent_metric_trace_enabled": bool(shadow_full_depth),
                "latent_metric_trace_eps": float(latent_only_eps),
                "shadow_trace": shadow_trace_records,
                "shadow_trace_complete": False if shadow_full_depth else None,
                "shadow_tail_start_iteration": None,
                "shadow_tail_iteration_count": 0 if shadow_full_depth else None,
                "shadow_error": shadow_error,
                "shadow_production_snapshot": production_snapshot,
            }

            if profile_coda_cost:
                run_one_iteration_ms_total = sum(run_one_iteration_ms_list)
                get_output_ms_total = sum(get_output_ms_list)
                coda_ms_total = sum(coda_ms_list)
                output_proj_ms_total = sum(output_proj_ms_list)
                # Ratio denominator is recurrent-core time plus all _get_output() calls.
                # The uncached baseline therefore includes the final return-path Coda call.
                profiled_recurrent_ms_total = run_one_iteration_ms_total + get_output_ms_total
                self.last_recurrence_debug.update({
                    "run_one_iteration_ms_list": run_one_iteration_ms_list,
                    "get_output_ms_list": get_output_ms_list,
                    "coda_ms_list": coda_ms_list,
                    "output_proj_ms_list": output_proj_ms_list,
                    "convergence_check_ms_list": convergence_check_ms_list,
                    "get_output_call_count": len(get_output_ms_list),
                    "coda_ms_total": coda_ms_total,
                    "get_output_ms_total": get_output_ms_total,
                    "run_one_iteration_ms_total": run_one_iteration_ms_total,
                    "output_proj_ms_total": output_proj_ms_total,
                    "coda_time_ratio_total": (
                        coda_ms_total / profiled_recurrent_ms_total if profiled_recurrent_ms_total else 0.0
                    ),
                })

            if shadow_full_depth and shadow_error is None:
                shadow_start_output = (
                    curr_output if curr_output is not None else final_output
                )
                shadow_result = run_shadow_tail(
                    self,
                    state=state,
                    current_output=shadow_start_output,
                    actual_iter=actual_iter,
                    max_iter=max_iter,
                    prelude_out=prelude_out,
                    h_a=h_a,
                    h_t=h_t,
                    p=p,
                    eps=latent_only_eps,
                )
                shadow_trace_records.extend(shadow_result["records"])
                self.last_recurrence_debug.update(
                    {
                        "shadow_trace": shadow_trace_records,
                        "shadow_trace_complete": shadow_result["completed"],
                        "shadow_tail_start_iteration": shadow_result[
                            "tail_start_iteration"
                        ],
                        "shadow_tail_iteration_count": shadow_result[
                            "tail_iteration_count"
                        ],
                        "shadow_error": shadow_result["error"],
                    }
                )
            return final_output, actual_iter, final_kl

        # 여기까지가 metric 추가한 adaptive branch


        # 기존 Fixed branch
        # Fixed iterations
        # if num_iter is not None:
        #     total = num_iter
        # elif self.training and self.cfg.random_iterations:
        #     total = self.sample_iterations()
        # else:
        #     total = self.cfg.mean_recurrence

        # k = self.cfg.backprop_depth
        # n_no_grad = max(0, total - k)

        # if n_no_grad > 0:
        #     with torch.no_grad():
        #         for _ in range(n_no_grad):
        #             state = self._run_one_iteration(state, prelude_out, h_a, h_t, p)

        # for _ in range(min(k, total)):
        #     state = self._run_one_iteration(state, prelude_out, h_a, h_t, p)

        # return self._get_output(state, h_a, h_t, p)

        # metric 측정용 fixed branch
        if num_iter is not None:
            total = num_iter
        elif self.training and self.cfg.random_iterations:
            total = self.sample_iterations()
        else:
            total = self.cfg.mean_recurrence

        # Inference-time fixed logging
        if not self.training:
            prev_output = None
            conv_score_list = []
            action_delta_list = []
            first_converged_k_1e_4 = None
            first_converged_k_5e_4 = None
            final_conv_score = None
            curr_output = None
            actual_iter = 0

            with rdvla_range("RDVLA/action_head/recurrent_loop_total"):
                with torch.no_grad():
                    for it in range(total):
                        state = self._run_one_iteration(state, prelude_out, h_a, h_t, p)
                        if capture_warm_start_candidates:
                            warm_start_candidate_states.append(state.detach())
                        with rdvla_range("RDVLA/action_head/get_output_each_iter"):
                            curr_output = self._get_output(state, h_a, h_t, p)
                        actual_iter = it + 1

                        if prev_output is not None:
                            with rdvla_range("RDVLA/action_head/stop_check/mse_compute"):
                                diff = curr_output - prev_output
                                mse_tensor = torch.mean(diff ** 2)
                            with rdvla_range("RDVLA/action_head/stop_check/item_sync"):
                                mse = mse_tensor.item()
                            with rdvla_range("RDVLA/action_head/stop_check/mse_compute"):
                                l2_tensor = torch.norm(diff.float())
                            with rdvla_range("RDVLA/action_head/stop_check/item_sync"):
                                l2 = l2_tensor.item()

                            conv_score_list.append(mse)
                            action_delta_list.append(l2)
                            final_conv_score = mse

                            with rdvla_range("RDVLA/action_head/stop_check/condition"):
                                if first_converged_k_1e_4 is None and mse < 1e-4:
                                    first_converged_k_1e_4 = actual_iter
                                if first_converged_k_5e_4 is None and mse < 5e-4:
                                    first_converged_k_5e_4 = actual_iter

                        prev_output = curr_output.detach()

            self.last_recurrence_debug = {
                "strategy": "fixed",
                "requested_recurrence_strategy": requested_recurrence_strategy or "fixed",
                "canonical_recurrence_strategy": canonical_recurrence_strategy or "fixed",
                "canonical_metric_name": "fixed",
                "action_mse_threshold": None,
                "latent_precheck_mode": latent_precheck_mode,
                "latent_precheck_trace_level_requested": latent_precheck_trace_level,
                "latent_precheck_trace_level_applied": None if latent_precheck_mode == "legacy" else latent_precheck_trace_level,
                "latent_precheck_trace_collected": False,
                "threshold": None,
                "fixed_K": int(total),
                "K_t": int(total),
                "max_iter": int(total),
                "adaptive_stop": False,
                "metric_name": "mse_between_action_outputs",
                "iteration_mse": conv_score_list,
                "iteration_metric_values": conv_score_list,
                "conv_score_list": conv_score_list,
                "action_delta_list": action_delta_list,
                "first_converged_k_1e_4": first_converged_k_1e_4,
                "first_converged_k_5e_4": first_converged_k_5e_4,
                "final_mse": final_conv_score,
                "final_conv_score": final_conv_score,
                "stop_reason": None,
                "canonical_stop_reason": None,
                "warm_start_min_iter_configured": warm_start_min_iter_configured,
                "effective_min_iter": int(effective_min_iter),
                "warm_start_state_used": cached_state_used,
                "min_iter_gate_block_count": 0,
                "first_threshold_satisfied_k": None,
            }

            with rdvla_range("RDVLA/action_head/final_get_output"):
                final_output = curr_output
            self._store_warm_start_candidate(
                warm_start_candidate_states, actual_iter, warm_start_source
            )
            return final_output, actual_iter, final_conv_score

        # Training-time fixed branch: 기존 코드 유지
        k = self.cfg.backprop_depth
        n_no_grad = max(0, total - k)

        with rdvla_range("RDVLA/action_head/recurrent_loop_total"):
            if n_no_grad > 0:
                with torch.no_grad():
                    for _ in range(n_no_grad):
                        state = self._run_one_iteration(state, prelude_out, h_a, h_t, p)

            for _ in range(min(k, total)):
                state = self._run_one_iteration(state, prelude_out, h_a, h_t, p)

        with rdvla_range("RDVLA/action_head/final_get_output"):
            return self._get_output(state, h_a, h_t, p)

        #여기까지가 수정한 metric 측정용 fixed branch


class ActionHeadRecurrent(nn.Module):
    def __init__(self, hidden_dim=896, action_dim=7, cfg=None):
        super().__init__()
        if cfg is None:
            cfg = RecurrentConfigInternal(hidden_dim=hidden_dim, action_dim=action_dim)
        elif isinstance(cfg, dict):
            cfg = RecurrentConfigInternal(**cfg)
        self.cfg = cfg
        self.hidden_dim = hidden_dim
        self.action_dim = action_dim
        self.num_task_tokens = 512
        self.model = VLARecurrent(cfg)

    def forward(self, x, h_a=None, h_t=None, p=None, num_iter=None, **kwargs):
        return self.model(h_a, h_t, p, num_iter=num_iter, **kwargs)

    def predict_action(self, actions_hidden_states, proprio=None, proprio_projector=None,
                       phase="Inference", num_iter=None, convergence_strategy=None,
                       kl_thresh=0.001, cos_thresh=0.999, max_iter=32,
                       warm_start_state=None, enable_warm_start: bool = False,
                       warm_start_source: str = "s1",
                       warm_start_min_iter: int = 2,
                       validate_warm_start_finite: bool = False,
                       profile_coda_cost=False, use_cached_final_output=False,
                       use_latent_precheck=False, latent_precheck_thresh=0.12,
                       latent_precheck_min_iter=2, latent_precheck_force_interval=0,
                       latent_precheck_mode="legacy", latent_precheck_trace_level="off",
                       latent_precheck_warm_thresh=None,
                       latent_precheck_max_skip_iters=0,
                       latent_precheck_confirmation_mode="next_iter",
                       nonfinite_policy="legacy",
                       shadow_full_depth=False,
                       capture_action_head_workload=False,
                       latent_only_metric="raw_mse",
                       latent_only_cold_threshold=0.0,
                       latent_only_warm_threshold=0.0,
                       latent_only_min_iter=2,
                       latent_only_eps=1e-8,
                       **kwargs):
        canonicalize_recurrence_strategy(convergence_strategy)
        validate_latent_only_configuration(
            convergence_strategy,
            metric=latent_only_metric,
            cold_threshold=latent_only_cold_threshold,
            warm_threshold=latent_only_warm_threshold,
            min_iter=latent_only_min_iter,
            eps=latent_only_eps,
            use_latent_precheck=use_latent_precheck,
            latent_precheck_mode=latent_precheck_mode,
            shadow_full_depth=shadow_full_depth,
            use_cached_final_output=use_cached_final_output,
        )
        validate_latent_precheck_configuration(
            latent_precheck_mode,
            latent_precheck_trace_level,
            use_latent_precheck,
            origin_aware_implemented=True,
            warm_threshold=latent_precheck_warm_thresh,
            max_skip_iters=latent_precheck_max_skip_iters,
            confirmation_mode=latent_precheck_confirmation_mode,
            warm_start_source=warm_start_source,
            recurrence_strategy=convergence_strategy,
            use_warm_start=enable_warm_start,
            min_iter=latent_precheck_min_iter,
            nonfinite_policy=nonfinite_policy,
            shadow_full_depth=shadow_full_depth,
        )
        with rdvla_range("RDVLA/action_head/wrapper_total"):
            B = actions_hidden_states.shape[0]
            proprio_input = proprio.reshape(B, -1).to(torch.bfloat16)
            with rdvla_range("RDVLA/action_head/proprio_projector"):
                proprio_features = proprio_projector(proprio_input).unsqueeze(1)
            with rdvla_range("RDVLA/action_head/split_h_t_h_a"):
                h_t = actions_hidden_states[:, :, :self.num_task_tokens, :]
                h_a = actions_hidden_states[:, :, self.num_task_tokens:, :]
            with rdvla_range("RDVLA/action_head/vla_recurrent_total"):
                result = self.model(h_a, h_t, proprio_features, num_iter=num_iter,
                                 convergence_strategy=convergence_strategy, kl_thresh=kl_thresh,
                                 cos_thresh=cos_thresh, max_iter=max_iter,
                                 warm_start_state=warm_start_state,
                                 enable_warm_start=enable_warm_start,
                                 warm_start_source=warm_start_source,
                                 warm_start_min_iter=warm_start_min_iter,
                                 validate_warm_start_finite=validate_warm_start_finite,
                                 profile_coda_cost=profile_coda_cost,
                                 use_cached_final_output=use_cached_final_output,
                                 use_latent_precheck=use_latent_precheck,
                                 latent_precheck_mode=latent_precheck_mode,
                                 latent_precheck_trace_level=latent_precheck_trace_level,
                                 latent_precheck_thresh=latent_precheck_thresh,
                                 latent_precheck_min_iter=latent_precheck_min_iter,
                                 latent_precheck_force_interval=latent_precheck_force_interval,
                                 latent_precheck_warm_thresh=latent_precheck_warm_thresh,
                                 latent_precheck_max_skip_iters=latent_precheck_max_skip_iters,
                                 latent_precheck_confirmation_mode=latent_precheck_confirmation_mode,
                                 nonfinite_policy=nonfinite_policy,
                                 shadow_full_depth=shadow_full_depth,
                                 capture_action_head_workload=capture_action_head_workload,
                                 latent_only_metric=latent_only_metric,
                                 latent_only_cold_threshold=latent_only_cold_threshold,
                                 latent_only_warm_threshold=latent_only_warm_threshold,
                                 latent_only_min_iter=latent_only_min_iter,
                                 latent_only_eps=latent_only_eps)
            if capture_action_head_workload:
                metadata = self.model.last_inference_metadata
                selected_initial_state = metadata.pop("_workload_selected_initial_state")
                actual_origin = (
                    "ACTUAL_WARM"
                    if metadata["warm_start"].get("state_used") is True
                    else "COLD"
                )
                metadata["action_head_workload"] = build_action_head_workload(
                    actions_hidden_states=actions_hidden_states,
                    proprio_input=proprio_input,
                    proprio_features=proprio_features,
                    incoming_warm_start_state=warm_start_state,
                    selected_initial_state=selected_initial_state,
                    actual_origin=actual_origin,
                )
            return result
