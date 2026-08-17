import math
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from configs.rdvla_precheck import (
    canonicalize_recurrence_strategy,
    validate_fixed_terminal_only_configuration,
    validate_latent_only_configuration,
    validate_latent_precheck_configuration,
)
from prismatic.models.fixed_terminal_only import run_fixed_terminal_only
from prismatic.models.latent_only_stopping import run_latent_only_adaptive
from prismatic.models.scalar_policy_stopping_runtime import (
    run_scalar_policy_adaptive,
)
from prismatic.models.scalar_stopping_policy import (
    PreparedScalarTaskPolicy,
    SUPPORTED_SCALAR_EXECUTION_MODES,
    validate_scalar_runtime_configuration,
)
from prismatic.models.action_delta_gate import (
    ACTION_DELTA_GATE_DIAGNOSTIC_RETURN_MODES,
    ACTION_DELTA_NONCONVERGENCE_CODA_COST_MS,
    ACTION_DELTA_NONCONVERGENCE_MIN_TERMINAL_ITER,
    ACTION_DELTA_NONCONVERGENCE_SCORER_COST_MS,
    ACTION_DELTA_NONCONVERGENCE_THRESHOLD,
    ActionDeltaGateCorrectionError,
    NonFiniteActionDeltaGateError,
    PreparedActionDeltaGate,
    build_action_delta_gate_corrected_output,
    evaluate_action_delta_gate,
    validate_action_delta_deferred_backfill_configuration,
    validate_action_delta_nonconvergence_filter_configuration,
    validate_action_delta_gate_runtime_configuration,
)
from prismatic.models.action_delta_gate_shadow import (
    ACTION_DELTA_GATE_SHADOW_SCHEMA_VERSION,
    build_action_delta_gate_shadow_transition,
    validate_action_delta_gate_shadow_configuration,
)
from prismatic.models.action_head_workload import build_action_head_workload
from prismatic.models.origin_aware_scheduler import (
    NonFiniteOriginAwareInferenceError,
    run_origin_aware_adaptive,
)
from prismatic.models.numerical_retry import run_cold_full_coda_retry
from prismatic.models.shadow_trace import (
    build_shadow_trace_record,
    capture_raw_shadow_tensor,
    run_shadow_tail,
)
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


def _action_delta_gate_exact_audit_metrics(
    anchor_output,
    terminal_output,
    predicted_delta_action,
):
    """Build JSON-safe float32 diagnostics for one exact action transition."""

    if not all(
        torch.is_tensor(value)
        for value in (
            anchor_output,
            terminal_output,
            predicted_delta_action,
        )
    ):
        raise ValueError("exact audit actions must be tensors")
    if anchor_output.ndim not in (2, 3):
        raise ValueError(
            "exact audit anchor action must have rank 2 or rank 3 with "
            "a leading batch dimension"
        )
    if terminal_output.ndim != anchor_output.ndim:
        raise ValueError(
            "exact audit action rank mismatch: "
            f"anchor={anchor_output.ndim}, terminal={terminal_output.ndim}"
        )
    if tuple(terminal_output.shape) != tuple(anchor_output.shape):
        raise ValueError(
            "exact audit action shape mismatch: "
            f"anchor={tuple(anchor_output.shape)}, "
            f"terminal={tuple(terminal_output.shape)}"
        )
    if tuple(predicted_delta_action.shape) != tuple(anchor_output.shape):
        raise ValueError(
            "exact audit predicted delta shape mismatch: "
            f"anchor={tuple(anchor_output.shape)}, "
            f"predicted_delta={tuple(predicted_delta_action.shape)}"
        )
    has_leading_batch = anchor_output.ndim == 3
    if has_leading_batch and anchor_output.shape[0] != 1:
        raise ValueError(
            "exact audit leading action batch dimension must have size 1"
        )

    anchor = anchor_output.detach().float()
    terminal = terminal_output.detach().float()
    predicted_delta = predicted_delta_action.detach().float()
    if not bool(torch.isfinite(anchor).all().item()):
        raise ValueError("exact audit anchor action is non-finite")
    if not bool(torch.isfinite(terminal).all().item()):
        raise ValueError("exact audit terminal action is non-finite")
    if not bool(torch.isfinite(predicted_delta).all().item()):
        raise ValueError("exact audit predicted delta action is non-finite")

    delta = terminal - anchor
    predicted_corrected = anchor + predicted_delta
    correction_error = terminal - predicted_corrected
    if not bool(torch.isfinite(delta).all().item()):
        raise ValueError("exact audit delta action is non-finite")
    if not bool(torch.isfinite(predicted_corrected).all().item()):
        raise ValueError("exact audit predicted corrected action is non-finite")
    if not bool(torch.isfinite(correction_error).all().item()):
        raise ValueError("exact audit correction error is non-finite")
    metric_delta = delta.squeeze(0) if has_leading_batch else delta
    metric_correction_error = (
        correction_error.squeeze(0) if has_leading_batch else correction_error
    )
    squared = metric_delta.square()
    absolute = metric_delta.abs()
    correction_squared = metric_correction_error.square()
    correction_absolute = metric_correction_error.abs()
    full_mse = squared.mean()
    l2 = torch.linalg.vector_norm(metric_delta)
    max_abs = absolute.max()
    per_step_mse = squared.mean(dim=-1)
    per_step_max_abs = absolute.amax(dim=-1)
    per_dim_mse = squared.mean(dim=-2)
    per_dim_max_abs = absolute.amax(dim=-2)
    correction_full_mse = correction_squared.mean()
    correction_l2 = torch.linalg.vector_norm(metric_correction_error)
    correction_max_abs = correction_absolute.max()
    correction_per_step_mse = correction_squared.mean(dim=-1)
    correction_per_step_max_abs = correction_absolute.amax(dim=-1)
    correction_per_dim_mse = correction_squared.mean(dim=-2)
    correction_per_dim_max_abs = correction_absolute.amax(dim=-2)
    prefix_step_count = min(5, int(metric_delta.shape[-2]))
    anchor_reuse_prefix_mse = squared[:prefix_step_count].mean()
    correction_prefix_mse = correction_squared[:prefix_step_count].mean()
    metric_tensors = (
        full_mse,
        l2,
        max_abs,
        per_step_mse,
        per_step_max_abs,
        per_dim_mse,
        per_dim_max_abs,
        correction_full_mse,
        correction_l2,
        correction_max_abs,
        correction_per_step_mse,
        correction_per_step_max_abs,
        correction_per_dim_mse,
        correction_per_dim_max_abs,
        anchor_reuse_prefix_mse,
        correction_prefix_mse,
    )
    if not all(bool(torch.isfinite(value).all().item()) for value in metric_tensors):
        raise ValueError("exact audit action metrics are non-finite")

    def safe_ratio(numerator, denominator):
        numerator_value = float(numerator.item())
        denominator_value = float(denominator.item())
        if denominator_value > 0.0:
            return numerator_value / denominator_value
        return 1.0 if numerator_value == 0.0 else None

    return {
        "action_delta_gate_exact_audit_action_shape": list(anchor.shape),
        "action_delta_gate_exact_audit_metric_action_shape": list(
            metric_delta.shape
        ),
        "action_delta_gate_exact_audit_leading_batch_dim_squeezed": bool(
            has_leading_batch
        ),
        "action_delta_gate_exact_audit_full_mse": float(full_mse.item()),
        "action_delta_gate_exact_audit_l2": float(l2.item()),
        "action_delta_gate_exact_audit_max_abs": float(max_abs.item()),
        "action_delta_gate_exact_audit_per_step_mse": (
            per_step_mse.cpu().tolist()
        ),
        "action_delta_gate_exact_audit_per_step_max_abs": (
            per_step_max_abs.cpu().tolist()
        ),
        "action_delta_gate_exact_audit_per_dim_mse": (
            per_dim_mse.cpu().tolist()
        ),
        "action_delta_gate_exact_audit_per_dim_max_abs": (
            per_dim_max_abs.cpu().tolist()
        ),
        "action_delta_gate_exact_audit_anchor_action": anchor.cpu().tolist(),
        "action_delta_gate_exact_audit_terminal_action": terminal.cpu().tolist(),
        "action_delta_gate_exact_audit_delta_action": delta.cpu().tolist(),
        "action_delta_gate_exact_audit_predicted_delta_action": (
            predicted_delta.cpu().tolist()
        ),
        "action_delta_gate_exact_audit_predicted_corrected_action": (
            predicted_corrected.cpu().tolist()
        ),
        "action_delta_gate_exact_audit_correction_full_mse": float(
            correction_full_mse.item()
        ),
        "action_delta_gate_exact_audit_correction_l2": float(
            correction_l2.item()
        ),
        "action_delta_gate_exact_audit_correction_max_abs": float(
            correction_max_abs.item()
        ),
        "action_delta_gate_exact_audit_correction_per_step_mse": (
            correction_per_step_mse.cpu().tolist()
        ),
        "action_delta_gate_exact_audit_correction_per_step_max_abs": (
            correction_per_step_max_abs.cpu().tolist()
        ),
        "action_delta_gate_exact_audit_correction_per_dim_mse": (
            correction_per_dim_mse.cpu().tolist()
        ),
        "action_delta_gate_exact_audit_correction_per_dim_max_abs": (
            correction_per_dim_max_abs.cpu().tolist()
        ),
        "action_delta_gate_exact_audit_prefix_step_count": prefix_step_count,
        "action_delta_gate_exact_audit_anchor_reuse_prefix_mse": float(
            anchor_reuse_prefix_mse.item()
        ),
        "action_delta_gate_exact_audit_correction_prefix_mse": float(
            correction_prefix_mse.item()
        ),
        "action_delta_gate_exact_audit_correction_full_mse_ratio": safe_ratio(
            correction_full_mse,
            full_mse,
        ),
        "action_delta_gate_exact_audit_correction_prefix_mse_ratio": safe_ratio(
            correction_prefix_mse,
            anchor_reuse_prefix_mse,
        ),
    }


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
                collect_preconvergence_raw_shadow: bool = False,
                preconvergence_raw_shadow_max_depth: int = 32,
                capture_action_head_workload: bool = False,
                latent_only_metric: str = "raw_mse",
                latent_only_cold_threshold: float = 0.0,
                latent_only_warm_threshold: float = 0.0,
                latent_only_min_iter: int = 2,
                latent_only_eps: float = 1e-8,
                scalar_task_policy=None,
                scalar_policy_execution_mode: str = "direct",
                use_action_delta_gate: bool = False,
                action_delta_gate=None,
                action_delta_gate_max_skip: int = 1,
                action_delta_gate_min_terminal_iter: int = 2,
                action_delta_gate_exact_coda_audit: bool = False,
                action_delta_gate_return_mode: str = "anchor",
                collect_action_delta_gate_shadow: bool = False,
                use_action_delta_nonconvergence_filter: bool = False,
                use_action_delta_deferred_backfill_filter: bool = False,
                **kwargs) -> torch.Tensor:
        requested_recurrence_strategy = convergence_strategy
        canonical_recurrence_strategy = canonicalize_recurrence_strategy(convergence_strategy)
        validate_fixed_terminal_only_configuration(
            canonical_recurrence_strategy,
            recurrent_num_iter=num_iter,
            recurrence_max_iter=max_iter,
        )
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
        validate_scalar_runtime_configuration(
            canonical_recurrence_strategy,
            task_policy=scalar_task_policy,
            execution_mode=scalar_policy_execution_mode,
            use_warm_start=enable_warm_start,
            warm_start_source=warm_start_source,
            use_latent_precheck=use_latent_precheck,
            latent_precheck_mode=latent_precheck_mode,
            latent_precheck_trace_level=latent_precheck_trace_level,
            shadow_full_depth=shadow_full_depth,
            collect_preconvergence_raw_shadow=(
                collect_preconvergence_raw_shadow
            ),
            use_cached_final_output=use_cached_final_output,
            max_iter=max_iter,
        )
        validate_action_delta_gate_runtime_configuration(
            enabled=use_action_delta_gate,
            canonical_recurrence_strategy=canonical_recurrence_strategy,
            prepared_gate=action_delta_gate,
            batch_size=h_a.size(0),
            use_warm_start=enable_warm_start,
            warm_start_source=warm_start_source,
            warm_start_min_iter=warm_start_min_iter,
            use_latent_precheck=use_latent_precheck,
            latent_precheck_mode=latent_precheck_mode,
            latent_precheck_trace_level=latent_precheck_trace_level,
            shadow_full_depth=shadow_full_depth,
            collect_preconvergence_raw_shadow=collect_preconvergence_raw_shadow,
            use_cached_final_output=use_cached_final_output,
            max_skip=action_delta_gate_max_skip,
            min_terminal_iter=action_delta_gate_min_terminal_iter,
            exact_coda_audit=action_delta_gate_exact_coda_audit,
            return_mode=action_delta_gate_return_mode,
        )
        validate_action_delta_gate_shadow_configuration(
            enabled=collect_action_delta_gate_shadow,
            production_gate_enabled=use_action_delta_gate,
            canonical_recurrence_strategy=canonical_recurrence_strategy,
            prepared_gate=action_delta_gate,
            batch_size=h_a.size(0),
            use_warm_start=enable_warm_start,
            warm_start_source=warm_start_source,
            warm_start_min_iter=warm_start_min_iter,
            use_latent_precheck=use_latent_precheck,
            latent_precheck_mode=latent_precheck_mode,
            latent_precheck_trace_level=latent_precheck_trace_level,
            shadow_full_depth=shadow_full_depth,
            collect_preconvergence_raw_shadow=collect_preconvergence_raw_shadow,
            use_cached_final_output=use_cached_final_output,
            min_terminal_iter=action_delta_gate_min_terminal_iter,
        )
        validate_action_delta_nonconvergence_filter_configuration(
            enabled=use_action_delta_nonconvergence_filter,
            production_gate_enabled=use_action_delta_gate,
            shadow_collection_enabled=collect_action_delta_gate_shadow,
            canonical_recurrence_strategy=canonical_recurrence_strategy,
            prepared_gate=action_delta_gate,
            batch_size=h_a.size(0),
            use_warm_start=enable_warm_start,
            warm_start_source=warm_start_source,
            warm_start_min_iter=warm_start_min_iter,
            use_latent_precheck=use_latent_precheck,
            latent_precheck_mode=latent_precheck_mode,
            latent_precheck_trace_level=latent_precheck_trace_level,
            shadow_full_depth=shadow_full_depth,
            collect_preconvergence_raw_shadow=collect_preconvergence_raw_shadow,
            use_cached_final_output=use_cached_final_output,
            profile_coda_cost=profile_coda_cost,
        )
        validate_action_delta_deferred_backfill_configuration(
            enabled=use_action_delta_deferred_backfill_filter,
            max_skip_filter_enabled=use_action_delta_nonconvergence_filter,
            production_gate_enabled=use_action_delta_gate,
            shadow_collection_enabled=collect_action_delta_gate_shadow,
            canonical_recurrence_strategy=canonical_recurrence_strategy,
            prepared_gate=action_delta_gate,
            batch_size=h_a.size(0),
            use_warm_start=enable_warm_start,
            warm_start_source=warm_start_source,
            warm_start_min_iter=warm_start_min_iter,
            use_latent_precheck=use_latent_precheck,
            latent_precheck_mode=latent_precheck_mode,
            latent_precheck_trace_level=latent_precheck_trace_level,
            shadow_full_depth=shadow_full_depth,
            collect_preconvergence_raw_shadow=collect_preconvergence_raw_shadow,
            use_cached_final_output=use_cached_final_output,
            profile_coda_cost=profile_coda_cost,
        )
        if latent_precheck_mode == "origin_aware" and self.training:
            raise ValueError("latent_precheck_mode='origin_aware' is inference-only")
        if shadow_full_depth and self.training:
            raise ValueError("shadow_full_depth is inference-only")
        if collect_preconvergence_raw_shadow:
            if not shadow_full_depth:
                raise ValueError("raw preconvergence collection requires shadow_full_depth=True")
            if self.training:
                raise ValueError("raw preconvergence collection is inference-only")
            if preconvergence_raw_shadow_max_depth != max_iter:
                raise ValueError(
                    "raw preconvergence maximum depth must equal recurrence max_iter"
                )
            if latent_precheck_mode != "off" or use_latent_precheck:
                raise ValueError("raw preconvergence collection requires clean pre-check off mode")
            if canonical_recurrence_strategy != "adjacent_action_mse":
                raise ValueError("raw preconvergence collection requires adjacent action-MSE stopping")
        if canonical_recurrence_strategy == "latent_only" and self.training:
            raise ValueError("recurrence_strategy='latent_only' is inference-only")
        if canonical_recurrence_strategy == "scalar_policy" and self.training:
            raise ValueError("recurrence_strategy='scalar_policy' is inference-only")
        if use_action_delta_gate and self.training:
            raise ValueError("Action-Delta Gate is inference-only")
        if collect_action_delta_gate_shadow and self.training:
            raise ValueError("Action-Delta Gate shadow collection is inference-only")
        if use_action_delta_nonconvergence_filter and self.training:
            raise ValueError("Action-Delta non-convergence filter is inference-only")
        if use_action_delta_deferred_backfill_filter and self.training:
            raise ValueError("Action-Delta deferred/backfill filter is inference-only")

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
                    or canonical_recurrence_strategy == "scalar_policy"
                    or canonical_recurrence_strategy == "fixed_terminal_only"
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
        latent_dynamics_warm_anchor = (
            state.detach().clone()
            if shadow_full_depth and cached_state_used
            else None
        )
        warm_start_min_iter_configured = int(warm_start_min_iter)
        effective_min_iter = (
            max(2, warm_start_min_iter_configured)
            if cached_state_used
            else 2
        )
        self.last_recurrence_debug = None
        self._last_get_output_timing = None

        if canonical_recurrence_strategy == "fixed_terminal_only":
            if self.training:
                raise ValueError("recurrence_strategy='fixed_terminal_only' is inference-only")
            actual_origin = "ACTUAL_WARM" if cached_state_used else "COLD"
            return run_fixed_terminal_only(
                self,
                state,
                prelude_out,
                h_a,
                h_t,
                p,
                fixed_k=num_iter,
                max_iter=max_iter,
                actual_origin=actual_origin,
                requested_recurrence_strategy=requested_recurrence_strategy,
                profile_coda_cost=profile_coda_cost,
                capture_warm_start_candidates=capture_warm_start_candidates,
                warm_start_candidate_states=warm_start_candidate_states,
                warm_start_source=warm_start_source,
            )

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

        scalar_policy_cold_fallback = False
        if canonical_recurrence_strategy == "scalar_policy" and not self.training:
            actual_origin = "ACTUAL_WARM" if cached_state_used else "COLD"

            if cached_state_used:
                return run_scalar_policy_adaptive(
                    self,
                    state,
                    prelude_out,
                    h_a,
                    h_t,
                    p,
                    policy=scalar_task_policy,
                    execution_mode=scalar_policy_execution_mode,
                    max_iter=max_iter,
                    actual_origin=actual_origin,
                    requested_recurrence_strategy=(
                        requested_recurrence_strategy
                    ),
                    profile_coda_cost=profile_coda_cost,
                    capture_warm_start_candidates=(
                        capture_warm_start_candidates
                    ),
                    warm_start_candidate_states=(
                        warm_start_candidate_states
                    ),
                    warm_start_source=warm_start_source,
                    warm_start_min_iter_configured=(
                        warm_start_min_iter_configured
                    ),
                )

            # The first prediction has no cached midpoint state.
            # Preserve the production adjacent action-MSE path so that
            # it can create the warm-start candidate for the next call.
            scalar_policy_cold_fallback = True
            canonical_recurrence_strategy = "adjacent_action_mse"


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
            shadow_previous_update = None
            raw_production_states = []
            raw_production_actions = []
            get_output_call_count = 0
            action_delta_gate_requested = bool(use_action_delta_gate)
            action_delta_gate_applied = bool(
                action_delta_gate_requested and cached_state_used
            )
            action_delta_gate_enabled_for_prediction = action_delta_gate_applied
            action_delta_gate_anchor_state = None
            action_delta_gate_anchor_output = None
            action_delta_gate_anchor_iteration = None
            action_delta_gate_return_output = None
            action_delta_gate_triggered = False
            action_delta_gate_terminal_iteration = None
            action_delta_gate_returned_action_source_iteration = None
            action_delta_gate_fallback_reason = None
            action_delta_gate_score_trace = []
            action_delta_gate_predictor_ms_list = []
            action_delta_gate_predicted_trigger_count = 0
            action_delta_gate_exact_terminal_accepted_trigger_count = 0
            action_delta_gate_oracle_confirm_accepted_count = 0
            action_delta_gate_oracle_confirm_rejected_false_safe_count = 0
            action_delta_gate_exact_confirmation_trace = []
            action_delta_gate_diagnostic_coda_call_count = 0
            action_delta_gate_diagnostic_coda_iterations = []
            action_delta_gate_diagnostic_get_output_ms_list = []
            action_delta_gate_mode_is_diagnostic = (
                action_delta_gate_return_mode
                in ACTION_DELTA_GATE_DIAGNOSTIC_RETURN_MODES
            )
            action_delta_gate_exact_audit = {
                "action_delta_gate_exact_audit_enabled": bool(
                    action_delta_gate_exact_coda_audit
                ),
                "action_delta_gate_exact_audit_performed": False,
                "action_delta_gate_exact_audit_anchor_iteration": None,
                "action_delta_gate_exact_audit_terminal_iteration": None,
                "action_delta_gate_exact_audit_full_mse": None,
                "action_delta_gate_exact_audit_l2": None,
                "action_delta_gate_exact_audit_max_abs": None,
                "action_delta_gate_exact_audit_per_step_mse": None,
                "action_delta_gate_exact_audit_per_step_max_abs": None,
                "action_delta_gate_exact_audit_per_dim_mse": None,
                "action_delta_gate_exact_audit_per_dim_max_abs": None,
                "action_delta_gate_exact_audit_anchor_action": None,
                "action_delta_gate_exact_audit_terminal_action": None,
                "action_delta_gate_exact_audit_delta_action": None,
                "action_delta_gate_exact_audit_predicted_delta_action": None,
                "action_delta_gate_exact_audit_predicted_corrected_action": None,
                "action_delta_gate_exact_audit_correction_full_mse": None,
                "action_delta_gate_exact_audit_correction_l2": None,
                "action_delta_gate_exact_audit_correction_max_abs": None,
                "action_delta_gate_exact_audit_correction_per_step_mse": None,
                "action_delta_gate_exact_audit_correction_per_step_max_abs": None,
                "action_delta_gate_exact_audit_correction_per_dim_mse": None,
                "action_delta_gate_exact_audit_correction_per_dim_max_abs": None,
                "action_delta_gate_exact_audit_prefix_step_count": None,
                "action_delta_gate_exact_audit_anchor_reuse_prefix_mse": None,
                "action_delta_gate_exact_audit_correction_prefix_mse": None,
                "action_delta_gate_exact_audit_correction_full_mse_ratio": None,
                "action_delta_gate_exact_audit_correction_prefix_mse_ratio": None,
                "action_delta_gate_exact_audit_action_shape": None,
                "action_delta_gate_exact_audit_metric_action_shape": None,
                "action_delta_gate_exact_audit_leading_batch_dim_squeezed": None,
                "action_delta_gate_exact_audit_get_output_ms": 0.0,
                "action_delta_gate_exact_audit_get_output_call_count": 0,
                "action_delta_gate_exact_audit_error": None,
            }
            action_delta_gate_shadow_applied = bool(
                collect_action_delta_gate_shadow and cached_state_used
            )
            action_delta_gate_shadow_anchor_state = None
            action_delta_gate_shadow_anchor_output = None
            action_delta_gate_shadow_anchor_iteration = None
            action_delta_gate_shadow_previous_latent_delta = None
            action_delta_gate_shadow_transitions = []
            action_delta_gate_shadow_error = None
            action_delta_gate_shadow_exact_outputs = []
            action_delta_gate_shadow_exact_output_iterations = []
            action_delta_nonconvergence_requested = bool(
                use_action_delta_nonconvergence_filter
            )
            action_delta_nonconvergence_applied = bool(
                action_delta_nonconvergence_requested and cached_state_used
            )
            action_delta_nonconvergence_enabled_for_prediction = (
                action_delta_nonconvergence_applied
            )
            action_delta_nonconvergence_anchor_state = None
            action_delta_nonconvergence_anchor_output = None
            action_delta_nonconvergence_anchor_iteration = None
            action_delta_nonconvergence_force_exact_next = False
            action_delta_nonconvergence_pending_event = None
            action_delta_nonconvergence_events = []
            action_delta_nonconvergence_score_trace = []
            action_delta_nonconvergence_predictor_ms_list = []
            action_delta_nonconvergence_predicted_event_count = 0
            action_delta_nonconvergence_skip_count = 0
            action_delta_nonconvergence_forced_coda_count = 0
            action_delta_nonconvergence_consecutive_skip_prevention_count = 0
            action_delta_nonconvergence_max_iter_skip_prevention_count = 0
            action_delta_nonconvergence_first_divergence_iteration = None
            action_delta_nonconvergence_fallback_reason = None
            action_delta_deferred_requested = bool(
                use_action_delta_deferred_backfill_filter
            )
            action_delta_deferred_applied = bool(
                action_delta_deferred_requested and cached_state_used
            )
            action_delta_deferred_enabled_for_prediction = (
                action_delta_deferred_applied
            )
            action_delta_deferred_previous_latent_state = None
            action_delta_deferred_last_exact_iteration = None
            action_delta_deferred_retained_state = None
            action_delta_deferred_retained_iteration = None
            action_delta_deferred_open_run = None
            action_delta_deferred_confirmation_run_pending = None
            action_delta_deferred_backfill_output = None
            action_delta_deferred_runs = []
            action_delta_deferred_score_trace = []
            action_delta_deferred_predictor_ms_list = []
            action_delta_deferred_high_score_count = 0
            action_delta_deferred_backfill_call_count = 0
            action_delta_deferred_backfill_get_output_ms_list = []
            action_delta_deferred_backfill_coda_ms_list = []
            action_delta_deferred_current_coda_call_count = 0
            action_delta_deferred_current_get_output_ms_list = []
            action_delta_deferred_current_coda_ms_list = []
            action_delta_deferred_exact_stop_mse_trace = []
            action_delta_deferred_max_iter_fallback_count = 0
            action_delta_deferred_fallback_reason = None

            run_one_iteration_ms_list = []
            # Output timing lists include the final return-path call unless the
            # terminal output from the adaptive loop is returned directly.
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
                        if collect_preconvergence_raw_shadow:
                            # Retain only a detached reference. The CPU copy happens
                            # after all production-visible values have been frozen.
                            raw_production_states.append(state.detach())
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
                        action_delta_gate_diagnostic_trigger_pending = None
                        gate_predicted_delta = None
                        action_delta_gate_shadow_pending = None
                        action_delta_nonconvergence_forced_confirmation = False
                        action_delta_deferred_backfill_output = None
                        action_delta_deferred_comparison_iteration = None
                        action_delta_deferred_suppress_stop_comparison = False
                        if action_delta_nonconvergence_force_exact_next:
                            # max_skip=1: the iteration immediately following a
                            # diagnostic skip is always routed through the normal
                            # exact-Coda path, with no predictor evaluation.
                            should_call_coda = True
                            action_delta_nonconvergence_forced_confirmation = True
                            action_delta_nonconvergence_forced_coda_count += 1
                            action_delta_nonconvergence_consecutive_skip_prevention_count += 1
                        with rdvla_range("RDVLA/action_head/coda_stop_check_total"):
                            if (
                                should_call_coda
                                and action_delta_gate_shadow_applied
                                and action_delta_gate_shadow_error is None
                                and action_delta_gate_shadow_anchor_state is not None
                                and actual_iter >= action_delta_gate_min_terminal_iter
                            ):
                                try:
                                    (
                                        shadow_gate_score,
                                        _shadow_predicted_trigger,
                                        shadow_predicted_delta,
                                    ) = evaluate_action_delta_gate(
                                        action_delta_gate,
                                        action_delta_gate_shadow_anchor_state,
                                        state,
                                        return_pred_delta=True,
                                    )
                                    action_delta_gate_shadow_pending = {
                                        "anchor_iteration": int(
                                            action_delta_gate_shadow_anchor_iteration
                                        ),
                                        "terminal_iteration": int(actual_iter),
                                        "anchor_state": action_delta_gate_shadow_anchor_state,
                                        "current_state": state.detach(),
                                        "anchor_output": action_delta_gate_shadow_anchor_output,
                                        "predicted_delta": shadow_predicted_delta,
                                        "score": float(shadow_gate_score),
                                        "previous_latent_delta_bfloat16": (
                                            action_delta_gate_shadow_previous_latent_delta
                                        ),
                                    }
                                except Exception as exc:
                                    # Shadow instrumentation must never alter the
                                    # exact Warm-only recurrence path.
                                    action_delta_gate_shadow_error = (
                                        f"{type(exc).__name__}: {exc}"
                                    )
                            if (
                                should_call_coda
                                and action_delta_deferred_enabled_for_prediction
                                and action_delta_deferred_previous_latent_state is not None
                                and actual_iter
                                >= ACTION_DELTA_NONCONVERGENCE_MIN_TERMINAL_ITER
                            ):
                                predictor_start = time.perf_counter()
                                deferred_score = None
                                deferred_high = False
                                try:
                                    with rdvla_range(
                                        "RDVLA/action_head/"
                                        "action_delta_deferred_backfill_filter_total"
                                    ):
                                        deferred_score, _unused_gate_decision = (
                                            evaluate_action_delta_gate(
                                                action_delta_gate,
                                                action_delta_deferred_previous_latent_state,
                                                state,
                                            )
                                        )
                                    deferred_high = bool(
                                        deferred_score
                                        >= ACTION_DELTA_NONCONVERGENCE_THRESHOLD
                                    )
                                except NonFiniteActionDeltaGateError as exc:
                                    # An indeterminate transition is handled as
                                    # requiring exact adjacent confirmation.
                                    action_delta_deferred_fallback_reason = str(exc)
                                    action_delta_deferred_enabled_for_prediction = False
                                predictor_end = time.perf_counter()
                                action_delta_deferred_predictor_ms_list.append(
                                    (predictor_end - predictor_start) * 1000.0
                                )
                                score_record = {
                                    "anchor_terminal_iteration": int(actual_iter - 1),
                                    "terminal_iteration": int(actual_iter),
                                    "score": (
                                        float(deferred_score)
                                        if deferred_score is not None
                                        else None
                                    ),
                                    "predicted_nonconverged": bool(deferred_high),
                                    "coda_deferred": False,
                                    "max_iter_exact_fallback": False,
                                    "fallback_reason": (
                                        action_delta_deferred_fallback_reason
                                        if deferred_score is None
                                        else None
                                    ),
                                }
                                action_delta_deferred_score_trace.append(score_record)

                                if deferred_high and actual_iter < max_iter:
                                    should_call_coda = False
                                    score_record["coda_deferred"] = True
                                    action_delta_deferred_high_score_count += 1
                                    if action_delta_deferred_open_run is None:
                                        action_delta_deferred_open_run = {
                                            "prediction_identity": None,
                                            "start_terminal_iteration": int(actual_iter),
                                            "end_terminal_iteration": int(actual_iter),
                                            "run_length": 0,
                                            "scores": [],
                                            "backfilled_terminal_iteration": None,
                                            "confirming_current_terminal_iteration": None,
                                            "exact_adjacent_confirmation_mse": None,
                                            "stopped_at_confirmation": None,
                                            "truly_eliminated_coda_calls": None,
                                            "closed_by_max_iter_exact_fallback": False,
                                        }
                                    action_delta_deferred_open_run[
                                        "end_terminal_iteration"
                                    ] = int(actual_iter)
                                    action_delta_deferred_open_run["run_length"] += 1
                                    action_delta_deferred_open_run["scores"].append(
                                        float(deferred_score)
                                    )
                                    # Only the final state of a high-score run is
                                    # needed to reconstruct the later adjacent pair.
                                    action_delta_deferred_retained_state = state.detach()
                                    action_delta_deferred_retained_iteration = int(
                                        actual_iter
                                    )
                                else:
                                    if deferred_high and actual_iter >= max_iter:
                                        # Terminal output must be exact. The high
                                        # score cannot suppress Coda at max_iter.
                                        action_delta_deferred_max_iter_fallback_count += 1
                                        score_record["max_iter_exact_fallback"] = True
                                        if action_delta_deferred_open_run is not None:
                                            # The last executed exact action can be
                                            # older than S[k-1] here. Execute exact
                                            # Coda(S[k]) for the return contract, but
                                            # never feed that non-adjacent pair to the
                                            # convergence criterion.
                                            action_delta_deferred_suppress_stop_comparison = (
                                                True
                                            )
                                            action_delta_deferred_open_run.update(
                                                {
                                                    "confirming_current_terminal_iteration": int(
                                                        actual_iter
                                                    ),
                                                    "truly_eliminated_coda_calls": int(
                                                        action_delta_deferred_open_run[
                                                            "run_length"
                                                        ]
                                                    ),
                                                    "closed_by_max_iter_exact_fallback": True,
                                                }
                                            )
                                            action_delta_deferred_runs.append(
                                                action_delta_deferred_open_run
                                            )
                                            action_delta_deferred_open_run = None
                                            action_delta_deferred_retained_state = None
                                            action_delta_deferred_retained_iteration = None
                                    elif action_delta_deferred_open_run is not None:
                                        # Low or non-finite score: reconstruct only
                                        # a[k-1], then the normal path below executes
                                        # a[k] and applies the original adjacent MSE.
                                        action_delta_deferred_confirmation_run_pending = (
                                            action_delta_deferred_open_run
                                        )
                                        action_delta_deferred_open_run = None
                                        backfill_start = time.perf_counter()
                                        try:
                                            with rdvla_range(
                                                "RDVLA/action_head/"
                                                "action_delta_deferred_backfill_coda"
                                            ):
                                                action_delta_deferred_backfill_output = (
                                                    self._get_output(
                                                        action_delta_deferred_retained_state,
                                                        h_a,
                                                        h_t,
                                                        p,
                                                        profile=profile_coda_cost,
                                                    )
                                                )
                                            get_output_call_count += 1
                                            action_delta_deferred_backfill_call_count += 1
                                            action_delta_deferred_comparison_iteration = int(
                                                action_delta_deferred_retained_iteration
                                            )
                                            if profile_coda_cost:
                                                append_get_output_timing()
                                                action_delta_deferred_backfill_get_output_ms_list.append(
                                                    float(
                                                        self._last_get_output_timing[
                                                            "get_output_ms"
                                                        ]
                                                    )
                                                )
                                                action_delta_deferred_backfill_coda_ms_list.append(
                                                    float(
                                                        self._last_get_output_timing[
                                                            "coda_ms"
                                                        ]
                                                    )
                                                )
                                            if not bool(
                                                torch.isfinite(
                                                    action_delta_deferred_backfill_output
                                                ).all().item()
                                            ):
                                                action_delta_deferred_fallback_reason = (
                                                    "deferred/backfill exact output is non-finite"
                                                )
                                                action_delta_deferred_enabled_for_prediction = False
                                        except Exception as exc:
                                            action_delta_deferred_backfill_output = None
                                            action_delta_deferred_suppress_stop_comparison = True
                                            action_delta_deferred_fallback_reason = (
                                                f"backfill {type(exc).__name__}: {exc}"
                                            )
                                            action_delta_deferred_enabled_for_prediction = False
                                        finally:
                                            _unused_backfill_wall_ms = (
                                                time.perf_counter() - backfill_start
                                            ) * 1000.0
                            if (
                                should_call_coda
                                and action_delta_nonconvergence_enabled_for_prediction
                                and not action_delta_nonconvergence_forced_confirmation
                                and action_delta_nonconvergence_anchor_state is not None
                                and actual_iter
                                >= ACTION_DELTA_NONCONVERGENCE_MIN_TERMINAL_ITER
                            ):
                                predictor_start = time.perf_counter()
                                nonconvergence_score = None
                                predicted_nonconvergence = False
                                try:
                                    with rdvla_range(
                                        "RDVLA/action_head/"
                                        "action_delta_nonconvergence_filter_total"
                                    ):
                                        (
                                            nonconvergence_score,
                                            _unused_convergence_decision,
                                        ) = evaluate_action_delta_gate(
                                            action_delta_gate,
                                            action_delta_nonconvergence_anchor_state,
                                            state,
                                        )
                                    predicted_nonconvergence = bool(
                                        nonconvergence_score
                                        >= ACTION_DELTA_NONCONVERGENCE_THRESHOLD
                                    )
                                except NonFiniteActionDeltaGateError as exc:
                                    # Fail closed: execute exact Coda now and
                                    # disable later diagnostic skips.
                                    action_delta_nonconvergence_fallback_reason = str(
                                        exc
                                    )
                                    action_delta_nonconvergence_enabled_for_prediction = (
                                        False
                                    )
                                predictor_end = time.perf_counter()
                                action_delta_nonconvergence_predictor_ms_list.append(
                                    (predictor_end - predictor_start) * 1000.0
                                )
                                action_delta_nonconvergence_score_trace.append(
                                    {
                                        "last_exact_anchor_iteration": int(
                                            action_delta_nonconvergence_anchor_iteration
                                        ),
                                        "terminal_iteration": int(actual_iter),
                                        "score": (
                                            float(nonconvergence_score)
                                            if nonconvergence_score is not None
                                            else None
                                        ),
                                        "predicted_nonconvergence": bool(
                                            predicted_nonconvergence
                                        ),
                                        "coda_skipped": False,
                                        "fallback_reason": (
                                            action_delta_nonconvergence_fallback_reason
                                            if nonconvergence_score is None
                                            else None
                                        ),
                                    }
                                )
                                if predicted_nonconvergence:
                                    action_delta_nonconvergence_predicted_event_count += 1
                                    if actual_iter < max_iter:
                                        # Never end a prediction on a skipped
                                        # Coda: reserve the next iteration for
                                        # mandatory exact confirmation.
                                        should_call_coda = False
                                        action_delta_nonconvergence_force_exact_next = True
                                        action_delta_nonconvergence_skip_count += 1
                                        action_delta_nonconvergence_score_trace[-1][
                                            "coda_skipped"
                                        ] = True
                                        event = {
                                            "last_exact_anchor_iteration": int(
                                                action_delta_nonconvergence_anchor_iteration
                                            ),
                                            "skipped_terminal_iteration": int(
                                                actual_iter
                                            ),
                                            "forced_exact_terminal_iteration": None,
                                            "score": float(nonconvergence_score),
                                            "exact_mse_at_forced_confirmation": None,
                                            "stopping_occurred_at_forced_confirmation": None,
                                            "extra_recurrent_iterations_after_skip": None,
                                            "iterations_from_skip_to_prediction_end": None,
                                        }
                                        action_delta_nonconvergence_events.append(event)
                                        action_delta_nonconvergence_pending_event = event
                                        if (
                                            action_delta_nonconvergence_first_divergence_iteration
                                            is None
                                        ):
                                            action_delta_nonconvergence_first_divergence_iteration = int(
                                                actual_iter
                                            )
                                    else:
                                        action_delta_nonconvergence_max_iter_skip_prevention_count += 1
                            if (
                                should_call_coda
                                and action_delta_gate_enabled_for_prediction
                                and action_delta_gate_anchor_state is not None
                                and actual_iter >= action_delta_gate_min_terminal_iter
                            ):
                                predictor_start = (
                                    time.perf_counter()
                                    if profile_coda_cost
                                    else None
                                )
                                try:
                                    with rdvla_range(
                                        "RDVLA/action_head/action_delta_gate_total"
                                    ):
                                        if (
                                            action_delta_gate_exact_coda_audit
                                            or action_delta_gate_return_mode
                                            == "predicted_correction"
                                        ):
                                            (
                                                gate_score,
                                                gate_triggered,
                                                gate_predicted_delta,
                                            ) = evaluate_action_delta_gate(
                                                action_delta_gate,
                                                action_delta_gate_anchor_state,
                                                state,
                                                return_pred_delta=True,
                                            )
                                        else:
                                            gate_score, gate_triggered = evaluate_action_delta_gate(
                                                action_delta_gate,
                                                action_delta_gate_anchor_state,
                                                state,
                                            )
                                except NonFiniteActionDeltaGateError as exc:
                                    gate_score = None
                                    gate_triggered = False
                                    action_delta_gate_fallback_reason = str(exc)
                                    action_delta_gate_enabled_for_prediction = False
                                if profile_coda_cost:
                                    # evaluate_action_delta_gate() has already
                                    # synchronized its one-scalar decision
                                    # payload. Avoid adding profiling-only CUDA
                                    # synchronizations around the gate.
                                    predictor_end = time.perf_counter()
                                    action_delta_gate_predictor_ms_list.append(
                                        (predictor_end - predictor_start) * 1000.0
                                    )
                                action_delta_gate_score_trace.append(
                                    {
                                        "anchor_iteration": int(
                                            action_delta_gate_anchor_iteration
                                        ),
                                        "terminal_iteration": int(actual_iter),
                                        "score": (
                                            float(gate_score)
                                            if gate_score is not None
                                            else None
                                        ),
                                        "triggered": bool(gate_triggered),
                                    }
                                )
                                if gate_triggered:
                                    action_delta_gate_predicted_trigger_count += 1
                                    if action_delta_gate_mode_is_diagnostic:
                                        # Diagnostic control modes deliberately
                                        # execute the current exact Coda through
                                        # the normal recurrence path below.
                                        action_delta_gate_diagnostic_trigger_pending = (
                                            action_delta_gate_return_mode
                                        )
                                    elif (
                                        action_delta_gate_return_mode
                                        == "predicted_correction"
                                    ):
                                        try:
                                            action_delta_gate_return_output = (
                                                build_action_delta_gate_corrected_output(
                                                    action_delta_gate_anchor_output,
                                                    gate_predicted_delta,
                                                )
                                            )
                                        except ActionDeltaGateCorrectionError as exc:
                                            gate_triggered = False
                                            action_delta_gate_fallback_reason = str(exc)
                                            action_delta_gate_enabled_for_prediction = False
                                    else:
                                        action_delta_gate_return_output = (
                                            action_delta_gate_anchor_output
                                        )
                                if gate_triggered and not action_delta_gate_mode_is_diagnostic:
                                    action_delta_gate_triggered = True
                                    action_delta_gate_terminal_iteration = int(
                                        actual_iter
                                    )
                                    action_delta_gate_returned_action_source_iteration = int(
                                        action_delta_gate_anchor_iteration
                                    )
                                    adaptive_stop = True
                                    stop_reason = "action_delta_gate"
                                    should_call_coda = False
                                    if action_delta_gate_exact_coda_audit:
                                        action_delta_gate_exact_audit.update(
                                            {
                                                "action_delta_gate_exact_audit_performed": True,
                                                "action_delta_gate_exact_audit_anchor_iteration": int(
                                                    action_delta_gate_anchor_iteration
                                                ),
                                                "action_delta_gate_exact_audit_terminal_iteration": int(
                                                    actual_iter
                                                ),
                                                "action_delta_gate_exact_audit_get_output_call_count": 1,
                                            }
                                        )
                                        audit_start = self._sync_time()
                                        audit_output = None
                                        try:
                                            with rdvla_range(
                                                "RDVLA/action_head/"
                                                "action_delta_gate_exact_coda_audit"
                                            ):
                                                audit_output = self._get_output(
                                                    state,
                                                    h_a,
                                                    h_t,
                                                    p,
                                                    profile=False,
                                                )
                                        except Exception as exc:
                                            action_delta_gate_exact_audit[
                                                "action_delta_gate_exact_audit_error"
                                            ] = (
                                                f"{type(exc).__name__}: {exc}"
                                            )
                                        finally:
                                            audit_end = self._sync_time()
                                            action_delta_gate_exact_audit[
                                                "action_delta_gate_exact_audit_get_output_ms"
                                            ] = (
                                                (audit_end - audit_start) * 1000.0
                                            )
                                        if (
                                            action_delta_gate_exact_audit[
                                                "action_delta_gate_exact_audit_error"
                                            ]
                                            is None
                                        ):
                                            try:
                                                action_delta_gate_exact_audit.update(
                                                    _action_delta_gate_exact_audit_metrics(
                                                        action_delta_gate_anchor_output,
                                                        audit_output,
                                                        gate_predicted_delta,
                                                    )
                                                )
                                            except Exception as exc:
                                                action_delta_gate_exact_audit[
                                                    "action_delta_gate_exact_audit_error"
                                                ] = (
                                                    f"{type(exc).__name__}: {exc}"
                                                )

                            if should_call_coda:
                                with rdvla_range("RDVLA/action_head/coda_stop_get_output"):
                                    with rdvla_range("RDVLA/action_head/get_output_each_iter"):
                                        curr_output = self._get_output(
                                            state,
                                            h_a,
                                            h_t,
                                            p,
                                            profile=profile_coda_cost,
                                        )
                                        get_output_call_count += 1
                                if collect_action_delta_gate_shadow:
                                    action_delta_gate_shadow_exact_outputs.append(
                                        curr_output.detach()
                                    )
                                    action_delta_gate_shadow_exact_output_iterations.append(
                                        int(actual_iter)
                                    )
                                if profile_coda_cost:
                                    append_get_output_timing()
                                    convergence_check_start = self._sync_time()

                                if action_delta_deferred_applied:
                                    action_delta_deferred_current_coda_call_count += 1
                                    if profile_coda_cost:
                                        action_delta_deferred_current_get_output_ms_list.append(
                                            float(
                                                self._last_get_output_timing[
                                                    "get_output_ms"
                                                ]
                                            )
                                        )
                                        action_delta_deferred_current_coda_ms_list.append(
                                            float(
                                                self._last_get_output_timing[
                                                    "coda_ms"
                                                ]
                                            )
                                        )

                                if action_delta_gate_diagnostic_trigger_pending is not None:
                                    action_delta_gate_diagnostic_coda_call_count += 1
                                    action_delta_gate_diagnostic_coda_iterations.append(
                                        int(actual_iter)
                                    )
                                    if profile_coda_cost:
                                        action_delta_gate_diagnostic_get_output_ms_list.append(
                                            float(
                                                self._last_get_output_timing[
                                                    "get_output_ms"
                                                ]
                                            )
                                        )

                                action_mse_previous_output = prev_output
                                action_mse_previous_iteration = (
                                    action_delta_deferred_last_exact_iteration
                                )
                                if action_delta_deferred_backfill_output is not None:
                                    action_mse_previous_output = (
                                        action_delta_deferred_backfill_output
                                    )
                                    action_mse_previous_iteration = (
                                        action_delta_deferred_comparison_iteration
                                    )
                                if action_delta_deferred_suppress_stop_comparison:
                                    action_mse_previous_output = None

                                if action_mse_previous_output is not None:
                                    with rdvla_range("RDVLA/action_head/stop_check/mse_compute"):
                                        diff = curr_output - action_mse_previous_output
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
                                                action_mse_previous_output.flatten(), curr_output.flatten(), dim=0
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
                                                        stop_reason = (
                                                            "adjacent_action_mse_cold_fallback"
                                                            if scalar_policy_cold_fallback
                                                            else requested_recurrence_strategy
                                                        )
                                                else:
                                                    min_iter_gate_block_count += 1

                                    if action_delta_deferred_applied:
                                        adjacent_comparison = bool(
                                            action_mse_previous_iteration
                                            == actual_iter - 1
                                        )
                                        if not adjacent_comparison:
                                            raise RuntimeError(
                                                "deferred/backfill stop comparison is not adjacent"
                                            )
                                        action_delta_deferred_exact_stop_mse_trace.append(
                                            {
                                                "anchor_terminal_iteration": int(
                                                    action_mse_previous_iteration
                                                ),
                                                "terminal_iteration": int(actual_iter),
                                                "exact_adjacent_mse": float(action_mse),
                                                "stopped": bool(adaptive_stop),
                                            }
                                        )

                                if (
                                    action_delta_deferred_confirmation_run_pending
                                    is not None
                                ):
                                    pending_run = (
                                        action_delta_deferred_confirmation_run_pending
                                    )
                                    pending_run.update(
                                        {
                                            "backfilled_terminal_iteration": int(
                                                action_delta_deferred_comparison_iteration
                                            )
                                            if action_delta_deferred_comparison_iteration
                                            is not None
                                            else None,
                                            "confirming_current_terminal_iteration": int(
                                                actual_iter
                                            ),
                                            "exact_adjacent_confirmation_mse": (
                                                float(action_mse)
                                                if action_mse is not None
                                                else None
                                            ),
                                            "stopped_at_confirmation": bool(
                                                adaptive_stop
                                            ),
                                            "truly_eliminated_coda_calls": int(
                                                pending_run["run_length"] - 1
                                            )
                                            if action_delta_deferred_backfill_output
                                            is not None
                                            else int(pending_run["run_length"]),
                                        }
                                    )
                                    action_delta_deferred_runs.append(pending_run)
                                    action_delta_deferred_confirmation_run_pending = None
                                    action_delta_deferred_retained_state = None
                                    action_delta_deferred_retained_iteration = None

                                if action_delta_nonconvergence_forced_confirmation:
                                    if action_delta_nonconvergence_pending_event is None:
                                        raise RuntimeError(
                                            "forced Action-Delta non-convergence Coda "
                                            "has no pending skip event"
                                        )
                                    action_delta_nonconvergence_pending_event.update(
                                        {
                                            "forced_exact_terminal_iteration": int(
                                                actual_iter
                                            ),
                                            "exact_mse_at_forced_confirmation": (
                                                float(action_mse)
                                                if action_mse is not None
                                                else None
                                            ),
                                            "stopping_occurred_at_forced_confirmation": bool(
                                                adaptive_stop
                                            ),
                                            "extra_recurrent_iterations_after_skip": int(
                                                actual_iter
                                                - action_delta_nonconvergence_pending_event[
                                                    "skipped_terminal_iteration"
                                                ]
                                            ),
                                        }
                                    )
                                    action_delta_nonconvergence_force_exact_next = False
                                    action_delta_nonconvergence_pending_event = None

                                if action_delta_gate_shadow_pending is not None:
                                    try:
                                        action_delta_gate_shadow_transitions.append(
                                            build_action_delta_gate_shadow_transition(
                                                action_delta_gate,
                                                anchor_iteration=action_delta_gate_shadow_pending[
                                                    "anchor_iteration"
                                                ],
                                                terminal_iteration=actual_iter,
                                                anchor_state=action_delta_gate_shadow_pending[
                                                    "anchor_state"
                                                ],
                                                current_state=action_delta_gate_shadow_pending[
                                                    "current_state"
                                                ],
                                                anchor_output=action_delta_gate_shadow_pending[
                                                    "anchor_output"
                                                ],
                                                current_output=curr_output,
                                                predicted_delta=action_delta_gate_shadow_pending[
                                                    "predicted_delta"
                                                ],
                                                score=action_delta_gate_shadow_pending[
                                                    "score"
                                                ],
                                                exact_adjacent_action_mse=action_mse,
                                                recurrence_mse_threshold=kl_thresh,
                                                previous_transition=(
                                                    action_delta_gate_shadow_transitions[-1]
                                                    if action_delta_gate_shadow_transitions
                                                    else None
                                                ),
                                                previous_latent_delta_bfloat16=(
                                                    action_delta_gate_shadow_pending[
                                                        "previous_latent_delta_bfloat16"
                                                    ]
                                                ),
                                            )
                                        )
                                    except Exception as exc:
                                        action_delta_gate_shadow_error = (
                                            f"{type(exc).__name__}: {exc}"
                                        )

                                if action_delta_gate_diagnostic_trigger_pending is not None:
                                    exact_safe = bool(
                                        action_mse is not None
                                        and action_mse < kl_thresh
                                        and actual_iter >= effective_min_iter
                                    )
                                    confirmation_record = {
                                        "mode": str(
                                            action_delta_gate_diagnostic_trigger_pending
                                        ),
                                        "anchor_iteration": int(
                                            action_delta_gate_anchor_iteration
                                        ),
                                        "terminal_iteration": int(actual_iter),
                                        "exact_adjacent_mse": (
                                            float(action_mse)
                                            if action_mse is not None
                                            else None
                                        ),
                                        "exact_safe": exact_safe,
                                        "accepted": False,
                                    }

                                    if action_delta_gate_exact_coda_audit:
                                        action_delta_gate_exact_audit.update(
                                            {
                                                "action_delta_gate_exact_audit_performed": True,
                                                "action_delta_gate_exact_audit_anchor_iteration": int(
                                                    action_delta_gate_anchor_iteration
                                                ),
                                                "action_delta_gate_exact_audit_terminal_iteration": int(
                                                    actual_iter
                                                ),
                                                # The diagnostic control mode's
                                                # exact Coda is reused; there is
                                                # no additional audit-only call.
                                                "action_delta_gate_exact_audit_get_output_call_count": 0,
                                                "action_delta_gate_exact_audit_get_output_ms": 0.0,
                                            }
                                        )
                                        try:
                                            action_delta_gate_exact_audit.update(
                                                _action_delta_gate_exact_audit_metrics(
                                                    action_delta_gate_anchor_output,
                                                    curr_output,
                                                    gate_predicted_delta,
                                                )
                                            )
                                        except Exception as exc:
                                            action_delta_gate_exact_audit[
                                                "action_delta_gate_exact_audit_error"
                                            ] = f"{type(exc).__name__}: {exc}"

                                    if (
                                        action_delta_gate_diagnostic_trigger_pending
                                        == "exact_terminal"
                                    ):
                                        confirmation_record["accepted"] = True
                                        action_delta_gate_exact_terminal_accepted_trigger_count += 1
                                        action_delta_gate_triggered = True
                                        action_delta_gate_terminal_iteration = int(
                                            actual_iter
                                        )
                                        action_delta_gate_returned_action_source_iteration = int(
                                            actual_iter
                                        )
                                        action_delta_gate_return_output = curr_output
                                        adaptive_stop = True
                                        stop_reason = "action_delta_gate"
                                    elif exact_safe:
                                        confirmation_record["accepted"] = True
                                        action_delta_gate_oracle_confirm_accepted_count += 1
                                        action_delta_gate_triggered = True
                                        action_delta_gate_terminal_iteration = int(
                                            actual_iter
                                        )
                                        action_delta_gate_returned_action_source_iteration = int(
                                            actual_iter
                                        )
                                        action_delta_gate_return_output = curr_output
                                        # The normal adjacent-action-MSE branch
                                        # above already applied the original
                                        # adaptive-stop reason and semantics.
                                    else:
                                        action_delta_gate_oracle_confirm_rejected_false_safe_count += 1

                                    action_delta_gate_exact_confirmation_trace.append(
                                        confirmation_record
                                    )

                                if shadow_full_depth:
                                    shadow_trace_records.append(
                                        build_shadow_trace_record(
                                            iteration=actual_iter,
                                            phase="production",
                                            previous_state=shadow_previous_state,
                                            current_state=state,
                                            previous_output=prev_output,
                                            current_output=curr_output,
                                            previous_update=shadow_previous_update,
                                            warm_anchor=latent_dynamics_warm_anchor,
                                            eps=latent_only_eps,
                                        )
                                    )
                                    shadow_record = shadow_trace_records[-1]
                                    if shadow_record["latent_dynamics_error"] is not None:
                                        shadow_error = {
                                            "iteration": int(actual_iter),
                                            "stage": "latent_dynamics",
                                            **shadow_record["latent_dynamics_error"],
                                        }
                                    elif not shadow_record["state_finite"]:
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
                                    if actual_iter >= 2 and shadow_record["state_finite"]:
                                        shadow_previous_update = (
                                            state.float() - shadow_previous_state.float()
                                        ).detach().clone()

                                if collect_preconvergence_raw_shadow:
                                    raw_production_actions.append(curr_output.detach())

                                prev_output = curr_output.detach()
                                if action_delta_deferred_applied:
                                    action_delta_deferred_last_exact_iteration = int(
                                        actual_iter
                                    )
                                    if not bool(
                                        torch.isfinite(curr_output).all().item()
                                    ):
                                        action_delta_deferred_fallback_reason = (
                                            "deferred/backfill current exact output is non-finite"
                                        )
                                        action_delta_deferred_enabled_for_prediction = False
                                if action_delta_gate_shadow_applied:
                                    # This diagnostic anchor follows every exact
                                    # Coda, including all iterations before
                                    # eligibility.  It is independent of the
                                    # production gate and legacy prev_state.
                                    if action_delta_gate_shadow_anchor_state is not None:
                                        action_delta_gate_shadow_previous_latent_delta = (
                                            state.float()
                                            - action_delta_gate_shadow_anchor_state.float()
                                        ).to(torch.bfloat16).detach()
                                    action_delta_gate_shadow_anchor_state = state.detach()
                                    action_delta_gate_shadow_anchor_output = curr_output.detach()
                                    action_delta_gate_shadow_anchor_iteration = int(
                                        actual_iter
                                    )
                                if (
                                    action_delta_gate_enabled_for_prediction
                                    and not adaptive_stop
                                ):
                                    if not bool(
                                        torch.isfinite(curr_output).all().item()
                                    ):
                                        action_delta_gate_fallback_reason = (
                                            "Action-Delta Gate anchor output is non-finite"
                                        )
                                        action_delta_gate_enabled_for_prediction = False
                                    else:
                                        action_delta_gate_anchor_state = state.detach()
                                        action_delta_gate_anchor_output = curr_output.detach()
                                        action_delta_gate_anchor_iteration = int(actual_iter)
                                if action_delta_nonconvergence_enabled_for_prediction:
                                    if not bool(
                                        torch.isfinite(curr_output).all().item()
                                    ):
                                        action_delta_nonconvergence_fallback_reason = (
                                            "Action-Delta non-convergence exact anchor "
                                            "output is non-finite"
                                        )
                                        action_delta_nonconvergence_enabled_for_prediction = (
                                            False
                                        )
                                    else:
                                        # Update after every executed exact Coda,
                                        # including all pre-eligibility calls and
                                        # forced confirmations. Skipped states are
                                        # never installed as anchors.
                                        action_delta_nonconvergence_anchor_state = (
                                            state.detach()
                                        )
                                        action_delta_nonconvergence_anchor_output = (
                                            curr_output.detach()
                                        )
                                        action_delta_nonconvergence_anchor_iteration = int(
                                            actual_iter
                                        )
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
                        if action_delta_deferred_applied:
                            # This adjacency anchor advances on every recurrent
                            # state, including iterations whose Coda was deferred.
                            action_delta_deferred_previous_latent_state = state.detach()
                        if adaptive_stop:
                            break

            if stop_reason is None and actual_iter >= max_iter:
                with rdvla_range("RDVLA/action_head/stop_reason_update"):
                    stop_reason = "max_iter"

            if action_delta_nonconvergence_force_exact_next:
                raise RuntimeError(
                    "Action-Delta non-convergence prediction ended without its "
                    "mandatory exact-Coda confirmation"
                )
            if (
                action_delta_nonconvergence_forced_coda_count
                != action_delta_nonconvergence_skip_count
            ):
                raise RuntimeError(
                    "Action-Delta non-convergence skip/forced-Coda accounting "
                    "is inconsistent"
                )
            if (
                len(action_delta_nonconvergence_events)
                != action_delta_nonconvergence_skip_count
            ):
                raise RuntimeError(
                    "Action-Delta non-convergence event/skip accounting is inconsistent"
                )
            for event in action_delta_nonconvergence_events:
                event["iterations_from_skip_to_prediction_end"] = int(
                    actual_iter - event["skipped_terminal_iteration"]
                )

            if (
                action_delta_deferred_open_run is not None
                or action_delta_deferred_confirmation_run_pending is not None
                or action_delta_deferred_retained_state is not None
            ):
                raise RuntimeError(
                    "deferred/backfill prediction ended with unresolved deferred state"
                )
            action_delta_deferred_eliminated_count = sum(
                int(run["truly_eliminated_coda_calls"] or 0)
                for run in action_delta_deferred_runs
            )
            if (
                action_delta_deferred_eliminated_count
                != action_delta_deferred_high_score_count
                - action_delta_deferred_backfill_call_count
            ):
                raise RuntimeError(
                    "deferred/backfill eliminated-Coda accounting is inconsistent"
                )

            reuse_terminal_output = bool(
                not action_delta_gate_triggered
                and
                curr_output is not None
                and (
                    use_cached_final_output
                    or scalar_policy_cold_fallback
                )
            )

            with rdvla_range("RDVLA/action_head/final_get_output"):
                if action_delta_gate_triggered:
                    final_output = action_delta_gate_return_output
                elif reuse_terminal_output:
                    final_output = curr_output
                else:
                    final_output = self._get_output(
                        state,
                        h_a,
                        h_t,
                        p,
                        profile=profile_coda_cost,
                    )
                    get_output_call_count += 1
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
                    "cached_final_output_reused": reuse_terminal_output,
                }
            else:
                production_snapshot = None

            action_delta_nonconvergence_estimated_gross_ms = (
                action_delta_nonconvergence_skip_count
                * ACTION_DELTA_NONCONVERGENCE_CODA_COST_MS
            )
            action_delta_nonconvergence_estimated_scorer_ms = (
                len(action_delta_nonconvergence_score_trace)
                * ACTION_DELTA_NONCONVERGENCE_SCORER_COST_MS
            )
            action_delta_deferred_estimated_gross_ms = (
                action_delta_deferred_eliminated_count
                * ACTION_DELTA_NONCONVERGENCE_CODA_COST_MS
            )
            action_delta_deferred_estimated_scorer_ms = (
                len(action_delta_deferred_score_trace)
                * ACTION_DELTA_NONCONVERGENCE_SCORER_COST_MS
            )

            self.last_recurrence_debug = {
                "strategy": convergence_strategy,
                "requested_recurrence_strategy": requested_recurrence_strategy,
                "canonical_recurrence_strategy": canonical_recurrence_strategy,
                "canonical_metric_name": canonical_recurrence_strategy,
                "scalar_policy_requested": (
                    requested_recurrence_strategy == "scalar_policy"
                ),
                "scalar_policy_applied": False,
                "scalar_policy_cold_fallback": bool(
                    scalar_policy_cold_fallback
                ),
                "scalar_policy_execution_mode": (
                    scalar_policy_execution_mode
                    if scalar_policy_cold_fallback
                    else None
                ),
                "scalar_policy_task_id": (
                    int(scalar_task_policy.task_id)
                    if scalar_policy_cold_fallback
                    else None
                ),
                "scalar_policy_outer_fold": (
                    int(scalar_task_policy.outer_fold)
                    if scalar_policy_cold_fallback
                    else None
                ),
                "scalar_policy_threshold": (
                    float(scalar_task_policy.threshold)
                    if scalar_policy_cold_fallback
                    else None
                ),
                "scalar_policy_gate_iteration": None,
                "scalar_policy_terminal_iteration": None,
                "scalar_policy_score_call_count": 0,
                "scalar_policy_score_trace": [],
                "use_action_delta_deferred_backfill_filter": bool(
                    use_action_delta_deferred_backfill_filter
                ),
                "action_delta_deferred_backfill_filter_requested": bool(
                    action_delta_deferred_requested
                ),
                "action_delta_deferred_backfill_filter_applied": bool(
                    action_delta_deferred_applied
                ),
                "action_delta_deferred_backfill_filter_development_only": True,
                "action_delta_deferred_backfill_filter_efficiency_eligible": False,
                "action_delta_deferred_backfill_filter_threshold": float(
                    ACTION_DELTA_NONCONVERGENCE_THRESHOLD
                ),
                "action_delta_deferred_backfill_filter_min_terminal_iter": int(
                    ACTION_DELTA_NONCONVERGENCE_MIN_TERMINAL_ITER
                ),
                "action_delta_deferred_backfill_filter_score_call_count": len(
                    action_delta_deferred_score_trace
                ),
                "action_delta_deferred_backfill_filter_score_trace": (
                    action_delta_deferred_score_trace
                ),
                "action_delta_deferred_backfill_filter_predictor_ms_list": (
                    action_delta_deferred_predictor_ms_list
                ),
                "action_delta_deferred_backfill_filter_predictor_ms_total": sum(
                    action_delta_deferred_predictor_ms_list
                ),
                "action_delta_deferred_backfill_filter_high_score_deferred_call_count": int(
                    action_delta_deferred_high_score_count
                ),
                "action_delta_deferred_backfill_filter_consecutive_run_lengths": [
                    int(run["run_length"]) for run in action_delta_deferred_runs
                ],
                "action_delta_deferred_backfill_filter_runs": (
                    action_delta_deferred_runs
                ),
                "action_delta_deferred_backfill_filter_backfill_coda_call_count": int(
                    action_delta_deferred_backfill_call_count
                ),
                "action_delta_deferred_backfill_filter_backfill_get_output_ms_list": (
                    action_delta_deferred_backfill_get_output_ms_list
                ),
                "action_delta_deferred_backfill_filter_backfill_get_output_ms_total": sum(
                    action_delta_deferred_backfill_get_output_ms_list
                ),
                "action_delta_deferred_backfill_filter_backfill_coda_ms_list": (
                    action_delta_deferred_backfill_coda_ms_list
                ),
                "action_delta_deferred_backfill_filter_backfill_coda_ms_total": sum(
                    action_delta_deferred_backfill_coda_ms_list
                ),
                "action_delta_deferred_backfill_filter_current_state_coda_call_count": int(
                    action_delta_deferred_current_coda_call_count
                ),
                "action_delta_deferred_backfill_filter_current_get_output_ms_list": (
                    action_delta_deferred_current_get_output_ms_list
                ),
                "action_delta_deferred_backfill_filter_current_get_output_ms_total": sum(
                    action_delta_deferred_current_get_output_ms_list
                ),
                "action_delta_deferred_backfill_filter_current_coda_ms_list": (
                    action_delta_deferred_current_coda_ms_list
                ),
                "action_delta_deferred_backfill_filter_current_coda_ms_total": sum(
                    action_delta_deferred_current_coda_ms_list
                ),
                "action_delta_deferred_backfill_filter_truly_eliminated_coda_call_count": int(
                    action_delta_deferred_eliminated_count
                ),
                "action_delta_deferred_backfill_filter_total_exact_coda_call_count": int(
                    get_output_call_count
                ),
                "action_delta_deferred_backfill_filter_recurrent_K": int(actual_iter),
                "action_delta_deferred_backfill_filter_exact_stop_mse_trace": (
                    action_delta_deferred_exact_stop_mse_trace
                ),
                "action_delta_deferred_backfill_filter_unresolved_max_iter_fallback_count": int(
                    action_delta_deferred_max_iter_fallback_count
                ),
                "action_delta_deferred_backfill_filter_fallback_reason": (
                    action_delta_deferred_fallback_reason
                ),
                "action_delta_deferred_backfill_filter_fixed_scorer_cost_ms_per_call": float(
                    ACTION_DELTA_NONCONVERGENCE_SCORER_COST_MS
                ),
                "action_delta_deferred_backfill_filter_fixed_coda_cost_ms_per_call": float(
                    ACTION_DELTA_NONCONVERGENCE_CODA_COST_MS
                ),
                "action_delta_deferred_backfill_filter_fixed_estimated_scorer_cost_ms": float(
                    action_delta_deferred_estimated_scorer_ms
                ),
                "action_delta_deferred_backfill_filter_fixed_estimated_coda_savings_ms": float(
                    action_delta_deferred_estimated_gross_ms
                ),
                "action_delta_deferred_backfill_filter_fixed_estimated_net_savings_ms": float(
                    action_delta_deferred_estimated_gross_ms
                    - action_delta_deferred_estimated_scorer_ms
                ),
                "use_action_delta_nonconvergence_filter": bool(
                    use_action_delta_nonconvergence_filter
                ),
                "action_delta_nonconvergence_filter_requested": bool(
                    action_delta_nonconvergence_requested
                ),
                "action_delta_nonconvergence_filter_applied": bool(
                    action_delta_nonconvergence_applied
                ),
                "action_delta_nonconvergence_filter_development_only": True,
                "action_delta_nonconvergence_filter_efficiency_eligible": False,
                "action_delta_nonconvergence_filter_threshold": float(
                    ACTION_DELTA_NONCONVERGENCE_THRESHOLD
                ),
                "action_delta_nonconvergence_filter_min_terminal_iter": int(
                    ACTION_DELTA_NONCONVERGENCE_MIN_TERMINAL_ITER
                ),
                "action_delta_nonconvergence_filter_max_skip": 1,
                "action_delta_nonconvergence_filter_score_call_count": len(
                    action_delta_nonconvergence_score_trace
                ),
                "action_delta_nonconvergence_filter_score_trace": (
                    action_delta_nonconvergence_score_trace
                ),
                "action_delta_nonconvergence_filter_predicted_event_count": int(
                    action_delta_nonconvergence_predicted_event_count
                ),
                "action_delta_nonconvergence_filter_actual_coda_skip_count": int(
                    action_delta_nonconvergence_skip_count
                ),
                "action_delta_nonconvergence_filter_forced_next_coda_call_count": int(
                    action_delta_nonconvergence_forced_coda_count
                ),
                "action_delta_nonconvergence_filter_consecutive_skip_prevention_count": int(
                    action_delta_nonconvergence_consecutive_skip_prevention_count
                ),
                "action_delta_nonconvergence_filter_max_iter_skip_prevention_count": int(
                    action_delta_nonconvergence_max_iter_skip_prevention_count
                ),
                "action_delta_nonconvergence_filter_exact_coda_call_count": int(
                    get_output_call_count
                ),
                "action_delta_nonconvergence_filter_recurrent_K": int(actual_iter),
                "action_delta_nonconvergence_filter_first_trajectory_divergence_terminal_iteration": (
                    action_delta_nonconvergence_first_divergence_iteration
                ),
                "action_delta_nonconvergence_filter_events": (
                    action_delta_nonconvergence_events
                ),
                "action_delta_nonconvergence_filter_fallback_reason": (
                    action_delta_nonconvergence_fallback_reason
                ),
                "action_delta_nonconvergence_filter_predictor_ms_list": (
                    action_delta_nonconvergence_predictor_ms_list
                ),
                "action_delta_nonconvergence_filter_predictor_ms_total": sum(
                    action_delta_nonconvergence_predictor_ms_list
                ),
                "action_delta_nonconvergence_filter_estimate_scorer_cost_ms_per_call": float(
                    ACTION_DELTA_NONCONVERGENCE_SCORER_COST_MS
                ),
                "action_delta_nonconvergence_filter_estimate_coda_cost_ms_per_call": float(
                    ACTION_DELTA_NONCONVERGENCE_CODA_COST_MS
                ),
                "action_delta_nonconvergence_filter_estimated_gross_coda_savings_ms": float(
                    action_delta_nonconvergence_estimated_gross_ms
                ),
                "action_delta_nonconvergence_filter_estimated_scorer_cost_ms": float(
                    action_delta_nonconvergence_estimated_scorer_ms
                ),
                "action_delta_nonconvergence_filter_estimated_net_savings_ms": float(
                    action_delta_nonconvergence_estimated_gross_ms
                    - action_delta_nonconvergence_estimated_scorer_ms
                ),
                "action_delta_nonconvergence_filter_measured_net_savings_ms": None,
                "action_delta_nonconvergence_filter_measured_net_savings_status": (
                    "requires_paired_warm_only_counterfactual"
                ),
                "use_action_delta_gate": bool(use_action_delta_gate),
                "action_delta_gate_requested": action_delta_gate_requested,
                "action_delta_gate_applied": action_delta_gate_applied,
                "action_delta_gate_threshold": (
                    float(action_delta_gate.threshold)
                    if action_delta_gate_requested
                    else None
                ),
                "action_delta_gate_outer_fold": (
                    int(action_delta_gate.outer_fold)
                    if action_delta_gate_requested
                    else None
                ),
                "action_delta_gate_held_out_task_ids": (
                    list(action_delta_gate.held_out_task_ids)
                    if action_delta_gate_requested
                    else []
                ),
                "action_delta_gate_score_call_count": len(
                    action_delta_gate_score_trace
                ),
                "action_delta_gate_min_terminal_iter": int(
                    action_delta_gate_min_terminal_iter
                ),
                "action_delta_gate_return_mode": str(
                    action_delta_gate_return_mode
                ),
                "action_delta_gate_first_eligible_terminal_iteration": (
                    int(action_delta_gate_min_terminal_iter)
                    if action_delta_gate_applied
                    else None
                ),
                "action_delta_gate_score_trace": action_delta_gate_score_trace,
                "action_delta_gate_predicted_trigger_count": int(
                    action_delta_gate_predicted_trigger_count
                ),
                "action_delta_gate_exact_terminal_accepted_trigger_count": int(
                    action_delta_gate_exact_terminal_accepted_trigger_count
                ),
                "action_delta_gate_oracle_confirm_accepted_count": int(
                    action_delta_gate_oracle_confirm_accepted_count
                ),
                "action_delta_gate_oracle_confirm_rejected_false_safe_count": int(
                    action_delta_gate_oracle_confirm_rejected_false_safe_count
                ),
                "action_delta_gate_exact_confirmation_trace": (
                    action_delta_gate_exact_confirmation_trace
                ),
                "action_delta_gate_diagnostic_coda_call_count": int(
                    action_delta_gate_diagnostic_coda_call_count
                ),
                "action_delta_gate_diagnostic_coda_iterations": (
                    action_delta_gate_diagnostic_coda_iterations
                ),
                "action_delta_gate_diagnostic_get_output_ms_list": (
                    action_delta_gate_diagnostic_get_output_ms_list
                ),
                "action_delta_gate_diagnostic_get_output_ms_total": sum(
                    action_delta_gate_diagnostic_get_output_ms_list
                ),
                "action_delta_gate_mode_is_diagnostic": bool(
                    action_delta_gate_mode_is_diagnostic
                ),
                "action_delta_gate_efficiency_eligible": bool(
                    not action_delta_gate_mode_is_diagnostic
                ),
                "action_delta_gate_triggered": action_delta_gate_triggered,
                "action_delta_gate_anchor_iteration": (
                    action_delta_gate_returned_action_source_iteration
                    if (
                        action_delta_gate_triggered
                        and not action_delta_gate_mode_is_diagnostic
                    )
                    else action_delta_gate_anchor_iteration
                ),
                "action_delta_gate_terminal_iteration": action_delta_gate_terminal_iteration,
                "action_delta_gate_returned_action_source_iteration": (
                    action_delta_gate_returned_action_source_iteration
                ),
                "action_delta_gate_skipped_coda_count": int(
                    action_delta_gate_triggered
                    and not action_delta_gate_mode_is_diagnostic
                ),
                "action_delta_gate_returned_predicted_correction": bool(
                    action_delta_gate_triggered
                    and action_delta_gate_return_mode == "predicted_correction"
                ),
                "action_delta_gate_returned_anchor": bool(
                    action_delta_gate_triggered
                    and action_delta_gate_return_mode == "anchor"
                ),
                "action_delta_gate_fallback_reason": action_delta_gate_fallback_reason,
                "action_delta_gate_predictor_ms_list": (
                    action_delta_gate_predictor_ms_list
                ),
                "action_delta_gate_predictor_ms_total": sum(
                    action_delta_gate_predictor_ms_list
                ),
                "collect_action_delta_gate_shadow": bool(
                    collect_action_delta_gate_shadow
                ),
                "action_delta_gate_shadow_applied": bool(
                    action_delta_gate_shadow_applied
                ),
                "action_delta_gate_shadow_schema_version": (
                    ACTION_DELTA_GATE_SHADOW_SCHEMA_VERSION
                    if collect_action_delta_gate_shadow
                    else None
                ),
                "action_delta_gate_shadow_eligible_row_count": len(
                    action_delta_gate_shadow_transitions
                ),
                "action_delta_gate_shadow_error": action_delta_gate_shadow_error,
                **action_delta_gate_exact_audit,
                "action_delta_gate_returned_previous_coda": bool(
                    action_delta_gate_triggered
                    and action_delta_gate_return_mode == "anchor"
                ),
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
                "canonical_stop_reason": (
                    "action_delta_gate"
                    if (
                        action_delta_gate_triggered
                        and action_delta_gate_return_mode != "oracle_confirm"
                    )
                    else (
                        canonical_recurrence_strategy
                        if adaptive_stop
                        else stop_reason
                    )
                ),
                "coda_call_count": int(get_output_call_count),
                "get_output_call_count": int(get_output_call_count),
                "final_state_coda_executed": bool(
                    (
                        action_delta_gate_triggered
                        and action_delta_gate_mode_is_diagnostic
                    )
                    or (
                        not action_delta_gate_triggered
                        and not reuse_terminal_output
                    )
                ),
                "returned_cached_final_output": reuse_terminal_output,
                "profiling_enabled": bool(profile_coda_cost),
                "use_cached_final_output": bool(use_cached_final_output),
                "warm_start_min_iter_configured": warm_start_min_iter_configured,
                "effective_min_iter": int(effective_min_iter),
                "warm_start_state_used": cached_state_used,
                "min_iter_gate_block_count": int(min_iter_gate_block_count),
                "first_threshold_satisfied_k": first_threshold_satisfied_k,
                "shadow_full_depth_enabled": bool(shadow_full_depth),
                "latent_metric_trace_enabled": bool(shadow_full_depth),
                "latent_dynamics_trace_enabled": bool(shadow_full_depth),
                "latent_dynamics_warm_anchor_available": bool(
                    shadow_full_depth and cached_state_used
                ),
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
                if len(get_output_ms_list) != get_output_call_count:
                    raise RuntimeError(
                        "profiled and production get_output call counts differ: "
                        f"profiled={len(get_output_ms_list)}, "
                        f"production={get_output_call_count}"
                    )
                self.last_recurrence_debug.update({
                    "run_one_iteration_ms_list": run_one_iteration_ms_list,
                    "get_output_ms_list": get_output_ms_list,
                    "coda_ms_list": coda_ms_list,
                    "output_proj_ms_list": output_proj_ms_list,
                    "convergence_check_ms_list": convergence_check_ms_list,
                    "get_output_call_count": int(get_output_call_count),
                    "coda_ms_total": coda_ms_total,
                    "get_output_ms_total": get_output_ms_total,
                    "run_one_iteration_ms_total": run_one_iteration_ms_total,
                    "output_proj_ms_total": output_proj_ms_total,
                    "coda_time_ratio_total": (
                        coda_ms_total / profiled_recurrent_ms_total if profiled_recurrent_ms_total else 0.0
                    ),
                    "action_delta_deferred_backfill_filter_recurrent_ms_total": (
                        run_one_iteration_ms_total
                    ),
                    "action_delta_deferred_backfill_filter_coda_ms_total": (
                        coda_ms_total
                    ),
                    "action_delta_deferred_backfill_filter_get_output_ms_total": (
                        get_output_ms_total
                    ),
                    "action_delta_deferred_backfill_filter_actual_inference_component_ms_total": (
                        run_one_iteration_ms_total
                        + get_output_ms_total
                        + sum(action_delta_deferred_predictor_ms_list)
                    ),
                    "action_delta_nonconvergence_filter_recurrent_ms_total": (
                        run_one_iteration_ms_total
                    ),
                    "action_delta_nonconvergence_filter_coda_ms_total": (
                        coda_ms_total
                    ),
                    "action_delta_nonconvergence_filter_get_output_ms_total": (
                        get_output_ms_total
                    ),
                    "action_delta_nonconvergence_filter_measured_gross_coda_savings_proxy_ms": (
                        action_delta_nonconvergence_skip_count
                        * (get_output_ms_total / get_output_call_count)
                        if get_output_call_count
                        else 0.0
                    ),
                    "action_delta_nonconvergence_filter_measured_net_savings_proxy_ms": (
                        (
                            action_delta_nonconvergence_skip_count
                            * (get_output_ms_total / get_output_call_count)
                            if get_output_call_count
                            else 0.0
                        )
                        - sum(action_delta_nonconvergence_predictor_ms_list)
                    ),
                })

            if collect_action_delta_gate_shadow:
                exact_outputs = [
                    capture_raw_shadow_tensor(value)
                    for value in action_delta_gate_shadow_exact_outputs
                ]
                self.last_inference_metadata["action_delta_gate_shadow"] = {
                    "schema_version": ACTION_DELTA_GATE_SHADOW_SCHEMA_VERSION,
                    "collection_applied": bool(action_delta_gate_shadow_applied),
                    "ineligible_reason": (
                        None
                        if action_delta_gate_shadow_applied
                        else "cold_origin"
                    ),
                    "min_terminal_iteration": int(
                        action_delta_gate_min_terminal_iter
                    ),
                    "gate_threshold": float(action_delta_gate.threshold),
                    "transitions": action_delta_gate_shadow_transitions,
                    "error": action_delta_gate_shadow_error,
                    "production_trace": {
                        "K_t": int(actual_iter),
                        "stop_reason": stop_reason,
                        "adaptive_stop": bool(adaptive_stop),
                        "returned_normalized_action": capture_raw_shadow_tensor(
                            final_output
                        ),
                        "exact_coda_output_iterations": (
                            action_delta_gate_shadow_exact_output_iterations
                        ),
                        "exact_coda_outputs": (
                            torch.stack(exact_outputs, dim=0)
                            if exact_outputs
                            else torch.empty(0)
                        ),
                        "exact_coda_call_count": int(get_output_call_count),
                        "iteration_mse": [
                            float(value) for value in conv_score_list
                        ],
                    },
                }

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
                    previous_update=shadow_previous_update,
                    warm_anchor=latent_dynamics_warm_anchor,
                    eps=latent_only_eps,
                    collect_raw=collect_preconvergence_raw_shadow,
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
                if collect_preconvergence_raw_shadow and shadow_result["completed"]:
                    states = [
                        capture_raw_shadow_tensor(value)
                        for value in raw_production_states
                    ] + shadow_result["raw_states"]
                    actions = [
                        capture_raw_shadow_tensor(value)
                        for value in raw_production_actions
                    ] + shadow_result["raw_actions"]
                    depth = int(preconvergence_raw_shadow_max_depth)
                    if len(states) != depth or len(actions) != depth:
                        raise RuntimeError(
                            "raw preconvergence trajectory does not cover the requested depth"
                        )
                    action_mse = [None] * (depth + 1)
                    action_mse_phase = [None] * (depth + 1)
                    action_mse_source = [None] * (depth + 1)
                    for k, value in enumerate(conv_score_list, start=2):
                        action_mse[k] = float(value)
                        action_mse_phase[k] = "production"
                        action_mse_source[k] = "production_native_bf16"
                    for record in shadow_result["records"]:
                        k = int(record["k"])
                        action_mse[k] = float(record["action_mse"])
                        action_mse_phase[k] = "shadow_tail"
                        action_mse_source[k] = "shadow_tail_fp32"
                    self.last_inference_metadata["preconvergence_raw_shadow"] = {
                        "actual_origin": (
                            "ACTUAL_WARM" if cached_state_used else "COLD_PRIMARY"
                        ),
                        "production_terminal_k": int(actual_iter),
                        "maximum_shadow_depth": depth,
                        "valid_trajectory_length": len(states),
                        "action_mse_threshold": float(kl_thresh),
                        "tensors": {
                            "states": torch.stack(states, dim=0),
                            "actions": torch.stack(actions, dim=0),
                        },
                        "production_iteration_mse": [float(value) for value in conv_score_list],
                        "action_mse": action_mse,
                        "action_mse_phase": action_mse_phase,
                        "action_mse_source": action_mse_source,
                    }
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
        self._scalar_task_policy = None
        self._scalar_policy_execution_mode = None
        # Plain prepared runtime data, deliberately not an nn.Module or
        # registered buffer, so legacy checkpoint state_dict keys are stable.
        self._action_delta_gate = None

    def configure_scalar_task_policy(
        self,
        policy: PreparedScalarTaskPolicy,
        execution_mode: str,
    ) -> None:
        """Bind one held-out task policy before running that LIBERO task."""

        if not isinstance(policy, PreparedScalarTaskPolicy):
            raise TypeError(
                "configure_scalar_task_policy requires "
                "PreparedScalarTaskPolicy"
            )
        if execution_mode not in SUPPORTED_SCALAR_EXECUTION_MODES:
            raise ValueError(
                "scalar policy execution mode must be direct "
                "or confirm_next"
            )

        self._scalar_task_policy = policy
        self._scalar_policy_execution_mode = execution_mode

    def clear_scalar_task_policy(self) -> None:
        self._scalar_task_policy = None
        self._scalar_policy_execution_mode = None

    def configure_action_delta_gate(
        self,
        gate: PreparedActionDeltaGate,
    ) -> None:
        if not isinstance(gate, PreparedActionDeltaGate):
            raise TypeError(
                "configure_action_delta_gate requires PreparedActionDeltaGate"
            )
        expected = (
            self.cfg.hidden_dim,
            self.cfg.action_dim,
            self.cfg.action_chunk_len,
        )
        actual = (
            gate.hidden_dim,
            gate.action_dim,
            gate.action_chunk_len,
        )
        if actual != expected:
            raise ValueError(
                "Action-Delta Gate/action-head dimension mismatch: "
                f"expected={expected}, actual={actual}"
            )
        self._action_delta_gate = gate

    def clear_action_delta_gate(self) -> None:
        self._action_delta_gate = None

    def _resolve_scalar_runtime_policy(
        self,
        convergence_strategy,
        scalar_task_policy,
        scalar_policy_execution_mode,
    ):
        canonical = canonicalize_recurrence_strategy(
            convergence_strategy
        )

        if canonical != "scalar_policy":
            return (
                scalar_task_policy,
                scalar_policy_execution_mode
                if scalar_policy_execution_mode is not None
                else "direct",
            )

        if scalar_task_policy is None:
            scalar_task_policy = self._scalar_task_policy

        if scalar_policy_execution_mode is None:
            scalar_policy_execution_mode = (
                self._scalar_policy_execution_mode
            )

        if scalar_policy_execution_mode is None:
            scalar_policy_execution_mode = "direct"

        return (
            scalar_task_policy,
            scalar_policy_execution_mode,
        )

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
                       collect_preconvergence_raw_shadow=False,
                       preconvergence_raw_shadow_max_depth=32,
                       capture_action_head_workload=False,
                       latent_only_metric="raw_mse",
                       latent_only_cold_threshold=0.0,
                       latent_only_warm_threshold=0.0,
                       latent_only_min_iter=2,
                       latent_only_eps=1e-8,
                       scalar_task_policy=None,
                       scalar_policy_execution_mode=None,
                       use_action_delta_gate=False,
                       action_delta_gate=None,
                       action_delta_gate_max_skip=1,
                       action_delta_gate_min_terminal_iter=2,
                       action_delta_gate_exact_coda_audit=False,
                       action_delta_gate_return_mode="anchor",
                       collect_action_delta_gate_shadow=False,
                       use_action_delta_nonconvergence_filter=False,
                       use_action_delta_deferred_backfill_filter=False,
                       **kwargs):
        canonical_recurrence_strategy = canonicalize_recurrence_strategy(
            convergence_strategy
        )
        validate_fixed_terminal_only_configuration(
            canonical_recurrence_strategy,
            recurrent_num_iter=num_iter,
            recurrence_max_iter=max_iter,
        )
        (
            scalar_task_policy,
            scalar_policy_execution_mode,
        ) = self._resolve_scalar_runtime_policy(
            convergence_strategy,
            scalar_task_policy,
            scalar_policy_execution_mode,
        )
        if (
            use_action_delta_gate
            or collect_action_delta_gate_shadow
            or use_action_delta_nonconvergence_filter
            or use_action_delta_deferred_backfill_filter
        ) and action_delta_gate is None:
            action_delta_gate = self._action_delta_gate
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
        validate_scalar_runtime_configuration(
            canonical_recurrence_strategy,
            task_policy=scalar_task_policy,
            execution_mode=scalar_policy_execution_mode,
            use_warm_start=enable_warm_start,
            warm_start_source=warm_start_source,
            use_latent_precheck=use_latent_precheck,
            latent_precheck_mode=latent_precheck_mode,
            latent_precheck_trace_level=latent_precheck_trace_level,
            shadow_full_depth=shadow_full_depth,
            collect_preconvergence_raw_shadow=(
                collect_preconvergence_raw_shadow
            ),
            use_cached_final_output=use_cached_final_output,
            max_iter=max_iter,
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
                                 collect_preconvergence_raw_shadow=collect_preconvergence_raw_shadow,
                                 preconvergence_raw_shadow_max_depth=preconvergence_raw_shadow_max_depth,
                                 capture_action_head_workload=capture_action_head_workload,
                                 latent_only_metric=latent_only_metric,
                                 latent_only_cold_threshold=latent_only_cold_threshold,
                                 latent_only_warm_threshold=latent_only_warm_threshold,
                                 latent_only_min_iter=latent_only_min_iter,
                                 latent_only_eps=latent_only_eps,
                                 scalar_task_policy=scalar_task_policy,
                                 scalar_policy_execution_mode=(
                                     scalar_policy_execution_mode
                                 ),
                                 use_action_delta_gate=use_action_delta_gate,
                                 action_delta_gate=action_delta_gate,
                                 action_delta_gate_max_skip=(
                                     action_delta_gate_max_skip
                                 ),
                                 action_delta_gate_min_terminal_iter=(
                                     action_delta_gate_min_terminal_iter
                                 ),
                                 action_delta_gate_exact_coda_audit=(
                                     action_delta_gate_exact_coda_audit
                                 ),
                                 action_delta_gate_return_mode=(
                                     action_delta_gate_return_mode
                                 ),
                                 collect_action_delta_gate_shadow=(
                                     collect_action_delta_gate_shadow
                                 ),
                                 use_action_delta_nonconvergence_filter=(
                                     use_action_delta_nonconvergence_filter
                                 ),
                                 use_action_delta_deferred_backfill_filter=(
                                     use_action_delta_deferred_backfill_filter
                                 ))
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
