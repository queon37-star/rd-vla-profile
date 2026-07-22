import math
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
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
                kl_thresh: float = 0.001, cos_thresh: float = 0.999,
                max_iter: int = 32, profile_coda_cost: bool = False,
                use_cached_final_output: bool = False,
                use_latent_precheck: bool = False,
        latent_precheck_thresh: float = 0.12,
        latent_precheck_min_iter: int = 2,
        latent_precheck_force_interval: int = 0, **kwargs) -> torch.Tensor:
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

        state = self.init_state(B, device, dtype)
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

        if convergence_strategy in ("kl_divergence", "cosine_similarity") and not self.training:
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
                        if profile_coda_cost:
                            run_one_iteration_start = self._sync_time()
                        state = self._run_one_iteration(state, prelude_out, h_a, h_t, p)
                        if profile_coda_cost:
                            run_one_iteration_end = self._sync_time()
                            run_one_iteration_ms_list.append((run_one_iteration_end - run_one_iteration_start) * 1000.0)

                        actual_iter = it + 1
                        with rdvla_range("RDVLA/action_head/latent_precheck_total"):
                            latent_mse = None
                            latent_l2 = None
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

                        action_mse = None
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

                                        if convergence_strategy == "cosine_similarity":
                                            cos_sim_tensor = F.cosine_similarity(
                                                prev_output.flatten(), curr_output.flatten(), dim=0
                                            )
                                            with rdvla_range("RDVLA/action_head/stop_check/item_sync"):
                                                cos_sim = cos_sim_tensor.item()
                                            final_kl = 1 - cos_sim
                                            if cos_sim > cos_thresh:
                                                with rdvla_range("RDVLA/action_head/stop_reason_update"):
                                                    adaptive_stop = True
                                                    stop_reason = "cosine_similarity"
                                        elif convergence_strategy == "kl_divergence":
                                            final_kl = action_mse
                                            if action_mse < kl_thresh:
                                                with rdvla_range("RDVLA/action_head/stop_reason_update"):
                                                    adaptive_stop = True
                                                    stop_reason = "kl_divergence"

                                prev_output = curr_output.detach()
                                if profile_coda_cost:
                                    convergence_check_end = self._sync_time()
                                    convergence_check_ms_list.append((convergence_check_end - convergence_check_start) * 1000.0)

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

            self.last_recurrence_debug = {
                "strategy": convergence_strategy,
                "threshold": float(kl_thresh) if convergence_strategy == "kl_divergence" else float(cos_thresh),
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
                "latent_precheck_thresh": float(latent_precheck_thresh),
                "latent_precheck_min_iter": int(latent_precheck_min_iter),
                "latent_precheck_force_interval": int(latent_precheck_force_interval),
                "latent_precheck_coda_call_mask": latent_precheck_coda_call_mask,
                "latent_precheck_skipped_iters": latent_precheck_skipped_iters,
                "latent_precheck_called_iters": latent_precheck_called_iters,
                "latent_precheck_skip_count": len(latent_precheck_skipped_iters),
                "latent_precheck_call_count": len(latent_precheck_called_iters),
                "latent_precheck_skip_ratio": (
                    len(latent_precheck_skipped_iters) / len(latent_precheck_coda_call_mask)
                    if latent_precheck_coda_call_mask else 0.0
                ),
                "latent_precheck_decisions": latent_precheck_decisions,
                "first_converged_k_1e_4": first_converged_k_1e_4,
                "first_converged_k_5e_4": first_converged_k_5e_4,
                "final_mse": conv_score_list[-1] if conv_score_list else None,
                "final_conv_score": final_kl,
                "stop_reason": stop_reason,
                "profiling_enabled": bool(profile_coda_cost),
                "use_cached_final_output": bool(use_cached_final_output),
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
            }

            with rdvla_range("RDVLA/action_head/final_get_output"):
                final_output = curr_output
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
                       profile_coda_cost=False, use_cached_final_output=False,
                       use_latent_precheck=False, latent_precheck_thresh=0.12,
                       latent_precheck_min_iter=2, latent_precheck_force_interval=0, **kwargs):
        with rdvla_range("RDVLA/action_head/wrapper_total"):
            B = actions_hidden_states.shape[0]
            proprio = proprio.reshape(B, -1).to(torch.bfloat16)
            with rdvla_range("RDVLA/action_head/proprio_projector"):
                proprio_features = proprio_projector(proprio).unsqueeze(1)
            with rdvla_range("RDVLA/action_head/split_h_t_h_a"):
                h_t = actions_hidden_states[:, :, :self.num_task_tokens, :]
                h_a = actions_hidden_states[:, :, self.num_task_tokens:, :]
            with rdvla_range("RDVLA/action_head/vla_recurrent_total"):
                return self.model(h_a, h_t, proprio_features, num_iter=num_iter,
                                 convergence_strategy=convergence_strategy, kl_thresh=kl_thresh,
                                 cos_thresh=cos_thresh, max_iter=max_iter,
                                 profile_coda_cost=profile_coda_cost,
                                 use_cached_final_output=use_cached_final_output,
                                 use_latent_precheck=use_latent_precheck,
                                 latent_precheck_thresh=latent_precheck_thresh,
                                 latent_precheck_min_iter=latent_precheck_min_iter,
                                 latent_precheck_force_interval=latent_precheck_force_interval)
