"""Pure helpers for profiling the frozen Combined adaptive-Coda gate.

The module contains no runtime scheduler integration.  Its optimized path is a
standalone tensor implementation used only by the offline GPU microbenchmark.
"""

from __future__ import annotations

import csv
import hashlib
import inspect
import json
import math
import random
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
import torch.nn.functional as F

from prismatic.models.latent_dynamics import compute_latent_dynamics
from prismatic.models.latent_metrics import compute_latent_metrics
from scripts.analyze_latent_dynamics_features import canonical_json


SCHEMA_VERSION = 1
EXPECTED_TRACE_IDENTITY = (
    "11e9625e136e2c1c08255a020b10a4b6645f8136a9c49d6bbf383f30d987b268"
)
EXPECTED_MODEL_SOURCE_COMMIT = "975da9093b960b36910c5b3a84723d23bbed3873"
COMBINED_FEATURE_NAMES = (
    "raw_mse",
    "relative_mse",
    "relative_l2",
    "cosine_distance",
    "contraction_ratio",
    "update_turning_cosine",
    "acceleration_rms",
    "acceleration_ratio",
    "state_norm_ratio",
    "token_update_p50",
    "token_update_p90",
    "token_update_p95",
    "token_update_max",
    "token_update_cv",
    "token_update_energy_entropy",
    "token_update_top10_fraction",
    "warm_anchor_relative_l2",
    "warm_anchor_cosine_distance",
)
HISTORY_FEATURE_INDICES = (4, 5, 6, 7)
OPERATIONS = (
    "raw_mse_tensor",
    "raw_mse_decision",
    "combined_current_diagnostic",
    "combined_optimized_tensor",
    "combined_optimized_decision",
    "coda_get_output",
    "recurrent_one_iteration",
    "warm_start_sk1_action_head",
)
STATE_CASES = ("k2", "k_ge_3")
DEFAULT_ORDER_SEED = 20260803
TOLERANCE = {"rtol": 1e-5, "atol": 1e-6}


class CombinedGateMicrobenchmarkError(ValueError):
    """Raised when frozen inputs or profiling invariants are violated."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise CombinedGateMicrobenchmarkError(message)


def _scalar_close(left: float, right: float) -> bool:
    return math.isclose(
        float(left),
        float(right),
        rel_tol=TOLERANCE["rtol"],
        abs_tol=TOLERANCE["atol"],
    )


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass(frozen=True)
class SerializedCombinedModel:
    outer_fold: int
    held_out_task_ids: tuple[int, ...]
    feature_names: tuple[str, ...]
    expanded_feature_names: tuple[str, ...]
    imputation_medians: tuple[float, ...]
    scaling_mean: tuple[float, ...]
    scaling_scale: tuple[float, ...]
    weights: tuple[float, ...]
    bias: float
    threshold: float
    threshold_hex: str


@dataclass(frozen=True)
class CombinedModelTensors:
    imputation_medians: torch.Tensor
    scaling_mean: torch.Tensor
    scaling_scale: torch.Tensor
    weights: torch.Tensor
    bias: torch.Tensor
    threshold: torch.Tensor


@dataclass(frozen=True)
class StateCase:
    case_id: str
    iteration: int
    current_state: torch.Tensor
    previous_state: torch.Tensor
    previous_update: torch.Tensor | None
    warm_anchor: torch.Tensor


def _expected_expanded_names() -> tuple[str, ...]:
    values = []
    for index, name in enumerate(COMBINED_FEATURE_NAMES):
        values.append(name)
        if index in HISTORY_FEATURE_INDICES:
            values.append(f"{name}__available")
    return tuple(values)


def load_serialized_combined_models(
    artifact_dir: Path,
) -> tuple[dict[int, SerializedCombinedModel], dict[int, int], dict[str, Any]]:
    """Load fold-specific Combined parameters without fitting or selection."""

    artifact_dir = Path(artifact_dir)
    report_path = artifact_dir / "metric_report.json"
    model_path = artifact_dir / "model_summary.json"
    hashes_path = artifact_dir / "output_hashes.json"
    for path in (report_path, model_path, hashes_path):
        _require(path.is_file(), f"missing adaptive-Coda artifact: {path}")
    hashes = json.loads(hashes_path.read_text(encoding="utf-8"))["files"]
    for name, expected in hashes.items():
        _require(
            sha256_file(artifact_dir / name) == expected,
            f"adaptive-Coda artifact hash mismatch: {name}",
        )
    report = json.loads(report_path.read_text(encoding="utf-8"))
    summary = json.loads(model_path.read_text(encoding="utf-8"))
    _require(
        report["inputs"]["workload_identity_sha256"] == EXPECTED_TRACE_IDENTITY,
        "frozen trace identity mismatch",
    )
    _require(
        report["inputs"]["source_git_commit"] == EXPECTED_MODEL_SOURCE_COMMIT,
        "adaptive-Coda source commit mismatch",
    )
    _require(summary.get("global_model_fitted") is False, "unexpected global model")
    _require(summary.get("global_threshold_fitted") is False, "unexpected global threshold")
    models: dict[int, SerializedCombinedModel] = {}
    task_to_fold: dict[int, int] = {}
    for fold in summary["outer_folds"]:
        fold_id = int(fold["outer_fold"])
        artifact = fold["learned_models"]["combined"]["outer_training_refit"]
        selection = fold["learned_models"]["combined"]["threshold_selection"]
        preprocessor = artifact.get("preprocessor")
        model = artifact.get("model")
        _require(isinstance(preprocessor, Mapping), f"fold {fold_id}: missing preprocessor")
        _require(isinstance(model, Mapping), f"fold {fold_id}: missing logistic model")
        feature_names = tuple(artifact["feature_names"])
        expanded_names = tuple(preprocessor["expanded_feature_names"])
        _require(
            feature_names == COMBINED_FEATURE_NAMES,
            f"fold {fold_id}: feature order mismatch",
        )
        _require(expanded_names == _expected_expanded_names(), f"fold {fold_id}: expanded feature order mismatch")
        held_out = tuple(int(value) for value in fold["outer_held_out_task_ids"])
        serialized = SerializedCombinedModel(
            outer_fold=fold_id,
            held_out_task_ids=held_out,
            feature_names=feature_names,
            expanded_feature_names=expanded_names,
            imputation_medians=tuple(float(value) for value in preprocessor["imputation_medians"]),
            scaling_mean=tuple(float(value) for value in preprocessor["scaling_mean"]),
            scaling_scale=tuple(float(value) for value in preprocessor["scaling_scale"]),
            weights=tuple(float(value) for value in model["weights"]),
            bias=float(model["bias"]),
            threshold=float(selection["selected_threshold"]),
            threshold_hex=str(selection["selected_threshold_hex"]),
        )
        _require(len(serialized.imputation_medians) == 18, f"fold {fold_id}: imputation length")
        _require(len(serialized.weights) == 22, f"fold {fold_id}: coefficient length")
        _require(len(serialized.scaling_mean) == 22, f"fold {fold_id}: scaling length")
        _require(len(serialized.scaling_scale) == 22, f"fold {fold_id}: scaling length")
        _require(math.isfinite(serialized.threshold), f"fold {fold_id}: non-finite threshold")
        _require(float.fromhex(serialized.threshold_hex) == serialized.threshold, f"fold {fold_id}: threshold hex mismatch")
        models[fold_id] = serialized
        for task_id in held_out:
            _require(task_id not in task_to_fold, f"task {task_id}: duplicate fold mapping")
            task_to_fold[task_id] = fold_id
    _require(set(task_to_fold) == set(range(10)), "task-to-fold mapping must cover tasks 0..9")
    provenance = {
        "artifact_dir": str(artifact_dir.resolve()),
        "metric_report_sha256": sha256_file(report_path),
        "model_summary_sha256": sha256_file(model_path),
        "output_hashes_sha256": sha256_file(hashes_path),
        "workload_identity_sha256": EXPECTED_TRACE_IDENTITY,
        "model_source_git_commit": EXPECTED_MODEL_SOURCE_COMMIT,
        "models_refit": False,
        "thresholds_reselected": False,
    }
    return models, task_to_fold, provenance


def load_fixed_raw_mse_thresholds(artifact_dir: Path) -> dict[int, float]:
    summary = json.loads(
        (Path(artifact_dir) / "model_summary.json").read_text(encoding="utf-8")
    )
    values = {}
    for fold in summary["outer_folds"]:
        fold_id = int(fold["outer_fold"])
        threshold = float(fold["fixed_raw_mse_reference"]["threshold"])
        _require(math.isfinite(threshold), f"fold {fold_id}: non-finite fixed threshold")
        values[fold_id] = threshold
    _require(set(values) == set(range(5)), "fixed thresholds must cover five folds")
    return values


def model_to_device(
    model: SerializedCombinedModel, device: torch.device | str
) -> CombinedModelTensors:
    def tensor(values: Any) -> torch.Tensor:
        return torch.as_tensor(values, dtype=torch.float64, device=device)

    return CombinedModelTensors(
        imputation_medians=tensor(model.imputation_medians),
        scaling_mean=tensor(model.scaling_mean),
        scaling_scale=tensor(model.scaling_scale),
        weights=tensor(model.weights),
        bias=tensor(model.bias),
        threshold=tensor(model.threshold),
    )


def raw_mse_tensor(current_state: torch.Tensor, previous_state: torch.Tensor) -> torch.Tensor:
    """Compute only FP32 raw MSE and retain the result on the input device."""

    return torch.mean((current_state.float() - previous_state.float()).square())


def raw_mse_decision(
    current_state: torch.Tensor,
    previous_state: torch.Tensor,
    threshold: float,
) -> bool:
    _require(math.isfinite(float(threshold)), "raw-MSE threshold must be finite")
    return bool((raw_mse_tensor(current_state, previous_state) <= threshold).item())


def _optimized_combined_components(
    current_state: torch.Tensor,
    previous_state: torch.Tensor,
    previous_update: torch.Tensor | None,
    warm_anchor: torch.Tensor,
    model: CombinedModelTensors,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    current = current_state.float()
    previous = previous_state.float()
    update = current - previous
    flat_current = current.reshape(-1)
    flat_previous = previous.reshape(-1)
    flat_update = update.reshape(-1)
    raw_mse = torch.mean(update.square())
    previous_mse = torch.mean(previous.square())
    relative_mse = raw_mse / (previous_mse + eps)
    relative_l2 = torch.linalg.vector_norm(flat_update) / (
        torch.linalg.vector_norm(flat_previous) + eps
    )
    cosine_distance = 1.0 - F.cosine_similarity(
        flat_current, flat_previous, dim=0, eps=eps
    )
    update_rms = raw_mse.sqrt()
    previous_state_rms = previous_mse.sqrt()
    state_rms = torch.mean(current.square()).sqrt()
    state_norm_ratio = state_rms / (previous_state_rms + eps)

    if previous_update is None:
        history = torch.full((4,), torch.nan, dtype=torch.float32, device=current.device)
    else:
        prior_update = previous_update.float()
        prior_rms = torch.mean(prior_update.square()).sqrt()
        acceleration = update - prior_update
        history = torch.stack(
            (
                update_rms / (prior_rms + eps),
                F.cosine_similarity(
                    flat_update, prior_update.reshape(-1), dim=0, eps=eps
                ),
                torch.mean(acceleration.square()).sqrt(),
                torch.mean(acceleration.square()).sqrt() / (prior_rms + eps),
            )
        )

    token_update_rms = torch.mean(update.square(), dim=-1).sqrt().reshape(-1)
    token_mean = torch.mean(token_update_rms)
    token_cv = torch.std(token_update_rms, unbiased=False) / (token_mean + eps)
    token_energy = token_update_rms.square()
    total_energy = torch.sum(token_energy)
    probabilities = token_energy / (total_energy + eps)
    entropy_terms = torch.where(
        probabilities > 0,
        probabilities * torch.log(torch.clamp_min(probabilities, eps)),
        torch.zeros_like(probabilities),
    )
    token_count = token_update_rms.numel()
    if token_count > 1:
        entropy = -torch.sum(entropy_terms) / math.log(token_count)
    else:
        entropy = torch.zeros((), dtype=torch.float32, device=current.device)
    top_count = max(1, int(math.ceil(0.10 * token_count)))
    top_fraction = torch.topk(token_energy, top_count).values.sum() / (
        total_energy + eps
    )
    anchor = warm_anchor.float()
    anchor_difference = current - anchor
    anchor_relative_l2 = torch.linalg.vector_norm(anchor_difference.reshape(-1)) / (
        torch.linalg.vector_norm(anchor.reshape(-1)) + eps
    )
    anchor_cosine_distance = 1.0 - F.cosine_similarity(
        flat_current, anchor.reshape(-1), dim=0, eps=eps
    )
    raw_features = torch.cat(
        (
            torch.stack((raw_mse, relative_mse, relative_l2, cosine_distance)),
            history,
            state_norm_ratio.reshape(1),
            torch.stack(
                (
                    torch.quantile(token_update_rms, 0.50),
                    torch.quantile(token_update_rms, 0.90),
                    torch.quantile(token_update_rms, 0.95),
                    torch.max(token_update_rms),
                    token_cv,
                    torch.clamp(entropy, min=0.0, max=1.0),
                    torch.clamp(top_fraction, min=0.0, max=1.0),
                    anchor_relative_l2,
                    anchor_cosine_distance,
                )
            ),
        )
    )
    available = torch.isfinite(raw_features)
    raw_for_model = raw_features.to(dtype=model.imputation_medians.dtype)
    imputed = torch.where(available, raw_for_model, model.imputation_medians)
    expanded_values = []
    for index in range(len(COMBINED_FEATURE_NAMES)):
        expanded_values.append(imputed[index])
        if index in HISTORY_FEATURE_INDICES:
            expanded_values.append(
                available[index].to(dtype=model.imputation_medians.dtype)
            )
    expanded = torch.stack(expanded_values)
    normalized = (expanded - model.scaling_mean) / model.scaling_scale
    logit = torch.dot(normalized, model.weights) + model.bias
    probability = torch.sigmoid(logit)
    return raw_features, normalized, logit, probability


def combined_optimized_tensor(
    current_state: torch.Tensor,
    previous_state: torch.Tensor,
    previous_update: torch.Tensor | None,
    warm_anchor: torch.Tensor,
    model: CombinedModelTensors,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Return the Combined probability without any host-visible transfer."""

    return _optimized_combined_components(
        current_state, previous_state, previous_update, warm_anchor, model, eps
    )[3]


def combined_optimized_decision(
    current_state: torch.Tensor,
    previous_state: torch.Tensor,
    previous_update: torch.Tensor | None,
    warm_anchor: torch.Tensor,
    model: CombinedModelTensors,
    eps: float = 1e-8,
) -> bool:
    probability = combined_optimized_tensor(
        current_state, previous_state, previous_update, warm_anchor, model, eps
    )
    return bool((probability >= model.threshold).item())


def _cpu_preprocess_and_score(
    features: Sequence[float | None], model: SerializedCombinedModel
) -> dict[str, Any]:
    expanded = []
    for index, value in enumerate(features):
        available = value is not None and math.isfinite(float(value))
        expanded.append(float(value) if available else model.imputation_medians[index])
        if index in HISTORY_FEATURE_INDICES:
            expanded.append(1.0 if available else 0.0)
    normalized = (
        (np.asarray(expanded, dtype=np.float64) - np.asarray(model.scaling_mean))
        / np.asarray(model.scaling_scale)
    )
    logit = float(np.dot(normalized, np.asarray(model.weights)) + model.bias)
    probability = float(1.0 / (1.0 + math.exp(-logit)))
    return {
        "features": list(features),
        "expanded_features": expanded,
        "normalized_features": normalized.tolist(),
        "logit": logit,
        "probability": probability,
        "decision": probability >= model.threshold,
    }


def combined_current_diagnostic(
    current_state: torch.Tensor,
    previous_state: torch.Tensor,
    previous_update: torch.Tensor | None,
    warm_anchor: torch.Tensor,
    model: SerializedCombinedModel,
    eps: float = 1e-8,
) -> dict[str, Any]:
    """Exercise the existing JSON-scalar diagnostics and stored model semantics."""

    existing = compute_latent_metrics(current_state, previous_state, eps=eps)
    dynamics = compute_latent_dynamics(
        current_state,
        previous_state,
        previous_update=previous_update,
        warm_anchor=warm_anchor,
        eps=eps,
    )
    features = (
        existing["raw_mse"],
        existing["relative_mse"],
        existing["relative_l2"],
        existing["cosine_distance"],
        dynamics["contraction_ratio"],
        dynamics["update_turning_cosine"],
        dynamics["acceleration_rms"],
        dynamics["acceleration_ratio"],
        dynamics["state_norm_ratio"],
        dynamics["token_update_p50"],
        dynamics["token_update_p90"],
        dynamics["token_update_p95"],
        dynamics["token_update_max"],
        dynamics["token_update_cv"],
        dynamics["token_update_energy_entropy"],
        dynamics["token_update_top10_fraction"],
        dynamics["warm_anchor_relative_l2"],
        dynamics["warm_anchor_cosine_distance"],
    )
    return _cpu_preprocess_and_score(features, model)


def optimized_correctness_values(
    current_state: torch.Tensor,
    previous_state: torch.Tensor,
    previous_update: torch.Tensor | None,
    warm_anchor: torch.Tensor,
    model: CombinedModelTensors,
    eps: float = 1e-8,
) -> dict[str, Any]:
    raw, normalized, logit, probability = _optimized_combined_components(
        current_state, previous_state, previous_update, warm_anchor, model, eps
    )
    return {
        "features": raw.detach().cpu().tolist(),
        "normalized_features": normalized.detach().cpu().tolist(),
        "logit": float(logit.detach().cpu()),
        "probability": float(probability.detach().cpu()),
        "decision": bool((probability >= model.threshold).detach().cpu()),
    }


def assert_case_parity(
    state_case: StateCase,
    serialized: SerializedCombinedModel,
    tensors: CombinedModelTensors,
    eps: float = 1e-8,
) -> dict[str, Any]:
    before = {
        "current": state_case.current_state.detach().clone(),
        "previous": state_case.previous_state.detach().clone(),
        "anchor": state_case.warm_anchor.detach().clone(),
        "previous_update": (
            None
            if state_case.previous_update is None
            else state_case.previous_update.detach().clone()
        ),
    }
    reference = combined_current_diagnostic(
        state_case.current_state,
        state_case.previous_state,
        state_case.previous_update,
        state_case.warm_anchor,
        serialized,
        eps,
    )
    optimized = optimized_correctness_values(
        state_case.current_state,
        state_case.previous_state,
        state_case.previous_update,
        state_case.warm_anchor,
        tensors,
        eps,
    )
    raw = float(raw_mse_tensor(state_case.current_state, state_case.previous_state).detach().cpu())
    _require(_scalar_close(raw, reference["features"][0]), "raw-MSE parity failed")
    for index, (expected, actual) in enumerate(zip(reference["features"], optimized["features"])):
        if expected is None:
            _require(math.isnan(actual), f"feature {index}: expected null/NaN")
        else:
            _require(_scalar_close(expected, actual), f"feature {index}: parity failed")
    _require(
        np.allclose(
            reference["normalized_features"],
            optimized["normalized_features"],
            **TOLERANCE,
        ),
        "normalized-feature parity failed",
    )
    _require(_scalar_close(reference["logit"], optimized["logit"]), "logit parity failed")
    _require(
        _scalar_close(reference["probability"], optimized["probability"]),
        "probability parity failed",
    )
    _require(reference["decision"] == optimized["decision"], "decision parity failed")
    _require(torch.equal(before["current"], state_case.current_state), "current state mutated")
    _require(torch.equal(before["previous"], state_case.previous_state), "previous state mutated")
    _require(torch.equal(before["anchor"], state_case.warm_anchor), "warm anchor mutated")
    if state_case.previous_update is not None:
        _require(torch.equal(before["previous_update"], state_case.previous_update), "previous update mutated")
    return {
        "case_id": state_case.case_id,
        "iteration": state_case.iteration,
        "history_available": state_case.previous_update is not None,
        "raw_mse": raw,
        "probability": optimized["probability"],
        "decision": optimized["decision"],
        "finite": all(
            math.isfinite(value)
            for value in optimized["normalized_features"]
            + [optimized["logit"], optimized["probability"]]
        ),
        "inputs_bitwise_unchanged": True,
        "parity_passed": True,
    }


def optimized_host_transfer_audit() -> dict[str, int]:
    tensor_source = inspect.getsource(combined_optimized_tensor) + inspect.getsource(
        _optimized_combined_components
    )
    decision_source = inspect.getsource(combined_optimized_decision)
    return {
        "optimized_tensor_item_calls": tensor_source.count(".item("),
        "optimized_decision_item_calls": decision_source.count(".item("),
    }


def feature_family_tensors(
    current_state: torch.Tensor,
    previous_state: torch.Tensor,
    previous_update: torch.Tensor | None,
    warm_anchor: torch.Tensor,
    model: CombinedModelTensors,
    eps: float = 1e-8,
) -> dict[str, Any]:
    """Return untimed callables for a separate profiler decomposition pass."""

    current = current_state.float()
    previous = previous_state.float()
    update = current - previous
    raw_features = _optimized_combined_components(
        current_state, previous_state, previous_update, warm_anchor, model, eps
    )[0].detach()

    def existing_metrics() -> torch.Tensor:
        difference = current - previous
        raw = torch.mean(difference.square())
        return torch.stack(
            (
                raw,
                raw / (torch.mean(previous.square()) + eps),
                torch.linalg.vector_norm(difference.reshape(-1))
                / (torch.linalg.vector_norm(previous.reshape(-1)) + eps),
                1.0
                - F.cosine_similarity(
                    current.reshape(-1), previous.reshape(-1), dim=0, eps=eps
                ),
            )
        )

    def update_dynamics() -> torch.Tensor:
        state_ratio = torch.mean(current.square()).sqrt() / (
            torch.mean(previous.square()).sqrt() + eps
        )
        if previous_update is None:
            return state_ratio.reshape(1)
        prior = previous_update.float()
        prior_rms = torch.mean(prior.square()).sqrt()
        acceleration = update - prior
        return torch.stack(
            (
                torch.mean(update.square()).sqrt() / (prior_rms + eps),
                F.cosine_similarity(
                    update.reshape(-1), prior.reshape(-1), dim=0, eps=eps
                ),
                torch.mean(acceleration.square()).sqrt(),
                torch.mean(acceleration.square()).sqrt() / (prior_rms + eps),
                state_ratio,
            )
        )

    def token_quantiles() -> torch.Tensor:
        rms = torch.mean(update.square(), dim=-1).sqrt().reshape(-1)
        return torch.stack(
            (
                torch.quantile(rms, 0.50),
                torch.quantile(rms, 0.90),
                torch.quantile(rms, 0.95),
                torch.max(rms),
                torch.std(rms, unbiased=False) / (torch.mean(rms) + eps),
            )
        )

    def token_entropy_topk() -> torch.Tensor:
        rms = torch.mean(update.square(), dim=-1).sqrt().reshape(-1)
        energy = rms.square()
        total = torch.sum(energy)
        probabilities = energy / (total + eps)
        entropy = -torch.sum(
            torch.where(
                probabilities > 0,
                probabilities * torch.log(torch.clamp_min(probabilities, eps)),
                torch.zeros_like(probabilities),
            )
        ) / math.log(max(2, rms.numel()))
        top_count = max(1, int(math.ceil(0.10 * rms.numel())))
        top = torch.topk(energy, top_count).values.sum() / (total + eps)
        return torch.stack((entropy, top))

    def warm_anchor_features() -> torch.Tensor:
        anchor = warm_anchor.float()
        difference = current - anchor
        return torch.stack(
            (
                torch.linalg.vector_norm(difference.reshape(-1))
                / (torch.linalg.vector_norm(anchor.reshape(-1)) + eps),
                1.0
                - F.cosine_similarity(
                    current.reshape(-1), anchor.reshape(-1), dim=0, eps=eps
                ),
            )
        )

    def normalization_logistic_head() -> torch.Tensor:
        available = torch.isfinite(raw_features)
        raw_for_model = raw_features.to(dtype=model.imputation_medians.dtype)
        imputed = torch.where(available, raw_for_model, model.imputation_medians)
        expanded = []
        for index in range(18):
            expanded.append(imputed[index])
            if index in HISTORY_FEATURE_INDICES:
                expanded.append(
                    available[index].to(dtype=model.imputation_medians.dtype)
                )
        normalized = (torch.stack(expanded) - model.scaling_mean) / model.scaling_scale
        return torch.sigmoid(torch.dot(normalized, model.weights) + model.bias)

    return {
        "existing_metrics": existing_metrics,
        "update_dynamics": update_dynamics,
        "token_quantiles": token_quantiles,
        "token_entropy_top_k": token_entropy_topk,
        "warm_anchor_features": warm_anchor_features,
        "normalization_logistic_head": normalization_logistic_head,
    }


def deterministic_operation_order(
    *, workload_index: int, trial_index: int, case_index: int, seed: int
) -> list[str]:
    values = list(OPERATIONS)
    mixed_seed = int(seed) + 1_000_003 * workload_index + 10_007 * trial_index + 101 * case_index
    random.Random(mixed_seed).shuffle(values)
    return values


def summarize_latency_samples(samples: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    _require(bool(samples), "latency samples are empty")
    grouped: dict[tuple[str, str], list[Mapping[str, Any]]] = defaultdict(list)
    by_operation: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    per_workload: dict[tuple[str, str, str], list[Mapping[str, Any]]] = defaultdict(list)
    for row in samples:
        grouped[(str(row["operation"]), str(row["case_id"]))].append(row)
        by_operation[str(row["operation"])].append(row)
        per_workload[(str(row["workload_id"]), str(row["operation"]), str(row["case_id"]))].append(row)

    def stats(values: Sequence[float]) -> dict[str, float]:
        array = np.asarray(values, dtype=np.float64)
        mean = float(array.mean())
        std = float(array.std())
        return {
            "sample_count": int(array.size),
            "p50_ms": float(np.quantile(array, 0.50)),
            "p90_ms": float(np.quantile(array, 0.90)),
            "p95_ms": float(np.quantile(array, 0.95)),
            "mean_ms": mean,
            "std_ms": std,
            "coefficient_of_variation": std / mean if mean else 0.0,
        }

    aggregates = {}
    for (operation, case_id), rows in sorted(grouped.items()):
        aggregates[f"{operation}:{case_id}"] = {
            "operation": operation,
            "case_id": case_id,
            "cuda_event_device": stats([float(row["cuda_event_ms"]) for row in rows]),
            "synchronized_wall": stats([float(row["wall_time_ms"]) for row in rows]),
            "cuda_event_is_end_to_end_decision_latency": False,
        }
    workload_rows = []
    for (workload_id, operation, case_id), rows in sorted(per_workload.items()):
        workload_rows.append(
            {
                "workload_id": workload_id,
                "operation": operation,
                "case_id": case_id,
                "cuda_event_device": stats([float(row["cuda_event_ms"]) for row in rows]),
                "synchronized_wall": stats([float(row["wall_time_ms"]) for row in rows]),
            }
        )
    operation_case_ranking = sorted(
        aggregates,
        key=lambda key: aggregates[key]["synchronized_wall"]["p50_ms"],
    )
    operation_aggregates = {
        operation: {
            "cuda_event_device": stats(
                [float(row["cuda_event_ms"]) for row in rows]
            ),
            "synchronized_wall": stats(
                [float(row["wall_time_ms"]) for row in rows]
            ),
        }
        for operation, rows in sorted(by_operation.items())
    }
    operation_ranking = sorted(
        operation_aggregates,
        key=lambda key: operation_aggregates[key]["synchronized_wall"]["p50_ms"],
    )
    return {
        "timing_concepts": {
            "cuda_event_device": "device elapsed time only",
            "synchronized_wall": "host-visible synchronized end-to-end operation latency",
        },
        "aggregates": aggregates,
        "operation_aggregates_across_state_cases": operation_aggregates,
        "per_workload": workload_rows,
        "stable_operation_ranking_basis": (
            "per-operation synchronized-wall p50 pooled across k=2 and k>=3"
        ),
        "operation_ranking_fastest_to_slowest": operation_ranking,
        "operation_case_ranking_fastest_to_slowest": operation_case_ranking,
    }


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def load_actual_warm_workload_descriptors(
    manifest_path: Path,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Read unique real ACTUAL_WARM shards named by the frozen calibration manifest."""

    manifest_path = Path(manifest_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    run_root = Path(manifest["calibration_run_root"])
    descriptors = []
    for task_id in range(10):
        step_path = run_root / f"task{task_id}/steps.jsonl"
        for record in _read_jsonl(step_path):
            if record.get("action_head_workload_captured") is not True:
                continue
            if record.get("warm_start_used") is not True or record.get("initial_state_origin") != "cached":
                continue
            path = Path(record["action_head_workload_file"])
            if not path.is_absolute():
                path = step_path.parent / path
            identity = {
                "task_id": int(record["task_id"]),
                "episode_id": int(record["episode_id"]),
                "paired_trial_id": int(record["paired_trial_id"]),
                "prediction_step": int(record["prediction_step"]),
                "initial_state_id": int(record["initial_state_id"]),
                "episode_seed": int(record["episode_seed"]),
            }
            descriptors.append(
                {
                    "workload_id": (
                        f"task{identity['task_id']}:episode{identity['episode_id']}:"
                        f"prediction{identity['prediction_step']}"
                    ),
                    "identity": identity,
                    "actual_origin": "ACTUAL_WARM",
                    "path": path.resolve(),
                    "sha256": str(record["action_head_workload_sha256"]),
                }
            )
    descriptors.sort(
        key=lambda item: (
            item["identity"]["task_id"],
            item["identity"]["episode_id"],
            item["identity"]["prediction_step"],
        )
    )
    keys = [item["workload_id"] for item in descriptors]
    _require(len(keys) == len(set(keys)), "duplicate ACTUAL_WARM workload identity")
    provenance = {
        "manifest_path": str(manifest_path.resolve()),
        "manifest_sha256": sha256_file(manifest_path),
        "calibration_run_root": str(run_root.resolve()),
        "available_actual_warm_workload_count": len(descriptors),
        "counts_by_task": dict(
            sorted(Counter(str(item["identity"]["task_id"]) for item in descriptors).items())
        ),
    }
    return descriptors, provenance


def select_stratified_workloads(
    descriptors: Sequence[Mapping[str, Any]], count: int
) -> list[Mapping[str, Any]]:
    _require(count > 0, "workload count must be positive")
    _require(
        len(descriptors) >= count,
        f"requested {count} distinct ACTUAL_WARM workloads, but only {len(descriptors)} are available",
    )
    by_task: dict[int, list[Mapping[str, Any]]] = defaultdict(list)
    for descriptor in descriptors:
        _require(descriptor["actual_origin"] == "ACTUAL_WARM", "non-warm workload selected")
        by_task[int(descriptor["identity"]["task_id"])].append(descriptor)
    selected = []
    depth = 0
    while len(selected) < count:
        progressed = False
        for task_id in sorted(by_task):
            if depth < len(by_task[task_id]) and len(selected) < count:
                selected.append(by_task[task_id][depth])
                progressed = True
        _require(progressed, "cannot complete stratified workload selection")
        depth += 1
    return selected


def load_scheduler_replays(artifact_dir: Path) -> dict[str, list[dict[str, Any]]]:
    aliases = {
        "fixed_raw_mse_beta_0_05": "fixed_raw_mse",
        "combined": "combined",
    }
    rows = {value: [] for value in aliases.values()}
    with (Path(artifact_dir) / "oof_prediction_replays.csv").open(newline="", encoding="utf-8") as handle:
        for raw in csv.DictReader(handle):
            if raw["policy"] not in aliases:
                continue
            rows[aliases[raw["policy"]]].append(
                {
                    "task_id": int(raw["task_id"]),
                    "episode_id": int(raw["episode_id"]),
                    "prediction_id": int(raw["prediction_id"]),
                    "baseline_coda_calls": int(raw["baseline_coda_calls"]),
                    "scheduled_coda_calls": int(raw["scheduled_coda_calls"]),
                    "delta_k": int(raw["delta_k"]),
                    "trigger_k": int(raw["trigger_k"]),
                }
            )
    _require(all(len(value) == 2298 for value in rows.values()), "scheduler replay count mismatch")
    return rows


def gate_evaluations_per_prediction(replays: Sequence[Mapping[str, Any]]) -> float:
    _require(bool(replays), "scheduler replays are empty")
    evaluations = [int(item["trigger_k"]) - 1 for item in replays]
    _require(all(1 <= value <= 30 for value in evaluations), "invalid trigger-derived gate count")
    return float(np.mean(evaluations))


def project_policy_latency(
    replays: Sequence[Mapping[str, Any]],
    *,
    coda_latency_ms: float,
    recurrent_iteration_latency_ms: float,
    gate_decision_latency_ms: float,
    baseline_action_head_latency_ms: float,
) -> dict[str, Any]:
    count = len(replays)
    baseline_calls = sum(int(item["baseline_coda_calls"]) for item in replays)
    scheduled_calls = sum(int(item["scheduled_coda_calls"]) for item in replays)
    saved_per_prediction = (baseline_calls - scheduled_calls) / count
    mean_delta = float(np.mean([int(item["delta_k"]) for item in replays]))
    gate_evaluations = gate_evaluations_per_prediction(replays)
    gross = saved_per_prediction * coda_latency_ms
    recurrent_cost = mean_delta * recurrent_iteration_latency_ms
    overhead = gate_evaluations * gate_decision_latency_ms
    net = gross - recurrent_cost - overhead
    break_even = (
        (gross - recurrent_cost) / gate_evaluations if gate_evaluations else math.inf
    )
    return {
        "prediction_count": count,
        "baseline_coda_calls": baseline_calls,
        "scheduled_coda_calls": scheduled_calls,
        "saved_coda_calls_per_prediction": saved_per_prediction,
        "mean_delta_K": mean_delta,
        "gate_evaluations_per_prediction": gate_evaluations,
        "measured_coda_latency_ms": coda_latency_ms,
        "measured_recurrent_iteration_latency_ms": recurrent_iteration_latency_ms,
        "measured_gate_decision_latency_ms": gate_decision_latency_ms,
        "gross_coda_saving_ms": gross,
        "added_recurrent_cost_ms": recurrent_cost,
        "gate_overhead_ms": overhead,
        "projected_net_latency_change_ms": -net,
        "projected_net_saving_ms": net,
        "projected_percentage_change_vs_warm_start_SK1": (
            -net / baseline_action_head_latency_ms
        ),
        "break_even_gate_latency_ms": break_even,
        "measured_to_break_even_gate_latency_ratio": (
            gate_decision_latency_ms / break_even if break_even > 0 else math.inf
        ),
        "projection_only_not_end_to_end_LIBERO": True,
    }


def bootstrap_projection(
    replays: Sequence[Mapping[str, Any]],
    latency_samples: Mapping[str, Sequence[float]],
    *,
    draws: int = 2000,
    seed: int = DEFAULT_ORDER_SEED,
) -> dict[str, Any]:
    _require(draws >= 100, "bootstrap requires at least 100 draws")
    rng = np.random.default_rng(seed)
    count = len(replays)
    arrays = {
        key: np.asarray(value, dtype=np.float64) for key, value in latency_samples.items()
    }
    _require(
        set(arrays) == {"coda", "recurrent", "gate", "baseline"},
        "bootstrap latency sample families are incomplete",
    )
    values = []
    percentages = []
    for _ in range(draws):
        replay_indices = rng.integers(0, count, size=count)
        sampled = [replays[index] for index in replay_indices]
        measured = {
            name: float(array[rng.integers(0, len(array), size=len(array))].mean())
            for name, array in arrays.items()
        }
        projection = project_policy_latency(
            sampled,
            coda_latency_ms=measured["coda"],
            recurrent_iteration_latency_ms=measured["recurrent"],
            gate_decision_latency_ms=measured["gate"],
            baseline_action_head_latency_ms=measured["baseline"],
        )
        values.append(projection["projected_net_saving_ms"])
        percentages.append(-projection["projected_percentage_change_vs_warm_start_SK1"])
    return {
        "draws": draws,
        "seed": seed,
        "projected_net_saving_ms_95_percentile_interval": [
            float(np.quantile(values, 0.025)),
            float(np.quantile(values, 0.975)),
        ],
        "projected_net_saving_fraction_vs_SK1_95_percentile_interval": [
            float(np.quantile(percentages, 0.025)),
            float(np.quantile(percentages, 0.975)),
        ],
    }


def compare_independent_runs(
    current: Mapping[str, Any], previous: Mapping[str, Any]
) -> dict[str, Any]:
    current_aggregates = current["aggregates"]
    previous_aggregates = previous["aggregates"]
    shared = sorted(set(current_aggregates) & set(previous_aggregates))
    comparisons = []
    for key in shared:
        current_p50 = float(current_aggregates[key]["synchronized_wall"]["p50_ms"])
        previous_p50 = float(previous_aggregates[key]["synchronized_wall"]["p50_ms"])
        relative = abs(current_p50 - previous_p50) / previous_p50
        comparisons.append(
            {
                "operation_case": key,
                "current_p50_ms": current_p50,
                "previous_p50_ms": previous_p50,
                "absolute_relative_difference": relative,
                "warning_over_10_percent": relative > 0.10,
            }
        )
    current_operations = current["operation_aggregates_across_state_cases"]
    previous_operations = previous["operation_aggregates_across_state_cases"]
    stable_pair_checks = []
    operation_names = sorted(set(current_operations) & set(previous_operations))
    for left_index, left in enumerate(operation_names):
        for right in operation_names[left_index + 1 :]:
            previous_left = float(
                previous_operations[left]["synchronized_wall"]["p50_ms"]
            )
            previous_right = float(
                previous_operations[right]["synchronized_wall"]["p50_ms"]
            )
            relative_separation = abs(previous_left - previous_right) / min(
                previous_left, previous_right
            )
            if relative_separation <= 0.10:
                continue
            current_left = float(
                current_operations[left]["synchronized_wall"]["p50_ms"]
            )
            current_right = float(
                current_operations[right]["synchronized_wall"]["p50_ms"]
            )
            stable_pair_checks.append(
                (previous_left < previous_right) == (current_left < current_right)
            )
    return {
        "identical_operation_ranking": (
            current["operation_ranking_fastest_to_slowest"]
            == previous["operation_ranking_fastest_to_slowest"]
        ),
        "stable_operation_ranking_with_10_percent_equivalence": all(
            stable_pair_checks
        ),
        "stable_ranking_definition": (
            "preserve ordering for operation-family p50 values separated by more than 10%; "
            "near-ties are treated as equivalent"
        ),
        "comparisons": comparisons,
        "warning_count": sum(item["warning_over_10_percent"] for item in comparisons),
    }


def write_json(path: Path, value: Any) -> None:
    path.write_text(canonical_json(value), encoding="utf-8")
