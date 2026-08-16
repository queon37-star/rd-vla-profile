"""Versioned storage for deployment-matched Action-Delta Gate shadows."""

from __future__ import annotations

import hashlib
import json
import math
import os
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch

from prismatic.models.action_delta_gate_shadow import (
    ACTION_DELTA_GATE_SHADOW_DEVELOPMENT_TASK_IDS,
    ACTION_DELTA_GATE_SHADOW_FEATURE_NAMES,
    ACTION_DELTA_GATE_SHADOW_SCHEMA_VERSION,
)
from prismatic.models.action_head_workload import sha256_file


ACTION_DELTA_GATE_SHADOW_COLLECTION_MODE = "deployment_matched_pre_coda_shadow"


class ActionDeltaGateShadowCollectionError(ValueError):
    """Raised when a shadow dataset is incomplete or internally inconsistent."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ActionDeltaGateShadowCollectionError(message)


def canonical_json_sha256(value: Any) -> str:
    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def tensor_sha256(value: torch.Tensor) -> str:
    _require(torch.is_tensor(value), "tensor hash input must be a tensor")
    tensor = value.detach().to(device="cpu", copy=True).contiguous()
    return hashlib.sha256(tensor.view(torch.uint8).numpy().tobytes()).hexdigest()


def tensor_metadata(value: torch.Tensor) -> dict[str, Any]:
    _require(torch.is_tensor(value), "tensor metadata input must be a tensor")
    return {
        "shape": list(value.shape),
        "dtype": str(value.dtype),
        "stride": list(value.stride()),
        "contiguous": bool(value.is_contiguous()),
    }


def globally_unique_trajectory_id(
    *,
    initial_state_manifest_sha256: str,
    task_id: int,
    initial_state_id: int,
    episode_seed: int,
) -> str:
    _require(
        isinstance(initial_state_manifest_sha256, str)
        and len(initial_state_manifest_sha256) == 64,
        "initial-state manifest SHA-256 is required for global identity",
    )
    material = {
        "namespace": "rdvla-action-delta-shadow-trajectory-v1",
        "initial_state_manifest_sha256": initial_state_manifest_sha256,
        "task_id": int(task_id),
        "initial_state_id": int(initial_state_id),
        "episode_seed": int(episode_seed),
    }
    return f"adgs-{canonical_json_sha256(material)}"


def _cpu_tensor(value: Any, name: str) -> torch.Tensor:
    if not torch.is_tensor(value):
        value = torch.as_tensor(value)
    tensor = value.detach().to(device="cpu", copy=True).contiguous()
    _require(bool(torch.isfinite(tensor.float()).all().item()), f"{name} is non-finite")
    return tensor


def _decorate_tensors(record: dict[str, Any]) -> None:
    tensors = record.get("tensors")
    _require(isinstance(tensors, Mapping) and tensors, "transition tensors are missing")
    record["tensor_metadata"] = {
        name: tensor_metadata(value) for name, value in tensors.items()
    }
    record["tensor_sha256"] = {
        name: tensor_sha256(value) for name, value in tensors.items()
    }


def _validate_transition(record: Mapping[str, Any]) -> None:
    _require(
        record.get("schema_version") == ACTION_DELTA_GATE_SHADOW_SCHEMA_VERSION,
        "transition schema mismatch",
    )
    anchor = record.get("anchor_iteration")
    terminal = record.get("terminal_iteration")
    _require(
        isinstance(anchor, int) and terminal == anchor + 1,
        "transition iteration identity is not adjacent",
    )
    for name in (
        "gate_score",
        "gate_threshold",
        "exact_adjacent_action_mse",
        "recurrence_mse_threshold",
        "residual",
    ):
        value = record.get(name)
        _require(isinstance(value, (int, float)) and math.isfinite(float(value)), f"invalid {name}")
    score = float(record["gate_score"])
    threshold = float(record["gate_threshold"])
    exact_mse = float(record["exact_adjacent_action_mse"])
    recurrence_threshold = float(record["recurrence_mse_threshold"])
    predicted_trigger = score <= threshold
    exact_safe = exact_mse < recurrence_threshold
    _require(record.get("predicted_trigger") is predicted_trigger, "predicted-trigger label mismatch")
    _require(record.get("exact_safe") is exact_safe, "exact-safe label mismatch")
    _require(record.get("false_safe") is (predicted_trigger and not exact_safe), "false-safe label mismatch")
    _require(float(record["residual"]) == exact_mse - score, "residual mismatch")
    features = record.get("features")
    _require(isinstance(features, Mapping), "runtime features are missing")
    _require(set(features) == set(ACTION_DELTA_GATE_SHADOW_FEATURE_NAMES), "runtime feature schema mismatch")
    _require(float(features["predicted_action_delta_mse"]) == score, "feature score mismatch")
    _require(int(features["terminal_iteration"]) == terminal, "feature terminal iteration mismatch")
    tensors = record.get("tensors")
    metadata = record.get("tensor_metadata")
    hashes = record.get("tensor_sha256")
    _require(isinstance(tensors, Mapping), "transition tensors are missing")
    _require(set(tensors) == {
        "anchor_state",
        "current_state",
        "latent_delta_bfloat16",
        "anchor_action",
        "exact_terminal_action",
        "predicted_delta_action",
        "previous_latent_delta_bfloat16",
    }, "transition tensor schema mismatch")
    for name, tensor in tensors.items():
        _require(torch.is_tensor(tensor) and tensor.device.type == "cpu", f"{name} must be a CPU tensor")
        _require(bool(torch.isfinite(tensor.float()).all().item()), f"{name} is non-finite")
        _require(metadata.get(name) == tensor_metadata(tensor), f"{name} metadata mismatch")
        _require(hashes.get(name) == tensor_sha256(tensor), f"{name} hash mismatch")
    _require(tensors["latent_delta_bfloat16"].dtype == torch.bfloat16, "latent delta must be BF16")
    previous_latent_delta = tensors["previous_latent_delta_bfloat16"]
    _require(
        previous_latent_delta.dtype == torch.bfloat16,
        "previous latent delta must be BF16",
    )
    _require(
        previous_latent_delta.numel() == 0
        or tuple(previous_latent_delta.shape)
        == tuple(tensors["latent_delta_bfloat16"].shape),
        "previous latent delta shape mismatch",
    )
    reproduced_delta = (
        tensors["current_state"].float() - tensors["anchor_state"].float()
    ).to(torch.bfloat16)
    _require(torch.equal(reproduced_delta, tensors["latent_delta_bfloat16"]), "BF16 latent delta mismatch")
    reproduced_score = float(tensors["predicted_delta_action"].float().square().mean().item())
    _require(reproduced_score == score, "predicted delta does not reproduce score")
    reproduced_exact = float(
        (tensors["exact_terminal_action"] - tensors["anchor_action"]).square().mean().item()
    )
    _require(reproduced_exact == exact_mse, "exact action pair does not reproduce label")


def build_shadow_prediction_payload(
    raw_shadow: Mapping[str, Any],
    *,
    task_id: int,
    task_name: str,
    episode_id: int,
    initial_state_id: int,
    paired_trial_id: int,
    episode_seed: int,
    prediction_id: int,
    environment_timestep: int,
    initial_state_manifest_sha256: str,
    protocol_identity: Mapping[str, Any],
    warm_start_metadata: Mapping[str, Any],
    returned_action: Any,
) -> dict[str, Any]:
    """Attach deployment identity and immutable parity evidence to model data."""

    _require(
        raw_shadow.get("schema_version") == ACTION_DELTA_GATE_SHADOW_SCHEMA_VERSION,
        "model shadow schema mismatch",
    )
    trajectory_id = globally_unique_trajectory_id(
        initial_state_manifest_sha256=initial_state_manifest_sha256,
        task_id=task_id,
        initial_state_id=initial_state_id,
        episode_seed=episode_seed,
    )
    production = dict(raw_shadow.get("production_trace") or {})
    returned_normalized = _cpu_tensor(
        production.get("returned_normalized_action"), "returned normalized action"
    )
    exact_outputs = _cpu_tensor(production.get("exact_coda_outputs"), "exact Coda outputs")
    returned_action_tensor = _cpu_tensor(returned_action, "returned action")
    k_value = int(production.get("K_t", 0))
    exact_iterations = list(production.get("exact_coda_output_iterations") or [])
    iteration_mse = list(production.get("iteration_mse") or [])
    _require(k_value >= 1, "production K must be positive")
    _require(exact_iterations == list(range(1, k_value + 1)), "exact Coda iterations must be 1..K")
    _require(exact_outputs.shape[0] == k_value, "exact Coda output count must equal K")
    _require(int(production.get("exact_coda_call_count", -1)) == k_value, "production Coda count must equal K")
    _require(len(iteration_mse) == max(0, k_value - 1), "iteration-MSE count must equal K-1")
    _require(torch.equal(returned_normalized, exact_outputs[-1]), "returned normalized action must be terminal exact Coda")

    transitions = []
    for raw_transition in raw_shadow.get("transitions", []):
        transition = dict(raw_transition)
        transition["tensors"] = {
            name: _cpu_tensor(value, name)
            for name, value in transition["tensors"].items()
        }
        transition_identity = {
            "trajectory_id": trajectory_id,
            "task_id": int(task_id),
            "episode_id": int(episode_id),
            "initial_state_id": int(initial_state_id),
            "episode_seed": int(episode_seed),
            "action_prediction_index": int(prediction_id),
            "environment_timestep": int(environment_timestep),
            "anchor_iteration": int(transition["anchor_iteration"]),
            "terminal_iteration": int(transition["terminal_iteration"]),
        }
        transition["identity"] = transition_identity
        transition["transition_id"] = (
            f"adgst-{canonical_json_sha256(transition_identity)}"
        )
        transition["warm_start_source_iteration"] = warm_start_metadata.get(
            "source_iteration"
        )
        transition["warm_start_source_K"] = warm_start_metadata.get("source_K")
        _decorate_tensors(transition)
        _validate_transition(transition)
        transitions.append(transition)

    prediction_identity = {
        "trajectory_id": trajectory_id,
        "task_id": int(task_id),
        "episode_id": int(episode_id),
        "initial_state_id": int(initial_state_id),
        "paired_trial_id": int(paired_trial_id),
        "episode_seed": int(episode_seed),
        "action_prediction_index": int(prediction_id),
        "environment_timestep": int(environment_timestep),
    }
    parity = {
        "shadow_values_used_for_control": False,
        "warm_only_control_contract_verified": True,
        "K_t": k_value,
        "stop_reason": production.get("stop_reason"),
        "adaptive_stop": bool(production.get("adaptive_stop")),
        "returned_action_sha256": tensor_sha256(returned_action_tensor),
        "returned_normalized_action_sha256": tensor_sha256(returned_normalized),
        "exact_coda_outputs_sha256": tensor_sha256(exact_outputs),
        "exact_coda_call_count": k_value,
        "iteration_mse": [float(value) for value in iteration_mse],
    }
    return {
        "schema_version": ACTION_DELTA_GATE_SHADOW_SCHEMA_VERSION,
        "collection_mode": ACTION_DELTA_GATE_SHADOW_COLLECTION_MODE,
        "identity": prediction_identity,
        "prediction_id": f"adgsp-{canonical_json_sha256(prediction_identity)}",
        "task_name": str(task_name),
        "protocol_identity": dict(protocol_identity),
        "warm_start_metadata": dict(warm_start_metadata),
        "collection_applied": bool(raw_shadow.get("collection_applied")),
        "ineligible_reason": raw_shadow.get("ineligible_reason"),
        "min_terminal_iteration": int(raw_shadow["min_terminal_iteration"]),
        "gate_threshold": float(raw_shadow["gate_threshold"]),
        "transitions": transitions,
        "collection_error": raw_shadow.get("error"),
        "production_parity": parity,
        "production_tensors": {
            "returned_action": returned_action_tensor,
            "returned_normalized_action": returned_normalized,
            "exact_coda_outputs": exact_outputs,
        },
    }


def _quantiles(values: Sequence[float]) -> dict[str, float | None]:
    if not values:
        return {name: None for name in ("p0", "p25", "p50", "p75", "p90", "p95", "p99", "p100")}
    quantiles = np.percentile(np.asarray(values, dtype=np.float64), [0, 25, 50, 75, 90, 95, 99, 100])
    return {
        name: float(value)
        for name, value in zip(
            ("p0", "p25", "p50", "p75", "p90", "p95", "p99", "p100"),
            quantiles,
        )
    }


def summarize_shadow_predictions(predictions: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    by_task: dict[int, list[Mapping[str, Any]]] = defaultdict(list)
    for prediction in predictions:
        by_task[int(prediction["identity"]["task_id"])].append(prediction)

    def summarize(items: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
        transitions = [transition for item in items for transition in item["transitions"]]
        triggers = [row for row in transitions if row["predicted_trigger"]]
        safe_triggers = [row for row in triggers if row["exact_safe"]]
        false_safe = [row for row in triggers if row["false_safe"]]
        trajectory_ids = {item["identity"]["trajectory_id"] for item in items}
        successful = {
            item["identity"]["trajectory_id"]
            for item in items
            if item.get("episode_success") is True
        }
        return {
            "trajectories": len(trajectory_ids),
            "successful_trajectories": len(successful),
            "predictions": len(items),
            "eligible_rows": len(transitions),
            "predicted_triggers": len(triggers),
            "exact_safe_triggers": len(safe_triggers),
            "false_safe_triggers": len(false_safe),
            "false_safe_rate_among_predicted_triggers": (
                len(false_safe) / len(triggers) if triggers else None
            ),
            "residual_quantiles": _quantiles([float(row["residual"]) for row in transitions]),
            "terminal_iteration_distribution": dict(
                sorted(Counter(str(row["terminal_iteration"]) for row in transitions).items())
            ),
        }

    return {
        "by_task": {
            str(task_id): summarize(items)
            for task_id, items in sorted(by_task.items())
        },
        "aggregate": summarize(predictions),
    }


def _atomic_write_json(path: Path, value: Mapping[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    try:
        temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


class ActionDeltaGateShadowWriter:
    """Atomic, non-overwriting writer for a complete Phase-A collection."""

    def __init__(
        self,
        output_dir: Path,
        *,
        shard_size: int,
        expected_task_ids: Sequence[int],
        expected_trajectories_per_task: int,
        source_commit: str,
        artifact_identity: Mapping[str, Any],
        checkpoint_identity: Mapping[str, Any],
        initial_state_manifest_identity: Mapping[str, Any],
        configuration: Mapping[str, Any],
        run_identity: Mapping[str, Any],
    ) -> None:
        _require(isinstance(shard_size, int) and not isinstance(shard_size, bool) and shard_size >= 1, "shard size must be >= 1")
        expected = tuple(int(task) for task in expected_task_ids)
        _require(bool(expected) and len(set(expected)) == len(expected), "expected task IDs must be unique")
        _require(set(expected).issubset(ACTION_DELTA_GATE_SHADOW_DEVELOPMENT_TASK_IDS), "Task 4/5 cannot enter shadow calibration")
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        if any(self.output_dir.iterdir()):
            raise FileExistsError(f"refusing to reuse non-empty shadow directory: {self.output_dir}")
        self.shard_size = shard_size
        self.expected_task_ids = expected
        self.expected_trajectories_per_task = int(expected_trajectories_per_task)
        self.source_commit = str(source_commit)
        self.artifact_identity = dict(artifact_identity)
        self.checkpoint_identity = dict(checkpoint_identity)
        self.initial_state_manifest_identity = dict(initial_state_manifest_identity)
        self.configuration = dict(configuration)
        self.run_identity = dict(run_identity)
        self._buffer: list[dict[str, Any]] = []
        self._predictions: list[dict[str, Any]] = []
        self._descriptors: list[dict[str, Any]] = []
        self._prediction_ids: set[str] = set()
        self._transition_ids: set[str] = set()
        self._shard_index = 0
        self._write_manifest(complete=False)

    @property
    def manifest_path(self) -> Path:
        return self.output_dir / "manifest.json"

    def add_episode(self, predictions: Sequence[Mapping[str, Any]], *, success: bool) -> None:
        for source in predictions:
            prediction = dict(source)
            prediction["episode_success"] = bool(success)
            prediction_id = str(prediction["prediction_id"])
            _require(prediction_id not in self._prediction_ids, f"duplicate prediction identity: {prediction_id}")
            self._prediction_ids.add(prediction_id)
            for transition in prediction["transitions"]:
                transition_id = str(transition["transition_id"])
                _require(transition_id not in self._transition_ids, f"duplicate transition identity: {transition_id}")
                self._transition_ids.add(transition_id)
                _validate_transition(transition)
            _require(prediction.get("collection_error") is None, "shadow predictor/recording error is not calibratable")
            _require(prediction["production_parity"]["warm_only_control_contract_verified"] is True, "Warm-only parity contract failed")
            self._buffer.append(prediction)
            self._predictions.append(prediction)
            if len(self._buffer) >= self.shard_size:
                self.flush()

    def flush(self) -> None:
        if not self._buffer:
            return
        name = f"action_delta_shadow_{self._shard_index:05d}.pt"
        path = self.output_dir / name
        if path.exists():
            raise FileExistsError(f"refusing to overwrite shadow shard: {path}")
        temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
        try:
            torch.save(
                {
                    "schema_version": ACTION_DELTA_GATE_SHADOW_SCHEMA_VERSION,
                    "collection_mode": ACTION_DELTA_GATE_SHADOW_COLLECTION_MODE,
                    "predictions": self._buffer,
                },
                temporary,
            )
            os.replace(temporary, path)
        finally:
            if temporary.exists():
                temporary.unlink()
        shard_hash = sha256_file(path)
        for index, prediction in enumerate(self._buffer):
            identity = prediction["identity"]
            self._descriptors.append(
                {
                    "prediction_id": prediction["prediction_id"],
                    "trajectory_id": identity["trajectory_id"],
                    "task_id": identity["task_id"],
                    "initial_state_id": identity["initial_state_id"],
                    "episode_seed": identity["episode_seed"],
                    "action_prediction_index": identity["action_prediction_index"],
                    "environment_timestep": identity["environment_timestep"],
                    "eligible_row_count": len(prediction["transitions"]),
                    "episode_success": prediction["episode_success"],
                    "production_parity": prediction["production_parity"],
                    "shard_path": name,
                    "shard_index": index,
                    "shard_sha256": shard_hash,
                }
            )
        self._buffer = []
        self._shard_index += 1
        self._write_manifest(complete=False)

    def _manifest(self, *, complete: bool) -> dict[str, Any]:
        task_state_seed = sorted(
            {
                (
                    int(item["task_id"]),
                    int(item["initial_state_id"]),
                    int(item["episode_seed"]),
                    str(item["trajectory_id"]),
                )
                for item in self._descriptors
            }
        )
        summary = summarize_shadow_predictions(self._predictions)
        parity = {
            "shadow_values_used_for_control": False,
            "prediction_contract_checks_passed": len(self._predictions),
            "prediction_contract_check_failures": 0,
            "verified_fields": [
                "returned_actions",
                "recurrent_K",
                "exact_Coda_outputs",
                "Warm-only_adaptive_stop",
            ],
        }
        manifest = {
            "schema_version": ACTION_DELTA_GATE_SHADOW_SCHEMA_VERSION,
            "collection_mode": ACTION_DELTA_GATE_SHADOW_COLLECTION_MODE,
            "complete": bool(complete),
            "source_commit": self.source_commit,
            "artifact_identity": self.artifact_identity,
            "checkpoint_identity": self.checkpoint_identity,
            "initial_state_manifest_identity": self.initial_state_manifest_identity,
            "configuration": self.configuration,
            "configuration_sha256": canonical_json_sha256(self.configuration),
            "collector_schema_version": ACTION_DELTA_GATE_SHADOW_SCHEMA_VERSION,
            "expected_task_ids": list(self.expected_task_ids),
            "expected_trajectories_per_task": self.expected_trajectories_per_task,
            "task_state_seed_list": [
                {
                    "task_id": task,
                    "initial_state_id": state,
                    "episode_seed": seed,
                    "trajectory_id": trajectory,
                }
                for task, state, seed, trajectory in task_state_seed
            ],
            "run_identity": self.run_identity,
            "prediction_count": len(self._descriptors),
            "transition_count": sum(item["eligible_row_count"] for item in self._descriptors),
            "shard_count": self._shard_index,
            "summary": summary,
            "production_parity": parity,
            "dataset_identity_sha256": canonical_json_sha256(
                [
                    {
                        "prediction_id": item["prediction_id"],
                        "shard_sha256": item["shard_sha256"],
                    }
                    for item in self._descriptors
                ]
            ),
            "predictions": self._descriptors,
        }
        return manifest

    def _write_manifest(self, *, complete: bool) -> None:
        _atomic_write_json(self.manifest_path, self._manifest(complete=complete))

    def finalize(self) -> Path:
        self.flush()
        _require(bool(self._descriptors), "cannot finalize an empty shadow collection")
        summary = summarize_shadow_predictions(self._predictions)
        observed_tasks = set(int(task) for task in summary["by_task"])
        _require(observed_tasks == set(self.expected_task_ids), "Phase-A task coverage is incomplete")
        for task_id in self.expected_task_ids:
            trajectories = summary["by_task"][str(task_id)]["trajectories"]
            _require(
                trajectories == self.expected_trajectories_per_task,
                f"task {task_id} trajectory count mismatch: expected={self.expected_trajectories_per_task}, actual={trajectories}",
            )
        self._write_manifest(complete=True)
        return self.manifest_path


def load_action_delta_gate_shadow_collection(
    manifest_path: Path,
    *,
    require_complete: bool = True,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Validate a saved collection and return its manifest and predictions."""

    path = Path(manifest_path)
    manifest = json.loads(path.read_text(encoding="utf-8"))
    _require(
        manifest.get("schema_version") == ACTION_DELTA_GATE_SHADOW_SCHEMA_VERSION,
        "manifest schema mismatch",
    )
    _require(
        manifest.get("collection_mode") == ACTION_DELTA_GATE_SHADOW_COLLECTION_MODE,
        "manifest collection mode mismatch",
    )
    if require_complete:
        _require(manifest.get("complete") is True, "collection manifest is incomplete")
    expected_tasks = tuple(int(value) for value in manifest.get("expected_task_ids", []))
    _require(
        set(expected_tasks).issubset(ACTION_DELTA_GATE_SHADOW_DEVELOPMENT_TASK_IDS),
        "manifest contains a forbidden calibration task",
    )
    _require(
        manifest.get("configuration_sha256")
        == canonical_json_sha256(manifest.get("configuration")),
        "configuration hash mismatch",
    )

    predictions: list[dict[str, Any]] = []
    prediction_ids: set[str] = set()
    transition_ids: set[str] = set()
    shard_cache: dict[str, Mapping[str, Any]] = {}
    for descriptor in manifest.get("predictions", []):
        shard_path = path.parent / descriptor["shard_path"]
        _require(shard_path.is_file(), f"missing shadow shard: {shard_path}")
        _require(
            sha256_file(shard_path) == descriptor["shard_sha256"],
            f"shadow shard hash mismatch: {shard_path}",
        )
        key = str(shard_path)
        if key not in shard_cache:
            shard_cache[key] = torch.load(
                shard_path, map_location="cpu", weights_only=True
            )
        shard = shard_cache[key]
        _require(
            shard.get("schema_version") == ACTION_DELTA_GATE_SHADOW_SCHEMA_VERSION,
            "shard schema mismatch",
        )
        _require(
            shard.get("collection_mode") == ACTION_DELTA_GATE_SHADOW_COLLECTION_MODE,
            "shard collection mode mismatch",
        )
        index = int(descriptor["shard_index"])
        items = shard.get("predictions", [])
        _require(0 <= index < len(items), "shard prediction index is out of range")
        prediction = dict(items[index])
        prediction_id = str(prediction["prediction_id"])
        _require(prediction_id == descriptor["prediction_id"], "prediction descriptor mismatch")
        _require(prediction_id not in prediction_ids, "duplicate prediction identity")
        prediction_ids.add(prediction_id)
        _require(
            int(prediction["identity"]["task_id"]) in expected_tasks,
            "prediction task is outside the declared development set",
        )
        parity = prediction["production_parity"]
        tensors = prediction["production_tensors"]
        _require(parity["shadow_values_used_for_control"] is False, "shadow control contamination")
        _require(
            tensor_sha256(tensors["returned_action"])
            == parity["returned_action_sha256"],
            "returned action hash mismatch",
        )
        _require(
            tensor_sha256(tensors["returned_normalized_action"])
            == parity["returned_normalized_action_sha256"],
            "normalized returned action hash mismatch",
        )
        _require(
            tensor_sha256(tensors["exact_coda_outputs"])
            == parity["exact_coda_outputs_sha256"],
            "exact Coda output hash mismatch",
        )
        _require(
            torch.equal(
                tensors["returned_normalized_action"],
                tensors["exact_coda_outputs"][-1],
            ),
            "returned normalized action differs from terminal exact Coda",
        )
        for transition in prediction["transitions"]:
            _validate_transition(transition)
            transition_id = str(transition["transition_id"])
            _require(transition_id not in transition_ids, "duplicate transition identity")
            transition_ids.add(transition_id)
        predictions.append(prediction)

    _require(
        len(predictions) == int(manifest.get("prediction_count", -1)),
        "manifest prediction count mismatch",
    )
    _require(
        len(transition_ids) == int(manifest.get("transition_count", -1)),
        "manifest transition count mismatch",
    )
    _require(
        len(shard_cache) == int(manifest.get("shard_count", -1)),
        "manifest shard count mismatch",
    )
    _require(
        summarize_shadow_predictions(predictions) == manifest.get("summary"),
        "manifest summary mismatch",
    )
    expected_identity = canonical_json_sha256(
        [
            {
                "prediction_id": item["prediction_id"],
                "shard_sha256": item["shard_sha256"],
            }
            for item in manifest.get("predictions", [])
        ]
    )
    _require(
        expected_identity == manifest.get("dataset_identity_sha256"),
        "dataset identity hash mismatch",
    )
    return manifest, predictions
