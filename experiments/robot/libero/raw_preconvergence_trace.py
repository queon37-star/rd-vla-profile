"""Versioned, fail-closed storage for optional raw preconvergence shadows."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

import torch

from prismatic.models.action_head_workload import sha256_file


RAW_PRECONVERGENCE_SCHEMA_VERSION = 2
RAW_PRECONVERGENCE_COLLECTION_MODE = "optional_post_production_shadow"
RAW_PRECONVERGENCE_ORIGINS = {"ACTUAL_WARM", "COLD_PRIMARY", "COLD_RETRY"}
RAW_PRECONVERGENCE_TENSORS = ("states", "actions")


class RawPreconvergenceTraceError(ValueError):
    """Raised when raw shadow data is incomplete, corrupt, or incompatible."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise RawPreconvergenceTraceError(message)


def canonical_json_sha256(value: Any) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def tensor_content_sha256(tensor: torch.Tensor) -> str:
    _require(torch.is_tensor(tensor), "tensor hash input must be a tensor")
    value = tensor.detach().to(device="cpu", copy=True).contiguous()
    return hashlib.sha256(value.view(torch.uint8).numpy().tobytes()).hexdigest()


def tensor_metadata(tensor: torch.Tensor) -> dict[str, Any]:
    _require(torch.is_tensor(tensor), "tensor metadata input must be a tensor")
    return {
        "shape": list(tensor.shape),
        "stride": list(tensor.stride()),
        "dtype": str(tensor.dtype),
        "layout": str(tensor.layout),
        "contiguous": bool(tensor.is_contiguous()),
    }


def current_source_commit(repo_root: Optional[Path] = None) -> str:
    root = Path(repo_root or Path(__file__).resolve().parents[3])
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def checkpoint_identity(checkpoint: Path) -> dict[str, Any]:
    """Hash every checkpoint file once, outside action-inference timing."""

    root = Path(checkpoint).resolve()
    _require(root.exists(), f"checkpoint does not exist: {root}")
    files = [root] if root.is_file() else sorted(path for path in root.rglob("*") if path.is_file())
    entries = []
    for path in files:
        relative = path.name if root.is_file() else str(path.relative_to(root))
        entries.append(
            {
                "path": relative,
                "size_bytes": int(path.stat().st_size),
                "sha256": sha256_file(path),
            }
        )
    _require(bool(entries), f"checkpoint contains no files: {root}")
    return {
        "path": str(root),
        "files": entries,
        "sha256": canonical_json_sha256(entries),
    }


def _atomic_write_json(path: Path, value: Mapping[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    try:
        temporary.write_text(
            json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _validate_prediction_payload(payload: Mapping[str, Any]) -> None:
    _require(payload.get("schema_version") == RAW_PRECONVERGENCE_SCHEMA_VERSION, "raw prediction schema mismatch")
    identity = payload.get("identity")
    _require(isinstance(identity, Mapping), "raw prediction identity is missing")
    for field in ("task_id", "episode_id", "prediction_id", "timestep"):
        value = identity.get(field)
        _require(isinstance(value, int) and not isinstance(value, bool) and value >= 0, f"invalid identity field: {field}")
    origin = payload.get("actual_origin")
    _require(origin in RAW_PRECONVERGENCE_ORIGINS, f"invalid raw origin: {origin!r}")
    checkpoint = payload.get("checkpoint_identity")
    _require(
        isinstance(checkpoint, Mapping)
        and isinstance(checkpoint.get("path"), str)
        and isinstance(checkpoint.get("sha256"), str)
        and bool(checkpoint.get("sha256")),
        "checkpoint identity is missing",
    )
    _require(isinstance(payload.get("source_commit"), str) and bool(payload.get("source_commit")), "source commit is missing")
    _require(isinstance(payload.get("run_identity"), Mapping), "run identity is missing")
    maximum_depth = int(payload.get("maximum_shadow_depth", 0))
    valid_length = int(payload.get("valid_trajectory_length", 0))
    baseline_k = int(payload.get("production_terminal_k", 0))
    _require(maximum_depth >= 2, "maximum shadow depth must be at least 2")
    _require(valid_length == maximum_depth, "raw trajectory must cover the full requested depth")
    _require(1 <= baseline_k <= maximum_depth, "invalid production terminal K")

    tensors = payload.get("tensors")
    metadata = payload.get("tensor_metadata")
    hashes = payload.get("tensor_sha256")
    _require(isinstance(tensors, Mapping), "raw tensors are missing")
    _require(isinstance(metadata, Mapping), "raw tensor metadata is missing")
    _require(isinstance(hashes, Mapping), "raw tensor hashes are missing")
    _require(set(tensors) == set(RAW_PRECONVERGENCE_TENSORS), "raw tensor fields are incomplete")
    for name in RAW_PRECONVERGENCE_TENSORS:
        tensor = tensors[name]
        _require(torch.is_tensor(tensor) and tensor.device.type == "cpu", f"{name} must be a CPU tensor")
        _require(tensor.shape[0] == maximum_depth, f"{name} trajectory length mismatch")
        _require(bool(torch.isfinite(tensor.float()).all().item()), f"{name} contains non-finite values")
        _require(metadata.get(name) == tensor_metadata(tensor), f"{name} metadata mismatch")
        _require(hashes.get(name) == tensor_content_sha256(tensor), f"{name} content hash mismatch")

    mse = payload.get("action_mse")
    sources = payload.get("action_mse_source")
    phases = payload.get("action_mse_phase")
    _require(isinstance(mse, Sequence) and len(mse) == maximum_depth + 1, "action-MSE length mismatch")
    _require(isinstance(sources, Sequence) and len(sources) == maximum_depth + 1, "MSE-source length mismatch")
    _require(isinstance(phases, Sequence) and len(phases) == maximum_depth + 1, "MSE-phase length mismatch")
    _require(mse[0] is None and mse[1] is None, "k=0/1 action MSE must be null")
    for k in range(2, maximum_depth + 1):
        expected_source = "production_native_bf16" if k <= baseline_k else "shadow_tail_fp32"
        expected_phase = "production" if k <= baseline_k else "shadow_tail"
        _require(sources[k] == expected_source, f"k={k}: action-MSE source mismatch")
        _require(phases[k] == expected_phase, f"k={k}: action-MSE phase mismatch")
        value = mse[k]
        _require(isinstance(value, (int, float)) and torch.isfinite(torch.tensor(float(value))).item(), f"k={k}: invalid action MSE")
    production_mse = payload.get("production_iteration_mse")
    _require(
        isinstance(production_mse, Sequence)
        and len(production_mse) == max(0, baseline_k - 1),
        "native production iteration-MSE length mismatch",
    )
    for offset, value in enumerate(production_mse, start=2):
        _require(float(value) == float(mse[offset]), f"k={offset}: native production MSE mismatch")
    threshold = float(payload.get("action_mse_threshold", float("nan")))
    _require(torch.isfinite(torch.tensor(threshold)).item(), "invalid action-MSE threshold")
    first_hit = next((k for k in range(2, maximum_depth + 1) if float(mse[k]) < threshold), None)
    expected_terminal = first_hit if first_hit is not None else maximum_depth
    _require(expected_terminal == baseline_k, "production terminal K is not the strict first action-MSE hit/max fallback")


def build_prediction_payload(
    raw_payload: Mapping[str, Any],
    *,
    task_id: int,
    task_name: str,
    episode_id: int,
    timestep: int,
    prediction_id: int,
    protocol_identity: Mapping[str, Any],
    warm_start_metadata: Mapping[str, Any],
    checkpoint: Mapping[str, Any],
    source_commit: str,
    run_identity: Mapping[str, Any],
    returned_action_sha256: str,
    rng_state_before_sha256: Optional[str],
    rng_state_after_sha256: Optional[str],
) -> dict[str, Any]:
    payload = dict(raw_payload)
    tensors = payload.get("tensors")
    _require(isinstance(tensors, Mapping), "model raw payload tensors are missing")
    payload["tensor_metadata"] = {
        name: tensor_metadata(tensors[name]) for name in RAW_PRECONVERGENCE_TENSORS
    }
    payload["tensor_sha256"] = {
        name: tensor_content_sha256(tensors[name]) for name in RAW_PRECONVERGENCE_TENSORS
    }
    payload.update(
        {
            "schema_version": RAW_PRECONVERGENCE_SCHEMA_VERSION,
            "identity": {
                "task_id": int(task_id),
                "episode_id": int(episode_id),
                "prediction_id": int(prediction_id),
                "timestep": int(timestep),
            },
            "task_name": str(task_name),
            "protocol_identity": dict(protocol_identity),
            "initial_warm_state_metadata": dict(warm_start_metadata),
            "checkpoint_identity": {
                "path": checkpoint.get("path"),
                "sha256": checkpoint.get("sha256"),
            },
            "source_commit": str(source_commit),
            "run_identity": dict(run_identity),
            "production_parity": {
                "returned_action_sha256": returned_action_sha256,
                "rng_state_before_sha256": rng_state_before_sha256,
                "rng_state_after_sha256": rng_state_after_sha256,
            },
        }
    )
    _validate_prediction_payload(payload)
    return payload


class RawPreconvergenceShardWriter:
    """Write grouped CPU prediction payloads with atomic, non-overwriting shards."""

    def __init__(
        self,
        output_dir: Path,
        *,
        shard_size: int,
        maximum_shadow_depth: int,
        source_commit: str,
        checkpoint: Mapping[str, Any],
        run_identity: Mapping[str, Any],
    ) -> None:
        _require(isinstance(shard_size, int) and not isinstance(shard_size, bool) and shard_size >= 1, "shard_size must be an integer >= 1")
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        if any(self.output_dir.iterdir()):
            raise FileExistsError(f"refusing to reuse non-empty raw shadow directory: {self.output_dir}")
        self.shard_size = shard_size
        self.maximum_shadow_depth = int(maximum_shadow_depth)
        self.source_commit = str(source_commit)
        self.checkpoint = dict(checkpoint)
        self.run_identity = dict(run_identity)
        self._buffer: list[dict[str, Any]] = []
        self._descriptors: list[dict[str, Any]] = []
        self._seen: set[tuple[int, int, int]] = set()
        self._shard_index = 0
        self._write_manifest(complete=False)

    @property
    def manifest_path(self) -> Path:
        return self.output_dir / "manifest.json"

    def add(self, payload: Mapping[str, Any]) -> None:
        _validate_prediction_payload(payload)
        identity = payload["identity"]
        key = (int(identity["task_id"]), int(identity["episode_id"]), int(identity["prediction_id"]))
        _require(key not in self._seen, f"duplicate raw prediction identity: {key}")
        self._seen.add(key)
        self._buffer.append(dict(payload))
        if len(self._buffer) >= self.shard_size:
            self.flush()

    def flush(self) -> None:
        if not self._buffer:
            return
        name = f"raw_shadow_{self._shard_index:05d}.pt"
        path = self.output_dir / name
        if path.exists():
            raise FileExistsError(f"refusing to overwrite raw shadow shard: {path}")
        shard = {
            "schema_version": RAW_PRECONVERGENCE_SCHEMA_VERSION,
            "collection_mode": RAW_PRECONVERGENCE_COLLECTION_MODE,
            "predictions": self._buffer,
        }
        temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
        try:
            torch.save(shard, temporary)
            os.replace(temporary, path)
        finally:
            if temporary.exists():
                temporary.unlink()
        shard_hash = sha256_file(path)
        for index, prediction in enumerate(self._buffer):
            identity = prediction["identity"]
            self._descriptors.append(
                {
                    "task_id": int(identity["task_id"]),
                    "episode_id": int(identity["episode_id"]),
                    "prediction_id": int(identity["prediction_id"]),
                    "timestep": int(identity["timestep"]),
                    "actual_origin": prediction["actual_origin"],
                    "shard_path": name,
                    "shard_index": index,
                    "shard_sha256": shard_hash,
                    "tensor_sha256": prediction["tensor_sha256"],
                    "tensor_metadata": prediction["tensor_metadata"],
                }
            )
        self._buffer = []
        self._shard_index += 1
        self._write_manifest(complete=False)

    def _manifest(self, *, complete: bool) -> dict[str, Any]:
        tasks = Counter(str(item["task_id"]) for item in self._descriptors)
        origins = Counter(item["actual_origin"] for item in self._descriptors)
        identity_material = [
            {
                "key": [item["task_id"], item["episode_id"], item["prediction_id"]],
                "tensor_sha256": item["tensor_sha256"],
            }
            for item in self._descriptors
        ]
        return {
            "schema_version": RAW_PRECONVERGENCE_SCHEMA_VERSION,
            "collection_mode": RAW_PRECONVERGENCE_COLLECTION_MODE,
            "complete": bool(complete),
            "source_commit": self.source_commit,
            "checkpoint_identity": self.checkpoint,
            "run_identity": self.run_identity,
            "maximum_shadow_depth": self.maximum_shadow_depth,
            "shard_size": self.shard_size,
            "prediction_count": len(self._descriptors),
            "shard_count": self._shard_index,
            "task_counts": dict(sorted(tasks.items())),
            "origin_counts": dict(sorted(origins.items())),
            "trace_set_sha256": canonical_json_sha256(identity_material),
            "sequences": self._descriptors,
        }

    def _write_manifest(self, *, complete: bool) -> None:
        _atomic_write_json(self.manifest_path, self._manifest(complete=complete))

    def finalize(self) -> Path:
        self.flush()
        _require(bool(self._descriptors), "cannot finalize an empty raw shadow collection")
        self._write_manifest(complete=True)
        return self.manifest_path


def load_and_validate_manifests(
    manifest_paths: Sequence[Path],
    *,
    require_complete: bool = True,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Validate manifests and shards, returning a compact report and predictions."""

    predictions: list[dict[str, Any]] = []
    seen: set[tuple[int, int, int]] = set()
    manifest_hashes = []
    for input_path in manifest_paths:
        path = Path(input_path)
        manifest = json.loads(path.read_text(encoding="utf-8"))
        _require(manifest.get("schema_version") == RAW_PRECONVERGENCE_SCHEMA_VERSION, f"manifest schema mismatch: {path}")
        _require(manifest.get("collection_mode") == RAW_PRECONVERGENCE_COLLECTION_MODE, f"manifest collection mode mismatch: {path}")
        if require_complete:
            _require(manifest.get("complete") is True, f"partial raw collection manifest: {path}")
        manifest_hashes.append({"path": str(path.resolve()), "sha256": sha256_file(path)})
        shard_cache: dict[str, Mapping[str, Any]] = {}
        descriptors = manifest.get("sequences", [])
        local_predictions = []
        for descriptor in descriptors:
            key = (int(descriptor["task_id"]), int(descriptor["episode_id"]), int(descriptor["prediction_id"]))
            _require(key not in seen, f"duplicate raw identity: {key}")
            seen.add(key)
            shard_path = path.parent / descriptor["shard_path"]
            _require(shard_path.is_file(), f"missing raw shard: {shard_path}")
            actual_hash = sha256_file(shard_path)
            _require(actual_hash == descriptor["shard_sha256"], f"corrupt raw shard: {shard_path}")
            if str(shard_path) not in shard_cache:
                shard_cache[str(shard_path)] = torch.load(shard_path, map_location="cpu", weights_only=True)
            shard = shard_cache[str(shard_path)]
            _require(shard.get("schema_version") == RAW_PRECONVERGENCE_SCHEMA_VERSION, f"raw shard schema mismatch: {shard_path}")
            index = int(descriptor["shard_index"])
            items = shard.get("predictions", [])
            _require(0 <= index < len(items), f"raw shard index out of range: {key}")
            prediction = items[index]
            _validate_prediction_payload(prediction)
            identity = prediction["identity"]
            _require(key == (identity["task_id"], identity["episode_id"], identity["prediction_id"]), f"raw shard identity mismatch: {key}")
            _require(prediction["tensor_sha256"] == descriptor["tensor_sha256"], f"raw tensor hash descriptor mismatch: {key}")
            predictions.append(prediction)
            local_predictions.append(prediction)
            _require(prediction["source_commit"] == manifest.get("source_commit"), f"source commit mismatch: {key}")
            _require(prediction["checkpoint_identity"].get("sha256") == manifest.get("checkpoint_identity", {}).get("sha256"), f"checkpoint identity mismatch: {key}")
            _require(prediction["run_identity"] == manifest.get("run_identity"), f"run identity mismatch: {key}")
            _require(int(prediction["maximum_shadow_depth"]) == int(manifest.get("maximum_shadow_depth", -1)), f"maximum shadow depth mismatch: {key}")
        _require(int(manifest.get("prediction_count", -1)) == len(descriptors), f"manifest prediction count mismatch: {path}")
        _require(int(manifest.get("shard_count", -1)) == len(shard_cache), f"manifest shard count mismatch: {path}")
        local_tasks = Counter(str(item["identity"]["task_id"]) for item in local_predictions)
        local_origins = Counter(item["actual_origin"] for item in local_predictions)
        _require(dict(sorted(local_tasks.items())) == manifest.get("task_counts"), f"manifest task counts mismatch: {path}")
        _require(dict(sorted(local_origins.items())) == manifest.get("origin_counts"), f"manifest origin counts mismatch: {path}")
        identity_material = [
            {
                "key": [item["task_id"], item["episode_id"], item["prediction_id"]],
                "tensor_sha256": item["tensor_sha256"],
            }
            for item in descriptors
        ]
        _require(canonical_json_sha256(identity_material) == manifest.get("trace_set_sha256"), f"manifest trace-set hash mismatch: {path}")

    tasks = Counter(str(item["identity"]["task_id"]) for item in predictions)
    origins = Counter(item["actual_origin"] for item in predictions)
    compact = {
        "schema_version": RAW_PRECONVERGENCE_SCHEMA_VERSION,
        "collection_mode": RAW_PRECONVERGENCE_COLLECTION_MODE,
        "complete": True,
        "source_manifests": manifest_hashes,
        "prediction_count": len(predictions),
        "task_counts": dict(sorted(tasks.items())),
        "origin_counts": dict(sorted(origins.items())),
        "missing_prediction_identity_count": 0,
        "duplicate_prediction_identity_count": 0,
        "nonfinite_tensor_count": 0,
        "trace_set_sha256": canonical_json_sha256(
            [
                {
                    "key": [item["identity"]["task_id"], item["identity"]["episode_id"], item["identity"]["prediction_id"]],
                    "tensor_sha256": item["tensor_sha256"],
                }
                for item in predictions
            ]
        ),
    }
    return compact, predictions
