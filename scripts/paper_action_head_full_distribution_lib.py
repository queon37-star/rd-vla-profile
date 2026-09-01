"""Utilities for full-distribution RD-VLA Action-head workload replay.

The paper checkpoint only consumes VLM hidden layers 11 and 23 in the Action
head. Capturing every VLM hidden layer for every policy query would therefore
store a large amount of data that can never affect ActionHeadRecurrent. This
module defines a fail-closed compact workload format that keeps only the unique
VLM layers actually indexed by the frozen Action head, while retaining the
exact proprio input, projected proprio feature, incoming warm cache, and
selected initial latent state required for deterministic replay.

Replay expands the selected layers back to their original layer indices before
calling the unmodified ActionHeadRecurrent.predict_action(). Expansion and
host-to-device transfer are intentionally outside the timed region.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional, Sequence

import torch


COMPACT_WORKLOAD_SCHEMA_VERSION = 1
COMPACT_WORKLOAD_TYPE = "rdvla_paper_action_head_full_distribution_compact"
FROZEN_SPATIAL_REQUIRED_VLM_LAYERS = (11, 23)
FROZEN_SPATIAL_NUM_TASK_TOKENS = 512


class FullDistributionWorkloadError(ValueError):
    """Raised when a full-distribution capture/replay contract is violated."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def tensor_nbytes(tensor: Optional[torch.Tensor]) -> int:
    if tensor is None:
        return 0
    return int(tensor.numel()) * int(tensor.element_size())


def payload_tensor_nbytes(tensors: Mapping[str, Optional[torch.Tensor]]) -> int:
    return int(sum(tensor_nbytes(tensor) for tensor in tensors.values()))


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise FullDistributionWorkloadError(message)


def _tensor_metadata(tensor: Optional[torch.Tensor]) -> Optional[dict[str, Any]]:
    if tensor is None:
        return None
    _require(torch.is_tensor(tensor), "compact workload value must be a tensor or null")
    _require(tensor.device.type == "cpu", "compact workload tensors must be on CPU")
    _require(tensor.is_contiguous(), "compact workload tensors must be contiguous")
    _require(bool(torch.isfinite(tensor).all().item()), "compact workload tensor is non-finite")
    return {
        "shape": list(tensor.shape),
        "stride": list(tensor.stride()),
        "dtype": str(tensor.dtype),
        "contiguous": True,
    }


def action_head_required_vlm_layers(action_head) -> tuple[int, ...]:
    cfg = action_head.cfg
    layers = sorted(
        {
            int(layer)
            for group in (
                cfg.prelude_vlm_layers,
                cfg.recurrent_vlm_layers,
                cfg.coda_vlm_layers,
            )
            for layer in group
        }
    )
    _require(bool(layers), "Action head does not consume any VLM hidden layer")
    return tuple(layers)


def validate_frozen_spatial_action_head(action_head) -> tuple[int, ...]:
    layers = action_head_required_vlm_layers(action_head)
    _require(
        layers == FROZEN_SPATIAL_REQUIRED_VLM_LAYERS,
        "frozen Spatial Action-head VLM layer contract changed: "
        f"expected={FROZEN_SPATIAL_REQUIRED_VLM_LAYERS}, actual={layers}",
    )
    _require(
        int(action_head.num_task_tokens) == FROZEN_SPATIAL_NUM_TASK_TOKENS,
        "frozen Spatial Action-head task-token count changed: "
        f"expected={FROZEN_SPATIAL_NUM_TASK_TOKENS}, actual={action_head.num_task_tokens}",
    )
    _require(int(action_head.cfg.hidden_dim) == 896, "frozen Spatial hidden_dim must be 896")
    _require(int(action_head.cfg.action_chunk_len) == 8, "frozen Spatial action_chunk_len must be 8")
    return layers


def compact_workload(
    workload: Mapping[str, Any],
    *,
    identity: Mapping[str, Any],
    layer_indices: Sequence[int],
    source_arm: str,
) -> tuple[dict[str, Any], dict[str, int]]:
    """Convert one exact production workload into a selected-layer CPU payload."""

    _require(isinstance(workload, Mapping), "source workload must be a mapping")
    actual_origin = workload.get("actual_origin")
    _require(actual_origin in {"COLD", "ACTUAL_WARM"}, "invalid source workload origin")
    source_tensors = workload.get("tensors")
    _require(isinstance(source_tensors, Mapping), "source workload tensors are missing")

    h = source_tensors.get("actions_hidden_states")
    _require(torch.is_tensor(h), "source actions_hidden_states is missing")
    _require(h.device.type == "cpu", "source actions_hidden_states must already be on CPU")
    _require(h.ndim == 4, "source actions_hidden_states must be rank 4")
    _require(h.is_contiguous(), "source actions_hidden_states must be contiguous")
    _require(bool(torch.isfinite(h).all().item()), "source actions_hidden_states is non-finite")

    normalized_layers = tuple(sorted({int(index) for index in layer_indices}))
    _require(bool(normalized_layers), "layer_indices must not be empty")
    _require(min(normalized_layers) >= 0, "VLM layer indices must be non-negative")
    _require(
        max(normalized_layers) < int(h.shape[1]),
        "selected VLM layer index exceeds source layer count",
    )
    selected_h = h[:, list(normalized_layers), :, :].contiguous()

    compact_tensors = {
        "selected_actions_hidden_states": selected_h,
        "proprio_input": source_tensors.get("proprio_input"),
        "proprio_features": source_tensors.get("proprio_features"),
        "incoming_warm_start_state": source_tensors.get("incoming_warm_start_state"),
        "selected_initial_state": source_tensors.get("selected_initial_state"),
    }
    # Detach/copy all non-hidden tensors so the payload owns its CPU storage.
    for name in tuple(compact_tensors):
        tensor = compact_tensors[name]
        if tensor is not None and name != "selected_actions_hidden_states":
            _require(torch.is_tensor(tensor), f"{name} must be a tensor or null")
            compact_tensors[name] = tensor.detach().to(device="cpu", copy=True).contiguous()

    metadata = {name: _tensor_metadata(tensor) for name, tensor in compact_tensors.items()}
    incoming = compact_tensors["incoming_warm_start_state"]
    selected = compact_tensors["selected_initial_state"]
    _require(torch.is_tensor(selected) and selected.ndim == 3, "selected_initial_state must be rank 3")
    if actual_origin == "COLD":
        _require(incoming is None, "COLD workload cannot contain an incoming warm cache")
    else:
        _require(torch.is_tensor(incoming) and incoming.ndim == 3, "ACTUAL_WARM requires a rank-3 warm cache")
        _require(torch.equal(incoming, selected), "accepted warm cache must equal selected initial state")

    payload = {
        "schema_version": COMPACT_WORKLOAD_SCHEMA_VERSION,
        "workload_type": COMPACT_WORKLOAD_TYPE,
        "source_arm": str(source_arm),
        "actual_origin": str(actual_origin),
        "identity": dict(identity),
        "layout": {
            "source_layer_count": int(h.shape[1]),
            "selected_layer_indices": list(normalized_layers),
            "token_count": int(h.shape[2]),
            "hidden_dim": int(h.shape[3]),
            "num_task_tokens": FROZEN_SPATIAL_NUM_TASK_TOKENS,
            "source_dtype": str(h.dtype),
        },
        "tensors": compact_tensors,
        "tensor_metadata": metadata,
    }
    byte_stats = {
        "source_tensor_bytes": payload_tensor_nbytes(source_tensors),
        "source_actions_hidden_states_bytes": tensor_nbytes(h),
        "compact_tensor_bytes": payload_tensor_nbytes(compact_tensors),
        "compact_actions_hidden_states_bytes": tensor_nbytes(selected_h),
    }
    return payload, byte_stats


def save_compact_workload(
    path: Path,
    workload: Mapping[str, Any],
    *,
    identity: Mapping[str, Any],
    layer_indices: Sequence[int],
    source_arm: str,
) -> dict[str, Any]:
    """Atomically save one compact workload and return a replay descriptor."""

    output_path = Path(path)
    if output_path.exists():
        raise FileExistsError(f"Refusing to overwrite compact workload: {output_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload, byte_stats = compact_workload(
        workload,
        identity=identity,
        layer_indices=layer_indices,
        source_arm=source_arm,
    )
    temporary_path = output_path.with_name(f".{output_path.name}.tmp-{os.getpid()}")
    try:
        torch.save(payload, temporary_path)
        os.replace(temporary_path, output_path)
    finally:
        if temporary_path.exists():
            temporary_path.unlink()
    digest = sha256_file(output_path)
    return {
        "path": str(output_path.resolve()),
        "sha256": digest,
        "file_bytes": int(output_path.stat().st_size),
        "identity": dict(identity),
        "actual_origin": payload["actual_origin"],
        "source_arm": str(source_arm),
        "layout": dict(payload["layout"]),
        **byte_stats,
    }


def _validate_tensor_metadata(
    name: str,
    tensor: Optional[torch.Tensor],
    metadata: Optional[Mapping[str, Any]],
) -> None:
    if tensor is None:
        _require(metadata is None, f"{name}: null tensor must have null metadata")
        return
    _require(torch.is_tensor(tensor), f"{name}: value is not a tensor")
    _require(tensor.device.type == "cpu", f"{name}: tensor must be on CPU")
    _require(tensor.is_contiguous(), f"{name}: tensor must be contiguous")
    _require(bool(torch.isfinite(tensor).all().item()), f"{name}: tensor is non-finite")
    _require(isinstance(metadata, Mapping), f"{name}: metadata must be a mapping")
    expected = _tensor_metadata(tensor)
    for field in ("shape", "stride", "dtype", "contiguous"):
        _require(metadata.get(field) == expected[field], f"{name}: metadata field {field} mismatch")


def load_compact_workload(
    path: Path,
    *,
    expected_sha256: Optional[str] = None,
    expected_identity: Optional[Mapping[str, Any]] = None,
    expected_source_arm: Optional[str] = None,
    expected_origin: Optional[str] = None,
) -> dict[str, Any]:
    workload_path = Path(path)
    _require(workload_path.is_file(), f"missing compact workload: {workload_path}")
    if expected_sha256 is not None:
        actual = sha256_file(workload_path)
        _require(actual == expected_sha256, f"compact workload SHA mismatch: {actual} != {expected_sha256}")
    try:
        payload = torch.load(workload_path, map_location="cpu", weights_only=True)
    except Exception as exc:
        raise FullDistributionWorkloadError(f"cannot load compact workload: {workload_path}") from exc
    _require(isinstance(payload, dict), "compact workload root must be a mapping")
    _require(payload.get("schema_version") == COMPACT_WORKLOAD_SCHEMA_VERSION, "unsupported compact workload schema")
    _require(payload.get("workload_type") == COMPACT_WORKLOAD_TYPE, "unexpected compact workload type")
    if expected_identity is not None:
        _require(payload.get("identity") == dict(expected_identity), "compact workload identity mismatch")
    if expected_source_arm is not None:
        _require(payload.get("source_arm") == expected_source_arm, "compact workload source-arm mismatch")
    if expected_origin is not None:
        _require(payload.get("actual_origin") == expected_origin, "compact workload origin mismatch")

    actual_origin = payload.get("actual_origin")
    _require(actual_origin in {"COLD", "ACTUAL_WARM"}, "invalid compact workload origin")
    layout = payload.get("layout")
    tensors = payload.get("tensors")
    metadata = payload.get("tensor_metadata")
    _require(isinstance(layout, Mapping), "compact workload layout is missing")
    _require(isinstance(tensors, Mapping), "compact workload tensors are missing")
    _require(isinstance(metadata, Mapping), "compact workload metadata is missing")
    expected_fields = {
        "selected_actions_hidden_states",
        "proprio_input",
        "proprio_features",
        "incoming_warm_start_state",
        "selected_initial_state",
    }
    _require(set(tensors) == expected_fields, "compact workload tensor fields are incomplete")
    for name in expected_fields:
        _validate_tensor_metadata(name, tensors[name], metadata.get(name))

    selected_h = tensors["selected_actions_hidden_states"]
    _require(selected_h is not None and selected_h.ndim == 4, "selected hidden states must be rank 4")
    layer_indices = tuple(int(value) for value in layout.get("selected_layer_indices", []))
    _require(len(layer_indices) == int(selected_h.shape[1]), "selected layer count/layout mismatch")
    _require(tuple(sorted(set(layer_indices))) == layer_indices, "selected layer indices must be unique and sorted")
    _require(max(layer_indices) < int(layout["source_layer_count"]), "selected layer index exceeds source layer count")
    _require(int(selected_h.shape[2]) == int(layout["token_count"]), "token count/layout mismatch")
    _require(int(selected_h.shape[3]) == int(layout["hidden_dim"]), "hidden dim/layout mismatch")
    _require(int(layout["num_task_tokens"]) == FROZEN_SPATIAL_NUM_TASK_TOKENS, "task-token layout mismatch")

    incoming = tensors["incoming_warm_start_state"]
    selected_state = tensors["selected_initial_state"]
    _require(selected_state is not None and selected_state.ndim == 3, "selected initial state must be rank 3")
    if actual_origin == "COLD":
        _require(incoming is None, "COLD compact workload cannot contain warm cache")
    else:
        _require(incoming is not None and incoming.ndim == 3, "ACTUAL_WARM compact workload requires warm cache")
        _require(torch.equal(incoming, selected_state), "ACTUAL_WARM cache/selected-state mismatch")
    return payload


def expand_compact_tensors(
    payload: Mapping[str, Any],
    *,
    device: torch.device,
    expected_layer_indices: Optional[Sequence[int]] = None,
) -> dict[str, Optional[torch.Tensor]]:
    """Move a compact workload to GPU and reconstruct the original layer axis."""

    layout = payload["layout"]
    cpu_tensors = payload["tensors"]
    selected_cpu = cpu_tensors["selected_actions_hidden_states"]
    layer_indices = tuple(int(value) for value in layout["selected_layer_indices"])
    if expected_layer_indices is not None:
        _require(
            layer_indices == tuple(int(value) for value in expected_layer_indices),
            f"compact layer set mismatch: expected={tuple(expected_layer_indices)}, actual={layer_indices}",
        )

    selected = selected_cpu.to(device=device, non_blocking=False)
    B, _, T, D = selected.shape
    source_layer_count = int(layout["source_layer_count"])
    full = torch.zeros(
        (B, source_layer_count, T, D),
        dtype=selected.dtype,
        device=device,
    )
    for compact_index, source_index in enumerate(layer_indices):
        full[:, source_index, :, :].copy_(selected[:, compact_index, :, :])

    result: dict[str, Optional[torch.Tensor]] = {
        "actions_hidden_states": full.contiguous(),
    }
    for name in (
        "proprio_input",
        "proprio_features",
        "incoming_warm_start_state",
        "selected_initial_state",
    ):
        tensor = cpu_tensors[name]
        result[name] = None if tensor is None else tensor.to(device=device, non_blocking=False).contiguous()
    return result


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as stream:
        for line_number, raw in enumerate(stream, start=1):
            text = raw.strip()
            if not text:
                continue
            try:
                value = json.loads(text)
            except json.JSONDecodeError as exc:
                raise FullDistributionWorkloadError(f"invalid JSONL at {path}:{line_number}") from exc
            _require(isinstance(value, dict), f"JSONL record at {path}:{line_number} must be an object")
            records.append(value)
    return records


def append_jsonl(path: Path, records: Iterable[Mapping[str, Any]]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        for record in records:
            stream.write(json.dumps(dict(record), ensure_ascii=False) + "\n")
