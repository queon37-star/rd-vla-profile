#!/usr/bin/env python3
"""Validate the diagnostic latent-dynamics contract in LIBERO steps JSONL."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

EXISTING_SCALAR_FIELDS = (
    "raw_mse",
    "relative_mse",
    "relative_l2",
    "cosine_distance",
    "adjacent_action_mse",
)
LATENT_DYNAMICS_FIELDS = (
    "update_rms",
    "contraction_ratio",
    "update_turning_cosine",
    "acceleration_rms",
    "acceleration_ratio",
    "token_update_p50",
    "token_update_p90",
    "token_update_p95",
    "token_update_max",
    "token_update_cv",
    "token_update_energy_entropy",
    "token_update_top10_fraction",
    "state_rms",
    "state_norm_ratio",
    "warm_anchor_relative_l2",
    "warm_anchor_cosine_distance",
)
HISTORY_DEPENDENT_FIELDS = (
    "contraction_ratio",
    "update_turning_cosine",
    "acceleration_rms",
    "acceleration_ratio",
)
WARM_ANCHOR_FIELDS = (
    "warm_anchor_relative_l2",
    "warm_anchor_cosine_distance",
)
CORE_IDENTITY_FIELDS = (
    "task_id",
    "episode_id",
    "prediction_step",
)
PROTOCOL_IDENTITY_FIELDS = (
    "paired_trial_id",
    "initial_state_id",
    "episode_seed",
)
TRACE_REQUIRED_FIELDS = frozenset(
    (
        "iteration_index",
        "phase",
        "actual_origin",
        "action_mse_below_0_001",
        "baseline_stopping_iteration",
        "task_id",
        "episode_id",
        "prediction_id",
        *EXISTING_SCALAR_FIELDS,
        *LATENT_DYNAMICS_FIELDS,
    )
)


class LatentDynamicsContractError(ValueError):
    """Raised when a steps record violates the diagnostic trace contract."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise LatentDynamicsContractError(message)


def _finite_scalar(value: Any, context: str) -> float:
    _require(
        isinstance(value, (int, float)) and not isinstance(value, bool),
        f"{context} must be a numeric scalar",
    )
    result = float(value)
    _require(math.isfinite(result), f"{context} must be finite")
    return result


def _identity(record: Mapping[str, Any]) -> dict[str, int | None]:
    missing = [field for field in CORE_IDENTITY_FIELDS if record.get(field) is None]
    _require(not missing, "missing core workload identity fields: " + ", ".join(missing))
    missing_protocol_keys = [
        field for field in PROTOCOL_IDENTITY_FIELDS if field not in record
    ]
    _require(
        not missing_protocol_keys,
        "missing protocol identity fields: " + ", ".join(missing_protocol_keys),
    )
    _require(
        record.get("prediction_step") == record.get("action_prediction_index"),
        "prediction_step/action_prediction_index mismatch",
    )
    identity = {field: int(record[field]) for field in CORE_IDENTITY_FIELDS}
    identity.update(
        {
            field: None if record[field] is None else int(record[field])
            for field in PROTOCOL_IDENTITY_FIELDS
        }
    )
    return identity


def validate_step_record(record: Mapping[str, Any]) -> dict[str, Any]:
    identity = _identity(record)
    context = (
        f"task={identity['task_id']} episode={identity['episode_id']} "
        f"prediction={identity['prediction_step']}"
    )
    _require(record.get("latent_metric_trace_enabled") is True, f"{context}: trace disabled")
    _require(
        record.get("latent_dynamics_trace_enabled") is True,
        f"{context}: latent dynamics disabled",
    )
    _require(
        record.get("shadow_full_depth_enabled") is True,
        f"{context}: full-depth shadow disabled",
    )
    _require(
        record.get("shadow_trace_complete") is True,
        f"{context}: shadow trace incomplete",
    )
    _require(record.get("shadow_error") is None, f"{context}: shadow error present")
    _require(
        int(record.get("max_recurrent_iteration")) == 32,
        f"{context}: max_recurrent_iteration must be 32",
    )
    origin = record.get("actual_origin")
    _require(origin in {"ACTUAL_WARM", "COLD"}, f"{context}: invalid origin")
    _require(
        record.get("latent_dynamics_warm_anchor_available")
        == (origin == "ACTUAL_WARM"),
        f"{context}: warm-anchor availability/origin mismatch",
    )

    trace = record.get("latent_metric_trace")
    _require(isinstance(trace, list), f"{context}: latent_metric_trace must be a list")
    _require(len(trace) == 31, f"{context}: trace length must be 31")
    _require(
        [item.get("iteration_index") for item in trace] == list(range(2, 33)),
        f"{context}: iteration indices must be exactly 2..32",
    )
    baseline_k = int(record["K_t"])
    for item in trace:
        iteration = int(item["iteration_index"])
        item_context = f"{context} k={iteration}"
        missing = TRACE_REQUIRED_FIELDS - set(item)
        _require(not missing, f"{item_context}: missing fields {sorted(missing)}")
        _require(item["task_id"] == identity["task_id"], f"{item_context}: task mismatch")
        _require(
            item["episode_id"] == identity["episode_id"],
            f"{item_context}: episode mismatch",
        )
        _require(
            item["prediction_id"] == identity["prediction_step"],
            f"{item_context}: prediction mismatch",
        )
        _require(item["actual_origin"] == origin, f"{item_context}: origin mismatch")
        _require(
            item["baseline_stopping_iteration"] == baseline_k,
            f"{item_context}: baseline K mismatch",
        )
        _require(
            isinstance(item["action_mse_below_0_001"], bool),
            f"{item_context}: action label must be boolean",
        )
        for field in EXISTING_SCALAR_FIELDS:
            _finite_scalar(item[field], f"{item_context}.{field}")
        for field in LATENT_DYNAMICS_FIELDS:
            value = item[field]
            if field in HISTORY_DEPENDENT_FIELDS and iteration == 2:
                _require(value is None, f"{item_context}.{field} must be null")
            elif field in WARM_ANCHOR_FIELDS and origin == "COLD":
                _require(value is None, f"{item_context}.{field} must be null")
            else:
                _finite_scalar(value, f"{item_context}.{field}")
        for field in (
            "token_update_energy_entropy",
            "token_update_top10_fraction",
        ):
            value = float(item[field])
            _require(0.0 <= value <= 1.0, f"{item_context}.{field} outside [0, 1]")
    return {"identity": identity, "transition_count": len(trace)}


def validate_records(
    records: Iterable[Mapping[str, Any]],
    *,
    expected_identity_sha256: str | None = None,
) -> dict[str, Any]:
    summaries = [validate_step_record(record) for record in records]
    _require(bool(summaries), "no step records found")
    identities = [summary["identity"] for summary in summaries]
    identity_keys = [
        (item["task_id"], item["episode_id"], item["prediction_step"])
        for item in identities
    ]
    _require(len(identity_keys) == len(set(identity_keys)), "duplicate workload identity")
    identity_payload = json.dumps(
        identities, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    identity_sha256 = hashlib.sha256(identity_payload).hexdigest()
    if expected_identity_sha256 is not None:
        _require(
            identity_sha256 == expected_identity_sha256,
            "workload identity SHA-256 mismatch",
        )
    return {
        "schema_version": 1,
        "passed": True,
        "prediction_count": len(summaries),
        "transition_count": sum(item["transition_count"] for item in summaries),
        "workload_identity_sha256": identity_sha256,
    }


def load_jsonl(paths: Sequence[Path]) -> list[Mapping[str, Any]]:
    records = []
    for path in paths:
        for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if not line.strip():
                continue
            value = json.loads(line)
            _require(isinstance(value, dict), f"{path}:{line_number}: expected object")
            records.append(value)
    return records


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", action="append", required=True, type=Path)
    parser.add_argument("--expected-identity-sha256")
    parser.add_argument("--output", type=Path)
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    result = validate_records(
        load_jsonl(args.steps),
        expected_identity_sha256=args.expected_identity_sha256,
    )
    result["inputs"] = [str(path.resolve()) for path in args.steps]
    payload = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload, encoding="utf-8")
    print(payload, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
