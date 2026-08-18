"""Counterfactual replay of lazy-prefix exact confirmation from frozen-v1 logs."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence


DIAGNOSTIC_Q = 0.0015


class LazyPrefixReplayError(ValueError):
    pass


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise LazyPrefixReplayError(message)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _exact_stop_by_terminal(record: Mapping[str, Any]) -> dict[int, dict[str, Any]]:
    trace = record.get(
        "action_delta_deferred_backfill_filter_exact_stop_mse_trace", []
    )
    result = {}
    for item in trace:
        terminal = int(item["terminal_iteration"])
        _require(terminal not in result, "duplicate exact-stop terminal iteration")
        result[terminal] = dict(item)
    return result


def replay_prediction(
    record: Mapping[str, Any], *, threshold: float = DIAGNOSTIC_Q
) -> dict[str, Any]:
    """Estimate lazy-prefix calls while preserving the logged recurrent K."""

    k = int(record.get("recurrent_iteration_count", record.get("K_t")))
    _require(k >= 1, "recurrent K must be positive")
    applied = bool(
        record.get("action_delta_deferred_backfill_filter_applied", False)
    )
    runtime_policy = record.get("action_delta_deferred_runtime_policy")
    _require(
        runtime_policy in (None, "frozen_v1"),
        "input is not a frozen-v1 deferred/backfill log",
    )
    score_trace = list(
        record.get("action_delta_deferred_backfill_filter_score_trace", [])
    )
    frozen_scorer_calls = int(
        record.get(
            "action_delta_deferred_backfill_filter_score_call_count",
            len(score_trace),
        )
    )
    _require(
        frozen_scorer_calls == len(score_trace),
        "frozen-v1 scorer count differs from score trace length",
    )
    frozen_coda_calls = int(
        record.get(
            "action_delta_deferred_backfill_filter_total_exact_coda_call_count",
            k,
        )
    )
    _require(0 <= frozen_coda_calls <= k, "invalid frozen-v1 Coda count")

    identity = {
        name: record.get(name)
        for name in (
            "task_id",
            "episode_id",
            "episode_seed",
            "initial_state_id",
            "action_prediction_index",
            "timestep",
            "actual_origin",
            "_source_jsonl",
            "_source_line",
        )
    }
    if not applied:
        _require(not score_trace, "ineligible prediction contains scorer calls")
        return {
            **identity,
            "recurrent_K": k,
            "filter_applied": False,
            "first_nonhigh_terminal_iteration": None,
            "predicted_scorer_calls": 0,
            "predicted_exact_coda_calls": k,
            "predicted_eliminated_coda_calls": 0,
            "first_coda_avoided": False,
            "exact_only_would_begin": False,
            "first_confirmation_stopped": None,
            "frozen_v1_scorer_calls": frozen_scorer_calls,
            "frozen_v1_coda_calls": frozen_coda_calls,
            "warm_reference_coda_calls": k,
        }

    terminals = [int(item["terminal_iteration"]) for item in score_trace]
    _require(
        terminals == list(range(2, 2 + len(terminals))),
        "frozen-v1 score trace must be contiguous from terminal 2",
    )
    _require(len(score_trace) <= max(k - 1, 0), "too many scorer calls for K")

    first_nonhigh = None
    for item in score_trace:
        score = item.get("score")
        high = score is not None and float(score) >= threshold
        if not high:
            first_nonhigh = int(item["terminal_iteration"])
            break

    exact_by_terminal = _exact_stop_by_terminal(record)
    if first_nonhigh is None:
        _require(
            len(score_trace) == max(k - 1, 0),
            "all-high trace does not cover every eligible transition through K",
        )
        lazy_scorer_calls = max(k - 1, 0)
        lazy_exact_calls = 1
        confirmation_stopped = None
        exact_only = False
        first_coda_avoided = k > 1
    else:
        lazy_scorer_calls = first_nonhigh - 1
        _require(
            lazy_scorer_calls == len(
                [item for item in score_trace if int(item["terminal_iteration"]) <= first_nonhigh]
            ),
            "first-nonhigh scorer accounting is inconsistent",
        )
        _require(
            first_nonhigh in exact_by_terminal,
            "first non-high transition lacks exact adjacent confirmation",
        )
        confirmation_stopped = bool(exact_by_terminal[first_nonhigh]["stopped"])
        lazy_exact_calls = 2 + (0 if confirmation_stopped else k - first_nonhigh)
        exact_only = not confirmation_stopped
        first_coda_avoided = first_nonhigh > 2

    eliminated = k - lazy_exact_calls
    _require(lazy_exact_calls >= 1, "lazy policy must return an exact action")
    _require(eliminated >= 0, "lazy replay produced negative Coda savings")
    if first_nonhigh is not None:
        _require(
            lazy_scorer_calls == first_nonhigh - 1,
            "lazy scorer_calls must equal first_nonhigh_terminal - 1",
        )
        expected_exact = 2 + (0 if confirmation_stopped else k - first_nonhigh)
        _require(
            lazy_exact_calls == expected_exact,
            "lazy exact-only Coda accounting mismatch",
        )
    else:
        _require(lazy_exact_calls == 1, "all-high replay requires one exact terminal Coda")

    return {
        **identity,
        "recurrent_K": k,
        "filter_applied": True,
        "first_nonhigh_terminal_iteration": first_nonhigh,
        "predicted_scorer_calls": lazy_scorer_calls,
        "predicted_exact_coda_calls": lazy_exact_calls,
        "predicted_eliminated_coda_calls": eliminated,
        "first_coda_avoided": first_coda_avoided,
        "exact_only_would_begin": exact_only,
        "first_confirmation_stopped": confirmation_stopped,
        "frozen_v1_scorer_calls": frozen_scorer_calls,
        "frozen_v1_coda_calls": frozen_coda_calls,
        "warm_reference_coda_calls": k,
    }


def analyze_records(
    records: Sequence[Mapping[str, Any]], *, threshold: float = DIAGNOSTIC_Q
) -> dict[str, Any]:
    predictions = [replay_prediction(record, threshold=threshold) for record in records]
    aggregate = {
        "prediction_count": len(predictions),
        "filter_applied_prediction_count": sum(
            int(item["filter_applied"]) for item in predictions
        ),
        "original_frozen_v1_scorer_calls": sum(
            int(item["frozen_v1_scorer_calls"]) for item in predictions
        ),
        "lazy_prefix_scorer_calls": sum(
            int(item["predicted_scorer_calls"]) for item in predictions
        ),
        "warm_reference_coda_calls": sum(
            int(item["warm_reference_coda_calls"]) for item in predictions
        ),
        "frozen_v1_coda_calls": sum(
            int(item["frozen_v1_coda_calls"]) for item in predictions
        ),
        "lazy_prefix_expected_coda_calls": sum(
            int(item["predicted_exact_coda_calls"]) for item in predictions
        ),
        "lazy_prefix_expected_eliminated_coda_calls": sum(
            int(item["predicted_eliminated_coda_calls"]) for item in predictions
        ),
        "first_coda_avoided_prediction_count": sum(
            int(item["first_coda_avoided"]) for item in predictions
        ),
        "exact_only_would_begin_prediction_count": sum(
            int(item["exact_only_would_begin"]) for item in predictions
        ),
    }
    aggregate["scorer_call_reduction"] = (
        aggregate["original_frozen_v1_scorer_calls"]
        - aggregate["lazy_prefix_scorer_calls"]
    )
    aggregate["expected_coda_reduction_vs_warm"] = (
        aggregate["warm_reference_coda_calls"]
        - aggregate["lazy_prefix_expected_coda_calls"]
    )
    aggregate["expected_additional_coda_reduction_vs_frozen_v1"] = (
        aggregate["frozen_v1_coda_calls"]
        - aggregate["lazy_prefix_expected_coda_calls"]
    )
    _require(
        aggregate["expected_coda_reduction_vs_warm"]
        == aggregate["lazy_prefix_expected_eliminated_coda_calls"],
        "aggregate lazy Coda accounting mismatch",
    )
    return {
        "analysis_type": "lazy_prefix_exact_counterfactual_from_frozen_v1_logs",
        "diagnostic_only": True,
        "runtime_latency_claim": False,
        "threshold": float(threshold),
        "aggregate": aggregate,
        "predictions": predictions,
    }


def load_jsonl(paths: Sequence[Path]) -> list[dict[str, Any]]:
    records = []
    for path in paths:
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if line.strip():
                    value = json.loads(line)
                    value["_source_jsonl"] = str(path)
                    value["_source_line"] = line_number
                    records.append(value)
    return records


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="+", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    paths = [path.resolve() for path in args.inputs]
    records = load_jsonl(paths)
    results = analyze_records(records)
    results["sources"] = [
        {"path": str(path), "sha256": _sha256_file(path)} for path in paths
    ]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(results, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    summary = results["aggregate"]
    print("Lazy-prefix exact counterfactual (offline; not runtime latency)")
    print(json.dumps(summary, indent=2, sort_keys=True))
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
