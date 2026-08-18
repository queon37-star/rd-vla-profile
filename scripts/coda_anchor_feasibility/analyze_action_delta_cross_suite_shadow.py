#!/usr/bin/env python3
"""Analyze diagnostic zero-shot cross-suite Action-Delta shadow logs."""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from prismatic.models.action_delta_cross_suite_shadow import (
    ACTION_DELTA_CROSS_SUITE_ANALYSIS_TYPE,
)
from prismatic.models.action_delta_gate import (
    ACTION_DELTA_NONCONVERGENCE_THRESHOLD,
)


def _rankdata(values: np.ndarray) -> np.ndarray:
    """Return average ranks for ties without requiring scipy."""

    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=np.float64)
    start = 0
    while start < len(values):
        end = start + 1
        while end < len(values) and values[order[end]] == values[order[start]]:
            end += 1
        ranks[order[start:end]] = (start + end - 1) / 2.0 + 1.0
        start = end
    return ranks


def _correlation(left: np.ndarray, right: np.ndarray) -> float | None:
    if len(left) < 2 or np.ptp(left) == 0.0 or np.ptp(right) == 0.0:
        return None
    return float(np.corrcoef(left, right)[0, 1])


def _quantile(values: np.ndarray, probability: float) -> float | None:
    return float(np.quantile(values, probability)) if len(values) else None


def summarize_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    exact_rows = [
        row
        for row in rows
        if math.isfinite(float(row["exact_adjacent_action_mse"]))
    ]
    finite_rows = [
        row
        for row in rows
        if row.get("score") is not None
        and math.isfinite(float(row["score"]))
        and math.isfinite(float(row["exact_adjacent_action_mse"]))
    ]
    scores = np.asarray([row["score"] for row in finite_rows], dtype=np.float64)
    paired_exact = np.asarray(
        [row["exact_adjacent_action_mse"] for row in finite_rows],
        dtype=np.float64,
    )
    exact = np.asarray(
        [row["exact_adjacent_action_mse"] for row in exact_rows],
        dtype=np.float64,
    )
    high_count = int(
        sum(bool(row["high_predicted_nonconvergence"]) for row in finite_rows)
    )
    exact_safe_count = int(sum(bool(row["exact_safe"]) for row in exact_rows))
    violation_count = int(
        sum(bool(row["high_exact_safe_violation"]) for row in finite_rows)
    )
    count = len(rows)
    finite_count = len(finite_rows)
    return {
        "transition_count": count,
        "finite_score_count": finite_count,
        "spearman_score_exact_mse": _correlation(
            _rankdata(scores), _rankdata(paired_exact)
        ) if finite_count else None,
        "pearson_score_exact_mse": _correlation(scores, paired_exact),
        "score_mean": float(scores.mean()) if finite_count else None,
        "score_median": float(np.median(scores)) if finite_count else None,
        "score_quantiles": {
            name: _quantile(scores, probability)
            for name, probability in (
                ("p50", 0.50),
                ("p90", 0.90),
                ("p95", 0.95),
                ("p99", 0.99),
            )
        },
        "exact_mse_mean": float(exact.mean()) if len(exact) else None,
        "exact_mse_median": float(np.median(exact)) if len(exact) else None,
        "high_predicted_nonconvergence_count": high_count,
        "high_predicted_nonconvergence_rate": (
            high_count / finite_count if finite_count else None
        ),
        "exact_safe_count": exact_safe_count,
        "exact_safe_rate": exact_safe_count / len(exact_rows) if exact_rows else None,
        "high_exact_safe_violation_count": violation_count,
        "high_exact_safe_violation_rate": (
            violation_count / finite_count if finite_count else None
        ),
        "high_prediction_exact_nonconverged_precision": (
            (high_count - violation_count) / high_count if high_count else None
        ),
    }


def load_rows(paths: Iterable[Path]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in paths:
        with path.open("r", encoding="utf-8") as stream:
            for line_number, line in enumerate(stream, start=1):
                if not line.strip():
                    continue
                record = json.loads(line)
                payload = record.get("action_delta_cross_suite_shadow")
                if payload is None:
                    continue
                if payload.get("analysis_type") != ACTION_DELTA_CROSS_SUITE_ANALYSIS_TYPE:
                    raise ValueError(
                        f"{path}:{line_number}: unexpected analysis_type"
                    )
                if not payload.get("diagnostic_only"):
                    raise ValueError(f"{path}:{line_number}: record is not diagnostic-only")
                if payload.get("production_efficiency_claim") is not False:
                    raise ValueError(
                        f"{path}:{line_number}: production efficiency claim must be false"
                    )
                if payload.get("predictor_training_suite") != "libero_spatial":
                    raise ValueError(
                        f"{path}:{line_number}: predictor training suite must be libero_spatial"
                    )
                if float(payload.get("high_side_threshold")) != (
                    ACTION_DELTA_NONCONVERGENCE_THRESHOLD
                ):
                    raise ValueError(
                        f"{path}:{line_number}: high-side threshold is not frozen at 0.0015"
                    )
                artifact_sha256 = str(
                    payload.get("predictor_artifact_sha256", "")
                ).lower()
                if len(artifact_sha256) != 64 or any(
                    character not in "0123456789abcdef"
                    for character in artifact_sha256
                ):
                    raise ValueError(
                        f"{path}:{line_number}: predictor artifact SHA-256 is invalid"
                    )
                for transition in payload.get("transitions", []):
                    score = transition.get("score")
                    if score is not None and math.isfinite(float(score)):
                        expected_high = bool(
                            float(score)
                            >= ACTION_DELTA_NONCONVERGENCE_THRESHOLD
                        )
                        exact_safe = bool(
                            float(transition["exact_adjacent_action_mse"])
                            < 0.001
                        )
                        if transition.get(
                            "high_predicted_nonconvergence"
                        ) is not expected_high:
                            raise ValueError(
                                f"{path}:{line_number}: high-side label mismatch"
                            )
                        if transition.get("exact_safe") is not exact_safe:
                            raise ValueError(
                                f"{path}:{line_number}: exact-safe label mismatch"
                            )
                        if transition.get(
                            "high_exact_safe_violation"
                        ) is not bool(expected_high and exact_safe):
                            raise ValueError(
                                f"{path}:{line_number}: violation label mismatch"
                            )
                    rows.append(
                        {
                            **transition,
                            "evaluation_suite": payload["evaluation_suite"],
                            "task_id": int(payload["task_id"]),
                        }
                    )
    return rows


def analyze_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_suite: dict[str, list[dict[str, Any]]] = defaultdict(list)
    by_suite_task: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        suite = str(row["evaluation_suite"])
        task_id = int(row["task_id"])
        by_suite[suite].append(row)
        by_suite_task[(suite, task_id)].append(row)
    return {
        "analysis_type": ACTION_DELTA_CROSS_SUITE_ANALYSIS_TYPE,
        "diagnostic_only": True,
        "production_efficiency_claim": False,
        "frozen_high_side_threshold": ACTION_DELTA_NONCONVERGENCE_THRESHOLD,
        "aggregate": summarize_rows(rows),
        "by_suite": {
            suite: summarize_rows(group)
            for suite, group in sorted(by_suite.items())
        },
        "by_suite_task": {
            f"{suite}/task_{task_id}": summarize_rows(group)
            for (suite, task_id), group in sorted(by_suite_task.items())
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="+", type=Path, help="prediction JSONL logs")
    parser.add_argument("--output", type=Path, required=True, help="output JSON path")
    args = parser.parse_args()

    results = analyze_rows(load_rows(args.inputs))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
