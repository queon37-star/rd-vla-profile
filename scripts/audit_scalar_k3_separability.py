#!/usr/bin/env python3
"""Audit k=3 scalar score separation without retraining."""

from __future__ import annotations

import argparse
import json
import math
import statistics
import subprocess
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = REPO_ROOT / "benchmark_results/preconvergence_trigger/seed7/calibration_depth_audit/report.json"
DEFAULT_PROTOCOL = REPO_ROOT / "experiments/robot/libero/manifests/scalar_k3_separability_audit_v1.json"


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    require(isinstance(value, dict), f"JSON root must be an object: {path}")
    return value


def git_commit() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, check=True,
        capture_output=True, text=True,
    ).stdout.strip()


def sha256_file(path: Path) -> str:
    import hashlib
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def percentile(values: Sequence[float], q: float) -> float | None:
    if not values:
        return None
    ordered = sorted(float(v) for v in values)
    if len(ordered) == 1:
        return ordered[0]
    p = (len(ordered) - 1) * q
    lo, hi = math.floor(p), math.ceil(p)
    if lo == hi:
        return ordered[lo]
    return ordered[lo] * (hi - p) + ordered[hi] * (p - lo)


def numeric_summary(values: Sequence[float]) -> dict[str, Any]:
    values = [float(v) for v in values]
    require(all(math.isfinite(v) for v in values), "non-finite score")
    return {
        "count": len(values),
        "mean": statistics.fmean(values) if values else None,
        "median": statistics.median(values) if values else None,
        "p05": percentile(values, 0.05),
        "p25": percentile(values, 0.25),
        "p75": percentile(values, 0.75),
        "p95": percentile(values, 0.95),
        "minimum": min(values) if values else None,
        "maximum": max(values) if values else None,
    }


def roc_auc(positives: Sequence[float], negatives: Sequence[float]) -> float | None:
    if not positives or not negatives:
        return None
    wins = 0.0
    for positive in positives:
        for negative in negatives:
            wins += float(positive > negative) + 0.5 * float(positive == negative)
    return wins / (len(positives) * len(negatives))


def average_precision(positives: Sequence[float], negatives: Sequence[float]) -> float | None:
    if not positives or not negatives:
        return None
    labeled = [(float(v), 1) for v in positives] + [(float(v), 0) for v in negatives]
    labeled.sort(reverse=True)
    true_positive = 0
    total = 0.0
    for rank, (_, label) in enumerate(labeled, start=1):
        if label:
            true_positive += 1
            total += true_positive / rank
    return total / len(positives)


def discrimination(rows, positive_max_k: int, negative_min_k: int, field: str):
    positive = [float(r[field]) for r in rows if int(r["K_action"]) <= positive_max_k]
    negative = [float(r[field]) for r in rows if int(r["K_action"]) >= negative_min_k]
    return {
        "positive_definition": f"K_action<={positive_max_k}",
        "negative_definition": f"K_action>={negative_min_k}",
        "positive_count": len(positive),
        "negative_count": len(negative),
        "roc_auc_high_score_means_safe": roc_auc(positive, negative),
        "average_precision_safe_class": average_precision(positive, negative),
    }


def trigger_metrics(rows, offset: float, protocol: Mapping[str, Any]):
    safe_max = int(protocol["safe_gate_max_K_action"])
    unsafe_min = int(protocol["unsafe_gate_min_K_action"])
    severe_min = int(protocol["severe_unsafe_min_K_action"])
    very_min = int(protocol["very_severe_unsafe_min_K_action"])

    def group(minimum=None, maximum=None):
        return [r for r in rows if (minimum is None or int(r["K_action"]) >= minimum)
                and (maximum is None or int(r["K_action"]) <= maximum)]

    def rate(items):
        return (sum(float(r["k3_score_margin"]) >= offset for r in items) / len(items)) if items else None

    safe = group(maximum=safe_max)
    unsafe = group(minimum=unsafe_min)
    severe = group(minimum=severe_min)
    very = group(minimum=very_min)
    triggered = [r for r in rows if float(r["k3_score_margin"]) >= offset]
    safe_triggered = sum(float(r["k3_score_margin"]) >= offset for r in safe)
    return {
        "margin_offset": offset,
        "overall_trigger_rate": len(triggered) / len(rows),
        "safe_prediction_count": len(safe),
        "safe_trigger_recall": rate(safe),
        "unsafe_prediction_count": len(unsafe),
        "unsafe_false_trigger_rate": rate(unsafe),
        "severe_prediction_count": len(severe),
        "severe_false_trigger_rate": rate(severe),
        "very_severe_prediction_count": len(very),
        "very_severe_false_trigger_rate": rate(very),
        "trigger_precision_safe": safe_triggered / len(triggered) if triggered else None,
    }


def sweep(rows, protocol):
    margins = sorted({float(r["k3_score_margin"]) for r in rows}, reverse=True)
    offsets = [math.nextafter(margins[0], math.inf)] + margins
    points = [trigger_metrics(rows, value, protocol) for value in offsets]
    selected = {}
    for limit in protocol["severe_false_trigger_limits"]:
        feasible = [p for p in points if p["severe_false_trigger_rate"] <= float(limit)]
        best = max(feasible, key=lambda p: (p["safe_trigger_recall"], -p["overall_trigger_rate"], p["margin_offset"]))
        selected[f"severe_FPR<={float(limit):.2f}"] = best
    return {"candidate_offset_count": len(offsets), "selected_descriptive_operating_points": selected}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.output.exists() and not args.overwrite:
        raise FileExistsError(f"refusing to overwrite: {args.output}")

    protocol = load_json(args.protocol)
    source = load_json(args.input)
    require(source.get("formal_run") is True, "input depth audit is not formal")
    require(source.get("latency_reporting_scope") == protocol["latency_scope"], "latency scope mismatch")
    validation = source["validation"]
    require(validation["prediction_count"] == protocol["expected_prediction_count"], "prediction count mismatch")
    require(validation["actual_warm_count"] == protocol["expected_actual_warm_count"], "warm count mismatch")
    require(validation["task_ids"] == protocol["expected_task_ids"], "task coverage mismatch")

    rows = [r for r in source["prediction_rows"] if r["actual_origin"] == "ACTUAL_WARM"]
    require(len(rows) == protocol["expected_actual_warm_count"], "warm row count mismatch")
    for row in rows:
        score = float(row["k3_scalar_score"])
        threshold = float(row["task_threshold"])
        margin = float(row["k3_score_margin"])
        require(all(math.isfinite(v) for v in (score, threshold, margin)), "non-finite score")
        require(math.isclose(score - threshold, margin, rel_tol=1e-7, abs_tol=1e-7), "margin mismatch")
        require(bool(row["k3_triggered"]) == (margin >= 0.0), "decision mismatch")

    safe_max = int(protocol["safe_gate_max_K_action"])
    unsafe_min = int(protocol["unsafe_gate_min_K_action"])
    severe_min = int(protocol["severe_unsafe_min_K_action"])
    very_min = int(protocol["very_severe_unsafe_min_K_action"])
    cohort_defs = {
        "safe_K<=4": lambda k: k <= safe_max,
        "unsafe_K=5": lambda k: unsafe_min <= k < severe_min,
        "severe_K=6-7": lambda k: severe_min <= k < very_min,
        "very_severe_K>=8": lambda k: k >= very_min,
    }
    cohorts = {}
    for name, accept in cohort_defs.items():
        selected = [r for r in rows if accept(int(r["K_action"]))]
        cohorts[name] = {
            "prediction_count": len(selected),
            "score": numeric_summary([r["k3_scalar_score"] for r in selected]),
            "margin": numeric_summary([r["k3_score_margin"] for r in selected]),
            "deployed_trigger_rate": sum(bool(r["k3_triggered"]) for r in selected) / len(selected) if selected else None,
        }

    by_task = defaultdict(list)
    for row in rows:
        by_task[int(row["task_id"])].append(row)
    per_task = {}
    task_aucs = []
    for task_id in range(10):
        metrics = discrimination(by_task[task_id], safe_max, severe_min, "k3_scalar_score")
        metrics["deployed_operating_point"] = trigger_metrics(by_task[task_id], 0.0, protocol)
        if metrics["roc_auc_high_score_means_safe"] is not None:
            task_aucs.append(metrics["roc_auc_high_score_means_safe"])
        per_task[str(task_id)] = metrics

    outcomes = {}
    for success, name in ((True, "success"), (False, "failure")):
        selected = [r for r in rows if bool(r["success"]) is success]
        outcomes[name] = {
            "episode_count": len({(r["task_id"], r["episode_id"]) for r in selected}),
            **trigger_metrics(selected, 0.0, protocol),
        }

    report = {
        "schema_version": 1,
        "formal_run": True,
        "code_git_commit": git_commit(),
        "latency_reporting_scope": protocol["latency_scope"],
        "protocol": protocol,
        "inputs": {
            "depth_audit_report": {"path": str(args.input.resolve()), "sha256": sha256_file(args.input)},
            "protocol_manifest": {"path": str(args.protocol.resolve()), "sha256": sha256_file(args.protocol)},
        },
        "validation": {"actual_warm_count": len(rows), "task_ids": list(range(10)), "nonfinite_score_count": 0},
        "cohort_score_distributions": cohorts,
        "deployed_operating_point": trigger_metrics(rows, 0.0, protocol),
        "threshold_free_discrimination": {
            "safe_vs_all_unsafe_score": discrimination(rows, safe_max, unsafe_min, "k3_scalar_score"),
            "safe_vs_severe_score": discrimination(rows, safe_max, severe_min, "k3_scalar_score"),
            "safe_vs_very_severe_score": discrimination(rows, safe_max, very_min, "k3_scalar_score"),
            "safe_vs_severe_margin": discrimination(rows, safe_max, severe_min, "k3_score_margin"),
        },
        "uniform_margin_offset_sweep": sweep(rows, protocol),
        "per_task_safe_vs_severe": {
            "tasks": per_task,
            "task_macro_roc_auc": statistics.fmean(task_aucs) if task_aucs else None,
            "task_minimum_roc_auc": min(task_aucs) if task_aucs else None,
            "task_maximum_roc_auc": max(task_aucs) if task_aucs else None,
        },
        "success_failure_deployed_point": outcomes,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")

    deployed = report["deployed_operating_point"]
    auc = report["threshold_free_discrimination"]["safe_vs_severe_score"]["roc_auc_high_score_means_safe"]
    print("Formal run: True")
    print(f"Actual-warm predictions: {len(rows)}")
    print(f"Deployed k=3 severe false-trigger rate: {100.0 * deployed['severe_false_trigger_rate']:.3f}%")
    print(f"Safe-vs-severe k=3 score ROC-AUC: {auc:.6f}")
    for name, point in report["uniform_margin_offset_sweep"]["selected_descriptive_operating_points"].items():
        print(f"{name}: safe recall={100.0 * point['safe_trigger_recall']:.3f}%, severe FPR={100.0 * point['severe_false_trigger_rate']:.3f}%, margin offset={point['margin_offset']:.6g}")
    print(f"Wrote: {args.output}")


if __name__ == "__main__":
    main()
