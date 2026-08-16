import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch


FEATURES = [
    "delta_rms",
    "relative_delta_rms",
    "max_abs_delta",
    "cosine_distance",
]


def rankdata(x):
    """Simple average-rank implementation for Spearman correlation."""
    x = np.asarray(x)
    order = np.argsort(x, kind="mergesort")
    ranks = np.empty(len(x), dtype=np.float64)

    i = 0
    while i < len(x):
        j = i + 1
        while j < len(x) and x[order[j]] == x[order[i]]:
            j += 1

        avg_rank = 0.5 * (i + j - 1) + 1.0
        ranks[order[i:j]] = avg_rank
        i = j

    return ranks


def spearman(x, y):
    rx = rankdata(x)
    ry = rankdata(y)

    if np.std(rx) == 0 or np.std(ry) == 0:
        return float("nan")

    return float(np.corrcoef(rx, ry)[0, 1])


def compute_features(s_k, s_next):
    s_k = s_k.float().reshape(-1)
    s_next = s_next.float().reshape(-1)

    delta = s_next - s_k

    delta_rms = torch.sqrt(torch.mean(delta * delta)).item()

    base_rms = torch.sqrt(torch.mean(s_k * s_k)).item()
    relative_delta_rms = delta_rms / max(base_rms, 1e-12)

    max_abs_delta = torch.max(torch.abs(delta)).item()

    denom = torch.linalg.vector_norm(s_k) * torch.linalg.vector_norm(s_next)
    if denom.item() == 0:
        cosine_distance = 0.0
    else:
        cosine_similarity = torch.dot(s_k, s_next) / denom
        cosine_distance = (1.0 - cosine_similarity).item()

    return {
        "delta_rms": delta_rms,
        "relative_delta_rms": relative_delta_rms,
        "max_abs_delta": max_abs_delta,
        "cosine_distance": cosine_distance,
    }


def choose_precision_threshold(values, labels, target_precision):
    """
    Lower scalar value => predict safe.

    Choose the threshold with maximum train coverage while satisfying
    required precision.
    """
    values = np.asarray(values, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.bool_)

    order = np.argsort(values)
    v = values[order]
    y = labels[order]

    cum_safe = np.cumsum(y.astype(np.int64))
    counts = np.arange(1, len(v) + 1)
    precision = cum_safe / counts

    valid = np.where(precision >= target_precision)[0]

    if len(valid) == 0:
        return None

    best_idx = valid[-1]
    return float(v[best_idx])


def metrics_for_predictions(pred_safe, target_safe):
    pred_safe = np.asarray(pred_safe, dtype=np.bool_)
    target_safe = np.asarray(target_safe, dtype=np.bool_)

    tp = int(np.sum(pred_safe & target_safe))
    fp = int(np.sum(pred_safe & ~target_safe))
    fn = int(np.sum(~pred_safe & target_safe))
    tn = int(np.sum(~pred_safe & ~target_safe))

    predicted_safe = tp + fp
    actual_safe = tp + fn
    total = len(target_safe)

    precision = tp / predicted_safe if predicted_safe else None
    recall = tp / actual_safe if actual_safe else None
    coverage = predicted_safe / total if total else None
    false_safe_fraction = fp / predicted_safe if predicted_safe else None

    return {
        "tp": tp,
        "fp_false_safe": fp,
        "fn": fn,
        "tn": tn,
        "precision": precision,
        "safe_recall": recall,
        "row_coverage": coverage,
        "false_safe_fraction_among_predicted_safe": false_safe_fraction,
    }


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--raw-root",
        type=Path,
        default=Path(
            "benchmark_results/preconvergence_trigger/"
            "raw_shadow_calibration_seed7"
        ),
    )

    parser.add_argument(
        "--index",
        type=Path,
        default=Path(
            "benchmark_results/coda_anchor_feasibility/"
            "transition_index.jsonl"
        ),
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "benchmark_results/coda_anchor_feasibility/"
            "scalar_baseline_results.json"
        ),
    )

    args = parser.parse_args()

    rows = []
    with args.index.open() as f:
        for line in f:
            rows.append(json.loads(line))

    print(f"Loaded transition rows: {len(rows)}")

    by_shard = defaultdict(list)
    for row_id, row in enumerate(rows):
        by_shard[row["shard"]].append((row_id, row))

    feature_rows = [None] * len(rows)

    for shard_idx, shard_name in enumerate(sorted(by_shard)):
        shard_path = args.raw_root / shard_name

        shard = torch.load(
            shard_path,
            map_location="cpu",
            weights_only=False,
        )

        for row_id, row in by_shard[shard_name]:
            p = shard["predictions"][row["prediction_index_in_shard"]]

            k = int(row["k"])

            # tensor index 0 = iteration 1
            s_k = p["tensors"]["states"][k - 1]
            s_next = p["tensors"]["states"][k]

            features = compute_features(s_k, s_next)

            feature_rows[row_id] = {
                **row,
                **features,
            }

        print(
            f"[{shard_idx + 1:03d}/{len(by_shard):03d}] "
            f"{shard_name}"
        )

    target_mse = np.asarray(
        [r["target_action_mse"] for r in feature_rows],
        dtype=np.float64,
    )

    target_safe = np.asarray(
        [r["target_safe"] for r in feature_rows],
        dtype=np.bool_,
    )

    folds = np.asarray(
        [r["fold"] for r in feature_rows],
        dtype=np.int64,
    )

    results = {
        "row_count": len(feature_rows),
        "safe_count": int(np.sum(target_safe)),
        "unsafe_count": int(np.sum(~target_safe)),
        "features": {},
    }

    precision_targets = [0.95, 0.99, 0.995]

    for feature_name in FEATURES:
        values = np.asarray(
            [r[feature_name] for r in feature_rows],
            dtype=np.float64,
        )

        feature_result = {
            "spearman_vs_next_action_mse": spearman(
                values,
                target_mse,
            ),
            "oof": {},
        }

        for precision_target in precision_targets:
            all_pred = np.zeros(len(feature_rows), dtype=np.bool_)
            fold_thresholds = {}

            for fold in sorted(np.unique(folds)):
                train_mask = folds != fold
                test_mask = folds == fold

                threshold = choose_precision_threshold(
                    values[train_mask],
                    target_safe[train_mask],
                    precision_target,
                )

                fold_thresholds[str(int(fold))] = threshold

                if threshold is not None:
                    all_pred[test_mask] = (
                        values[test_mask] <= threshold
                    )

            feature_result["oof"][str(precision_target)] = {
                "fold_thresholds": fold_thresholds,
                "metrics": metrics_for_predictions(
                    all_pred,
                    target_safe,
                ),
            }

        results["features"][feature_name] = feature_result

    args.output.parent.mkdir(parents=True, exist_ok=True)

    with args.output.open("w") as f:
        json.dump(results, f, indent=2)

    print("\n=== RESULTS ===")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
