import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import torch


FOLD_MAP = {
    0: 0,
    1: 1,
    2: 2,
    3: 3,
    4: 4,
    5: 4,
    6: 3,
    7: 2,
    8: 1,
    9: 0,
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
        "--output",
        type=Path,
        default=Path(
            "benchmark_results/coda_anchor_feasibility/"
            "transition_index.jsonl"
        ),
    )
    parser.add_argument(
        "--summary",
        type=Path,
        default=Path(
            "benchmark_results/coda_anchor_feasibility/"
            "transition_index_summary.json"
        ),
    )
    parser.add_argument(
        "--origin",
        default="ACTUAL_WARM",
        choices=["ACTUAL_WARM", "COLD_PRIMARY", "ALL"],
    )
    args = parser.parse_args()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.summary.parent.mkdir(parents=True, exist_ok=True)

    shard_paths = sorted(args.raw_root.glob("raw_shadow_*.pt"))
    if not shard_paths:
        raise RuntimeError(f"No raw shards found under {args.raw_root}")

    origin_counts = Counter()
    task_prediction_counts = Counter()
    task_row_counts = Counter()
    fold_row_counts = Counter()
    safe_counts = Counter()

    prediction_count = 0
    row_count = 0

    # This Phase-A index uses only the causal prefix:
    #
    #   k < production_terminal_k
    #
    # Target at row k is whether the *next* transition,
    # a_(k+1) - a_k, satisfies the original RD-VLA
    # action-MSE stopping threshold.
    #
    # No post-convergence shadow-tail row is admitted.

    with args.output.open("w") as fout:
        for shard_path in shard_paths:
            shard = torch.load(
                shard_path,
                map_location="cpu",
                weights_only=False,
            )

            if shard.get("schema_version") != 2:
                raise RuntimeError(
                    f"Unexpected schema in {shard_path}: "
                    f"{shard.get('schema_version')}"
                )

            predictions = shard["predictions"]

            for pred_index, p in enumerate(predictions):
                origin = p["actual_origin"]
                origin_counts[origin] += 1

                if args.origin != "ALL" and origin != args.origin:
                    continue

                identity = p["identity"]
                task_id = int(identity["task_id"])
                terminal_k = int(p["production_terminal_k"])
                threshold = float(p["action_mse_threshold"])
                valid_len = int(p["valid_trajectory_length"])

                states = p["tensors"]["states"]
                actions = p["tensors"]["actions"]

                if states.shape[0] != valid_len:
                    raise RuntimeError(
                        f"State length mismatch: {identity}"
                    )
                if actions.shape[0] != valid_len:
                    raise RuntimeError(
                        f"Action length mismatch: {identity}"
                    )

                if terminal_k < 2 or terminal_k > valid_len:
                    raise RuntimeError(
                        f"Invalid terminal K={terminal_k}: {identity}"
                    )

                # Production K must be the first authoritative hit.
                authoritative = p["action_mse"]

                first_hit = None
                for kk in range(2, terminal_k + 1):
                    value = authoritative[kk]
                    if value is None:
                        raise RuntimeError(
                            f"Missing production MSE at k={kk}: {identity}"
                        )
                    if float(value) < threshold:
                        first_hit = kk
                        break

                if first_hit != terminal_k:
                    raise RuntimeError(
                        f"First-hit mismatch for {identity}: "
                        f"first_hit={first_hit}, terminal_k={terminal_k}"
                    )

                prediction_count += 1
                task_prediction_counts[task_id] += 1

                # Row k predicts the missing next Coda result at k+1.
                #
                # k=1 ... K-1 are all causally available.
                # The final row k=K-1 is the true-safe convergence row.
                for k in range(1, terminal_k):
                    next_k = k + 1

                    next_mse = authoritative[next_k]
                    if next_mse is None:
                        raise RuntimeError(
                            f"Missing target MSE at k={next_k}: {identity}"
                        )

                    target_safe = float(next_mse) < threshold

                    record = {
                        "shard": shard_path.name,
                        "prediction_index_in_shard": pred_index,
                        "task_id": task_id,
                        "fold": FOLD_MAP[task_id],
                        "episode_id": int(identity["episode_id"]),
                        "prediction_id": int(identity["prediction_id"]),
                        "timestep": int(identity["timestep"]),
                        "actual_origin": origin,
                        "production_terminal_k": terminal_k,
                        "k": k,
                        "next_k": next_k,
                        "target_action_mse": float(next_mse),
                        "target_safe": bool(target_safe),
                        "threshold": threshold,
                    }

                    fout.write(json.dumps(record) + "\n")

                    row_count += 1
                    task_row_counts[task_id] += 1
                    fold_row_counts[FOLD_MAP[task_id]] += 1
                    safe_counts["safe" if target_safe else "unsafe"] += 1

    summary = {
        "raw_root": str(args.raw_root),
        "origin_filter": args.origin,
        "shard_count": len(shard_paths),
        "prediction_count": prediction_count,
        "row_count": row_count,
        "source_origin_counts": dict(origin_counts),
        "task_prediction_counts": {
            str(k): task_prediction_counts[k]
            for k in sorted(task_prediction_counts)
        },
        "task_row_counts": {
            str(k): task_row_counts[k]
            for k in sorted(task_row_counts)
        },
        "fold_row_counts": {
            str(k): fold_row_counts[k]
            for k in sorted(fold_row_counts)
        },
        "target_counts": dict(safe_counts),
        "task_fold_map": {
            str(k): v for k, v in FOLD_MAP.items()
        },
        "causal_contract": {
            "row_k_range": "1 <= k < production_terminal_k",
            "input": ["S_k", "S_(k+1)", "a_k"],
            "regression_target": "a_(k+1) - a_k",
            "safe_label":
                "authoritative action_mse[k+1] < threshold",
            "post_convergence_rows_included": False,
        },
    }

    with args.summary.open("w") as f:
        json.dump(summary, f, indent=2)

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
