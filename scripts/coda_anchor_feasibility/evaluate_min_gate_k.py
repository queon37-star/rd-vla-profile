import argparse
import json
from pathlib import Path

import numpy as np
import torch

from evaluate_linear_action_predictor import (
    set_seed,
    train_model,
    predict_rows,
)


MODEL_NAME = "delta_only"
METHOD = "cp95_false_safe_risk_0.01"

MIN_K_VALUES = [1, 2, 3, 4]


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--cache",
        type=Path,
        default=Path(
            "benchmark_results/"
            "coda_anchor_feasibility/"
            "action_delta_cache.pt"
        ),
    )

    parser.add_argument(
        "--calibration-results",
        type=Path,
        default=Path(
            "benchmark_results/"
            "coda_anchor_feasibility/"
            "multitask_risk_calibration_results.json"
        ),
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "benchmark_results/"
            "coda_anchor_feasibility/"
            "min_gate_k_sweep_results.json"
        ),
    )

    parser.add_argument(
        "--device",
        default=(
            "cuda"
            if torch.cuda.is_available()
            else "cpu"
        ),
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=60,
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=512,
    )

    parser.add_argument(
        "--lr",
        type=float,
        default=1e-3,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=7,
    )

    args = parser.parse_args()

    set_seed(args.seed)

    print("Loading cache:", args.cache)

    cache = torch.load(
        args.cache,
        map_location="cpu",
        weights_only=False,
    )

    with args.calibration_results.open() as f:
        calibration = json.load(f)

    folds = cache["folds"].numpy()
    task_ids = cache["task_ids"].numpy()
    target_safe = cache["target_safe"].numpy()
    trajectory_ids = cache["trajectory_ids"].numpy()
    ks = cache["ks"].numpy()

    aggregate = {
        min_k: {
            "trajectory_count": 0,
            "activated": 0,
            "correct": 0,
            "false_early": 0,
            "no_skip": 0,
            "folds": {},
        }
        for min_k in MIN_K_VALUES
    }

    # -------------------------------------------------------------
    # Reproduce each final outer-fold predictor.
    #
    # IMPORTANT:
    # We reuse the already-selected CP 1% threshold.
    # min_k is NOT re-calibrated.
    # -------------------------------------------------------------

    for test_fold in range(5):
        print()
        print("=" * 72)
        print("OUTER TEST FOLD:", test_fold)
        print("=" * 72)

        method_result = (
            calibration["outer_folds"]
            [str(test_fold)]
            ["methods"]
            [METHOD]
        )

        if method_result is None:
            raise RuntimeError(
                f"No threshold exists for {METHOD}, "
                f"fold={test_fold}"
            )

        threshold = float(
            method_result["threshold"]
        )

        train_idx = np.where(
            folds != test_fold
        )[0]

        test_idx = np.where(
            folds == test_fold
        )[0]

        test_tasks = sorted(
            np.unique(
                task_ids[test_idx]
            ).tolist()
        )

        print("test tasks:", test_tasks)
        print("threshold:", threshold)
        print("train rows:", len(train_idx))
        print("test rows:", len(test_idx))

        # Same final-model seed used in the previous
        # multi-task risk calibration experiment.
        model, stats = train_model(
            cache=cache,
            train_indices=train_idx,
            model_name=MODEL_NAME,
            device=args.device,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            seed=args.seed + 1000 + test_fold,
        )

        test_indices, pred_delta = predict_rows(
            cache=cache,
            row_indices=test_idx,
            model=model,
            stats=stats,
            model_name=MODEL_NAME,
            device=args.device,
            batch_size=args.batch_size,
        )

        pred_scores = (
            pred_delta
            .float()
            .pow(2)
            .mean(dim=(1, 2))
            .numpy()
        )

        # ---------------------------------------------------------
        # Group all rows belonging to the same prediction trajectory.
        # ---------------------------------------------------------

        by_trajectory = {}

        for local_i, global_idx in enumerate(test_indices):
            traj = int(
                trajectory_ids[global_idx]
            )

            by_trajectory.setdefault(
                traj,
                [],
            ).append(
                (
                    local_i,
                    int(global_idx),
                )
            )

        fold_stats = {
            min_k: {
                "trajectory_count": 0,
                "activated": 0,
                "correct": 0,
                "false_early": 0,
                "no_skip": 0,
            }
            for min_k in MIN_K_VALUES
        }

        # ---------------------------------------------------------
        # Sequential runtime replay.
        #
        # For each trajectory:
        # - examine k in increasing order
        # - reject k < min_k
        # - first score <= fixed threshold activates the gate
        # ---------------------------------------------------------

        for traj, members in by_trajectory.items():
            members = sorted(
                members,
                key=lambda x: int(
                    ks[x[1]]
                ),
            )

            for min_k in MIN_K_VALUES:
                fold_stats[min_k][
                    "trajectory_count"
                ] += 1

                activated = None

                for local_i, global_idx in members:
                    k = int(
                        ks[global_idx]
                    )

                    if k < min_k:
                        continue

                    score = float(
                        pred_scores[local_i]
                    )

                    if score <= threshold:
                        activated = (
                            local_i,
                            global_idx,
                            score,
                        )
                        break

                if activated is None:
                    fold_stats[min_k][
                        "no_skip"
                    ] += 1
                    continue

                _, global_idx, _ = activated

                fold_stats[min_k][
                    "activated"
                ] += 1

                if bool(
                    target_safe[global_idx]
                ):
                    fold_stats[min_k][
                        "correct"
                    ] += 1
                else:
                    fold_stats[min_k][
                        "false_early"
                    ] += 1

        print()
        print("MIN-K results for this fold:")

        for min_k in MIN_K_VALUES:
            s = fold_stats[min_k]

            precision = (
                s["correct"] / s["activated"]
                if s["activated"]
                else None
            )

            capture = (
                s["correct"]
                / s["trajectory_count"]
                if s["trajectory_count"]
                else None
            )

            print(
                f"  min_k={min_k}: "
                f"activated={s['activated']}, "
                f"correct={s['correct']}, "
                f"false_early={s['false_early']}, "
                f"no_skip={s['no_skip']}, "
                f"precision={precision}, "
                f"capture={capture}"
            )

            aggregate[min_k][
                "trajectory_count"
            ] += s["trajectory_count"]

            aggregate[min_k][
                "activated"
            ] += s["activated"]

            aggregate[min_k][
                "correct"
            ] += s["correct"]

            aggregate[min_k][
                "false_early"
            ] += s["false_early"]

            aggregate[min_k][
                "no_skip"
            ] += s["no_skip"]

            aggregate[min_k][
                "folds"
            ][str(test_fold)] = {
                **s,
                "test_tasks": test_tasks,
                "threshold": threshold,
                "activation_precision": precision,
                "safe_opportunity_capture": capture,
            }

        del model
        del stats

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # -------------------------------------------------------------
    # Aggregate summary
    # -------------------------------------------------------------

    result = {
        "model": MODEL_NAME,
        "method": METHOD,
        "min_k_values": MIN_K_VALUES,
        "threshold_recalibrated_for_min_k": False,
        "results": {},
    }

    baseline_correct = aggregate[1]["correct"]

    print()
    print("=" * 72)
    print("FINAL MIN-GATE-K SWEEP")
    print("=" * 72)

    for min_k in MIN_K_VALUES:
        s = aggregate[min_k]

        precision = (
            s["correct"] / s["activated"]
            if s["activated"]
            else None
        )

        capture = (
            s["correct"]
            / s["trajectory_count"]
            if s["trajectory_count"]
            else None
        )

        retention = (
            s["correct"] / baseline_correct
            if baseline_correct
            else None
        )

        result["results"][str(min_k)] = {
            **s,
            "activation_precision": precision,
            "safe_opportunity_capture": capture,
            "correct_skip_retention_vs_min_k1":
                retention,
        }

        print()
        print(f"min_k={min_k}")
        print(
            "  trajectory_count:",
            s["trajectory_count"],
        )
        print(
            "  activated:",
            s["activated"],
        )
        print(
            "  correct:",
            s["correct"],
        )
        print(
            "  false_early:",
            s["false_early"],
        )
        print(
            "  no_skip:",
            s["no_skip"],
        )
        print(
            "  activation_precision:",
            precision,
        )
        print(
            "  safe_opportunity_capture:",
            capture,
        )
        print(
            "  correct_skip_retention_vs_min_k1:",
            retention,
        )

    args.output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with args.output.open("w") as f:
        json.dump(
            result,
            f,
            indent=2,
        )

    print()
    print(
        "Results written to:",
        args.output,
    )


if __name__ == "__main__":
    main()
