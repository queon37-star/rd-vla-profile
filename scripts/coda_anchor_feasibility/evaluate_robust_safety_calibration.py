import argparse
import json
from pathlib import Path

import numpy as np
import torch

from evaluate_linear_action_predictor import (
    set_seed,
    train_model,
    predict_rows,
    row_metrics,
    replay_metrics,
)


QUANTILES = [0.99, 0.995, 0.999]
MODEL_NAME = "delta_only"


def conservative_quantile(values, q):
    """
    One-sided conservative empirical quantile.

    method='higher' selects an observed residual at or above
    the requested quantile rather than interpolating downward.
    """
    values = np.asarray(values, dtype=np.float64)

    if len(values) == 0:
        raise RuntimeError("Cannot calibrate on empty residual set")

    value = float(
        np.quantile(
            values,
            q,
            method="higher",
        )
    )

    # We only need protection against under-estimation.
    # Never let a negative residual quantile make the gate less conservative.
    return max(0.0, value)


def evaluate_with_bound(
    pred_scores,
    target_safe,
    trajectory_ids,
    ks,
    action_threshold,
    q_bound,
):
    effective_threshold = action_threshold - q_bound

    if effective_threshold <= 0.0:
        pred_safe = np.zeros(
            len(pred_scores),
            dtype=np.bool_,
        )

        # row_metrics/replay_metrics expect a numeric threshold.
        # Use -inf so no row activates.
        threshold_for_eval = -np.inf
    else:
        pred_safe = (
            np.asarray(pred_scores)
            <= effective_threshold
        )
        threshold_for_eval = effective_threshold

    row = row_metrics(
        pred_scores,
        target_safe,
        threshold_for_eval,
    )

    replay = replay_metrics(
        pred_scores,
        target_safe,
        trajectory_ids,
        ks,
        threshold_for_eval,
    )

    return {
        "q_bound": float(q_bound),
        "effective_predicted_mse_threshold": float(
            effective_threshold
        ),
        "row_metrics": row,
        "sequential_replay": replay,
    }


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
        "--output",
        type=Path,
        default=Path(
            "benchmark_results/"
            "coda_anchor_feasibility/"
            "robust_safety_calibration_results.json"
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

    parser.add_argument(
        "--action-threshold",
        type=float,
        default=0.001,
    )

    args = parser.parse_args()

    set_seed(args.seed)

    print("Loading cache:", args.cache)

    cache = torch.load(
        args.cache,
        map_location="cpu",
        weights_only=False,
    )

    folds = cache["folds"].numpy()
    task_ids = cache["task_ids"].numpy()
    target_safe = cache["target_safe"].numpy()
    target_mse = cache["target_mse"].numpy()
    trajectory_ids = cache["trajectory_ids"].numpy()
    ks = cache["ks"].numpy()

    n = len(folds)

    print("rows:", n)
    print("trajectories:", cache["trajectory_count"])
    print("device:", args.device)

    results = {
        "model": MODEL_NAME,
        "action_threshold": args.action_threshold,
        "protocol": {
            "outer_test_fold": "f",
            "outer_training_pool": "all other 4 task-pair folds",
            "inner_calibration":
                "4-way task-pair OOF within outer training pool",
            "final_predictor":
                "trained on all 4 non-test folds",
            "test_data_used_for_calibration": False,
            "quantiles": QUANTILES,
            "global_bound":
                "Q_q(true_mse - predicted_mse) over inner OOF rows",
            "task_robust_bound":
                "max over per-task Q_q(true_mse - predicted_mse)",
        },
        "outer_folds": {},
    }

    # OOF test predictions from the final outer models.
    final_oof_scores = np.full(
        n,
        np.nan,
        dtype=np.float64,
    )

    # Each outer test fold receives its own calibrated bounds.
    outer_bounds = {
        str(q): {
            "global": {},
            "task_robust": {},
        }
        for q in QUANTILES
    }

    # -----------------------------------------------------------------
    # Outer loop:
    # test fold is completely untouched until final evaluation.
    # -----------------------------------------------------------------

    for outer_test_fold in range(5):
        print()
        print("=" * 76)
        print("OUTER TEST FOLD:", outer_test_fold)
        print("=" * 76)

        non_test_folds = [
            f
            for f in range(5)
            if f != outer_test_fold
        ]

        outer_test_idx = np.where(
            folds == outer_test_fold
        )[0]

        outer_pool_idx = np.where(
            folds != outer_test_fold
        )[0]

        print(
            "non-test folds:",
            non_test_folds,
        )
        print(
            "outer pool rows:",
            len(outer_pool_idx),
        )
        print(
            "outer test rows:",
            len(outer_test_idx),
        )

        # -------------------------------------------------------------
        # Inner OOF predictions over all non-test tasks.
        #
        # Every calibration row must be predicted by a model that did
        # NOT train on its task-pair fold.
        # -------------------------------------------------------------

        inner_scores = np.full(
            n,
            np.nan,
            dtype=np.float64,
        )

        for inner_val_fold in non_test_folds:
            inner_train_folds = [
                f
                for f in non_test_folds
                if f != inner_val_fold
            ]

            inner_train_idx = np.where(
                np.isin(
                    folds,
                    inner_train_folds,
                )
            )[0]

            inner_val_idx = np.where(
                folds == inner_val_fold
            )[0]

            print()
            print(
                f"  Inner val fold {inner_val_fold}"
            )
            print(
                "    train folds:",
                inner_train_folds,
            )
            print(
                "    train rows:",
                len(inner_train_idx),
                "val rows:",
                len(inner_val_idx),
            )

            model, stats = train_model(
                cache=cache,
                train_indices=inner_train_idx,
                model_name=MODEL_NAME,
                device=args.device,
                epochs=args.epochs,
                batch_size=args.batch_size,
                lr=args.lr,
                seed=(
                    args.seed
                    + outer_test_fold * 100
                    + inner_val_fold
                ),
            )

            val_indices, val_pred = predict_rows(
                cache=cache,
                row_indices=inner_val_idx,
                model=model,
                stats=stats,
                model_name=MODEL_NAME,
                device=args.device,
                batch_size=args.batch_size,
            )

            val_scores = (
                val_pred
                .float()
                .pow(2)
                .mean(dim=(1, 2))
                .numpy()
            )

            inner_scores[val_indices] = val_scores

            del model
            del stats

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        if np.any(
            np.isnan(inner_scores[outer_pool_idx])
        ):
            raise RuntimeError(
                "Missing inner OOF calibration predictions"
            )

        # Signed residual:
        #
        #   positive => predictor UNDER-estimated action change
        #
        residuals = (
            target_mse[outer_pool_idx]
            - inner_scores[outer_pool_idx]
        )

        calibration_tasks = sorted(
            np.unique(
                task_ids[outer_pool_idx]
            ).tolist()
        )

        # -------------------------------------------------------------
        # Fit one final predictor on all 8 non-test tasks.
        # -------------------------------------------------------------

        print()
        print("  Training final outer model...")
        print(
            "    train folds:",
            non_test_folds,
        )

        final_model, final_stats = train_model(
            cache=cache,
            train_indices=outer_pool_idx,
            model_name=MODEL_NAME,
            device=args.device,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            seed=args.seed + 1000 + outer_test_fold,
        )

        test_indices, test_pred = predict_rows(
            cache=cache,
            row_indices=outer_test_idx,
            model=final_model,
            stats=final_stats,
            model_name=MODEL_NAME,
            device=args.device,
            batch_size=args.batch_size,
        )

        test_scores = (
            test_pred
            .float()
            .pow(2)
            .mean(dim=(1, 2))
            .numpy()
        )

        final_oof_scores[test_indices] = test_scores

        del final_model
        del final_stats

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        fold_result = {
            "test_fold": outer_test_fold,
            "test_tasks": sorted(
                np.unique(
                    task_ids[outer_test_idx]
                ).tolist()
            ),
            "calibration_tasks": calibration_tasks,
            "calibration_row_count": int(
                len(outer_pool_idx)
            ),
            "test_row_count": int(
                len(outer_test_idx)
            ),
            "residual_summary": {
                "mean": float(
                    np.mean(residuals)
                ),
                "median": float(
                    np.median(residuals)
                ),
                "positive_fraction": float(
                    np.mean(residuals > 0)
                ),
                "max": float(
                    np.max(residuals)
                ),
            },
            "quantiles": {},
        }

        # -------------------------------------------------------------
        # Calibrate global and task-robust one-sided bounds.
        # -------------------------------------------------------------

        for q in QUANTILES:
            q_key = str(q)

            global_q = conservative_quantile(
                residuals,
                q,
            )

            per_task_q = {}

            for task_id in calibration_tasks:
                task_mask = (
                    task_ids[outer_pool_idx]
                    == task_id
                )

                task_residuals = residuals[
                    task_mask
                ]

                per_task_q[str(task_id)] = (
                    conservative_quantile(
                        task_residuals,
                        q,
                    )
                )

            task_robust_q = max(
                per_task_q.values()
            )

            outer_bounds[q_key]["global"][
                str(outer_test_fold)
            ] = global_q

            outer_bounds[q_key]["task_robust"][
                str(outer_test_fold)
            ] = task_robust_q

            # Calibration OOF sanity check
            cal_global = evaluate_with_bound(
                pred_scores=inner_scores[
                    outer_pool_idx
                ],
                target_safe=target_safe[
                    outer_pool_idx
                ],
                trajectory_ids=trajectory_ids[
                    outer_pool_idx
                ],
                ks=ks[
                    outer_pool_idx
                ],
                action_threshold=args.action_threshold,
                q_bound=global_q,
            )

            cal_robust = evaluate_with_bound(
                pred_scores=inner_scores[
                    outer_pool_idx
                ],
                target_safe=target_safe[
                    outer_pool_idx
                ],
                trajectory_ids=trajectory_ids[
                    outer_pool_idx
                ],
                ks=ks[
                    outer_pool_idx
                ],
                action_threshold=args.action_threshold,
                q_bound=task_robust_q,
            )

            # Completely unseen outer tasks
            test_global = evaluate_with_bound(
                pred_scores=test_scores,
                target_safe=target_safe[
                    test_indices
                ],
                trajectory_ids=trajectory_ids[
                    test_indices
                ],
                ks=ks[
                    test_indices
                ],
                action_threshold=args.action_threshold,
                q_bound=global_q,
            )

            test_robust = evaluate_with_bound(
                pred_scores=test_scores,
                target_safe=target_safe[
                    test_indices
                ],
                trajectory_ids=trajectory_ids[
                    test_indices
                ],
                ks=ks[
                    test_indices
                ],
                action_threshold=args.action_threshold,
                q_bound=task_robust_q,
            )

            fold_result["quantiles"][q_key] = {
                "global": {
                    "q_bound": global_q,
                    "calibration": cal_global,
                    "test": test_global,
                },
                "task_robust": {
                    "q_bound": task_robust_q,
                    "per_task_q": per_task_q,
                    "calibration": cal_robust,
                    "test": test_robust,
                },
            }

            print()
            print(
                f"  q={q}: "
                f"global={global_q:.8f}, "
                f"task_robust={task_robust_q:.8f}"
            )
            print(
                "    test global replay:",
                test_global["sequential_replay"],
            )
            print(
                "    test robust replay:",
                test_robust["sequential_replay"],
            )

        results["outer_folds"][
            str(outer_test_fold)
        ] = fold_result

    if np.any(np.isnan(final_oof_scores)):
        raise RuntimeError(
            "Missing final outer-fold OOF predictions"
        )

    # -----------------------------------------------------------------
    # Aggregate unseen-task OOF replay.
    #
    # Every fold uses only the bound derived from its non-test tasks.
    # -----------------------------------------------------------------

    results["aggregate_oof"] = {}

    for q in QUANTILES:
        q_key = str(q)

        results["aggregate_oof"][q_key] = {}

        for method in [
            "global",
            "task_robust",
        ]:
            total_trajectories = 0
            total_activated = 0
            total_correct = 0
            total_false_early = 0
            total_no_skip = 0

            row_tp = 0
            row_fp = 0
            row_fn = 0
            row_tn = 0

            fold_details = {}

            for test_fold in range(5):
                mask = folds == test_fold

                bound = outer_bounds[
                    q_key
                ][method][str(test_fold)]

                evaluation = evaluate_with_bound(
                    pred_scores=final_oof_scores[mask],
                    target_safe=target_safe[mask],
                    trajectory_ids=trajectory_ids[mask],
                    ks=ks[mask],
                    action_threshold=args.action_threshold,
                    q_bound=bound,
                )

                row = evaluation["row_metrics"]
                replay = evaluation[
                    "sequential_replay"
                ]

                row_tp += row["tp"]
                row_fp += row["fp_false_safe"]
                row_fn += row["fn"]
                row_tn += row["tn"]

                total_trajectories += replay[
                    "trajectory_count"
                ]
                total_activated += replay[
                    "activated"
                ]
                total_correct += replay[
                    "correct_safe_stop"
                ]
                total_false_early += replay[
                    "false_early_stop"
                ]
                total_no_skip += replay[
                    "no_skip"
                ]

                fold_details[str(test_fold)] = {
                    "q_bound": bound,
                    "effective_threshold":
                        args.action_threshold - bound,
                    "replay": replay,
                }

            predicted_safe_rows = (
                row_tp + row_fp
            )

            results["aggregate_oof"][q_key][
                method
            ] = {
                "row_metrics": {
                    "tp": row_tp,
                    "fp_false_safe": row_fp,
                    "fn": row_fn,
                    "tn": row_tn,
                    "precision": (
                        row_tp / predicted_safe_rows
                        if predicted_safe_rows
                        else None
                    ),
                    "safe_recall": (
                        row_tp / (row_tp + row_fn)
                        if row_tp + row_fn
                        else None
                    ),
                    "row_coverage": (
                        predicted_safe_rows / n
                    ),
                },
                "sequential_replay": {
                    "trajectory_count":
                        total_trajectories,
                    "activated":
                        total_activated,
                    "correct_safe_stop":
                        total_correct,
                    "false_early_stop":
                        total_false_early,
                    "no_skip":
                        total_no_skip,
                    "activation_precision": (
                        total_correct
                        / total_activated
                        if total_activated
                        else None
                    ),
                    "safe_opportunity_capture": (
                        total_correct
                        / total_trajectories
                        if total_trajectories
                        else None
                    ),
                },
                "fold_details": fold_details,
            }

    args.output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with args.output.open("w") as f:
        json.dump(
            results,
            f,
            indent=2,
        )

    print()
    print("=" * 76)
    print("FINAL ROBUST-CALIBRATION SUMMARY")
    print("=" * 76)

    for q in QUANTILES:
        q_key = str(q)

        print()
        print(
            f"RESIDUAL QUANTILE = {q}"
        )

        for method in [
            "global",
            "task_robust",
        ]:
            r = results[
                "aggregate_oof"
            ][q_key][method]

            print()
            print(" ", method)

            print(
                "    row:",
                r["row_metrics"],
            )

            print(
                "    replay:",
                r["sequential_replay"],
            )

            print(
                "    fold bounds:",
                {
                    k: round(
                        v["q_bound"],
                        8,
                    )
                    for k, v
                    in r[
                        "fold_details"
                    ].items()
                },
            )

    print()
    print(
        "Results written to:",
        args.output,
    )


if __name__ == "__main__":
    main()
