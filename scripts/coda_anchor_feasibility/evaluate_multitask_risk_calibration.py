import argparse
import json
from pathlib import Path

import numpy as np
import torch
from scipy.stats import beta

from evaluate_linear_action_predictor import (
    set_seed,
    train_model,
    predict_rows,
    replay_metrics,
    row_metrics,
    choose_replay_threshold,
)


MODEL_NAME = "delta_only"

EMPIRICAL_PRECISION_TARGETS = [
    0.99,
    0.995,
]

# Maximum tolerated false-safe probability.
RISK_LIMITS = [
    0.02,   # >= 98% precision
    0.01,   # >= 99% precision
    0.005,  # >= 99.5% precision
]

CONFIDENCE = 0.95


def cp_upper_bound(failures, trials, confidence=0.95):
    """
    One-sided exact Clopper-Pearson upper confidence bound
    for the Bernoulli failure probability.
    """
    if trials <= 0:
        return None

    if failures >= trials:
        return 1.0

    alpha = 1.0 - confidence

    return float(
        beta.ppf(
            1.0 - alpha,
            failures + 1,
            trials - failures,
        )
    )


def candidate_thresholds(scores, max_candidates=1200):
    scores = np.asarray(scores, dtype=np.float64)

    unique = np.unique(scores)

    if len(unique) <= max_candidates:
        return unique

    qs = np.linspace(
        0.0,
        1.0,
        max_candidates,
    )

    return np.unique(
        np.quantile(
            unique,
            qs,
        )
    )


def choose_cp_threshold(
    scores,
    labels,
    trajectory_ids,
    ks,
    risk_limit,
    confidence=0.95,
):
    """
    Choose the most useful threshold whose one-sided
    upper confidence bound on false-early-stop probability
    is <= risk_limit.
    """

    best = None

    for threshold in candidate_thresholds(scores):
        replay = replay_metrics(
            scores,
            labels,
            trajectory_ids,
            ks,
            threshold,
        )

        n = replay["activated"]
        m = replay["false_early_stop"]

        if n == 0:
            continue

        upper = cp_upper_bound(
            failures=m,
            trials=n,
            confidence=confidence,
        )

        if upper is None or upper > risk_limit:
            continue

        # Primary objective: maximize useful Coda skips.
        # Tie break toward lower observed risk and
        # more conservative threshold.
        key = (
            replay["correct_safe_stop"],
            -m,
            -upper,
            -threshold,
        )

        if best is None or key > best["key"]:
            best = {
                "threshold": float(threshold),
                "cp_upper_false_safe_risk": upper,
                "calibration_replay": replay,
                "key": key,
            }

    if best is None:
        return None

    del best["key"]

    return best


def evaluate_threshold(
    scores,
    labels,
    trajectory_ids,
    ks,
    threshold,
):
    return {
        "row_metrics": row_metrics(
            scores,
            labels,
            threshold,
        ),
        "sequential_replay": replay_metrics(
            scores,
            labels,
            trajectory_ids,
            ks,
            threshold,
        ),
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
            "multitask_risk_calibration_results.json"
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

    cache = torch.load(
        args.cache,
        map_location="cpu",
        weights_only=False,
    )

    folds = cache["folds"].numpy()
    task_ids = cache["task_ids"].numpy()
    target_safe = cache["target_safe"].numpy()
    trajectory_ids = cache["trajectory_ids"].numpy()
    ks = cache["ks"].numpy()

    n_rows = len(folds)

    print("rows:", n_rows)
    print("trajectories:", cache["trajectory_count"])
    print("device:", args.device)

    results = {
        "model": MODEL_NAME,
        "protocol": {
            "outer_test_fold":
                "one task-pair fold completely held out",
            "inner_oof_calibration":
                "all 4 remaining folds predicted OOF",
            "final_model":
                "trained on all 4 non-test folds",
            "test_used_for_threshold_selection": False,
            "confidence": CONFIDENCE,
            "empirical_precision_targets":
                EMPIRICAL_PRECISION_TARGETS,
            "risk_limits": RISK_LIMITS,
        },
        "outer_folds": {},
    }

    final_oof_scores = np.full(
        n_rows,
        np.nan,
        dtype=np.float64,
    )

    chosen_thresholds = {}

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

        outer_pool_idx = np.where(
            folds != outer_test_fold
        )[0]

        outer_test_idx = np.where(
            folds == outer_test_fold
        )[0]

        print("non-test folds:", non_test_folds)
        print("pool rows:", len(outer_pool_idx))
        print("test rows:", len(outer_test_idx))

        # ---------------------------------------------------------
        # 1. Inner OOF prediction over all 8 non-test tasks.
        # ---------------------------------------------------------

        inner_oof_scores = np.full(
            n_rows,
            np.nan,
            dtype=np.float64,
        )

        for inner_val_fold in non_test_folds:
            train_folds = [
                f
                for f in non_test_folds
                if f != inner_val_fold
            ]

            train_idx = np.where(
                np.isin(
                    folds,
                    train_folds,
                )
            )[0]

            val_idx = np.where(
                folds == inner_val_fold
            )[0]

            print()
            print(
                f"  inner val={inner_val_fold}, "
                f"train={train_folds}"
            )

            model, stats = train_model(
                cache=cache,
                train_indices=train_idx,
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
                row_indices=val_idx,
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

            inner_oof_scores[
                val_indices
            ] = val_scores

            del model
            del stats

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        if np.any(
            np.isnan(
                inner_oof_scores[
                    outer_pool_idx
                ]
            )
        ):
            raise RuntimeError(
                "Missing inner OOF predictions"
            )

        cal_scores = inner_oof_scores[
            outer_pool_idx
        ]

        cal_labels = target_safe[
            outer_pool_idx
        ]

        cal_traj = trajectory_ids[
            outer_pool_idx
        ]

        cal_ks = ks[
            outer_pool_idx
        ]

        # ---------------------------------------------------------
        # 2. Threshold selection ONLY on non-test OOF trajectories.
        # ---------------------------------------------------------

        selected = {}

        for precision_target in (
            EMPIRICAL_PRECISION_TARGETS
        ):
            key = (
                f"empirical_precision_"
                f"{precision_target}"
            )

            chosen = choose_replay_threshold(
                cal_scores,
                cal_labels,
                cal_traj,
                cal_ks,
                precision_target,
            )

            selected[key] = chosen

        for risk_limit in RISK_LIMITS:
            key = (
                f"cp95_false_safe_risk_"
                f"{risk_limit}"
            )

            chosen = choose_cp_threshold(
                scores=cal_scores,
                labels=cal_labels,
                trajectory_ids=cal_traj,
                ks=cal_ks,
                risk_limit=risk_limit,
                confidence=CONFIDENCE,
            )

            selected[key] = chosen

        # ---------------------------------------------------------
        # 3. Final predictor: all 8 non-test tasks.
        # ---------------------------------------------------------

        print()
        print("  Training final outer model...")

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

        final_oof_scores[
            test_indices
        ] = test_scores

        del final_model
        del final_stats

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        fold_result = {
            "test_tasks": sorted(
                np.unique(
                    task_ids[
                        outer_test_idx
                    ]
                ).tolist()
            ),
            "calibration_tasks": sorted(
                np.unique(
                    task_ids[
                        outer_pool_idx
                    ]
                ).tolist()
            ),
            "methods": {},
        }

        for method, chosen in selected.items():
            if chosen is None:
                fold_result[
                    "methods"
                ][method] = None
                continue

            threshold = chosen["threshold"]

            cal_eval = evaluate_threshold(
                cal_scores,
                cal_labels,
                cal_traj,
                cal_ks,
                threshold,
            )

            test_eval = evaluate_threshold(
                test_scores,
                target_safe[test_indices],
                trajectory_ids[test_indices],
                ks[test_indices],
                threshold,
            )

            # Show calibration behavior separately for
            # each held-out task-pair fold.
            cal_group_replay = {}

            for f in non_test_folds:
                local = (
                    folds[
                        outer_pool_idx
                    ] == f
                )

                cal_group_replay[str(f)] = (
                    replay_metrics(
                        cal_scores[local],
                        cal_labels[local],
                        cal_traj[local],
                        cal_ks[local],
                        threshold,
                    )
                )

            fold_result[
                "methods"
            ][method] = {
                "threshold": threshold,
                "selection": chosen,
                "calibration": cal_eval,
                "calibration_by_fold":
                    cal_group_replay,
                "test": test_eval,
            }

            print()
            print(" ", method)
            print("    threshold:", threshold)
            print(
                "    calibration replay:",
                cal_eval["sequential_replay"],
            )

            if (
                "cp_upper_false_safe_risk"
                in chosen
            ):
                print(
                    "    CP upper risk:",
                    chosen[
                        "cp_upper_false_safe_risk"
                    ],
                )

            print(
                "    TEST replay:",
                test_eval["sequential_replay"],
            )

        results[
            "outer_folds"
        ][str(outer_test_fold)] = (
            fold_result
        )

        chosen_thresholds[
            str(outer_test_fold)
        ] = {
            method: (
                None
                if chosen is None
                else chosen["threshold"]
            )
            for method, chosen
            in selected.items()
        }

    if np.any(
        np.isnan(final_oof_scores)
    ):
        raise RuntimeError(
            "Missing final test OOF scores"
        )

    # -------------------------------------------------------------
    # 4. Aggregate completely held-out task results.
    # -------------------------------------------------------------

    method_names = set()

    for fold_result in results[
        "outer_folds"
    ].values():
        method_names.update(
            fold_result["methods"].keys()
        )

    aggregate = {}

    for method in sorted(method_names):
        trajectories = 0
        activated = 0
        correct = 0
        false_early = 0
        no_skip = 0

        row_tp = 0
        row_fp = 0
        row_fn = 0
        row_tn = 0

        per_fold = {}

        for test_fold in range(5):
            fold_result = results[
                "outer_folds"
            ][str(test_fold)]

            method_result = fold_result[
                "methods"
            ].get(method)

            mask = (
                folds == test_fold
            )

            if method_result is None:
                fold_traj = len(
                    np.unique(
                        trajectory_ids[
                            mask
                        ]
                    )
                )

                trajectories += fold_traj
                no_skip += fold_traj

                per_fold[str(test_fold)] = None
                continue

            row = method_result[
                "test"
            ]["row_metrics"]

            replay = method_result[
                "test"
            ]["sequential_replay"]

            row_tp += row["tp"]
            row_fp += row["fp_false_safe"]
            row_fn += row["fn"]
            row_tn += row["tn"]

            trajectories += replay[
                "trajectory_count"
            ]

            activated += replay[
                "activated"
            ]

            correct += replay[
                "correct_safe_stop"
            ]

            false_early += replay[
                "false_early_stop"
            ]

            no_skip += replay[
                "no_skip"
            ]

            per_fold[str(test_fold)] = {
                "threshold":
                    method_result["threshold"],
                "replay": replay,
            }

        aggregate[method] = {
            "row_metrics": {
                "tp": row_tp,
                "fp_false_safe": row_fp,
                "fn": row_fn,
                "tn": row_tn,
                "precision": (
                    row_tp / (row_tp + row_fp)
                    if row_tp + row_fp
                    else None
                ),
                "safe_recall": (
                    row_tp / (row_tp + row_fn)
                    if row_tp + row_fn
                    else None
                ),
                "row_coverage": (
                    (row_tp + row_fp)
                    / n_rows
                ),
            },
            "sequential_replay": {
                "trajectory_count":
                    trajectories,
                "activated":
                    activated,
                "correct_safe_stop":
                    correct,
                "false_early_stop":
                    false_early,
                "no_skip":
                    no_skip,
                "activation_precision": (
                    correct / activated
                    if activated
                    else None
                ),
                "safe_opportunity_capture": (
                    correct / trajectories
                    if trajectories
                    else None
                ),
            },
            "per_fold": per_fold,
        }

    results[
        "aggregate_outer_test"
    ] = aggregate

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
    print("FINAL MULTI-TASK RISK-CALIBRATION SUMMARY")
    print("=" * 76)

    for method, r in aggregate.items():
        print()
        print(method)
        print(
            "  row:",
            r["row_metrics"],
        )
        print(
            "  replay:",
            r[
                "sequential_replay"
            ],
        )

    print()
    print(
        "Results written to:",
        args.output,
    )


if __name__ == "__main__":
    main()
