import argparse
import json
from collections import Counter
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
        "--index",
        type=Path,
        default=Path(
            "benchmark_results/"
            "coda_anchor_feasibility/"
            "transition_index.jsonl"
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
            "false_early_audit_cp001.json"
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

    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=7)

    args = parser.parse_args()

    set_seed(args.seed)

    cache = torch.load(
        args.cache,
        map_location="cpu",
        weights_only=False,
    )

    with args.calibration_results.open() as f:
        calibration = json.load(f)

    rows = []

    with args.index.open() as f:
        for line in f:
            rows.append(json.loads(line))

    if len(rows) != cache["row_count"]:
        raise RuntimeError(
            f"index/cache row mismatch: "
            f"{len(rows)} vs {cache['row_count']}"
        )

    folds = cache["folds"].numpy()
    task_ids = cache["task_ids"].numpy()
    target_safe = cache["target_safe"].numpy()
    trajectory_ids = cache["trajectory_ids"].numpy()
    ks = cache["ks"].numpy()

    true_delta = cache["delta_actions"].float()

    false_cases = []
    correct_cases = []

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
            print("No feasible threshold.")
            continue

        threshold = float(
            method_result["threshold"]
        )

        train_idx = np.where(
            folds != test_fold
        )[0]

        test_idx = np.where(
            folds == test_fold
        )[0]

        print("threshold:", threshold)
        print("train rows:", len(train_idx))
        print("test rows:", len(test_idx))

        # Same seed as the original final outer model.
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

        # Group local test rows by trajectory.
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

        fold_false = 0
        fold_correct = 0

        for traj, members in by_trajectory.items():
            # Runtime first-hit ordering.
            members = sorted(
                members,
                key=lambda x: int(
                    ks[x[1]]
                ),
            )

            activated = None

            for local_i, global_idx in members:
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
                continue

            local_i, global_idx, pred_mse = activated

            row = rows[global_idx]

            is_safe = bool(
                target_safe[global_idx]
            )

            delta = true_delta[
                global_idx
            ]

            # If the gate stops here and returns cached a_k,
            # this is exactly the action difference from a_(k+1).
            fp32_delta_mse = float(
                delta.pow(2).mean().item()
            )

            max_abs_delta = float(
                delta.abs().max().item()
            )

            first_step = delta[0]

            first_step_mse = float(
                first_step.pow(2).mean().item()
            )

            first_step_max_abs = float(
                first_step.abs().max().item()
            )

            cartesian = delta[..., :6]
            gripper = delta[..., 6]

            cartesian_mse = float(
                cartesian.pow(2).mean().item()
            )

            gripper_mse = float(
                gripper.pow(2).mean().item()
            )

            prediction_error = (
                pred_delta[local_i].float()
                - delta
            )

            predictor_delta_mse = float(
                prediction_error
                .pow(2)
                .mean()
                .item()
            )

            actual_native_mse = float(
                row["target_action_mse"]
            )

            terminal_k = int(
                row["production_terminal_k"]
            )

            k = int(row["k"])
            next_k = int(row["next_k"])

            record = {
                "outer_test_fold": test_fold,
                "task_id": int(row["task_id"]),
                "episode_id": int(row["episode_id"]),
                "prediction_id": int(row["prediction_id"]),
                "timestep": int(row["timestep"]),
                "trajectory_id": traj,

                "k": k,
                "next_k": next_k,
                "production_terminal_k": terminal_k,

                "iterations_early":
                    terminal_k - next_k,

                "predicted_action_mse":
                    pred_mse,

                "gate_threshold":
                    threshold,

                "actual_action_mse_native":
                    actual_native_mse,

                "actual_action_mse_fp32":
                    fp32_delta_mse,

                "threshold_ratio":
                    actual_native_mse / 0.001,

                "underestimate_amount":
                    actual_native_mse
                    - pred_mse,

                "max_abs_action_delta":
                    max_abs_delta,

                "first_step_mse":
                    first_step_mse,

                "first_step_max_abs_delta":
                    first_step_max_abs,

                "cartesian_mse":
                    cartesian_mse,

                "gripper_mse":
                    gripper_mse,

                "predictor_delta_regression_mse":
                    predictor_delta_mse,

                "target_safe":
                    is_safe,
            }

            if is_safe:
                correct_cases.append(record)
                fold_correct += 1
            else:
                false_cases.append(record)
                fold_false += 1

        print(
            "correct safe activations:",
            fold_correct,
        )
        print(
            "false early activations:",
            fold_false,
        )

        del model
        del stats

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # ----------------------------------------------------------------
    # Summaries
    # ----------------------------------------------------------------

    task_false_counts = Counter(
        x["task_id"]
        for x in false_cases
    )

    early_counts = Counter(
        x["iterations_early"]
        for x in false_cases
    )

    result = {
        "method": METHOD,
        "false_case_count":
            len(false_cases),
        "correct_case_count":
            len(correct_cases),

        "false_counts_by_task": {
            str(k): v
            for k, v
            in sorted(
                task_false_counts.items()
            )
        },

        "false_counts_by_iterations_early": {
            str(k): v
            for k, v
            in sorted(
                early_counts.items()
            )
        },

        "false_cases": false_cases,
    }

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
    print("=" * 72)
    print("FALSE-EARLY AUDIT SUMMARY")
    print("=" * 72)

    print(
        "false cases:",
        len(false_cases),
    )

    print(
        "false counts by task:",
        result[
            "false_counts_by_task"
        ],
    )

    print(
        "false counts by iterations early:",
        result[
            "false_counts_by_iterations_early"
        ],
    )

    if false_cases:
        print()
        print("Individual failures:")

        for i, x in enumerate(
            false_cases,
            start=1,
        ):
            print()
            print(f"[{i}]")
            print(
                " task / episode / prediction / timestep:",
                x["task_id"],
                x["episode_id"],
                x["prediction_id"],
                x["timestep"],
            )
            print(
                " k -> next_k -> terminal_K:",
                x["k"],
                "->",
                x["next_k"],
                "->",
                x[
                    "production_terminal_k"
                ],
            )
            print(
                " iterations early:",
                x[
                    "iterations_early"
                ],
            )
            print(
                " predicted MSE:",
                x[
                    "predicted_action_mse"
                ],
            )
            print(
                " actual MSE:",
                x[
                    "actual_action_mse_native"
                ],
            )
            print(
                " threshold ratio:",
                x[
                    "threshold_ratio"
                ],
            )
            print(
                " max abs delta:",
                x[
                    "max_abs_action_delta"
                ],
            )
            print(
                " first-step MSE:",
                x[
                    "first_step_mse"
                ],
            )
            print(
                " cartesian MSE:",
                x[
                    "cartesian_mse"
                ],
            )
            print(
                " gripper MSE:",
                x[
                    "gripper_mse"
                ],
            )

    print()
    print(
        "Audit written to:",
        args.output,
    )


if __name__ == "__main__":
    main()
