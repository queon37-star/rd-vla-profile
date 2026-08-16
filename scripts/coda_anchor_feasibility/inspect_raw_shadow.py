import argparse
from pathlib import Path

import torch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--shard",
        type=Path,
        default=Path(
            "benchmark_results/preconvergence_trigger/"
            "raw_shadow_calibration_seed7/raw_shadow_00000.pt"
        ),
    )
    parser.add_argument("--prediction-index", type=int, default=0)
    args = parser.parse_args()

    shard = torch.load(args.shard, map_location="cpu", weights_only=False)

    print("schema_version:", shard["schema_version"])
    print("collection_mode:", shard["collection_mode"])
    print("predictions in shard:", len(shard["predictions"]))

    p = shard["predictions"][args.prediction_index]

    states = p["tensors"]["states"]
    actions = p["tensors"]["actions"]

    print("\n=== Identity ===")
    print(p["identity"])
    print("task:", p["task_name"])
    print("origin:", p["actual_origin"])

    print("\n=== Runtime metadata ===")
    print("production_terminal_k:", p["production_terminal_k"])
    print("maximum_shadow_depth:", p["maximum_shadow_depth"])
    print("valid_trajectory_length:", p["valid_trajectory_length"])
    print("threshold:", p["action_mse_threshold"])

    print("\n=== Tensor contracts ===")
    print("states :", states.shape, states.dtype)
    print("actions:", actions.shape, actions.dtype)

    print("\n=== Stored production MSE ===")
    print("production_iteration_mse:")
    for i, x in enumerate(p["production_iteration_mse"], start=2):
        print(f"  k={i:2d}: {x}")

    print("\n=== action_mse raw list ===")
    for i, x in enumerate(p["action_mse"][:10]):
        print(
            f"  index={i:2d}: mse={x}, "
            f"phase={p['action_mse_phase'][i]}, "
            f"source={p['action_mse_source'][i]}"
        )

    print("\n=== Recomputed adjacent action MSE ===")
    # actions[0] = a_1, actions[1] = a_2, ...
    # Compute in FP32 only for diagnostic comparison.
    recomputed = []

    for idx in range(1, actions.shape[0]):
        prev_a = actions[idx - 1].float()
        curr_a = actions[idx].float()

        mse = torch.mean((curr_a - prev_a) ** 2).item()

        k = idx + 1
        recomputed.append(mse)

        if k <= 10:
            print(f"  k={k:2d}: {mse:.10f}")

    threshold = p["action_mse_threshold"]

    first_hit = next(
        (k for k, mse in enumerate(recomputed, start=2) if mse < threshold),
        None,
    )

    print("\n=== Diagnostic ===")
    print("FP32 recomputed first hit:", first_hit)
    print("production terminal K     :", p["production_terminal_k"])

    print("\nNOTE:")
    print(
        "A mismatch near threshold is not automatically an error: "
        "production stopping used native values, while this diagnostic "
        "casts actions to FP32."
    )


if __name__ == "__main__":
    main()
