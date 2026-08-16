import argparse
import json
import math
import random
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn


MODEL_NAMES = [
    "delta_only",
    "delta_plus_anchor",
]

PRECISION_TARGETS = [0.95, 0.99, 0.995]


# ---------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def rankdata(x):
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


# ---------------------------------------------------------------------
# Cache construction
# ---------------------------------------------------------------------

def build_cache(raw_root, index_path, cache_path):
    print("Building action-delta cache...")

    rows = []

    with index_path.open() as f:
        for line in f:
            rows.append(json.loads(line))

    print("transition rows:", len(rows))

    by_shard = defaultdict(list)

    for row_id, row in enumerate(rows):
        by_shard[row["shard"]].append((row_id, row))

    n = len(rows)

    delta_states = torch.empty(
        (n, 8, 896),
        dtype=torch.bfloat16,
    )

    anchor_actions = torch.empty(
        (n, 8, 7),
        dtype=torch.bfloat16,
    )

    delta_actions = torch.empty(
        (n, 8, 7),
        dtype=torch.bfloat16,
    )

    task_ids = torch.empty(n, dtype=torch.int16)
    folds = torch.empty(n, dtype=torch.int8)
    ks = torch.empty(n, dtype=torch.int8)

    target_mse = torch.empty(n, dtype=torch.float32)
    target_safe = torch.empty(n, dtype=torch.bool)

    trajectory_ids = torch.empty(n, dtype=torch.int32)

    trajectory_map = {}
    next_trajectory_id = 0

    shard_names = sorted(by_shard)

    for shard_idx, shard_name in enumerate(shard_names):
        shard = torch.load(
            raw_root / shard_name,
            map_location="cpu",
            weights_only=False,
        )

        for row_id, row in by_shard[shard_name]:
            p = shard["predictions"][
                row["prediction_index_in_shard"]
            ]

            k = int(row["k"])

            # Tensor indexing:
            # index 0 -> iteration 1
            s_k = p["tensors"]["states"][k - 1]
            s_next = p["tensors"]["states"][k]

            a_k = p["tensors"]["actions"][k - 1]
            a_next = p["tensors"]["actions"][k]

            delta_s = (
                s_next.float() - s_k.float()
            ).to(torch.bfloat16)

            delta_a = (
                a_next.float() - a_k.float()
            ).to(torch.bfloat16)

            delta_states[row_id].copy_(delta_s.squeeze(0))
            anchor_actions[row_id].copy_(a_k.squeeze(0))
            delta_actions[row_id].copy_(delta_a.squeeze(0))

            task_ids[row_id] = int(row["task_id"])
            folds[row_id] = int(row["fold"])
            ks[row_id] = k

            target_mse[row_id] = float(
                row["target_action_mse"]
            )

            target_safe[row_id] = bool(
                row["target_safe"]
            )

            trajectory_key = (
                int(row["task_id"]),
                int(row["episode_id"]),
                int(row["prediction_id"]),
                int(row["timestep"]),
            )

            if trajectory_key not in trajectory_map:
                trajectory_map[trajectory_key] = (
                    next_trajectory_id
                )
                next_trajectory_id += 1

            trajectory_ids[row_id] = trajectory_map[
                trajectory_key
            ]

        print(
            f"[{shard_idx + 1:03d}/{len(shard_names):03d}] "
            f"{shard_name}"
        )

    cache = {
        "delta_states": delta_states,
        "anchor_actions": anchor_actions,
        "delta_actions": delta_actions,
        "task_ids": task_ids,
        "folds": folds,
        "ks": ks,
        "target_mse": target_mse,
        "target_safe": target_safe,
        "trajectory_ids": trajectory_ids,
        "trajectory_count": next_trajectory_id,
        "row_count": n,
    }

    cache_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    torch.save(cache, cache_path)

    print("cache written:", cache_path)
    print("rows:", n)
    print("trajectories:", next_trajectory_id)

    return cache


# ---------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------

class SharedTokenLinear(nn.Module):
    """
    One lightweight projection shared across the 8 action tokens.

    delta_only:
        896 -> 7

    delta_plus_anchor:
        896 + 7 -> 7
    """

    def __init__(self, model_name):
        super().__init__()

        if model_name == "delta_only":
            in_dim = 896

        elif model_name == "delta_plus_anchor":
            in_dim = 896 + 7

        else:
            raise ValueError(model_name)

        self.model_name = model_name

        self.linear = nn.Linear(
            in_dim,
            7,
        )

    def make_input(
        self,
        delta_s,
        anchor_a,
    ):
        if self.model_name == "delta_only":
            return delta_s

        return torch.cat(
            [delta_s, anchor_a],
            dim=-1,
        )

    def forward(
        self,
        x,
    ):
        return self.linear(x)


# ---------------------------------------------------------------------
# Normalization statistics
# ---------------------------------------------------------------------

@torch.no_grad()
def compute_stats(
    cache,
    row_indices,
    model_name,
    chunk_size=512,
):
    x_sum = None
    x_sq_sum = None

    y_sum = None
    y_sq_sum = None

    count = 0

    for start in range(0, len(row_indices), chunk_size):
        idx = row_indices[
            start:start + chunk_size
        ]

        ds = cache["delta_states"][idx].float()
        aa = cache["anchor_actions"][idx].float()
        da = cache["delta_actions"][idx].float()

        if model_name == "delta_only":
            x = ds
        else:
            x = torch.cat([ds, aa], dim=-1)

        # [rows, 8, features] -> token samples
        x = x.reshape(-1, x.shape[-1])
        y = da.reshape(-1, 7)

        if x_sum is None:
            x_sum = x.sum(dim=0)
            x_sq_sum = (x * x).sum(dim=0)

            y_sum = y.sum(dim=0)
            y_sq_sum = (y * y).sum(dim=0)

        else:
            x_sum += x.sum(dim=0)
            x_sq_sum += (x * x).sum(dim=0)

            y_sum += y.sum(dim=0)
            y_sq_sum += (y * y).sum(dim=0)

        count += x.shape[0]

    x_mean = x_sum / count
    y_mean = y_sum / count

    x_var = (
        x_sq_sum / count
        - x_mean * x_mean
    ).clamp_min(1e-10)

    y_var = (
        y_sq_sum / count
        - y_mean * y_mean
    ).clamp_min(1e-10)

    x_std = torch.sqrt(x_var)
    y_std = torch.sqrt(y_var)

    return {
        "x_mean": x_mean,
        "x_std": x_std,
        "y_mean": y_mean,
        "y_std": y_std,
    }


# ---------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------

def train_model(
    cache,
    train_indices,
    model_name,
    device,
    epochs,
    batch_size,
    lr,
    seed,
):
    set_seed(seed)

    model = SharedTokenLinear(
        model_name
    ).to(device)

    stats = compute_stats(
        cache,
        train_indices,
        model_name,
    )

    stats = {
        k: v.to(device)
        for k, v in stats.items()
    }

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=lr,
        weight_decay=1e-4,
    )

    loss_fn = nn.MSELoss()

    train_indices_np = np.asarray(
        train_indices,
        dtype=np.int64,
    )

    for epoch in range(epochs):
        perm = np.random.permutation(
            train_indices_np
        )

        epoch_loss = 0.0
        token_count = 0

        model.train()

        for start in range(
            0,
            len(perm),
            batch_size,
        ):
            batch_idx = perm[
                start:start + batch_size
            ]

            ds = cache["delta_states"][
                batch_idx
            ].float().to(device)

            aa = cache["anchor_actions"][
                batch_idx
            ].float().to(device)

            target = cache["delta_actions"][
                batch_idx
            ].float().to(device)

            if model_name == "delta_only":
                x = ds
            else:
                x = torch.cat(
                    [ds, aa],
                    dim=-1,
                )

            x = (
                x - stats["x_mean"]
            ) / stats["x_std"]

            target_norm = (
                target - stats["y_mean"]
            ) / stats["y_std"]

            pred_norm = model(x)

            loss = loss_fn(
                pred_norm,
                target_norm,
            )

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            n_tokens = (
                target.shape[0]
                * target.shape[1]
            )

            epoch_loss += (
                loss.item() * n_tokens
            )

            token_count += n_tokens

        if (
            epoch == 0
            or (epoch + 1) % 10 == 0
            or epoch + 1 == epochs
        ):
            print(
                f"    epoch "
                f"{epoch + 1:03d}/{epochs}: "
                f"loss="
                f"{epoch_loss / token_count:.6f}"
            )

    return model, stats


# ---------------------------------------------------------------------
# Prediction
# ---------------------------------------------------------------------

@torch.no_grad()
def predict_rows(
    cache,
    row_indices,
    model,
    stats,
    model_name,
    device,
    batch_size,
):
    model.eval()

    all_indices = []
    all_pred_delta = []

    row_indices = np.asarray(
        row_indices,
        dtype=np.int64,
    )

    for start in range(
        0,
        len(row_indices),
        batch_size,
    ):
        idx = row_indices[
            start:start + batch_size
        ]

        ds = cache["delta_states"][
            idx
        ].float().to(device)

        aa = cache["anchor_actions"][
            idx
        ].float().to(device)

        if model_name == "delta_only":
            x = ds
        else:
            x = torch.cat(
                [ds, aa],
                dim=-1,
            )

        x = (
            x - stats["x_mean"]
        ) / stats["x_std"]

        pred_norm = model(x)

        pred = (
            pred_norm * stats["y_std"]
            + stats["y_mean"]
        )

        all_indices.append(idx)

        all_pred_delta.append(
            pred.cpu()
        )

    return (
        np.concatenate(all_indices),
        torch.cat(all_pred_delta, dim=0),
    )


# ---------------------------------------------------------------------
# Replay metrics
# ---------------------------------------------------------------------

def replay_metrics(
    scores,
    labels,
    trajectory_ids,
    ks,
    threshold,
):
    scores = np.asarray(scores)
    labels = np.asarray(labels, dtype=np.bool_)
    trajectory_ids = np.asarray(trajectory_ids)
    ks = np.asarray(ks)

    activated = 0
    correct = 0
    false_early = 0
    no_skip = 0

    unique_trajectories = np.unique(
        trajectory_ids
    )

    for traj in unique_trajectories:
        mask = trajectory_ids == traj

        traj_scores = scores[mask]
        traj_labels = labels[mask]
        traj_ks = ks[mask]

        eligible = traj_scores <= threshold

        if not np.any(eligible):
            no_skip += 1
            continue

        eligible_indices = np.where(
            eligible
        )[0]

        # Runtime behavior:
        # first predicted-safe iteration wins.
        first_local = eligible_indices[
            np.argmin(
                traj_ks[eligible_indices]
            )
        ]

        activated += 1

        if traj_labels[first_local]:
            correct += 1
        else:
            false_early += 1

    precision = (
        correct / activated
        if activated
        else None
    )

    safe_capture = (
        correct / len(unique_trajectories)
        if len(unique_trajectories)
        else None
    )

    return {
        "trajectory_count": int(
            len(unique_trajectories)
        ),
        "activated": int(activated),
        "correct_safe_stop": int(correct),
        "false_early_stop": int(false_early),
        "no_skip": int(no_skip),
        "activation_precision": precision,
        "safe_opportunity_capture": safe_capture,
    }


def choose_replay_threshold(
    scores,
    labels,
    trajectory_ids,
    ks,
    target_precision,
):
    scores = np.asarray(scores)

    unique_scores = np.unique(scores)

    # Enough resolution for this dataset while avoiding
    # pathological O(N^2) work if predictions become dense.
    if len(unique_scores) > 1000:
        q = np.linspace(
            0.0,
            1.0,
            1000,
        )

        candidates = np.unique(
            np.quantile(
                unique_scores,
                q,
            )
        )
    else:
        candidates = unique_scores

    best = None

    for threshold in candidates:
        metrics = replay_metrics(
            scores,
            labels,
            trajectory_ids,
            ks,
            threshold,
        )

        precision = metrics[
            "activation_precision"
        ]

        if precision is None:
            continue

        if precision < target_precision:
            continue

        # Primary objective:
        # maximize number of correct Coda skips.
        #
        # Tie-break:
        # fewer false-safe activations.
        key = (
            metrics["correct_safe_stop"],
            -metrics["false_early_stop"],
            -threshold,
        )

        if best is None or key > best["key"]:
            best = {
                "threshold": float(
                    threshold
                ),
                "metrics": metrics,
                "key": key,
            }

    if best is None:
        return None

    del best["key"]

    return best


# ---------------------------------------------------------------------
# Row-level metrics
# ---------------------------------------------------------------------

def row_metrics(
    scores,
    labels,
    threshold,
):
    pred_safe = (
        np.asarray(scores)
        <= threshold
    )

    labels = np.asarray(
        labels,
        dtype=np.bool_,
    )

    tp = int(
        np.sum(pred_safe & labels)
    )

    fp = int(
        np.sum(pred_safe & ~labels)
    )

    fn = int(
        np.sum(~pred_safe & labels)
    )

    tn = int(
        np.sum(~pred_safe & ~labels)
    )

    pred_count = tp + fp

    precision = (
        tp / pred_count
        if pred_count
        else None
    )

    recall = (
        tp / (tp + fn)
        if tp + fn
        else None
    )

    return {
        "tp": tp,
        "fp_false_safe": fp,
        "fn": fn,
        "tn": tn,
        "precision": precision,
        "safe_recall": recall,
        "row_coverage": (
            pred_count / len(labels)
        ),
    }


# ---------------------------------------------------------------------
# Main experiment
# ---------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--raw-root",
        type=Path,
        default=Path(
            "benchmark_results/"
            "preconvergence_trigger/"
            "raw_shadow_calibration_seed7"
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
            "linear_action_predictor_results.json"
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
        "--rebuild-cache",
        action="store_true",
    )

    args = parser.parse_args()

    set_seed(args.seed)

    if (
        args.rebuild_cache
        or not args.cache.exists()
    ):
        cache = build_cache(
            args.raw_root,
            args.index,
            args.cache,
        )

    else:
        print(
            "Loading cache:",
            args.cache,
        )

        cache = torch.load(
            args.cache,
            map_location="cpu",
            weights_only=False,
        )

    print()
    print("=== CACHE ===")
    print("rows:", cache["row_count"])
    print(
        "trajectories:",
        cache["trajectory_count"],
    )
    print(
        "delta_states:",
        tuple(
            cache["delta_states"].shape
        ),
    )
    print(
        "anchor_actions:",
        tuple(
            cache["anchor_actions"].shape
        ),
    )
    print(
        "delta_actions:",
        tuple(
            cache["delta_actions"].shape
        ),
    )
    print("device:", args.device)

    folds = cache["folds"].numpy()
    target_safe = (
        cache["target_safe"].numpy()
    )
    target_mse = (
        cache["target_mse"].numpy()
    )
    trajectory_ids = (
        cache["trajectory_ids"].numpy()
    )
    ks = cache["ks"].numpy()

    results = {
        "protocol": {
            "outer_folds": 5,
            "test_fold": "f",
            "calibration_fold": "(f+1) mod 5",
            "train_folds": "remaining 3 folds",
            "precision_targets":
                PRECISION_TARGETS,
            "sequential_threshold_calibration":
                True,
        },
        "models": {},
    }

    for model_name in MODEL_NAMES:
        print()
        print("=" * 72)
        print("MODEL:", model_name)
        print("=" * 72)

        model_result = {
            "folds": {},
        }

        oof_scores = np.full(
            len(folds),
            np.nan,
            dtype=np.float64,
        )

        oof_pred_delta = torch.zeros(
            (
                len(folds),
                8,
                7,
            ),
            dtype=torch.float32,
        )

        fold_thresholds = {
            str(p): {}
            for p in PRECISION_TARGETS
        }

        for test_fold in range(5):
            calibration_fold = (
                test_fold + 1
            ) % 5

            train_mask = (
                (folds != test_fold)
                & (
                    folds
                    != calibration_fold
                )
            )

            calibration_mask = (
                folds == calibration_fold
            )

            test_mask = (
                folds == test_fold
            )

            train_idx = np.where(
                train_mask
            )[0]

            calibration_idx = np.where(
                calibration_mask
            )[0]

            test_idx = np.where(
                test_mask
            )[0]

            print()
            print(
                f"Outer fold {test_fold}:"
            )
            print(
                f"  train folds="
                f"{sorted(set(folds[train_idx]))}"
            )
            print(
                f"  calibration fold="
                f"{calibration_fold}"
            )
            print(
                f"  test fold={test_fold}"
            )
            print(
                f"  rows: "
                f"train={len(train_idx)}, "
                f"cal={len(calibration_idx)}, "
                f"test={len(test_idx)}"
            )

            model, stats = train_model(
                cache=cache,
                train_indices=train_idx,
                model_name=model_name,
                device=args.device,
                epochs=args.epochs,
                batch_size=args.batch_size,
                lr=args.lr,
                seed=args.seed + test_fold,
            )

            cal_indices, cal_pred = (
                predict_rows(
                    cache,
                    calibration_idx,
                    model,
                    stats,
                    model_name,
                    args.device,
                    args.batch_size,
                )
            )

            test_indices, test_pred = (
                predict_rows(
                    cache,
                    test_idx,
                    model,
                    stats,
                    model_name,
                    args.device,
                    args.batch_size,
                )
            )

            # Predicted action-change magnitude.
            cal_scores = (
                cal_pred
                .float()
                .pow(2)
                .mean(dim=(1, 2))
                .numpy()
            )

            test_scores = (
                test_pred
                .float()
                .pow(2)
                .mean(dim=(1, 2))
                .numpy()
            )

            oof_scores[
                test_indices
            ] = test_scores

            oof_pred_delta[
                test_indices
            ] = test_pred.float()

            fold_result = {
                "train_row_count":
                    int(len(train_idx)),
                "calibration_row_count":
                    int(len(calibration_idx)),
                "test_row_count":
                    int(len(test_idx)),
                "calibration_fold":
                    calibration_fold,
                "thresholds": {},
            }

            for target_precision in (
                PRECISION_TARGETS
            ):
                chosen = (
                    choose_replay_threshold(
                        cal_scores,
                        target_safe[
                            cal_indices
                        ],
                        trajectory_ids[
                            cal_indices
                        ],
                        ks[
                            cal_indices
                        ],
                        target_precision,
                    )
                )

                key = str(
                    target_precision
                )

                if chosen is None:
                    fold_result[
                        "thresholds"
                    ][key] = None

                    fold_thresholds[
                        key
                    ][str(test_fold)] = None

                    continue

                threshold = chosen[
                    "threshold"
                ]

                fold_thresholds[
                    key
                ][str(test_fold)] = (
                    threshold
                )

                test_row = row_metrics(
                    test_scores,
                    target_safe[
                        test_indices
                    ],
                    threshold,
                )

                test_replay = (
                    replay_metrics(
                        test_scores,
                        target_safe[
                            test_indices
                        ],
                        trajectory_ids[
                            test_indices
                        ],
                        ks[
                            test_indices
                        ],
                        threshold,
                    )
                )

                fold_result[
                    "thresholds"
                ][key] = {
                    "threshold":
                        threshold,
                    "calibration_replay":
                        chosen["metrics"],
                    "test_row_metrics":
                        test_row,
                    "test_replay_metrics":
                        test_replay,
                }

            model_result[
                "folds"
            ][str(test_fold)] = (
                fold_result
            )

        # -------------------------------------------------------------
        # Regression diagnostics over full OOF predictions
        # -------------------------------------------------------------

        assert not np.any(
            np.isnan(oof_scores)
        )

        true_delta = (
            cache["delta_actions"]
            .float()
        )

        error = (
            oof_pred_delta
            - true_delta
        )

        model_result[
            "oof_regression"
        ] = {
            "delta_action_mse":
                float(
                    error.pow(2)
                    .mean()
                    .item()
                ),
            "first_action_delta_mse":
                float(
                    error[:, 0]
                    .pow(2)
                    .mean()
                    .item()
                ),
            "mean_row_max_abs_error":
                float(
                    error.abs()
                    .amax(dim=(1, 2))
                    .mean()
                    .item()
                ),
            "spearman_predicted_vs_true_action_mse":
                spearman(
                    oof_scores,
                    target_mse,
                ),
        }

        # -------------------------------------------------------------
        # Aggregate OOF safety metrics.
        #
        # Each row uses the threshold learned from the calibration fold
        # corresponding to its held-out test fold.
        # -------------------------------------------------------------

        model_result[
            "oof_safety"
        ] = {}

        for target_precision in (
            PRECISION_TARGETS
        ):
            key = str(
                target_precision
            )

            pred_safe = np.zeros(
                len(folds),
                dtype=np.bool_,
            )

            valid_threshold = np.zeros(
                len(folds),
                dtype=np.bool_,
            )

            for test_fold in range(5):
                threshold = (
                    fold_thresholds[
                        key
                    ].get(
                        str(test_fold)
                    )
                )

                mask = (
                    folds == test_fold
                )

                if threshold is None:
                    continue

                valid_threshold[
                    mask
                ] = True

                pred_safe[
                    mask
                ] = (
                    oof_scores[
                        mask
                    ]
                    <= threshold
                )

            # Row aggregate
            tp = int(
                np.sum(
                    pred_safe
                    & target_safe
                )
            )

            fp = int(
                np.sum(
                    pred_safe
                    & ~target_safe
                )
            )

            fn = int(
                np.sum(
                    ~pred_safe
                    & target_safe
                )
            )

            tn = int(
                np.sum(
                    ~pred_safe
                    & ~target_safe
                )
            )

            predicted_count = (
                tp + fp
            )

            row_summary = {
                "tp": tp,
                "fp_false_safe": fp,
                "fn": fn,
                "tn": tn,
                "precision": (
                    tp / predicted_count
                    if predicted_count
                    else None
                ),
                "safe_recall": (
                    tp / (tp + fn)
                    if tp + fn
                    else None
                ),
                "row_coverage": (
                    predicted_count
                    / len(folds)
                ),
            }

            # Sequential OOF replay must use the fold-specific
            # threshold for each trajectory.
            activated = 0
            correct = 0
            false_early = 0
            no_skip = 0
            traj_count = 0

            for test_fold in range(5):
                threshold = (
                    fold_thresholds[
                        key
                    ].get(
                        str(test_fold)
                    )
                )

                mask = (
                    folds == test_fold
                )

                fold_traj_count = len(
                    np.unique(
                        trajectory_ids[
                            mask
                        ]
                    )
                )

                traj_count += (
                    fold_traj_count
                )

                if threshold is None:
                    no_skip += (
                        fold_traj_count
                    )
                    continue

                m = replay_metrics(
                    oof_scores[mask],
                    target_safe[mask],
                    trajectory_ids[mask],
                    ks[mask],
                    threshold,
                )

                activated += m[
                    "activated"
                ]

                correct += m[
                    "correct_safe_stop"
                ]

                false_early += m[
                    "false_early_stop"
                ]

                no_skip += m[
                    "no_skip"
                ]

            replay_summary = {
                "trajectory_count":
                    traj_count,
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
                    correct / traj_count
                    if traj_count
                    else None
                ),
            }

            model_result[
                "oof_safety"
            ][key] = {
                "fold_thresholds":
                    fold_thresholds[key],
                "row_metrics":
                    row_summary,
                "sequential_replay":
                    replay_summary,
            }

        results[
            "models"
        ][model_name] = (
            model_result
        )

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
    print("=" * 72)
    print("FINAL SUMMARY")
    print("=" * 72)

    for model_name in MODEL_NAMES:
        r = results[
            "models"
        ][model_name]

        print()
        print(
            f"MODEL: {model_name}"
        )

        print(
            "  regression delta-action MSE:",
            r["oof_regression"][
                "delta_action_mse"
            ],
        )

        print(
            "  Spearman(pred action MSE, true action MSE):",
            r["oof_regression"][
                "spearman_predicted_vs_true_action_mse"
            ],
        )

        for p in PRECISION_TARGETS:
            s = r[
                "oof_safety"
            ][str(p)]

            print(
                f"  target precision={p}"
            )

            print(
                "    row:",
                s["row_metrics"],
            )

            print(
                "    replay:",
                s[
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
