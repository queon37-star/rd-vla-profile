# One-step-ahead Coda trigger feasibility

## Status

This branch contains an offline-only implementation of the proposed
pre-convergence trigger. It does not change `action_heads.py`, runtime stopping,
Coda scheduling, model outputs, warm-cache behavior, or defaults.

The local artifact audit found that the required raw recurrent trajectory is
not currently recorded. Consequently no real model was fit, no task-OOF result
manifest was produced, and no GPU or LIBERO run was started. This is a blocked
data-feasibility state, not a negative model result and not a promotion result.

## Phase-0 artifact audit

The audited calibration population has 2,398 predictions and 74,338 full-depth
transitions: 2,298 `ACTUAL_WARM` predictions and 100 `COLD` predictions. Every
identity `(task_id, episode_id, prediction_id)` is unique, and every trace has
exactly the iterations 2 through 32.

| Required value | Local availability | Authoritative source or limitation |
|---|---|---|
| raw latent state `S_k` | missing | Neither learned JSONL, latent-dynamics JSONL, shadow JSONL, nor workload shards retain the trajectory. |
| tensor `S_k - S_(k-1)` | missing | Only scalar summaries such as raw MSE and update RMS are retained. |
| action `a_k` or vector action delta | missing | Existing traces retain action MSE/L2 scalars, not the action vector needed by the auxiliary target. |
| authoritative action MSE | available | Existing learned-probe dataset uses native BF16 `iteration_mse` through baseline K and FP32 shadow-tail MSE only afterward. |
| `K_action` | available | Strict first hit of authoritative `action_mse < 0.001`; all 2,398 first hits equal recorded baseline K. |
| task/episode/prediction identity | available | Unique for all 2,398 predictions. |
| actual origin | available | `ACTUAL_WARM=2,298`, `COLD=100`. |

The completed latent-dynamics JSONL is not an authoritative replacement for
production labels. Its `adjacent_action_mse` is a diagnostic FP32 value in the
production prefix as well as the shadow tail. Twelve values close to 0.001
produce a different first hit from native BF16 control flow. The builder
therefore joins raw shards only to the frozen learned-probe dataset and enforces
the production/shadow phase boundary.

There are 200 existing action-head workload shards, split into 100 cold and 100
actual-warm workloads. Their tensor fields are:

- `actions_hidden_states`
- `proprio_input`
- `proprio_features`
- `incoming_warm_start_state`
- `selected_initial_state`

They permit action-head input replay, but do not contain `S_1..S_32` or
`a_1..a_32`, cover only 100 of the 2,298 primary predictions, and therefore are
not silently treated as the requested OOF dataset.

## Dataset and leakage contract

For each prediction:

```text
K_action = min { k : authoritative_action_mse[k] < 0.001 }
y_k = 1  iff k == K_action - 1
y_k = 0  iff k <  K_action - 1
```

Rows at or after `K_action` are excluded from classification training. Since
the model needs both `delta_k` and `delta_(k-1)`, scoring begins at k=3. A target
at k=2 is reported as unavailable coverage in the dataset and OOF summaries;
no history value is invented.

The scorer for iteration k reads exactly `S_k`, `S_(k-1)`, and `S_(k-2)`.
Actions and action MSE are supervision/replay data only. Each applicable
prediction has total training weight one: its positive row has weight 0.5 and
all pre-target negatives share the other 0.5. The existing LIBERO Spatial
task-level five-fold split is reused. Normalization, model parameters, and the
threshold are fit on outer-training `ACTUAL_WARM` tasks only. Cold predictions
are a separate held-out descriptive report.

## Model

For each k>=3 the implementation computes:

```text
delta_k    = S_k - S_(k-1)
delta_prev = S_(k-1) - S_(k-2)
x_k = concat(mean_tokens(delta_k), mean_tokens(delta_prev))
h_k = Linear(x_k, rank)
trigger_logit = Linear(h_k, 1)
```

Ranks 4, 8, and 16 are evaluated. Each rank has a no-auxiliary variant and a
variant with `Linear(h_k, action_dim)` trained by SmoothL1 against the current
action delta `a_k-a_(k-1)`. The auxiliary vector is never an inference input.
No cosine, quantile, entropy, top-k, sorting, or other handcrafted metric is
computed. The tensor scorer returns a tensor and contains no `.item()` call.

For latent width D and bottleneck rank R, the no-auxiliary parameter count is:

```text
(2D * R + R) + (R + 1)
```

An auxiliary head of action width A adds `R*A + A` training parameters. The
reported inference FLOP estimate excludes the training-only auxiliary head.

## Threshold selection and exact scheduler replay

Each fold enumerates every unique training score plus the fail-closed threshold
immediately above the maximum score. Selection uses only outer-training
predictions, in this order:

1. require zero late or missed training triggers when feasible;
2. minimize exact `CONFIRM_NEXT` Coda calls;
3. minimize mean absolute trigger offset;
4. maximize threshold for deterministic tie-breaking.

The replay preserves forced Coda calls at k=1 and k=2. From k=3 onward:

```text
SEARCH negative  -> skip Coda and continue recurrence
SEARCH positive  -> run Coda at k and store a_k
CONFIRM_NEXT      -> force Coda at k+1
                     stop only if adjacent action MSE < 0.001
                     otherwise run Coda every iteration until convergence
```

The gate never declares convergence. If it never triggers, the max-iteration
state is decoded so a final action exists. Replay uses the full non-monotonic
authoritative action-label sequence, rather than combining terminal K values.
Reports include trigger categories/offsets, Coda calls and savings, delta K,
max-iteration change, actual-warm and cold results, per-task/task-macro results,
and latency terms for saved Coda, added recurrence, gate evaluations, and the
explicit gate-cost assumption.

## Minimal optional collection design

The missing data should be collected only behind a new calibration-only option
whose default is off. The minimal change is:

1. freeze the returned action, terminal K, warm-start candidate, cache, and RNG
   state exactly as `shadow_full_depth` already does;
2. in the detached full-depth diagnostic namespace, retain cloned `S_1..S_32`
   and `a_1..a_32` only long enough to write one binary shard per prediction;
3. store no tensor in JSONL; JSON contains only identity, shard path, SHA-256,
   tensor metadata, origin, and phase/protocol identity;
4. write a manifest with `collection_mode=optional_post_production_shadow` and
   `source_trace_set_sha256` equal to the authoritative learned dataset;
5. verify before/after equality for returned action, terminal K, cached state,
   warm candidate, Coda call count in the production prefix, and stop reason.

Required shard keys are `schema_version`, exact identity, `actual_origin`,
`states` with leading iteration dimension 32, and `actions` with leading
iteration dimension 32. The manifest must cover every authoritative prediction.
Collection must preserve native state/action dtypes and record shape, stride,
dtype, and layout metadata. This design has not been wired into runtime code on
this branch because the collection change and one-task smoke result must be
reviewed before a ten-task calibration is launched.

## Commands and expected outputs

After an optional raw collection manifest has been reviewed, build the derived
dataset without rerunning LIBERO:

```bash
python scripts/build_preconvergence_dataset.py \
  --raw-manifest benchmark_results/preconvergence_trigger/raw_shadow_v1/manifest.json \
  --authoritative-dataset-dir benchmark_results/learned_convergence_probe/20260801_seed7/dataset \
  --output-dir benchmark_results/preconvergence_trigger/seed7/dataset
```

Expected outputs are `manifest.json` and ignored binary
`preconvergence_dataset.pt`. The command fails closed on identity, SHA-256,
phase, first-hit, non-finite, shape, or coverage errors.

Train all six rank/auxiliary configurations with the frozen task split:

```bash
python scripts/train_preconvergence_trigger.py \
  --dataset-dir benchmark_results/preconvergence_trigger/seed7/dataset \
  --fold-manifest experiments/robot/libero/manifests/libero_spatial_task_oof_5fold_v1.json \
  --output-dir benchmark_results/preconvergence_trigger/seed7/training \
  --seed 7
```

Expected outputs are ignored `training_bundle.pt` and
`training_summary.json`. No global model or global threshold is fit.

Evaluate exact held-out replay with measured values, or explicitly identified
planning assumptions, supplied on the command line:

```bash
python scripts/evaluate_preconvergence_trigger.py \
  --dataset-dir benchmark_results/preconvergence_trigger/seed7/dataset \
  --training-bundle benchmark_results/preconvergence_trigger/seed7/training/training_bundle.pt \
  --fold-manifest experiments/robot/libero/manifests/libero_spatial_task_oof_5fold_v1.json \
  --coda-latency-ms '<measured-or-declared>' \
  --recurrent-iteration-latency-ms '<measured-or-declared>' \
  --gate-latency-ms '<measured-or-declared>' \
  --output benchmark_results/preconvergence_trigger/seed7/metric_report.json \
  --compact-manifest experiments/robot/libero/manifests/preconvergence_trigger_seed7_result_v1.json
```

The compact manifest must only be created after a real OOF run. If no model
passes every promotion check, online integration and the GPU microbenchmark
remain prohibited. If a model passes, a separate profiling-only benchmark must
measure tensor scoring, synchronized decision cost, kernels, peak memory, Coda,
recurrence, and break-even latency before any runtime proposal.
