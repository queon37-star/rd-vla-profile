# One-step-ahead Coda trigger feasibility

## Status

This branch contains an offline-only implementation of the proposed
pre-convergence trigger plus an explicit, disabled-by-default raw shadow
collector. The collector adds diagnostic plumbing to `action_heads.py` but
does not change runtime stopping, Coda scheduling, model outputs, warm-cache
behavior, or defaults.

The local artifact audit found that the required raw recurrent trajectory is
not present in the existing artifacts. Collection code is now available, but
no new raw shard, real model, task-OOF result, GPU run, or LIBERO run has been
produced. Feasibility remains blocked pending the documented smoke and formal
collection; this is not a negative model or promotion result.

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
produce a different first hit from native BF16 control flow. New schema-2 raw
shards therefore embed native production labels and FP32 shadow-tail labels
with explicit per-iteration source flags. The legacy schema-1 builder still
joins old-style raw shards to the frozen learned-probe dataset.

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

## Optional raw-shadow collection contract

Raw collection is wired only behind
`collect_preconvergence_raw_shadow=false` by default. It is accepted only for
the smoke or calibration protocol, clean adjacent action-MSE stopping,
midpoint warm-start, cached final output, and existing `shadow_full_depth`.
No production stopping or scheduling branch reads the collected payload.

The action head freezes terminal K, returned output, and midpoint candidate
before copying production-prefix tensors to CPU or continuing the existing
detached shadow tail. It stores native state/action dtype and values; it does
not pool, summarize, or cast tensors to FP32. Grouped shards are bounded by
`preconvergence_raw_shadow_shard_size` (default 32 predictions), written to a
temporary file, and atomically renamed. A non-empty output directory is never
reused. An interrupted manifest remains `complete=false` and is rejected by
the validator.

Schema version 2 stores `states[S_1..S_max]`, `actions[a_1..a_max]`, production
native-BF16 `iteration_mse`, shadow-tail FP32 diagnostic action MSE, and an
iteration-level source/phase vector. Every prediction also carries exact
task/episode/prediction/timestep identity, detailed origin
(`ACTUAL_WARM`, `COLD_PRIMARY`, or `COLD_RETRY`), terminal K, protocol and warm
metadata, source commit, run identity, checkpoint file hashes, tensor
shape/stride/dtype/layout, and tensor content SHA-256. JSONL retains hashes and
flags only; full latent tensors exist only in ignored binary shards.

The raw artifact creates its own `trace_set_sha256` from its prediction
identities and tensor hashes. It deliberately does not reuse the earlier
scalar trace-set SHA. The dataset builder accepts one or more schema-2 raw
manifests directly because authoritative BF16/FP32 labels are embedded in the
shards; its legacy schema-1 join remains supported.

For `[1,8,896]` BF16 states and `[1,8,7]` BF16 actions, 32 iterations occupy
about 451.5 KiB per prediction. At 2,398 predictions the raw tensors total
about 1.033 GiB; allow roughly 1.05--1.15 GiB including serialization and
metadata. A 32-prediction shard is approximately 14.1 MiB, or about 75 shards.

## Commands and expected outputs

Run the paired three-episode task-0 smoke first. These commands are for manual
GPU/LIBERO execution and are not run by unit tests:

```bash
# Collection off reference (full-depth diagnostics remain on for parity hashes).
python experiments/robot/libero/run_libero_eval.py \
  --pretrained_checkpoint outputs/12_24-24_24_Spatial_40k \
  --task_suite_name libero_spatial --task_id 0 --num_trials_per_task 3 \
  --evaluation_protocol_phase smoke \
  --initial_state_manifest_path experiments/robot/libero/manifests/libero_spatial_official_50_v1.json \
  --initial_states_path DEFAULT --reset_rng_each_episode True --seed 7 \
  --use_recurrent True --recurrence_strategy adjacent_action_mse \
  --recurrence_kl_thresh 0.001 --recurrence_max_iter 32 \
  --use_warm_start True --warm_start_source midpoint --warm_start_min_iter 2 \
  --use_cached_final_output True --use_latent_precheck False \
  --latent_precheck_mode off --shadow_full_depth True \
  --step_log_file benchmark_results/preconvergence_trigger/raw_shadow_smoke_off/steps.jsonl \
  --json_log_file benchmark_results/preconvergence_trigger/raw_shadow_smoke_off/result.json

# Collection on, using a fresh output directory.
python experiments/robot/libero/run_libero_eval.py \
  --pretrained_checkpoint outputs/12_24-24_24_Spatial_40k \
  --task_suite_name libero_spatial --task_id 0 --num_trials_per_task 3 \
  --evaluation_protocol_phase smoke \
  --initial_state_manifest_path experiments/robot/libero/manifests/libero_spatial_official_50_v1.json \
  --initial_states_path DEFAULT --reset_rng_each_episode True --seed 7 \
  --use_recurrent True --recurrence_strategy adjacent_action_mse \
  --recurrence_kl_thresh 0.001 --recurrence_max_iter 32 \
  --use_warm_start True --warm_start_source midpoint --warm_start_min_iter 2 \
  --use_cached_final_output True --use_latent_precheck False \
  --latent_precheck_mode off --shadow_full_depth True \
  --collect_preconvergence_raw_shadow True \
  --preconvergence_raw_shadow_max_depth 32 \
  --preconvergence_raw_shadow_shard_size 32 \
  --preconvergence_raw_shadow_dir benchmark_results/preconvergence_trigger/raw_shadow_smoke_on \
  --step_log_file benchmark_results/preconvergence_trigger/raw_shadow_smoke_on/steps.jsonl \
  --json_log_file benchmark_results/preconvergence_trigger/raw_shadow_smoke_on/result.json

python scripts/validate_preconvergence_raw_shards.py \
  --manifest benchmark_results/preconvergence_trigger/raw_shadow_smoke_on/manifest.json \
  --step-log benchmark_results/preconvergence_trigger/raw_shadow_smoke_on/steps.jsonl \
  --parity-step-log benchmark_results/preconvergence_trigger/raw_shadow_smoke_off/steps.jsonl \
  --expected-state-shape 32,1,8,896 --expected-state-dtype torch.bfloat16 \
  --expected-action-shape 32,1,8,7 --expected-action-dtype torch.bfloat16 \
  --output benchmark_results/preconvergence_trigger/raw_shadow_smoke_on/validation_report.json \
  --compact-manifest benchmark_results/preconvergence_trigger/raw_shadow_smoke_on/compact_manifest.json
```

After validator and parity review, verify that the builder reads the smoke
artifact without rerunning LIBERO:

```bash
python scripts/build_preconvergence_dataset.py \
  --raw-manifest benchmark_results/preconvergence_trigger/raw_shadow_smoke_on/manifest.json \
  --output-dir benchmark_results/preconvergence_trigger/raw_shadow_smoke_dataset
```

Expected outputs are `manifest.json` and ignored binary
`preconvergence_dataset.pt`. The command fails closed on identity, SHA-256,
phase, first-hit, non-finite, shape, or coverage errors.

Only after the smoke report is reviewed, the formal 10-task x 10-episode
collection is shown below. Run it from the reviewed collector commit so its
recorded `source_commit` is the new collector commit, not the earlier scalar
artifact commit:

```bash
python experiments/robot/libero/run_libero_eval.py \
  --pretrained_checkpoint outputs/12_24-24_24_Spatial_40k \
  --task_suite_name libero_spatial --num_trials_per_task 10 \
  --evaluation_protocol_phase calibration \
  --initial_state_manifest_path experiments/robot/libero/manifests/libero_spatial_official_50_v1.json \
  --initial_states_path DEFAULT --reset_rng_each_episode True --seed 7 \
  --use_recurrent True --recurrence_strategy adjacent_action_mse \
  --recurrence_kl_thresh 0.001 --recurrence_max_iter 32 \
  --use_warm_start True --warm_start_source midpoint --warm_start_min_iter 2 \
  --use_cached_final_output True --use_latent_precheck False \
  --latent_precheck_mode off --shadow_full_depth True \
  --collect_preconvergence_raw_shadow True \
  --preconvergence_raw_shadow_max_depth 32 \
  --preconvergence_raw_shadow_shard_size 32 \
  --preconvergence_raw_shadow_dir benchmark_results/preconvergence_trigger/raw_shadow_calibration_seed7 \
  --step_log_file benchmark_results/preconvergence_trigger/raw_shadow_calibration_seed7/steps.jsonl \
  --json_log_file benchmark_results/preconvergence_trigger/raw_shadow_calibration_seed7/result.json
```

The expected root is
`benchmark_results/preconvergence_trigger/raw_shadow_calibration_seed7/` with
`manifest.json`, grouped `raw_shadow_*.pt` shards, step/results logs, and a
separately generated validation report. The manifest reports ACTUAL_WARM and
cold counts separately; primary feasibility remains ACTUAL_WARM-only.

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
