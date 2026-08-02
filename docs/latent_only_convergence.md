# True latent-only convergence stopping

## Research question

Can change between consecutive recurrent latent states replace adjacent
action-output MSE as the stopping signal, so that recurrence performs exactly
one Coda decode on its terminal state? This experiment does not modify or
extend the origin-aware scheduler. It specifically excludes confirmation,
backfill, max-skip, and cached-action policies.

## Online semantics

Select the path with `--recurrence_strategy latent_only`. Each iteration first
advances the recurrent state. Starting at `latent_only_min_iter`, it compares
the new state with the immediately preceding state in FP32. A value at or below
the effective threshold stops recurrence; otherwise recurrence reaches
`recurrence_max_iter`. Coda runs once after the loop on `S_K_t`. Action-MSE is
never computed in this path.

Supported metrics are:

- `raw_mse = mean((s_k - s_prev)^2)`
- `relative_mse = raw_mse / (mean(s_prev^2) + eps)`
- `cosine_distance = 1 - cosine_similarity(flatten(s_k), flatten(s_prev))`
- `relative_l2 = norm(s_k - s_prev) / (norm(s_prev) + eps)`

`latent_only_cold_threshold` applies when no cached state was accepted.
`latent_only_warm_threshold` applies only when the supplied warm state was
actually accepted. Merely enabling warm-start does not select the warm
threshold. Existing S1, midpoint, and final candidate capture are preserved;
midpoint remains the experiment default. Non-finite recurrent states, metrics,
or the terminal Coda output raise before an invalid action can be returned.

Runtime defaults remain backward-compatible. The latent-only thresholds default
to zero and have no effect unless this strategy is explicitly selected.

## Calibration trace

The existing clean adjacent-action-MSE `shadow_full_depth` mode is the scalar
trace mode. Production stops exactly as before and freezes its returned action
and warm-start candidate before a detached diagnostic tail runs to `max_iter`.
The diagnostic tail is therefore for calibration only and must not be enabled
for a latent-only rollout.

Each eligible trace item contains the iteration, actual origin, all four latent
metrics, adjacent action-MSE, the `action_mse < 0.001` label, production baseline
K, and task/episode/prediction IDs. No latent tensors are written.

One task × one episode trace smoke collection (requires the local checkpoint and
LIBERO):

```bash
PYTHONPATH="$PWD/LIBERO" python experiments/robot/libero/run_libero_eval.py \
  --pretrained_checkpoint outputs/12_24-24_24_Spatial_40k \
  --task_suite_name libero_spatial --task_id 0 --num_trials_per_task 1 \
  --use_recurrent True --recurrence_strategy adjacent_action_mse \
  --recurrence_kl_thresh 0.001 --recurrence_max_iter 32 \
  --use_warm_start True --warm_start_source midpoint \
  --validate_warm_start_finite True --use_cached_final_output True \
  --use_latent_precheck False --latent_precheck_mode "'off'" \
  --latent_precheck_trace_level "'off'" --shadow_full_depth True \
  --latent_only_eps 1e-8 \
  --step_log_file benchmark_results/latent_only/trace_smoke/steps.jsonl \
  --json_log_file benchmark_results/latent_only/trace_smoke/result.json
```

## Offline task-level OOF evaluation

Pass every task JSONL with a repeated `--trace`. The evaluator uses the frozen
task-level 5-fold manifest. It selects separate `COLD` and `ACTUAL_WARM`
thresholds from the four training folds only. Scalar metrics are not normalized.
The held-out fold cannot affect threshold selection.

```bash
python scripts/evaluate_latent_only_metrics.py \
  --trace benchmark_results/latent_only/calibration/task0/steps.jsonl \
  --trace benchmark_results/latent_only/calibration/task1/steps.jsonl \
  --trace benchmark_results/latent_only/calibration/task2/steps.jsonl \
  --trace benchmark_results/latent_only/calibration/task3/steps.jsonl \
  --trace benchmark_results/latent_only/calibration/task4/steps.jsonl \
  --trace benchmark_results/latent_only/calibration/task5/steps.jsonl \
  --trace benchmark_results/latent_only/calibration/task6/steps.jsonl \
  --trace benchmark_results/latent_only/calibration/task7/steps.jsonl \
  --trace benchmark_results/latent_only/calibration/task8/steps.jsonl \
  --trace benchmark_results/latent_only/calibration/task9/steps.jsonl \
  --output benchmark_results/latent_only/calibration/metric_report.json
```

For every metric and origin, the report includes AUROC, AUPRC, fold thresholds,
false-convergence count, convergence capture/recall, mean and p95 `delta_K`,
early-stop rate, max-iteration rate, per-task results, and task-macro summaries.
`nominal_best_metric` is diagnostic only; `runtime_defaults_modified` remains
false. Freeze chosen thresholds in the rollout command rather than modifying
source defaults.

## Runtime smoke and screening

One task × one episode latent-only smoke run:

```bash
PYTHONPATH="$PWD/LIBERO" python experiments/robot/libero/run_libero_eval.py \
  --pretrained_checkpoint outputs/12_24-24_24_Spatial_40k \
  --task_suite_name libero_spatial --task_id 0 --num_trials_per_task 1 \
  --use_recurrent True --recurrence_strategy latent_only \
  --recurrence_max_iter 32 --latent_only_metric raw_mse \
  --latent_only_cold_threshold 0.01 --latent_only_warm_threshold 0.01 \
  --latent_only_min_iter 2 --latent_only_eps 1e-8 \
  --use_warm_start True --warm_start_source midpoint \
  --validate_warm_start_finite True --use_cached_final_output False \
  --use_latent_precheck False --latent_precheck_mode "'off'" \
  --latent_precheck_trace_level "'off'" --shadow_full_depth False \
  --step_log_file benchmark_results/latent_only/runtime_smoke/steps.jsonl \
  --json_log_file benchmark_results/latent_only/runtime_smoke/result.json
```

After calibration and threshold freezing, paired 10 tasks × 10 episodes
screening uses the screening partition. Run this only in the checkpoint/LIBERO
environment:

```bash
for task_id in {0..9}; do
  PYTHONPATH="$PWD/LIBERO" python experiments/robot/libero/run_libero_eval.py \
    --pretrained_checkpoint outputs/12_24-24_24_Spatial_40k \
    --task_suite_name libero_spatial --task_id "$task_id" \
    --evaluation_protocol_phase screening \
    --initial_state_manifest_path experiments/robot/libero/manifests/libero_spatial_official_50_v1.json \
    --initial_states_path DEFAULT --num_trials_per_task 10 --seed 7 \
    --reset_rng_each_episode True --use_recurrent True \
    --recurrence_strategy latent_only --recurrence_max_iter 32 \
    --latent_only_metric '<frozen_metric>' \
    --latent_only_cold_threshold '<frozen_cold_threshold>' \
    --latent_only_warm_threshold '<frozen_warm_threshold>' \
    --latent_only_min_iter 2 --latent_only_eps 1e-8 \
    --use_warm_start True --warm_start_source midpoint \
    --validate_warm_start_finite True --use_cached_final_output False \
    --use_latent_precheck False --latent_precheck_mode "'off'" \
    --latent_precheck_trace_level "'off'" --shadow_full_depth False \
    --step_log_file "benchmark_results/latent_only/screening/task${task_id}/steps.jsonl" \
    --json_log_file "benchmark_results/latent_only/screening/task${task_id}/result.json"
done
```

## Pass/fail criteria

A runtime prediction passes the mechanism checks only when `strategy` and
`execution_path` are `latent_only`, `coda_call_count == 1`, the selected metric
and effective threshold match its actual origin, and the stop reason is exactly
`latent_threshold` or `max_iter`. Any non-finite abort or entry into a pre-check
or origin-aware path fails.

Calibration passes the data contract when all eligible iterations contain the
complete scalar/label/identity fields and the evaluator leakage audit passes.
Metric promotion should require zero held-out false-convergence events and an
acceptable held-out convergence capture, `delta_K`, max-iteration rate, and
paired LIBERO success result. Thresholds must be frozen from calibration OOF
results before screening; screening and final rollouts must never tune them.
