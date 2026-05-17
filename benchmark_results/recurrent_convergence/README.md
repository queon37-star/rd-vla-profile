# Recurrent Convergence Metrics

This directory is the default output location for recurrent iteration convergence logs from
`experiments/robot/libero/run_libero_eval.py`.

## Files

- `*_predictions.jsonl`: one record per action prediction.
- `*_summary.json`: run-level summary with success/failure comparisons.

## Prediction Record Schema

```json
{
  "schema_version": 1,
  "task_id": 0,
  "task_name": "pick up the black bowl between the plate and the ramekin",
  "episode_id": 0,
  "timestep": 10,
  "action_prediction_index": 0,
  "recurrence_strategy": "kl_divergence",
  "recurrent_iteration_count": 6,
  "max_recurrent_iteration": 32,
  "adaptive_stop": true,
  "metric_name": "mse_between_action_outputs",
  "iteration_mse": [0.0123, 0.0041, 0.0007],
  "final_mse": 0.0007,
  "threshold": 0.001,
  "executed_actions_from_prediction": 5,
  "rollout_avg_iteration": 6.4,
  "rollout_max_iteration": 9,
  "rollout_min_iteration": 4,
  "success": true
}
```

## Smoke Test Command

Run a single rollout on one LIBERO task and write metrics to this directory:

```bash
python experiments/robot/libero/run_libero_eval.py \
  --pretrained_checkpoint outputs/12_24-24_24_Spatial_40k \
  --task_suite_name libero_spatial \
  --task_id 0 \
  --num_trials_per_task 1 \
  --use_recurrent True \
  --recurrence_strategy kl_divergence \
  --recurrence_max_iter 32 \
  --recurrence_kl_thresh 0.001 \
  --json_log_file benchmark_results/recurrent_convergence/smoke_results.json
```
