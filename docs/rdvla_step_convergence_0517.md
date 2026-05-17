# RD-VLA Step Convergence Analysis - 2026-05-17

## Purpose

This note preserves the first-pass convergence analysis for exploring step-level
reuse metrics in RD-VLA. The goal of this stage is logging and benchmark
inspection, not selecting the final reuse metric.

## Added Logging Metrics

The step-level JSONL log records one row per action prediction step. The added
fields include:

- `K_t`: recurrence iterations used for the prediction step.
- `final_conv_score`: final convergence score from the recurrent action head.
- `conv_score_list`: per-iteration MSE between consecutive recurrent outputs.
- `action_delta_list`: per-iteration L2 distance between consecutive recurrent
  outputs.
- `first_converged_k_1e_4`: first iteration whose MSE fell below `1e-4`.
- `first_converged_k_5e_4`: first iteration whose MSE fell below `5e-4`.
- `prev_action_delta`: L2 distance between the current final action chunk and
  the previous prediction step's final action chunk.
- `proprio_delta`: L2 distance between proprioception vectors at consecutive
  prediction steps.
- `latency_ms`: action prediction latency in milliseconds.

The metrics are emitted to `*_step_log.jsonl`, derived from `json_log_file` when
`step_log_file` is not explicitly set.

## LIBERO-Spatial Benchmark Summary

All runs used `num_trials_per_task=3` over 10 LIBERO-Spatial tasks, for 30
episodes per configuration.

| Configuration | Successes | Success Rate | Mean Iterations | Mean Prediction Steps |
| --- | ---: | ---: | ---: | ---: |
| Fixed K=8 | 28/30 | 93.33% | 8.00 | 21.90 |
| Fixed K=12 | 29/30 | 96.67% | 12.00 | 21.33 |
| Fixed K=16 | 29/30 | 96.67% | 16.00 | 20.87 |
| Adaptive threshold=1e-4 | 30/30 | 100.00% | 11.27 | 20.83 |
| Adaptive threshold=5e-4 | 29/30 | 96.67% | 8.25 | 21.20 |

## Fixed K Results

Fixed K=8 reduced recurrent compute but had the lowest success rate in this
trial set. Fixed K=12 and K=16 both reached 29/30 successes. K=16 did not
improve aggregate success over K=12 in these results, while requiring four more
recurrent iterations per prediction.

The fixed runs are useful as convergence baselines because they log
`conv_score_list`, `action_delta_list`, and first-threshold crossing information
without changing the stopping rule.

## Adaptive Results

Adaptive threshold `1e-4` reached 30/30 successes with a mean of 11.27 recurrent
iterations per prediction. Adaptive threshold `5e-4` reached 29/30 successes with
a mean of 8.25 iterations, close to Fixed K=8 compute while matching the
aggregate success rate of Fixed K=12 and K=16 in this trial set.

These adaptive runs still use a convergence/stopping metric close to the
original RD-VLA adaptive logic. They should be treated as evidence about
recurrence convergence behavior, not as a final step-reuse policy.

## Current Conclusion

No new reuse metric has been finalized in this stage. `prev_action_delta` is only
a candidate signal: it compares final action chunks across prediction steps, but
it is available after the current prediction is complete and therefore is not by
itself an early reuse decision metric.

## Next Steps

Implement reuse-oriented candidate metrics that are more directly connected to
prediction-step reuse decisions:

- `early_action_delta`: compare early current-step action outputs to the
  previous step's final action output.
- `state_delta_k`: measure recurrent state changes between `S_k` and `S_{k-1}`
  inside a prediction step.
- `prev_state_delta`: compare compact summaries of final recurrent state across
  consecutive prediction steps.
- `visual_feature_delta`: optional follow-up if VLM feature extraction can be
  logged cleanly without affecting inference outputs.
