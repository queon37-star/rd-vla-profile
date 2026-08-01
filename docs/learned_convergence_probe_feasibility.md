# Learned Convergence Probe Feasibility

## Decision

**Stop the study, fail closed.** The corrected primary policy is
`final_only_coda`: recurrence and the probe run until the first positive (or
`max_iter`), and Coda runs exactly once on that terminal latent. It reduces
Coda calls by 80.57%, but every probe still has false convergence and large
terminal delay. Moreover, with the frozen planning anchors, the extra recurrent
work outweighs the saved Coda work even before probe/control overhead. The
result is `safety_failed_efficiency_also_unfavorable`, and
`online_integration_worth_investigating` remains `false`. No online inference
code or `action_heads.py` was changed.

This is a baseline-conditioned offline feasibility result. It is not evidence
about closed-loop success or deployment latency.

## Frozen input and formal validation

Only this calibration run was consumed:

```text
benchmark_results/origin_aware_calibration/20260801_ca1b7d3_seed7_10x10
```

The existing formal validator was run before dataset construction and is also
mandatory inside the builder. It revalidated all workload shards and protocol
metadata:

| Check | Result |
|---|---:|
| Tasks | 10 |
| Episodes | 100 |
| Predictions | 2,398 |
| Actual-warm predictions | 2,298 |
| Cold predictions | 100 |
| Workload shards | 200 |
| Successful episodes | 92 |
| Complete ten-task gate | pass |

Provenance:

- source Git commit: `7c5ce2184c2e06f062bda616f607fc9b1c93762b`
- initial-state manifest SHA-256: `0e3c6609b719d6b0a05f79efd769dff67141b52d00b42d9e0bea904ecf493144`
- canonical calibration-validation payload SHA-256: `5d137b70260c68eceacd537709e733efd66e08188613674d7ec7c53a278d29af`
- ten-trace aggregate SHA-256: `7d183db1df095c423a5e3976dd96f9ef59991741623ae8c0d5d8f5ed94c98926`
- five-fold manifest SHA-256: `77f6b66a8a20736ce081c009dc5401a51433d96b64c580199495f0ff8ba27571`
- derived dataset SHA-256: `9a059b982cf9cc257abada4479acca3dd8098a8f7f2978779352a4d8b404d413`

The compact result manifest records each of the ten trace-file hashes. Smoke,
screening, and final data were not loaded. No rollout or calibration collection
was run.

## Dataset and label contract

The dataset has 74,338 finite scalar transitions, one sequence per prediction.
For every `k >= 2`, the label is `action_mse < 0.001`.

- On `k <= baseline_k`, `action_mse` is the recorded native control-flow
  `iteration_mse`.
- Only on `k > baseline_k` is `shadow_trace.action_mse` used; this is explicitly
  marked as an FP32 shadow-tail diagnostic.
- The trace has no original latent tensors and no recorded cosine scalar.
  Consequently no cosine feature was reconstructed or invented.

The 18 features are current `latent_mse` and `latent_l2`, iteration and
normalized iteration, one- and two-iteration latent history, one- and two-step
slopes and ratios, a two-step-history availability flag, and warm/cold origin.
The exact ordered schema and missing-history policy are in the result manifest.

## OOF and fitting protocol

The existing task manifest is used unchanged:

```text
fold 0: validation tasks 0, 9
fold 1: validation tasks 1, 8
fold 2: validation tasks 2, 7
fold 3: validation tasks 3, 6
fold 4: validation tasks 4, 5
```

Thus every prediction and every episode from one task remains entirely in one
fold. The recorded leakage audit reports zero prediction and episode overlap in
all five folds. Models are fit on both recorded origins so the origin flag is
available, but threshold selection and the primary gate use actual-warm only;
cold is reported separately.

For each fold, feature means/scales, model parameters, and the decision
threshold are fit from the four training folds only. The threshold candidates
are 1,025 train-score quantiles plus train-only endpoints and the strict
negative-score safety boundary. Among candidates meeting the 99.5% train
capture floor, false convergence is minimized first and expected decode
omission is maximized second. Every selected fold threshold attained zero train
false convergences and at least 99.5% train capture. Held-out results below were
not used to choose normalization, weights, or thresholds.

The v1 evaluator implemented `legacy_cached_action_precheck`: each negative
iteration generated Coda output, a positive at `k` skipped the current decode,
and the previous cached action was returned. That is preserved as a diagnostic.
It is not the proposed mechanism.

The corrected primary `final_only_coda` replay runs recurrence and the probe at
each eligible iteration, stops at the first positive or `max_iter`, and then
runs exactly one `Coda(S_terminal_k)`. With features beginning at `k = 2`, the
accounting is `recurrent_calls = terminal_k`, `probe_calls = terminal_k - 1`,
and `coda_decode_calls = 1`. Both policies use identical scores, thresholds,
labels, and stop points; their safety and terminal metrics are asserted equal.

## Primary actual-warm OOF results

All metrics cover 2,298 actual-warm predictions under `final_only_coda`.
Candidate Coda calls are 2,298 versus 11,839 baseline calls for every model,
a task-macro reduction of 80.572%.

| Model | False conv. | Capture | Precision | Recall | Coda reduction | Task-macro mean delta-K | Task-macro p95 delta-K | Max-iter rate delta | Params | Approx. FLOPs |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Latent-MSE threshold | 1 | 99.629% | 99.956% | 99.695% | 80.572% | 3.689 | 8.085 | +0.326 pp | 1 | 1 |
| Logistic regression | 2 | 99.915% | 99.913% | 99.913% | 80.572% | 3.376 | 7.500 | 0 | 19 | 40 |
| Class-weighted logistic | 2 | 99.915% | 99.913% | 99.913% | 80.572% | 3.341 | 7.560 | 0 | 19 | 40 |
| Tiny MLP, 18-16-1 | 4 | 99.842% | 99.826% | 99.826% | 80.572% | 3.134 | 7.765 | 0 | 321 | 628 |

The nominal diagnostic selection is the latent-MSE threshold because it has
the fewest OOF false convergences. This does not promote it: it still violates
false-convergence, mean delta-K, p95 delta-K, and max-iteration requirements.

Worst-task diagnostics make the lack of robustness explicit:

| Model | Worst capture task/value | Worst p95 delta-K task/value | Worst Coda reduction task/value | False-convergence tasks |
|---|---:|---:|---:|---|
| Latent-MSE threshold | task 0 / 96.739% | task 0 / 15.85 | task 4 / 77.138% | task 6: 1 |
| Logistic regression | task 0 / 99.457% | task 0 / 12.0 | task 4 / 77.138% | tasks 0, 5: 1 each |
| Class-weighted logistic | task 0 / 99.457% | task 0 / 12.0 | task 4 / 77.138% | tasks 0, 5: 1 each |
| Tiny MLP | task 6 / 99.099% | task 0 / 13.85 | task 4 / 77.138% | tasks 4, 6: 2 each |

## Policy comparison

The stop policy is identical, so safety, terminal-K, and recurrent/probe calls
match exactly. Only action generation and Coda accounting differ.

| Model | Final-only Coda reduction | Legacy Coda reduction |
|---|---:|---:|
| Latent-MSE threshold | 80.572% | -51.839% |
| Logistic regression | 80.572% | -46.234% |
| Class-weighted logistic | 80.572% | -45.491% |
| Tiny MLP | 80.572% | -41.086% |

The legacy negative results reproduce v1 and are diagnostic only. They measure
the cached-action pre-check, not the final-only mechanism. The v1 compact result
and model artifact remain byte-identical and are referenced by SHA-256 in v2.

## Cost model

The evaluator keeps the requested terms separate:
`T_baseline = K_baseline * (T_recurrent + T_coda) + baseline control`, and
`T_probe = K_probe * T_recurrent + N_probe * T_probe + T_coda + probe control`.

The frozen planning-anchor manifest supplies:

- recurrent: 3.56 ms/call;
- Coda: 1.83 ms/call;
- baseline action comparison: median 0.166328 ms/call.

These are planning anchors, not current-commit deployment measurements. With
probe, control, and synchronization latency set to zero, the mechanism-only
estimate is:

| Model | Mean recurrent calls | Mean probe calls | Candidate mean | Latency change vs baseline |
|---|---:|---:|---:|---:|
| Latent-MSE threshold | 8.902 | 7.902 | 33.522 ms | +19.293% |
| Logistic regression | 8.590 | 7.590 | 32.409 ms | +15.332% |
| Class-weighted logistic | 8.555 | 7.555 | 32.285 ms | +14.892% |
| Tiny MLP | 8.348 | 7.348 | 31.548 ms | +12.266% |

The mechanism-only baseline is 28.101 ms task-macro mean. Thus even the
zero-overhead upper bound is slower: extra recurrence costs more than the Coda
calls it removes under these anchors.

The conservative record includes the baseline comparison anchor and each
model's parameter/FLOP counts, but synchronized probe and GPU-CPU control
latencies have not been measured. Its total candidate latency and relative
reduction are therefore explicitly `null`, with status
`incomplete_unmeasured_probe_and_synchronization_latency`.

FLOPs are retained only as operation counts and are never converted to latency.
The known-cost headroom is already negative for every model, so any positive
unmeasured overhead can only worsen the planning estimate.

## Reliability and cold supplement

Ten-bin actual-warm transition reliability curves are stored in the compact
manifest. Expected calibration error is 0.00765 for logistic regression,
0.05612 for class-weighted logistic regression, and 0.00363 for the tiny MLP.
The raw latent-MSE baseline is not a probability and is marked not applicable.
Good row-level calibration does not translate to prompt, safe first-hit
scheduler behavior.

Cold results are supplementary (100 predictions). All four models have zero
cold false convergences and 100% capture, precision, and recall. Final-only
Coda calls are 100 versus 597 baseline calls, a task-macro reduction of 83.250%.
Cold results do not enter the primary gate.

## Gate outcome

The required gate is false convergence 0, task-macro capture at least 99.5%,
mean delta-K at most 0.25, task-macro p95 delta-K at most 1, no max-iteration
rate increase, at least 20% decode reduction (clearly beyond the prior 10.2%),
and finite evaluation for every task.

No model passes. All tasks are finite and evaluable and all models meet the
capture floor and 20% Coda-reduction target. However, every model violates
false-convergence, mean delta-K, and p95 delta-K requirements; the latent
threshold additionally increases the max-iteration rate. Safety therefore
fails closed. Separately, mechanism-only latency is unfavorable and the
conservative estimate is incomplete, so the combined decision is:

```text
safety_failed_efficiency_also_unfavorable
online_integration_worth_investigating = false
```

## Reproduction

Use the project `rdvla_env` environment (Python 3.10.16, NumPy 1.26.4).
The correction reuses the existing dataset, training bundle, fold thresholds,
normalization, and model artifact; it does not rebuild or retrain them:

```bash
python scripts/evaluate_learned_convergence_probe.py \
  --dataset-dir benchmark_results/learned_convergence_probe/20260801_seed7/dataset \
  --training-bundle benchmark_results/learned_convergence_probe/20260801_seed7/training_bundle.json \
  --output benchmark_results/learned_convergence_probe/20260801_seed7/final_only_coda_report_v2.json \
  --compact-manifest experiments/robot/libero/manifests/learned_convergence_probe_seed7_final_only_coda_result_v2.json \
  --overwrite
```

The 29 MB derived dataset and full report remain under ignored
`benchmark_results/`; only the compact v2 manifest is tracked. The v1 result
manifest and frozen model artifact are inputs and are not overwritten.
