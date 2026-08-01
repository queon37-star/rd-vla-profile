# Learned Convergence Probe Feasibility

## Decision

**Stop the study, fail closed.** None of the four scalar-only probes passes the
OOF scheduler gate. Every model produces at least one false convergence, mean
and p95 delta-K are far above their limits, and expected Coda decode calls
increase rather than decrease. `online_integration_worth_investigating` is
therefore `false`. No online inference code or `action_heads.py` was changed.

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

The sequential replay evaluates the probe before generating Coda output at
iteration `k`. A positive decision stops at `k`, omits that current decode, and
uses the previous cached action. A miss continues recurrent computation and
Coda decoding; max iteration forces a decode. This gives the probe its most
direct scheduler interpretation without changing online code.

## Primary actual-warm OOF results

All metrics cover 2,298 actual-warm predictions. Decode reduction is
task-macro; negative values mean more Coda calls than the baseline.

| Model | False conv. | Capture | Precision | Recall | Decode reduction | Mean delta-K | Task-macro p95 delta-K | Max-iter rate delta | Params | Approx. FLOPs |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Latent-MSE threshold | 1 | 99.629% | 99.956% | 99.695% | -51.839% | 3.689 | 8.085 | +0.326 pp | 1 | 1 |
| Logistic regression | 2 | 99.915% | 99.913% | 99.913% | -46.234% | 3.376 | 7.500 | 0 | 19 | 40 |
| Class-weighted logistic | 2 | 99.915% | 99.913% | 99.913% | -45.491% | 3.341 | 7.560 | 0 | 19 | 40 |
| Tiny MLP, 18-16-1 | 4 | 99.842% | 99.826% | 99.826% | -41.086% | 3.134 | 7.765 | 0 | 321 | 628 |

The nominal diagnostic selection is the latent-MSE threshold because it has
the fewest OOF false convergences. This does not promote it: it still fails five
scheduler gates and its decode-call count is 51.8% higher than baseline.

Worst-task diagnostics make the lack of robustness explicit:

| Model | Worst capture task/value | Worst p95 delta-K task/value | Worst decode task/value | False-convergence tasks |
|---|---:|---:|---:|---|
| Latent-MSE threshold | task 0 / 96.739% | task 0 / 15.85 | task 5 / -90.185% | task 6: 1 |
| Logistic regression | task 0 / 99.457% | task 0 / 12.0 | task 5 / -85.335% | tasks 0, 5: 1 each |
| Class-weighted logistic | task 0 / 99.457% | task 0 / 12.0 | task 5 / -86.721% | tasks 0, 5: 1 each |
| Tiny MLP | task 6 / 99.099% | task 0 / 13.85 | task 5 / -87.067% | tasks 4, 6: 2 each |

## Reliability and cold supplement

Ten-bin actual-warm transition reliability curves are stored in the compact
manifest. Expected calibration error is 0.00765 for logistic regression,
0.05612 for class-weighted logistic regression, and 0.00363 for the tiny MLP.
The raw latent-MSE baseline is not a probability and is marked not applicable.
Good row-level calibration does not translate to prompt, safe first-hit
scheduler behavior.

Cold results are supplementary (100 predictions). All four models have zero
cold false convergences and 100% capture, precision, and recall, but still
increase Coda calls: 40.03% for latent threshold, 8.71% for logistic, 5.03% for
class-weighted logistic, and 10.39% for the MLP. Cold results do not enter the
primary gate.

## Gate outcome

The required gate is false convergence 0, task-macro capture at least 99.5%,
mean delta-K at most 0.25, task-macro p95 delta-K at most 1, no max-iteration
rate increase, at least 20% decode reduction (clearly beyond the prior 10.2%),
and finite evaluation for every task.

No model passes. All tasks are finite and evaluable and all models meet the
capture floor, but every model violates false-convergence, delta-K, p95, and
decode-reduction requirements. The latent threshold additionally increases the
max-iteration rate. The correct outcome is therefore:

```text
stop_research_fail_closed
online_integration_worth_investigating = false
```

## Reproduction

Use the project `rdvla_env` environment (Python 3.10.16, NumPy 1.26.4):

```bash
python scripts/build_learned_convergence_dataset.py \
  --run-root benchmark_results/origin_aware_calibration/20260801_ca1b7d3_seed7_10x10 \
  --output-dir benchmark_results/learned_convergence_probe/20260801_seed7/dataset \
  --base-seed 7

python scripts/train_learned_convergence_probe.py \
  --dataset-dir benchmark_results/learned_convergence_probe/20260801_seed7/dataset \
  --output benchmark_results/learned_convergence_probe/20260801_seed7/training_bundle.json \
  --seed 7

python scripts/evaluate_learned_convergence_probe.py \
  --dataset-dir benchmark_results/learned_convergence_probe/20260801_seed7/dataset \
  --training-bundle benchmark_results/learned_convergence_probe/20260801_seed7/training_bundle.json \
  --output benchmark_results/learned_convergence_probe/20260801_seed7/report.json \
  --compact-manifest experiments/robot/libero/manifests/learned_convergence_probe_seed7_result_v1.json \
  --model-artifact experiments/robot/libero/manifests/learned_convergence_probe_seed7_model_v1.json
```

The 29 MB derived dataset and full report remain under ignored
