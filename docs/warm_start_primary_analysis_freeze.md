# Warm-start primary analysis freeze

## Decision

The confirmatory success-preservation analysis is frozen before the final warm-start rollout as follows.

- Estimand: equal-weight mean paired success difference across the ten fixed LIBERO Spatial tasks.
- Direction: midpoint warm-start minus cold-initialized adaptive baseline.
- Primary sample: 47 predeclared warm-start-outcome-unseen states per task, 470 paired trials total.
- Non-inferiority margin: 5 percentage points in absolute success probability.
- Test: pooled paired-trinomial profile-likelihood test of the null boundary `Delta <= -0.05`.
- Decision cutoff: one-sided profile-likelihood p-value strictly less than `0.045`.

The cutoff `0.045` is a finite-sample simulation calibration for this fixed design and predeclared probability grid. It is not presented as a new universal significance level. The nominal one-sided context remains 5%, while the smaller decision cutoff corrects the observed liberal behavior of the unadjusted profile-likelihood approximation at 470 pairs.

## Planning evidence

At 20,000 simulated studies per scenario, cutoff `0.045` met both planning gates across all predeclared null and true-difference-zero scenarios:

- worst estimated type-I error: 0.0488;
- worst Wilson 95% upper bound: 0.0519;
- minimum power when the true difference is zero: 0.8578;
- minimum power when the true difference is -1 percentage point: 0.6983, reported only as sensitivity.

The unadjusted `0.05` cutoff did not qualify:

- worst estimated type-I error: 0.0634;
- worst Wilson 95% upper bound: 0.0668.

Therefore `0.05` is rejected for the final primary test, and `0.045` is the largest qualified value in the predeclared cutoff grid.

## Margin rationale

A reduction greater than five successful episodes per 100 paired rollouts is treated as practically unacceptable. The same 5 percentage-point threshold was used previously in the scalar-stopping preservation evaluation, preserving continuity in the project's definition of a material closed-loop regression.

Margins of 5, 3, and 2 percentage points were examined before observing the final warm-start outcomes. The 5 percentage-point margin is frozen because it is both the maximum substantively acceptable loss and the only candidate that provides at least 80% planned power across the true-difference-zero scenario grid with 47 pairs per task. Statistical feasibility alone is not presented as the substantive justification.

The final paper must disclose that power drops below 80% in the worst planning scenario when the true warm-start effect is -1 percentage point. The design is powered under a true-difference-zero planning assumption, not under all small negative effects.

## State accounting and acquisition

Three calibration states per task were observed in the all-task warm-start preflight and are excluded from the primary confirmatory analysis. The remaining 47 state IDs per task are fixed in `state_accounting_v1.json`.

To avoid changing the already validated 10/10/30 online protocol, the final acquisition will execute the existing `calibration`, `screening`, and `final` phases for both arms, covering all 50 official states per task. The primary analyzer will then remove the predeclared three preflight state IDs from each task before computing the 470-pair estimand.

This means:

- executed rollout pairs: 50 per task, 500 total pairs, 1,000 episodes across two arms;
- primary-analysis pairs: 47 per task, 470 total pairs;
- excluded pairs: exactly the predeclared three warm-start-preflight states per task;
- no exclusion may depend on final success, latency, recurrence depth, or numerical outcome.

The allowed description is **warm-start-outcome-unseen states**, not **untouched states**, because these official initial states were used in earlier scalar studies.

## Secondary efficiency analysis

Efficiency endpoints remain descriptive secondary outcomes:

- predictions per episode;
- mean recurrence depth per prediction;
- recurrent and `_get_output` calls per episode;
- synchronized online policy-query latency per prediction;
- summed synchronized online policy-query time per episode.

The online timer surrounds `get_action` with CUDA synchronization and includes processing, VLM prediction, action policy/action head, and post-processing. It is not the same scope as the post-VLM fixed-Coda microprofile.

Episode-level cost differences cannot be interpreted as a pure warm-start kernel speedup because they also reflect query count, episode length, success, and divergent closed-loop trajectories.

## Frozen mechanisms

Both arms retain the original input-adaptive Action-MSE stopping rule:

- adjacent action-output MSE threshold: `0.001`;
- maximum recurrence depth: `32`;
- cached terminal output: enabled;
- executed actions per prediction: `5`;
- latent pre-check, scalar stopping, shadow recurrence, dynamic action horizon: disabled.

The only arm difference is the initial latent source. The baseline uses a fresh random latent for every prediction. The midpoint arm is cold on the first prediction and subsequently reuses the previous prediction's validated finite midpoint candidate.

The following changes are prohibited after this freeze: margin changes, cutoff changes, task-specific rescue rules, midpoint redefinition, warm-start minimum-iteration tuning, staleness gates, noise interpolation, latent/scalar stopping, and state selection based on final outcomes.

## Remaining authorization gates

This analysis freeze does not authorize the final rollout. Authorization still requires:

1. a frozen runtime source commit;
2. a frozen checkpoint tree digest;
3. a captured Python, PyTorch, CUDA, GPU, and package environment snapshot;
4. a validated final launcher that executes the three existing phases for both arms;
5. a validated final analyzer that excludes exactly the frozen 3 state IDs per task and applies the frozen profile-likelihood decision rule.
