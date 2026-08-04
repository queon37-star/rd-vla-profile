# Warm-Start Extended Preflight: Interpretation and Next Gate

Status: **planning diagnostic only; final run not authorized**

## 1. Runtime-contract result

The all-task extended preflight passed the paired runtime validator:

- 10 LIBERO Spatial tasks;
- 3 paired states per task;
- 30 paired outcomes in total;
- cold-only success: 2;
- midpoint-warm-only success: 1;
- observed warm-minus-cold success difference: -1/30 = -3.33 percentage points;
- no warm-state reset or non-finite event;
- paired initial-state and episode-seed identity verified;
- midpoint provenance, adaptive Action-MSE stopping, cached terminal output,
  Coda/get-output call accounting, and inactive-mechanism contracts verified.

The net success difference is one discordant pair. It is not evidence that
warm-start is inferior, non-inferior, or superior. The run was designed to
validate execution and logging, not to estimate the primary effect.

## 2. Descriptive efficiency result

The validator-v2 paired summaries reported the following mean warm-minus-cold
changes across the 30 pairs:

```text
predictions per episode:        +1.3333
recurrent calls per episode:   -53.0333
mean K per prediction:          -2.0624
inference time per episode:   -168.74 ms
```

Interpretation is deliberately limited:

1. Midpoint warm-start produced more policy queries per episode in this small
   sample.
2. The recurrence depth reduction was large enough that total recurrent and
   Coda/get-output calls per episode still decreased.
3. The sum of synchronized online policy-query latencies per episode also
   decreased in this sample.
4. The latency scope is the synchronized `get_action` call and includes image
   preparation/processor work, VLM prediction, action-policy computation, and
   action post-processing. It is not the post-VLM action-head microprofile.
5. The policies may follow different observation trajectories after their
   actions diverge. Therefore the episode-level differences are descriptive
   closed-loop outcomes, not a controlled decomposition of a single identical
   trajectory.
6. No relative percentage reduction is reported until the final arm totals and
   confidence procedure are frozen.

## 3. Repeatability result

The original task-0/task-5 preflight and the corresponding subset of the
all-task extended preflight used the same smoke phase, state IDs, and base seed.
Their behavioral traces matched under the repeatability audit for:

- success outcome;
- prediction count;
- terminal-K sequence;
- stop reason;
- Coda/get-output counts;
- midpoint-source metadata;
- convergence traces.

This supports repeatability of the recorded behavioral execution path. It does
not establish bitwise equality of returned actions because action hashes were
not recorded, and it does not require latency equality.

## 4. State accounting

Smoke uses the first three states of each task's calibration partition. Those
three states have now been observed under both warm and cold policies and are
excluded from the primary confirmatory analysis.

```text
official states per task:                  50
observed warm-start preflight states:       3
warm-start-outcome-unseen states:          47
total maximum primary pairs:              470
```

The allowed term is **warm-start-outcome-unseen states**, not `untouched
states`: the official states were used by prior scalar studies, although their
paired warm-start outcomes had not been inspected before this work.

A 10-state/task planning pilot is not run at this point because it would reduce
the maximum outcome-unseen primary set from 47 to 40 states/task.

## 5. Interval-method validation gate

Before a final run can be authorized, the one-sided paired non-inferiority
procedure must be evaluated at exactly 47 pairs/task. The validation grid must
include:

- the null boundary `Delta = -margin` for type-I error;
- homogeneous and task-heterogeneous discordance;
- balanced task-specific effect heterogeneity around the aggregate boundary;
- true differences above the boundary for power;
- Monte Carlo uncertainty for both type-I error and power.

Candidate procedures currently implemented for comparison are:

1. pooled paired-trinomial profile-likelihood test;
2. fixed-task stratified Wald lower bound;
3. task-stratified percentile bootstrap as an opt-in sensitivity method.

None is presumed valid before simulation. In particular:

- the pooled profile likelihood uses a pooled working model and must be stress
  tested under task heterogeneity;
- the stratified Wald method may be unstable under sparse discordance;
- the percentile bootstrap is not automatically promoted to the primary gate.

The project-level screening rule classifies null-boundary size as controlled
only when the Wilson 95% upper Monte Carlo bound is at most `alpha + 0.01`.
This rule is a planning criterion, not a universal statistical theorem.

## 6. Commands

Development check without nested bootstrap:

```bash
python -m py_compile scripts/validate_warm_start_interval_methods.py

python scripts/validate_warm_start_interval_methods.py \
  --output benchmark_results/warm_start_power/interval_validation_dev_v1.json \
  --outer-replicates 500
```

Planning-grade fast-method run:

```bash
python scripts/validate_warm_start_interval_methods.py \
  --output benchmark_results/warm_start_power/interval_validation_fast_v1.json \
  --outer-replicates 5000
```

The bootstrap sensitivity run is intentionally separate because it nests
bootstrap resampling inside the outer simulation. Its scenario grid and
replicate counts must be reduced so that the script's explicit workload guard
is satisfied.

## 7. Remaining authorization requirements

- candidate interval methods pass or fail the frozen simulation grid;
- one primary method is selected before observing the final outcomes;
- the non-inferiority margin is justified substantively and frozen;
- target power and feasible sample size are frozen at no more than 47 pairs/task;
- the warm-start-specific paired seed namespace is frozen;
- final source commit, checkpoint digest, launcher, and validator are frozen;
- the 30 preflight pairs remain excluded from the primary confirmatory analysis.
