# Warm-Start Power Methodology Correction

Status: **development diagnostic only; no margin or sample size selected**

## 1. What the first run established

The command using 50 outer simulations and 200 bootstrap replicates established
only that the script executes and produces the expected scenario grid. It is not
planning-grade power evidence.

At estimated power 0.80, 50 outer replicates have Monte Carlo standard error
`sqrt(0.8 * 0.2 / 50) = 0.0566`; an approximate 95% Monte Carlo half-width is
about 0.111. With 200 bootstrap replicates, a one-sided 5th percentile is
resolved by only about ten lower-tail samples. Therefore distinctions such as
20 versus 30 pairs per task in the development output are unstable.

No non-inferiority margin may be selected from this run.

## 2. Statistical correction

The current planner mirrors a task-stratified percentile-bootstrap lower bound.
That interval is now **provisional**, not frozen as the final confirmatory
method. Percentile bootstrap intervals for binary proportions can have poor
coverage, especially in sparse or boundary configurations. Matched-pair binary
risk-difference literature instead gives dedicated score/profile methods,
including Tango-type score intervals and related paired-proportion intervals.

Before final authorization, candidate interval procedures must be compared by
simulation under the planned fixed-task design. At minimum, the validation must
include:

1. type-I error at the non-inferiority boundary `Delta = -margin`;
2. sparse discordance and directional imbalance;
3. task-heterogeneous `p01/p10` profiles;
4. the exact candidate pairs-per-task values;
5. Monte Carlo uncertainty for both size and power.

The percentile-bootstrap result may remain a descriptive sensitivity analysis,
but it is not yet the sole primary gate.

Relevant paired-binary interval references:

- Newcombe (1998), *Statistics in Medicine*, DOI 10.1002/(SICI)1097-0258(19981130)17:22<2635::AID-SIM954>3.0.CO;2-C.
- Tang et al. (2010), *Statistics in Medicine*, DOI 10.1002/sim.3738.
- Yang, Sun, and Hardin (2013), *Statistics in Medicine*, DOI 10.1002/sim.5561.

## 3. Existing-result audit

The current file listing shows many cold 10-task runs, but midpoint warm-start
results appear to be task-0 pilots. File names alone cannot establish pairing.
The required episode fields are:

```text
paired_trial_id
initial_state_id
episode_seed
```

Run:

```bash
python -m py_compile scripts/audit_warm_start_pilot_candidates.py

python scripts/audit_warm_start_pilot_candidates.py \
  --root benchmark_results \
  --output benchmark_results/warm_start_power/pilot_candidate_audit_v1.json
```

The audit classifies result files as paired candidates, legacy/incomplete,
duplicate, or protocol-unverified. It never upgrades a legacy run into paired
evidence.

Expected interpretation from the current chronology is that most July warm-start
runs will be rejected as legacy/unpaired, but this must be verified from the
actual JSON metadata.

## 4. Decision order

1. Audit existing results.
2. If a valid ten-task cold/midpoint pair exists, build a planning-only task
   `p01/p10` profile.
3. Otherwise run a small, explicitly paired warm-start pilot that is excluded
   from confirmatory evidence.
4. Validate interval size at the candidate margins.
5. Run planning-grade power only for the empirically plausible profiles.
6. Freeze margin, target power, pairs per task, seed namespace, and interval
   method before the final run.
