#!/usr/bin/env python3
"""Additional interleaved 4-arm Action-head latency validation.

This runner extends the already completed 10-task x 5-state interleaved
component benchmark with ten *disjoint* official states per task, yielding an
additional 100 measured episodes per method (400 total measured episodes).

State selection
---------------
The previous 5-state run used positions:
    0, 10, 20, 30, 40

This additional run uses the disjoint positions:
    2, 7, 12, 17, 22, 27, 32, 37, 42, 47

These preserve the frozen manifest's 20% calibration / 20% screening / 60%
final partition proportions (2 / 2 / 6 states per task).  Pooling both runs
therefore gives 15 states/task = 150 episodes/method with 3 / 3 / 9 states from
the three partitions.

Order balancing
---------------
The first 5-state run rotated methods with shift = task_id mod 4, giving shift
counts 3/3/2/2 over ten tasks.  This additional run uses the complementary
rotation shift = (task_id + 2) mod 4, giving 2/2/3/3.  When the two runs are
pooled, order-position exposure is therefore nearly balanced rather than
repeating the same temporal-position imbalance.

All model, method, timer, and final Combined semantics are inherited unchanged
from ``profile_spatial_paper_action_head_10x5_interleaved.py``.  In particular,
Combined applies LDCE on cold-origin predictions as well as warm-origin ones,
and the reportable latency is synchronized wall-clock around
``action_head.predict_action`` only.  Policy-query latency from this live
profile must not be reported.
"""

from __future__ import annotations

import sys

import scripts.profile_spatial_paper_action_head_10x5_interleaved as runner


# Ten new official-state positions, strictly disjoint from the previous
# (0, 10, 20, 30, 40) set and preserving 2/2/6 partition proportions.
runner.STATE_POSITIONS = (2, 7, 12, 17, 22, 27, 32, 37, 42, 47)
runner.PROTOCOL = "libero-spatial-action-head-live-additional-10x10-4arm-interleaved-v1"
runner.SMOKE_PROTOCOL = (
    "libero-spatial-action-head-live-additional-1x1-4arm-interleaved-smoke-v1"
)
runner.DEFAULT_OUTPUT_ROOT = (
    "benchmark_results/paper_spatial_action_head_latency_interleaved_additional_10x10"
)


def _complementary_rotating_order(task_id: int) -> tuple[str, ...]:
    shift = (int(task_id) + 2) % len(runner.ARMS)
    return runner.ARMS[shift:] + runner.ARMS[:shift]


runner._rotating_order = _complementary_rotating_order


if __name__ == "__main__":
    sys.exit(runner.main())
