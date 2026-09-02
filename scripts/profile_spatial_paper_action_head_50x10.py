#!/usr/bin/env python3
"""Run the final live Action-head profile on all 50 official Spatial states.

This is the full-distribution closed-loop counterpart of
``profile_spatial_paper_action_head_10x5.py``. It reuses the validated live
component timer and the exact frozen 4-arm paper configuration, but profiles
all 50 official initial states for every LIBERO-Spatial task.

Measured scale:
    10 tasks x 50 episodes x 4 arms = 2,000 live episodes
plus one unmeasured warm-up rollout per arm.

Primary use of this run:
    * paired closed-loop success / recurrent-depth / Coda accounting audit
    * live Action-head wall-clock latency over the full prediction distribution

The Action-head timer intentionally synchronizes CUDA immediately before and
after ``action_head.predict_action``. Therefore the enclosing get_action timer
from this instrumented run is not the clean paper Policy-query latency metric;
use the profiling-off 2,000-episode paper run for that secondary metric.

All 50 state positions are selected from the frozen official-state manifest,
so there is no subsampling or workload replay in this measurement.
"""

from __future__ import annotations

import sys

import scripts.profile_spatial_paper_action_head as profiler


# Override only the formal sample set and protocol identity. The validated live
# timer, warm-up behavior, model configuration, method configuration, task-level
# resume logic, and result aggregation remain unchanged.
profiler.PROFILE_STATE_POSITIONS = tuple(range(50))
profiler.PROFILE_EPISODES_PER_TASK = len(profiler.PROFILE_STATE_POSITIONS)
profiler.PROFILE_PROTOCOL = "libero-spatial-action-head-live-50x10-4arm-v1"


if __name__ == "__main__":
    sys.exit(profiler.main())
