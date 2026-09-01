#!/usr/bin/env python3
"""Run the live Action-head validation on 10 LIBERO-Spatial tasks x 5 episodes.

This is a thin protocol wrapper around ``profile_spatial_paper_action_head.py``.
It keeps the same live timing boundary and frozen paper arm configurations, but
uses five official states per task instead of ten.

Selected positions in the frozen 50-state manifest are 0, 10, 20, 30, 40.
Given the manifest ordering calibration(10) / screening(10) / final(30), this
preserves the 20% / 20% / 60% partition proportions exactly: one calibration,
one screening, and three final states per task.

Measured scale:
    10 tasks x 5 episodes x 4 arms = 200 live episodes
plus one unmeasured warm-up rollout per arm.

The underlying profiler measures only ``action_head.predict_action`` with a
CUDA synchronization immediately before and after the call while the complete
LIBERO/VLM rollout is otherwise executed normally.
"""

from __future__ import annotations

import sys

import scripts.profile_spatial_paper_action_head as profiler


# Override only the evaluation sample set/protocol label. All model/method and
# timer logic remains in the validated live profiler.
profiler.PROFILE_STATE_POSITIONS = (0, 10, 20, 30, 40)
profiler.PROFILE_EPISODES_PER_TASK = len(profiler.PROFILE_STATE_POSITIONS)
profiler.PROFILE_PROTOCOL = "libero-spatial-action-head-live-10x5-4arm-v1"


if __name__ == "__main__":
    sys.exit(profiler.main())
