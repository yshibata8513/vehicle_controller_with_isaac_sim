"""Reference-path container expressed in the **vehicle (body) frame**.

A `Plan` is what the controller / policy sees: the next K waypoints
relative to the car's current pose. The planner internally maintains the
full world-frame course (e.g., a circle / S-curve / DLC); each step it
projects the lookahead window into the current body frame and emits a
`Plan`. World-frame state and the full course are NOT part of the
observation.

Body-frame convention (ISO / right-hand): +x forward, +y left, yaw about
+z. Waypoint values are signed offsets from the car's origin:
  x > 0 means waypoint is ahead, y > 0 means it is to the left.

The minimal core is (x, y, v) per waypoint; richer fields (per-waypoint
heading, curvature, course-coordinate s, ...) will be added when the
controllers actually need them.

Shapes are batched as (N, K) so the same Plan can carry per-env paths in
parallel envs. For single-env Phase 2 use, N=1.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor


@dataclass
class Plan:
    """K-point reference path in body frame (lookahead window)."""

    x: Tensor   # (N, K) body-frame longitudinal offset [m] (>0 = ahead)
    y: Tensor   # (N, K) body-frame lateral offset [m]      (>0 = left)
    v: Tensor   # (N, K) target speed at each waypoint [m/s]

    @property
    def num_envs(self) -> int:
        return self.x.shape[0]

    @property
    def num_points(self) -> int:
        return self.x.shape[1]

    @property
    def device(self) -> torch.device:
        return self.x.device
