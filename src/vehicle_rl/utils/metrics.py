"""Trajectory metrics shared by Phase 2 baseline and Phase 3 RL evaluation.

Two layers:

- `ProgressAccumulator` is a streaming, per-env arc-length progress tracker
  that runs inside the rollout loop. It owns the wrap correction and the
  per-step delta cap, so completion / on-track-progress stay honest even
  when the projection foot jumps near self-intersections or off-track
  excursions.

- `TrajectoryMetrics` (dataclass) + `summarize_trajectory(...)` produce the
  7-metric Phase 2 sanity report from per-step time-series. Phase 3 uses
  the same definitions per env, so RL-vs-baseline comparisons are
  apples-to-apples.

Keys in `TrajectoryMetrics` match the JSON output of `run_classical.py`.
"""
from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass

import numpy as np
import torch
from torch import Tensor


@dataclass
class TrajectoryMetrics:
    """Per-rollout trajectory summary (single env)."""
    duration_sec: float = 0.0
    n_steps: int = 0
    rms_lateral_error_m: float = 0.0
    max_lateral_error_m: float = 0.0
    completion_rate: float = 0.0
    on_track_progress_rate: float = 0.0
    traveled_arc_m: float = 0.0
    course_length_m: float = 0.0
    mean_speed_error_mps: float = 0.0
    max_yaw_rate_rad_s: float = 0.0
    max_roll_angle_deg: float = 0.0
    off_track_time_sec: float = 0.0
    off_track_threshold_m: float = 1.0


class ProgressAccumulator:
    """Streaming arc-length-progress tracker (per env).

    Each step:
      delta_s = s_proj_curr - s_proj_prev
      if loop:                                 # wrap correction
          delta_s += L if delta_s < -L/2
          delta_s -= L if delta_s >  L/2
      cap     = max(ds * cap_baseline_ds_factor,
                    |vx| * dt * cap_velocity_factor)
      delta_s = clamp(delta_s, -cap, +cap)     # filter argmin glitches
      forward = max(delta_s, 0)                # ignore retreats
      traveled += forward
      on_track += forward * (|lat| <= threshold)

    The cap is per-element so each env clamps with its own current speed.
    """

    def __init__(
        self,
        num_envs: int,
        *,
        total_length: float,
        ds: float,
        dt: float,
        is_loop: bool,
        device: torch.device | str = "cpu",
        off_track_threshold: float = 1.0,
        cap_velocity_factor: float = 2.0,
        cap_baseline_ds_factor: float = 2.0,
    ):
        if total_length <= 0.0:
            raise ValueError(f"total_length must be positive, got {total_length}")
        self.num_envs = num_envs
        self.L_total = float(total_length)
        self.ds = float(ds)
        self.dt = float(dt)
        self.is_loop = bool(is_loop)
        self.off_track_threshold = float(off_track_threshold)
        self._cap_v = float(cap_velocity_factor)
        self._cap_ds = float(cap_baseline_ds_factor)
        self.device = torch.device(device)

        self._has_prev = False
        self._s_prev = torch.zeros(num_envs, device=self.device)
        self._traveled = torch.zeros(num_envs, device=self.device)
        self._on_track = torch.zeros(num_envs, device=self.device)

    def update(self, s_proj: Tensor, vx: Tensor, lat_err: Tensor) -> None:
        """Accumulate one step of progress for all envs.

        Inputs are (N,) tensors aligned with `num_envs`. The first call only
        records `s_prev` and skips accumulation (no prior sample to diff
        against), so over T steps the accumulator integrates T-1 deltas.
        """
        if not self._has_prev:
            self._s_prev = s_proj.detach().clone()
            self._has_prev = True
            return

        delta = s_proj - self._s_prev
        if self.is_loop:
            half = self.L_total / 2.0
            delta = torch.where(delta < -half, delta + self.L_total, delta)
            delta = torch.where(delta >  half, delta - self.L_total, delta)
        cap = torch.clamp(vx.abs() * self.dt * self._cap_v,
                          min=self.ds * self._cap_ds)
        delta = torch.maximum(-cap, torch.minimum(cap, delta))
        forward = torch.clamp(delta, min=0.0)
        self._traveled = self._traveled + forward
        on_track_mask = lat_err.abs() <= self.off_track_threshold
        self._on_track = self._on_track + forward * on_track_mask.to(forward.dtype)
        self._s_prev = s_proj.detach().clone()

    def reset(self, env_ids: Tensor | None = None) -> None:
        """Restart accumulation. Partial reset zeros only `env_ids`."""
        if env_ids is None:
            self._has_prev = False
            self._traveled.zero_()
            self._on_track.zero_()
            self._s_prev.zero_()
        else:
            # Per-env "has_prev" is not tracked; the first post-reset update()
            # for these envs will diff against the old s_prev (now zero) and
            # might log one spurious clamped delta. For Phase 3 RL with
            # ~hundreds of steps per episode this is a sub-percent effect.
            self._traveled[env_ids] = 0.0
            self._on_track[env_ids] = 0.0
            self._s_prev[env_ids] = 0.0

    @property
    def traveled_arc(self) -> Tensor:
        """(N,) cumulative forward arc-length along the path [m]."""
        return self._traveled

    @property
    def on_track_arc(self) -> Tensor:
        """(N,) cumulative forward arc-length while |lat| <= threshold [m]."""
        return self._on_track

    def completion_rate(self) -> Tensor:
        """(N,) traveled_arc / course_length, clamped to [0, 1]."""
        return torch.clamp(self._traveled / self.L_total, max=1.0)

    def on_track_progress_rate(self) -> Tensor:
        """(N,) on_track_arc / course_length, clamped to [0, 1]."""
        return torch.clamp(self._on_track / self.L_total, max=1.0)


def summarize_trajectory(
    *,
    lat_err_m: np.ndarray,         # (T,) m
    vx_mps: np.ndarray,            # (T,) m/s, body-frame
    yaw_rate_rad_s: np.ndarray,    # (T,) rad/s
    roll_deg: np.ndarray,          # (T,) deg (matches CSV log convention)
    dt: float,
    target_speed: float,
    traveled_arc: float,
    on_track_arc: float,
    course_length: float,
    off_track_threshold: float = 1.0,
) -> TrajectoryMetrics:
    """Compute Phase 2's 7 metrics from per-step time-series + progress totals.

    Phase 2 callers feed CSV columns directly. Phase 3 callers can apply
    this per env after a vectorized rollout (slice (T, N) tensors per env).

    `traveled_arc` and `on_track_arc` come from `ProgressAccumulator` (or
    equivalent post-hoc computation); they cannot be recovered from the
    time-series alone because of the per-step glitch cap.
    """
    n = int(lat_err_m.shape[0])
    if n == 0:
        return TrajectoryMetrics(course_length_m=float(course_length),
                                 off_track_threshold_m=float(off_track_threshold))
    if course_length <= 0.0:
        raise ValueError(f"course_length must be positive, got {course_length}")

    abs_lat = np.abs(lat_err_m)
    return TrajectoryMetrics(
        duration_sec=float(n) * float(dt),
        n_steps=n,
        rms_lateral_error_m=float(np.sqrt(np.mean(lat_err_m ** 2))),
        max_lateral_error_m=float(abs_lat.max()),
        completion_rate=float(min(1.0, traveled_arc / course_length)),
        on_track_progress_rate=float(min(1.0, on_track_arc / course_length)),
        traveled_arc_m=float(traveled_arc),
        course_length_m=float(course_length),
        mean_speed_error_mps=float(np.mean(vx_mps - target_speed)),
        max_yaw_rate_rad_s=float(np.abs(yaw_rate_rad_s).max()),
        max_roll_angle_deg=float(np.abs(roll_deg).max()),
        off_track_time_sec=float(((abs_lat > off_track_threshold) * dt).sum()),
        off_track_threshold_m=float(off_track_threshold),
    )


def write_metrics_json(
    metrics: TrajectoryMetrics,
    out_path: str,
    **extra: object,
) -> None:
    """Write `metrics` as JSON to `out_path`, merging caller-supplied extras
    (run identifiers like course / mu / target_speed) at the top level.
    """
    payload = {**asdict(metrics), **extra}
    parent = os.path.dirname(out_path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2)
