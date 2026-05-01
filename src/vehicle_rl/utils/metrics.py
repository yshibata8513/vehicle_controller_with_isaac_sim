"""Vehicle dynamics metrics shared by Phase 1 sanity, Phase 2 baseline, Phase 3 RL eval.

Implementing once and reusing keeps Phase 2 vs Phase 3 comparisons apples-to-apples.
"""
from __future__ import annotations

import csv
import json
import math
import os
from dataclasses import asdict, dataclass


@dataclass
class TrajectoryMetrics:
    """Summary metrics for a single rollout against a reference path."""
    rms_lateral_error: float = 0.0
    max_lateral_error: float = 0.0
    completion_rate: float = 0.0
    mean_speed_error: float = 0.0
    max_yaw_rate: float = 0.0
    max_roll_angle: float = 0.0
    max_pitch_angle: float = 0.0
    off_track_time: float = 0.0
    n_samples: int = 0
    duration: float = 0.0


def quat_to_euler_xyz(qw: float, qx: float, qy: float, qz: float) -> tuple[float, float, float]:
    """Quaternion (w, x, y, z) → roll, pitch, yaw (radians, ZYX intrinsic)."""
    # roll (X)
    sinr_cosp = 2 * (qw * qx + qy * qz)
    cosr_cosp = 1 - 2 * (qx * qx + qy * qy)
    roll = math.atan2(sinr_cosp, cosr_cosp)
    # pitch (Y)
    sinp = 2 * (qw * qy - qz * qx)
    pitch = math.copysign(math.pi / 2, sinp) if abs(sinp) >= 1 else math.asin(sinp)
    # yaw (Z)
    siny_cosp = 2 * (qw * qz + qx * qy)
    cosy_cosp = 1 - 2 * (qy * qy + qz * qz)
    yaw = math.atan2(siny_cosp, cosy_cosp)
    return roll, pitch, yaw


def metrics_from_csv(csv_path: str, target_speed: float | None = None,
                     reference_path: list[tuple[float, float]] | None = None,
                     off_track_threshold: float = 1.0) -> TrajectoryMetrics:
    """Compute summary metrics from a per-step CSV (t, x, y, z, qw, qx, qy, qz, vx_world, vy_world, wz_world)."""
    rows: list[dict[str, float]] = []
    with open(csv_path, "r") as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append({k: float(v) for k, v in r.items()})
    if not rows:
        return TrajectoryMetrics()

    t = [r["t"] for r in rows]
    duration = t[-1] - t[0]
    n = len(rows)
    out = TrajectoryMetrics(n_samples=n, duration=duration)

    rolls, pitches, yaws = [], [], []
    speeds = []
    yaw_rates = []
    for r in rows:
        roll, pitch, yaw = quat_to_euler_xyz(r["qw"], r["qx"], r["qy"], r["qz"])
        rolls.append(roll); pitches.append(pitch); yaws.append(yaw)
        speeds.append(math.hypot(r["vx_world"], r["vy_world"]))
        yaw_rates.append(abs(r["wz_world"]))
    out.max_yaw_rate = max(yaw_rates)
    out.max_roll_angle = max(abs(r) for r in rolls)
    out.max_pitch_angle = max(abs(p) for p in pitches)

    if target_speed is not None:
        out.mean_speed_error = sum(abs(s - target_speed) for s in speeds) / n

    if reference_path is not None and len(reference_path) >= 2:
        lat_errs = []
        for r in rows:
            lat_errs.append(_lateral_error(r["x"], r["y"], reference_path))
        out.rms_lateral_error = math.sqrt(sum(e * e for e in lat_errs) / n)
        out.max_lateral_error = max(lat_errs)
        # Completion = reached final waypoint within tolerance (3 m)
        last_wp = reference_path[-1]
        final_dist = math.hypot(rows[-1]["x"] - last_wp[0], rows[-1]["y"] - last_wp[1])
        out.completion_rate = 1.0 if final_dist < 3.0 else 0.0
        # Off-track time = sum of dt where |lat| > threshold
        if len(t) > 1:
            dt = (t[-1] - t[0]) / (n - 1)
            out.off_track_time = sum(dt for e in lat_errs if e > off_track_threshold)
    return out


def _lateral_error(x: float, y: float, path: list[tuple[float, float]]) -> float:
    """Minimum perpendicular distance to a polyline."""
    best = float("inf")
    for i in range(len(path) - 1):
        x1, y1 = path[i]
        x2, y2 = path[i + 1]
        dx, dy = x2 - x1, y2 - y1
        seg2 = dx * dx + dy * dy
        if seg2 < 1e-12:
            d = math.hypot(x - x1, y - y1)
        else:
            t = max(0.0, min(1.0, ((x - x1) * dx + (y - y1) * dy) / seg2))
            px, py = x1 + t * dx, y1 + t * dy
            d = math.hypot(x - px, y - py)
        if d < best:
            best = d
    return best


def write_metrics_json(metrics: TrajectoryMetrics, out_path: str) -> None:
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(asdict(metrics), f, indent=2)
