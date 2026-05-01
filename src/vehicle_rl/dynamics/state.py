"""Vehicle state passed to dynamics submodels (case-B / Phase 1.5).

Wheel order convention: [FL, FR, RL, RR].
Frame convention (body): x forward, y left, z up. World: ENU-ish (z up).
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor


@dataclass
class VehicleState:
    pos_world: Tensor          # (N, 3) base_link world position
    quat_wxyz_world: Tensor    # (N, 4) wxyz quaternion (Isaac Lab convention)
    rot_body_to_world: Tensor  # (N, 3, 3) v_world = R @ v_body
    vel_world: Tensor          # (N, 3) linear velocity of CoG in world frame
    vel_body: Tensor           # (N, 3) linear velocity of CoG in body frame
    angvel_body: Tensor        # (N, 3) angular velocity (roll, pitch, yaw rates) in body frame
    rpy: Tensor                # (N, 3) roll, pitch, yaw extracted from rot
    delta_actual: Tensor       # (N,)   centerline steering angle (post first-order lag)
    omega_wheel: Tensor        # (N, 4) wheel spin rates [FL, FR, RL, RR] (rad/s)
    a_x_actual: Tensor         # (N,)   longitudinal accel command after first-order lag (m/s^2)
    a_y_estimate: Tensor       # (N,)   estimated lateral accel for Fz feedback (m/s^2)


def quat_wxyz_to_rotmat(q: Tensor) -> Tensor:
    """(N, 4) wxyz -> (N, 3, 3) rotation matrix (body -> world)."""
    w, x, y, z = q.unbind(dim=-1)
    R = torch.stack([
        torch.stack([1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)], dim=-1),
        torch.stack([2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)], dim=-1),
        torch.stack([2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)], dim=-1),
    ], dim=-2)  # (N, 3, 3)
    return R


def quat_wxyz_to_rpy(q: Tensor) -> Tensor:
    """(N, 4) wxyz -> (N, 3) roll, pitch, yaw."""
    w, x, y, z = q.unbind(dim=-1)
    sinr_cosp = 2 * (w * x + y * z)
    cosr_cosp = 1 - 2 * (x * x + y * y)
    roll = torch.atan2(sinr_cosp, cosr_cosp)
    sinp = torch.clamp(2 * (w * y - z * x), -1.0, 1.0)
    pitch = torch.asin(sinp)
    siny_cosp = 2 * (w * z + x * y)
    cosy_cosp = 1 - 2 * (y * y + z * z)
    yaw = torch.atan2(siny_cosp, cosy_cosp)
    return torch.stack([roll, pitch, yaw], dim=-1)
