"""Stateful physics wrapper: PhysX articulation + case-B dynamics modules.

Owns:
  - `sim` (SimulationContext) and `sedan` (Articulation) handles
  - first-order lag actuator state (pinion, a_x)
  - SteeringModel (pinion -> tire angle)
  - dynamics modules (StaticNormalLoadModel, LinearFrictionCircleTire,
    AttitudeDamper)
  - per-env per-wheel mu (Tensor)
  - centripetal feedback `a_y_estimate` for the normal-load model

Public surface:
  - `reset(env_ids=None, initial_pose=None) -> VehicleStateGT`
  - `step(action: VehicleAction) -> VehicleStateGT`
  - `mu` property (writable, shape (N, 4))

Phase 3's RL env (ManagerBased or Direct, decided in that phase) wraps this
class to provide the gym I/F. Path-following errors live in
`envs.sensors`, not here -- they are functions of (state, plan).
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from isaaclab.assets import Articulation
from isaaclab.sim import SimulationContext

from vehicle_rl.assets import (
    COG_Z_DEFAULT,
    DELTA_MAX_RAD,
    TOTAL_MASS,
    TRACK,
    WHEELBASE,
)
from vehicle_rl.dynamics import (
    AttitudeDamper,
    FirstOrderLagActuator,
    FixedRatioSteeringModel,
    LinearFrictionCircleTire,
    StaticNormalLoadModel,
    VehicleState,
    aggregate_tire_forces_to_base_link,
    quat_wxyz_to_rotmat,
    quat_wxyz_to_rpy,
)
from vehicle_rl.envs.types import VehicleAction, VehicleStateGT


@dataclass
class _SimulatorJointIds:
    """Joint and body indices, discovered once at construction time.

    Wheel order convention follows PLAN.md / dynamics modules: [FL, FR, RL, RR].
    """
    base: int
    fl_steer: int
    fr_steer: int
    fl_wheel: int
    fr_wheel: int
    rl_wheel: int
    rr_wheel: int

    @property
    def steer_ids(self) -> list[int]:
        return [self.fl_steer, self.fr_steer]

    @property
    def wheel_ids(self) -> list[int]:
        return [self.fl_wheel, self.fr_wheel, self.rl_wheel, self.rr_wheel]


class VehicleSimulator:
    """Sedan physics core (case-B Phase 1.5 step 1)."""

    def __init__(
        self,
        sim: SimulationContext,
        sedan: Articulation,
        *,
        device: torch.device | str | None = None,
        # Steering chain (pinion -> tire angle)
        steering_ratio: float,
        # First-order lag time constants and initial value
        tau_steer: float,
        tau_drive: float,
        tau_brake: float,
        actuator_initial_value: float,
        # Tire model
        cornering_stiffness: float,
        eps_vlong: float,
        # Longitudinal force split per axle. Each value is one of
        # {"rear", "front", "four_wheel"}; the tire-frame Fx command is
        # routed accordingly. Adapter validates the strings.
        fx_split_accel: str,
        fx_split_brake: str,
        # Geometry: front-axle distance from CoG (vehicle YAML).
        a_front: float,
        # Static normal load (z-drift PD just suppresses integration drift)
        z_drift_kp: float,
        z_drift_kd: float,
        # Virtual roll/pitch damper (yaw is intentionally undamped)
        k_roll: float,
        c_roll: float,
        k_pitch: float,
        c_pitch: float,
        # Initial mu (callers can override via the `mu` property after construction)
        mu_default: float,
        gravity: float,
    ):
        self.sim = sim
        self.sedan = sedan
        self.num_envs = sedan.num_instances
        self.device = torch.device(device) if device is not None else sedan.device
        self.dt = sim.get_physics_dt()

        # Vehicle constants. WHEELBASE / TRACK / TOTAL_MASS / COG_Z_DEFAULT
        # come from the assets module (single source of truth synced to YAML);
        # `a_front` is now an explicit kwarg so non-symmetric weight
        # distributions actually surface (review finding 3).
        self._mass = TOTAL_MASS
        self._L = WHEELBASE
        self._T = TRACK
        self._h_cg = COG_Z_DEFAULT
        self._a_front = float(a_front)
        self._gravity = gravity

        # Longitudinal-force split discriminators (validated by adapter).
        self._fx_split_accel = str(fx_split_accel)
        self._fx_split_brake = str(fx_split_brake)

        # Joint / body discovery (PLAN wheel order: FL, FR, RL, RR)
        self.joints = _SimulatorJointIds(
            base=sedan.find_bodies("base_link")[0][0],
            fl_steer=sedan.find_joints("front_left_steer_joint")[0][0],
            fr_steer=sedan.find_joints("front_right_steer_joint")[0][0],
            fl_wheel=sedan.find_joints("front_left_wheel_joint")[0][0],
            fr_wheel=sedan.find_joints("front_right_wheel_joint")[0][0],
            rl_wheel=sedan.find_joints("rear_left_wheel_joint")[0][0],
            rr_wheel=sedan.find_joints("rear_right_wheel_joint")[0][0],
        )

        # Actuator chain
        self.steer_act = FirstOrderLagActuator(
            self.num_envs, self.device,
            tau_pos=tau_steer,
            initial_value=actuator_initial_value,
        )
        self.drive_act = FirstOrderLagActuator(
            self.num_envs, self.device,
            tau_pos=tau_drive, tau_neg=tau_brake,
            initial_value=actuator_initial_value,
        )
        self.steering = FixedRatioSteeringModel(steering_ratio)

        # Dynamics modules
        self.normal_load = StaticNormalLoadModel(
            mass=self._mass, wheelbase=self._L, track=self._T,
            h_cg=self._h_cg, a_front=self._a_front, z_ref=self._h_cg,
            gravity=gravity, z_drift_kp=z_drift_kp, z_drift_kd=z_drift_kd,
        )
        self.tire = LinearFrictionCircleTire(
            cornering_stiffness=cornering_stiffness,
            wheelbase=self._L, track=self._T, a_front=self._a_front, h_cg=self._h_cg,
            eps_vlong=eps_vlong,
        )
        self.attitude = AttitudeDamper(
            k_roll=k_roll, c_roll=c_roll, k_pitch=k_pitch, c_pitch=c_pitch,
        )

        # Internal Python state (not in PhysX)
        self._mu = torch.full((self.num_envs, 4), float(mu_default), device=self.device)
        self._a_y_estimate = torch.zeros(self.num_envs, device=self.device)

        # Pre-allocate constant tensors to avoid per-step allocations
        self._zeros = torch.zeros(self.num_envs, device=self.device)

        # Cached per-step force / accel state. step() refreshes these after the
        # dynamics computation; reset(env_ids=...) only slice-updates the
        # selected envs so untouched envs keep their last-step values. This is
        # what makes partial reset's returned VehicleStateGT correct.
        self._Fz_static_full = self._compute_static_Fz()              # (N, 4), constant
        self._Fz = self._Fz_static_full.clone()                       # (N, 4)
        self._Fx = torch.zeros(self.num_envs, 4, device=self.device)
        self._Fy = torch.zeros(self.num_envs, 4, device=self.device)
        self._slip_angle = torch.zeros(self.num_envs, 4, device=self.device)
        self._ax_body = torch.zeros(self.num_envs, device=self.device)
        self._ay_body = torch.zeros(self.num_envs, device=self.device)

    # ------------------------------------------------------------------
    # Public properties

    @property
    def mu(self) -> Tensor:
        """Per-env per-wheel friction coefficient, shape (N, 4)."""
        return self._mu

    @mu.setter
    def mu(self, value: Tensor) -> None:
        if value.shape != (self.num_envs, 4):
            raise ValueError(f"mu must be shape ({self.num_envs}, 4), got {tuple(value.shape)}")
        self._mu = value.to(self.device)

    @property
    def pinion_max(self) -> float:
        """Pinion-angle physical limit derived from URDF tire-angle limit."""
        return float(self.steering.delta_to_pinion(torch.tensor(DELTA_MAX_RAD)).item())

    # ------------------------------------------------------------------
    # Reset / step

    def reset(
        self,
        *,
        env_ids: Tensor | None = None,
        initial_pose: Tensor | None = None,
    ) -> VehicleStateGT:
        """Reset selected envs to default articulation state (or override pose).

        initial_pose: optional (M, 7) of [x, y, z, qw, qx, qy, qz]. If given,
        M must match `len(env_ids)` (or `num_envs` if env_ids is None).

        Returns the full (N, ...) `VehicleStateGT`. For partial reset,
        untouched envs retain their previous-step PhysX pose, actuator
        state, and cached force/accel; only `env_ids` are reverted to the
        rest pose with static Fz and zero Fx/Fy/slip/accel.
        """
        if env_ids is None:
            env_ids_t = torch.arange(self.num_envs, device=self.device)
        else:
            env_ids_t = env_ids.to(self.device)

        # Slice default state to the selected envs so partial reset doesn't
        # clobber unrelated envs.
        default_state = self.sedan.data.default_root_state[env_ids_t].clone()
        if initial_pose is not None:
            default_state[:, :7] = initial_pose.to(default_state)

        self.sedan.write_root_pose_to_sim(default_state[:, :7], env_ids=env_ids_t)
        self.sedan.write_root_velocity_to_sim(default_state[:, 7:], env_ids=env_ids_t)

        jp = self.sedan.data.default_joint_pos[env_ids_t].clone()
        jv = self.sedan.data.default_joint_vel[env_ids_t].clone()
        self.sedan.write_joint_state_to_sim(jp, jv, env_ids=env_ids_t)
        self.sedan.reset(env_ids=env_ids_t)

        # Reset Python-side internal state (slice-update only). Omit
        # `value=` so each actuator reverts to its construction-time
        # `initial_value` (sourced from `actuator_lag.initial_value` in YAML).
        self.steer_act.reset(env_ids=env_ids_t)
        self.drive_act.reset(env_ids=env_ids_t)
        self._a_y_estimate[env_ids_t] = 0.0

        # Cached force/accel: revert reset envs to static/zero, leave others.
        self._Fz[env_ids_t] = self._Fz_static_full[env_ids_t]
        self._Fx[env_ids_t] = 0.0
        self._Fy[env_ids_t] = 0.0
        self._slip_angle[env_ids_t] = 0.0
        self._ax_body[env_ids_t] = 0.0
        self._ay_body[env_ids_t] = 0.0

        return self._build_state_gt_from_cache()

    def step(self, action: VehicleAction) -> VehicleStateGT:
        """Advance physics by one sim_dt with the given action.

        Phase 2 / standalone scripts call this. Phase 3 RL env uses
        `apply_action_to_physx()` instead so the DirectRLEnv outer loop
        owns sim.step / scene.update (decimation handling).
        """
        self.apply_action_to_physx(action)
        self.sedan.write_data_to_sim()
        self.sim.step()
        self.sedan.update(self.dt)
        return self._build_state_gt_from_cache()

    def apply_action_to_physx(self, action: VehicleAction) -> None:
        """Compute tire forces from the action and write them to PhysX
        buffers (external force/torque + steer joint target). Does NOT
        advance the sim or refresh `sedan.data`.

        Intended for `DirectRLEnv._apply_action`, which is called once per
        physics substep inside the env-step decimation loop. The cache
        (`self._Fz / _Fx / _Fy / _slip_angle / _ax_body / _ay_body`) is
        updated here so a subsequent `_build_state_gt_from_cache()` returns
        forces consistent with the PhysX state read after the next
        `sim.step()`.

        Numerically identical to the body of `step()` minus the three
        sim-stepping calls; bit-equivalence is preserved for the Phase 2
        smoke regression test (forces are computed from pre-step state in
        both code paths).
        """
        # 1) Actuator first-order lag (Python-side)
        pinion_actual = self.steer_act.step(action.pinion_target, self.dt)
        a_x_actual = self.drive_act.step(action.a_x_target, self.dt)
        delta_actual = self.steering.pinion_to_delta(pinion_actual)

        # 2) Read PhysX state (pre-step) and build dynamics input
        veh_state = self._read_vehicle_state(delta_actual, a_x_actual)

        # 3) Tire normal load (Fz) and tangential forces (Fx, Fy + slip_angle)
        Fz = self.normal_load.compute(veh_state)
        fx_command = self._distribute_fx(a_x_actual)
        Fx, Fy = self.tire.compute(veh_state, Fz, self._mu, fx_command)
        slip_angle = self._compute_slip_angles(veh_state, delta_actual)

        # 4) Aggregate forces/torques onto base_link, add attitude damper
        delta_per_wheel = self.tire.per_wheel_steer(delta_actual)
        F_body, tau_body = aggregate_tire_forces_to_base_link(
            Fx, Fy, Fz, delta_per_wheel, self.tire.tire_positions_body().to(self.device)
        )
        tau_body = tau_body + self.attitude.compute(veh_state)

        # 5) Apply external force/torque (body frame) and steering joint target
        forces = F_body.unsqueeze(1)        # (N, 1, 3)
        torques = tau_body.unsqueeze(1)
        self.sedan.set_external_force_and_torque(forces, torques, body_ids=[self.joints.base])
        steer_target = torch.stack([delta_actual, delta_actual], dim=-1)   # (N, 2)
        self.sedan.set_joint_position_target(steer_target, joint_ids=self.joints.steer_ids)

        # 6) Update centripetal feedback (uses pre-step values, matching the
        #    Phase 1.5 reference implementation)
        v_long = veh_state.vel_body[..., 0]
        wz = veh_state.angvel_body[..., 2]
        self._a_y_estimate = wz * v_long

        # 7) Cache forces/accel for the next state_gt build. Body-frame
        # accel approximates IMU specific-force; for small roll/pitch
        # (Phase 1.5 satisfies this) gravity nearly cancels Fz so
        # F_body[..., :2] / m is a faithful IMU-side ax/ay.
        self._Fz = Fz
        self._Fx = Fx
        self._Fy = Fy
        self._slip_angle = slip_angle
        self._ax_body = F_body[..., 0] / self._mass
        self._ay_body = F_body[..., 1] / self._mass

    def get_state(self) -> VehicleStateGT:
        """Return current `VehicleStateGT` from cached forces + post-step
        PhysX pose. Use after `apply_action_to_physx()` + sim has advanced.
        """
        return self._build_state_gt_from_cache()

    # ------------------------------------------------------------------
    # Private helpers

    def _read_vehicle_state(self, delta_actual: Tensor, a_x_actual: Tensor) -> VehicleState:
        """Pack PhysX state into the dynamics-internal VehicleState."""
        rs = self.sedan.data.root_state_w   # (N, 13)
        pos_world = rs[:, 0:3]
        quat = rs[:, 3:7]
        vel_world = rs[:, 7:10]
        angvel_world = rs[:, 10:13]

        R = quat_wxyz_to_rotmat(quat)
        Rt = R.transpose(-1, -2)
        vel_body = (Rt @ vel_world.unsqueeze(-1)).squeeze(-1)
        angvel_body = (Rt @ angvel_world.unsqueeze(-1)).squeeze(-1)
        rpy = quat_wxyz_to_rpy(quat)

        omega_wheel = self.sedan.data.joint_vel[:, self.joints.wheel_ids]   # (N, 4)

        return VehicleState(
            pos_world=pos_world,
            quat_wxyz_world=quat,
            rot_body_to_world=R,
            vel_world=vel_world,
            vel_body=vel_body,
            angvel_body=angvel_body,
            rpy=rpy,
            delta_actual=delta_actual,
            omega_wheel=omega_wheel,
            a_x_actual=a_x_actual,
            a_y_estimate=self._a_y_estimate,
        )

    def _distribute_fx(self, a_x_actual: Tensor) -> Tensor:
        """Split tire-frame longitudinal force across [FL, FR, RL, RR] axles.

        The accel/brake split is YAML-driven: each branch routes Fx to one
        of {"rear", "front", "four_wheel"}. PR 2 round-1 fix replaces the
        previously hard-coded RWD-accel / 4WD-brake behavior with the
        adapter-validated `fx_split_accel` / `fx_split_brake` strings.
        Returns (N, 4) tire-frame longitudinal force command.
        """
        Fx_total = self._mass * a_x_actual               # (N,)
        is_drive = (a_x_actual >= 0.0).unsqueeze(-1)     # (N, 1)
        fx_drive = self._fx_split_to_per_wheel(Fx_total, self._fx_split_accel)
        fx_brake = self._fx_split_to_per_wheel(Fx_total, self._fx_split_brake)
        return torch.where(is_drive, fx_drive, fx_brake)

    @staticmethod
    def _fx_split_to_per_wheel(Fx_total: Tensor, split: str) -> Tensor:
        """Translate one of {"rear", "front", "four_wheel"} to (N, 4) [FL, FR, RL, RR]."""
        zero = torch.zeros_like(Fx_total)
        half = Fx_total * 0.5
        quarter = Fx_total * 0.25
        if split == "rear":
            return torch.stack([zero, zero, half, half], dim=-1)
        if split == "front":
            return torch.stack([half, half, zero, zero], dim=-1)
        if split == "four_wheel":
            return torch.stack([quarter, quarter, quarter, quarter], dim=-1)
        # The adapter validates this at construction time; this is a defensive
        # fallback so any drift between adapter and simulator fails loudly.
        raise ValueError(f"unknown longitudinal_force_split value: {split!r}")

    def _compute_slip_angles(self, veh_state: VehicleState, delta_actual: Tensor) -> Tensor:
        """Per-wheel slip angle (tire frame), shape (N, 4) [rad].

        Mirrors LinearFrictionCircleTire.compute for logging / GT purposes.
        """
        r_body = self.tire.tire_positions_body().to(self.device)         # (4, 3)
        v_body = veh_state.vel_body.unsqueeze(1)                          # (N, 1, 3)
        w_body = veh_state.angvel_body.unsqueeze(1)
        r = r_body.unsqueeze(0).expand(self.num_envs, -1, -1)             # (N, 4, 3)
        v_tire_body = v_body.expand(-1, 4, -1) + torch.cross(
            w_body.expand(-1, 4, -1), r, dim=-1
        )

        delta_per_wheel = self.tire.per_wheel_steer(delta_actual)         # (N, 4)
        cos_d = torch.cos(delta_per_wheel)
        sin_d = torch.sin(delta_per_wheel)
        v_long = cos_d * v_tire_body[..., 0] + sin_d * v_tire_body[..., 1]
        v_lat = -sin_d * v_tire_body[..., 0] + cos_d * v_tire_body[..., 1]
        denom = torch.clamp(torch.abs(v_long), min=1e-2)
        return torch.atan(v_lat / denom)

    def _compute_static_Fz(self) -> Tensor:
        """Initial Fz at rest (a_x = a_y = 0). Used by reset()."""
        m, g = self._mass, self._gravity
        Fz_static_front = m * g * (self._L - self._a_front) / self._L / 2.0
        Fz_static_rear = m * g * self._a_front / self._L / 2.0
        Fz = torch.tensor(
            [Fz_static_front, Fz_static_front, Fz_static_rear, Fz_static_rear],
            device=self.device,
        )
        return Fz.unsqueeze(0).expand(self.num_envs, -1).contiguous()

    def _build_state_gt_from_cache(self) -> VehicleStateGT:
        """Pack current PhysX state + cached force/accel into VehicleStateGT.

        PhysX-derived fields (pose, velocity, omega_wheel) are read fresh from
        `self.sedan.data`. Actuator internals come from `self.steer_act` /
        `self.drive_act`. Force/accel come from per-step caches that
        `step()` refreshes and `reset()` slice-updates -- this is what makes
        partial reset coherent across reset and untouched envs.
        """
        pinion_actual = self.steer_act.value
        delta_actual = self.steering.pinion_to_delta(pinion_actual)
        a_x_actual = self.drive_act.value

        rs = self.sedan.data.root_state_w   # (N, 13)
        pos_xyz = rs[:, 0:3]
        quat = rs[:, 3:7]
        vel_world = rs[:, 7:10]
        angvel_world = rs[:, 10:13]
        R = quat_wxyz_to_rotmat(quat)
        Rt = R.transpose(-1, -2)
        vel_body = (Rt @ vel_world.unsqueeze(-1)).squeeze(-1)
        angvel_body = (Rt @ angvel_world.unsqueeze(-1)).squeeze(-1)
        rpy = quat_wxyz_to_rpy(quat)
        omega_wheel = self.sedan.data.joint_vel[:, self.joints.wheel_ids]

        return VehicleStateGT(
            pos_xyz=pos_xyz,
            quat_wxyz=quat,
            vel_world=vel_world,
            angvel_world=angvel_world,
            vel_body=vel_body,
            angvel_body=angvel_body,
            rpy=rpy,
            ax_body=self._ax_body,
            ay_body=self._ay_body,
            pinion_actual=pinion_actual,
            delta_actual=delta_actual,
            a_x_actual=a_x_actual,
            mu_per_wheel=self._mu,
            Fz_per_wheel=self._Fz,
            Fx_per_wheel=self._Fx,
            Fy_per_wheel=self._Fy,
            slip_angle=self._slip_angle,
            omega_wheel=omega_wheel,
        )
