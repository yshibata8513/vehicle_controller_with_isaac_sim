"""Sedan ArticulationCfg for vehicle_rl (case-B / Phase 1).

SEDAN_CFG is the real-vehicle-scale chassis used throughout the project:
  - PhysX rigid body for the chassis (base_link)
  - Visual-only wheels (no collision in URDF)
  - Steering joints kept for actuator integration (Phase 1.5 sends position
    targets after a first-order lag on delta_target)
  - Spin joints kept as state variables (used in Phase 1.5 step 2 for slip
    ratio computation; in step 1 they freewheel and are not load-bearing)

Tire forces are injected externally by `vehicle_rl.dynamics`. This module
only defines the articulation; it does not compute or apply forces.

The Phase 1a (legacy) `SEDAN_MIN_CFG` (4 cylinder wheels with PhysX contact)
has been moved to `legacy/phase1a/sedan_min_cfg.py`.
"""
import os

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg

from vehicle_rl import USD_DIR


# Joint name conventions (must match assets/urdf/sedan.urdf):
STEER_JOINT_REGEX = "front_(left|right)_steer_joint"
WHEEL_JOINT_REGEX = "(front|rear)_(left|right)_wheel_joint"

# Vehicle dimensions (kept in sync with the URDF):
WHEELBASE = 2.7        # m, front-rear axle distance
TRACK = 1.55           # m, left-right wheel distance
WHEEL_RADIUS = 0.33    # m
WHEEL_WIDTH = 0.225    # m
TOTAL_MASS = 1500.0    # kg (held in base_link; wheels are 0.1 kg dummies)
COG_Z_DEFAULT = 0.55   # m, base_link origin height at static ride height


SEDAN_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=os.path.join(USD_DIR, "sedan.usd"),
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            rigid_body_enabled=True,
            max_linear_velocity=100.0,
            max_angular_velocity=100.0,
            max_depenetration_velocity=10.0,
            enable_gyroscopic_forces=True,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False,
            solver_position_iteration_count=8,
            solver_velocity_iteration_count=1,
            sleep_threshold=0.005,
            stabilization_threshold=0.001,
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        # base_link CoG at static ride height. With Phase 1 gravity OFF and
        # case-B wheels having no collision, the chassis box bottom sits at
        # z = 0.55 - 0.35 = 0.20 m above the ground (no contact).
        pos=(0.0, 0.0, COG_Z_DEFAULT),
        joint_pos={
            "front_left_steer_joint": 0.0,
            "front_right_steer_joint": 0.0,
            "front_left_wheel_joint": 0.0,
            "front_right_wheel_joint": 0.0,
            "rear_left_wheel_joint": 0.0,
            "rear_right_wheel_joint": 0.0,
        },
    ),
    actuators={
        # Steering: position-controlled with stiff PD. Phase 1.5 will write
        # delta_actual (after a first-order lag) into the position target.
        "steer": ImplicitActuatorCfg(
            joint_names_expr=[STEER_JOINT_REGEX],
            stiffness=8000.0,
            damping=400.0,
            effort_limit_sim=500.0,
            velocity_limit_sim=3.0,
            friction=0.1,
        ),
        # Wheels: free-spin in step 1 (no torque injected by the tire model).
        # In step 2 (Fiala / slip ratio), torque is written via
        # set_joint_effort_target on these joints.
        "wheels": ImplicitActuatorCfg(
            joint_names_expr=[WHEEL_JOINT_REGEX],
            stiffness=0.0,
            damping=0.0,
            effort_limit_sim=400.0,
            velocity_limit_sim=200.0,
            friction=0.0,
        ),
    },
)
