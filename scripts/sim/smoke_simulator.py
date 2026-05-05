"""Smoke test: verify VehicleSimulator matches the Phase 1.5 reference
(`run_phase1_5.py`) on the straight scenario.

With `steering_ratio=1.0` the new pinion-based action chain becomes
numerically equivalent to the reference's direct delta-based chain
(pinion_target == delta_target). Same physics modules, same dt, same
PhysX state -> we expect identical numbers up to float-precision noise.

Reference numbers (from PLAN.md / metrics/phase1_5_straight_mu0.90.json):
    max_roll = 0.0085 deg
    max_pitch = 1.2914 deg
    z_drift = 0.85 mm
    sum_Fz_relative_error_max = 0.31 %  (peaks 4-5% during pitch dive)

Run from repository root (venv python):
    PY="/c/work/isaac/env_isaaclab/Scripts/python.exe"
    $PY scripts/sim/smoke_simulator.py --headless
"""
import argparse
import json
import math
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="VehicleSimulator vs run_phase1_5 smoke test")
parser.add_argument("--duration", type=float, default=8.0)
parser.add_argument("--mu", type=float, default=0.9)
parser.add_argument("--reference_json", type=str,
                    default="metrics/phase1_5_straight_mu0.90.json",
                    help="Reference run_phase1_5.py output for diff")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.enable_cameras = False  # smoke test doesn't need video

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# --- imports below this line require the sim app ---
import os

import torch
import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from isaaclab.sim import SimulationContext

from vehicle_rl.assets import SEDAN_CFG, TOTAL_MASS, COG_Z_DEFAULT
from vehicle_rl.envs import VehicleAction
from vehicle_rl.envs.simulator import VehicleSimulator


GRAVITY = 9.81


def main():
    device = args_cli.device

    sim_cfg = sim_utils.SimulationCfg(
        device=device,
        dt=1.0 / 200.0,
        gravity=(0.0, 0.0, -GRAVITY),
    )
    sim = SimulationContext(sim_cfg)
    sim.set_camera_view([7.0, -10.0, 3.5], [0.0, 0.0, 0.5])

    # Scene: ground + light + sedan (no camera; this is a numeric smoke test)
    sim_utils.GroundPlaneCfg().func("/World/defaultGroundPlane", sim_utils.GroundPlaneCfg())
    sim_utils.DomeLightCfg(intensity=3000.0, color=(0.75, 0.75, 0.75)).func(
        "/World/Light", sim_utils.DomeLightCfg(intensity=3000.0, color=(0.75, 0.75, 0.75))
    )

    sedan_cfg = SEDAN_CFG.copy()
    sedan_cfg.prim_path = "/World/Sedan"
    sedan = Articulation(cfg=sedan_cfg)
    sim.reset()

    # PR 3: route through make_simulator_kwargs(adapter)
    # PR 2 round-1 fix: pass the full new VehicleSimulator kwarg set inline
    # so smoke_simulator keeps booting at PR 2 merge. steering_ratio=1.0 keeps
    # the apples-to-apples comparison with run_phase1_5.py reference.
    vsim = VehicleSimulator(
        sim, sedan,
        steering_ratio=1.0,
        tau_steer=0.05,
        tau_drive=0.20,
        tau_brake=0.07,
        actuator_initial_value=0.0,
        cornering_stiffness=60000.0,
        eps_vlong=0.01,
        fx_split_accel="rear",
        fx_split_brake="four_wheel",
        a_front=2.7 / 2.0,        # WHEELBASE / 2.0 (symmetric)
        z_drift_kp=50000.0,
        z_drift_kd=5000.0,
        k_roll=80000.0,
        c_roll=8000.0,
        k_pitch=80000.0,
        c_pitch=8000.0,
        mu_default=args_cli.mu,
        gravity=GRAVITY,
    )
    state_gt = vsim.reset()

    print(f"[INFO] num_envs={vsim.num_envs}  dt={vsim.dt}  pinion_max={vsim.pinion_max:.4f} rad", flush=True)
    print(f"[INFO] joint_names = {sedan.joint_names}", flush=True)
    print(f"[INFO] body_names  = {sedan.body_names}", flush=True)

    sim_dt = vsim.dt
    n_steps = int(args_cli.duration / sim_dt)
    mg = TOTAL_MASS * GRAVITY

    max_roll_deg = 0.0
    max_pitch_deg = 0.0
    max_z_drift = 0.0
    max_sum_fz_err = 0.0

    zero = torch.zeros(1, device=device)
    for step in range(n_steps):
        t = step * sim_dt
        # straight scenario from run_phase1_5.scenario_command:
        #   a_x = 2 for t < 5, else 0;  delta = 0
        a_x_t = 2.0 if t < 5.0 else 0.0
        action = VehicleAction(
            pinion_target=zero,                                  # delta_target=0 (parallel)
            a_x_target=torch.full((1,), a_x_t, device=device),
        )
        state_gt = vsim.step(action)

        roll_deg = abs(math.degrees(float(state_gt.rpy[0, 0].item())))
        pitch_deg = abs(math.degrees(float(state_gt.rpy[0, 1].item())))
        z = float(state_gt.pos_xyz[0, 2].item())
        sum_fz = float(state_gt.Fz_per_wheel.sum().item())

        max_roll_deg = max(max_roll_deg, roll_deg)
        max_pitch_deg = max(max_pitch_deg, pitch_deg)
        max_z_drift = max(max_z_drift, abs(z - COG_Z_DEFAULT))
        max_sum_fz_err = max(max_sum_fz_err, abs(sum_fz - mg) / mg)

    sim_results = {
        "max_roll_deg": max_roll_deg,
        "max_pitch_deg": max_pitch_deg,
        "max_z_drift_m": max_z_drift,
        "max_sum_Fz_relative_error": max_sum_fz_err,
    }

    # Load reference
    ref_path = args_cli.reference_json
    if not os.path.isabs(ref_path):
        ref_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), ref_path)
    if not os.path.exists(ref_path):
        print(f"[WARN] reference not found at {ref_path}; printing absolute results only", flush=True)
        ref = None
    else:
        with open(ref_path) as f:
            ref = json.load(f)

    print("=" * 72, flush=True)
    print(f"VehicleSimulator vs run_phase1_5 reference (straight, mu={args_cli.mu})", flush=True)
    print(f"  duration = {args_cli.duration} s, n_steps = {n_steps}", flush=True)
    print("-" * 72, flush=True)
    if ref is None:
        for k, v in sim_results.items():
            print(f"  {k:35s} sim {v:.6g}", flush=True)
    else:
        keys = ["max_roll_deg", "max_pitch_deg", "max_z_drift_m", "max_sum_Fz_relative_error"]
        worst_rel = 0.0
        for k in keys:
            sv = sim_results[k]
            rv = float(ref.get(k, 0.0))
            denom = max(abs(rv), 1e-9)
            rel = abs(sv - rv) / denom
            worst_rel = max(worst_rel, rel)
            print(f"  {k:35s} sim {sv:14.6g}  ref {rv:14.6g}  rel_diff {rel*100:7.3f}%", flush=True)
        # PASS threshold: 1% relative diff. Higher would suggest a non-trivial
        # behavioral diff (not just float noise). Adjust if needed.
        passed = worst_rel < 0.01
        print("-" * 72, flush=True)
        print(f"  worst relative diff = {worst_rel*100:.3f}%   PASS = {passed}", flush=True)
    print("=" * 72, flush=True)
    return 0 if (ref is None or worst_rel < 0.01) else 1


if __name__ == "__main__":
    rc = main()
    simulation_app.close()
    sys.exit(rc)
