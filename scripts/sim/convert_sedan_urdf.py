"""Convert assets/urdf/sedan.urdf -> assets/usd/sedan.usd via Isaac Lab.

Run from repository root:
    python scripts/sim/convert_sedan_urdf.py [--force]

Mirrors the converter settings used for the Phase 1a (legacy) artefact
(`legacy/phase1a/assets/usd/config.yaml`), with Phase 1 (case-B) tweaks:
  - collision_from_visuals = False  (we deliberately omit <collision> on wheels;
    we do NOT want the visual cylinder to be auto-promoted to a collider)
  - replace_cylinders_with_capsules = False
  - merge_fixed_joints = False (keep the steer-link / wheel-link split for
    actuator routing)
"""
import argparse
import os

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="URDF -> USD conversion for sedan.urdf")
parser.add_argument("--force", action="store_true",
                    help="Regenerate USD even if it exists")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# Conversion does not need cameras.
args_cli.enable_cameras = False

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# --- Imports that require the sim app to be running ---
from isaaclab.sim.converters import UrdfConverter, UrdfConverterCfg

from vehicle_rl import URDF_DIR, USD_DIR


def main():
    urdf_path = os.path.join(URDF_DIR, "sedan.urdf")
    if not os.path.exists(urdf_path):
        raise FileNotFoundError(f"URDF not found: {urdf_path}")

    os.makedirs(USD_DIR, exist_ok=True)

    cfg = UrdfConverterCfg(
        asset_path=urdf_path,
        usd_dir=USD_DIR,
        usd_file_name="sedan.usd",
        force_usd_conversion=bool(args_cli.force),
        make_instanceable=True,
        fix_base=False,
        merge_fixed_joints=False,
        convert_mimic_joints_to_normal_joints=False,
        joint_drive=UrdfConverterCfg.JointDriveCfg(
            drive_type="force",
            target_type="position",
            gains=UrdfConverterCfg.JointDriveCfg.PDGainsCfg(
                stiffness=1000.0,
                damping=50.0,
            ),
        ),
        collider_type="convex_hull",
        self_collision=False,
        replace_cylinders_with_capsules=False,
        collision_from_visuals=False,  # Phase 1 (case-B): wheels are visual-only
    )

    print(f"[INFO] Converting {urdf_path}")
    print(f"[INFO]   -> {os.path.join(USD_DIR, 'sedan.usd')}")
    print(f"[INFO]   force_usd_conversion = {cfg.force_usd_conversion}")

    converter = UrdfConverter(cfg)
    print(f"[INFO] Done. USD path: {converter.usd_path}")


if __name__ == "__main__":
    main()
    simulation_app.close()
