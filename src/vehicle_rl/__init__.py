"""vehicle_rl: Vehicle motion-control RL on Isaac Sim / Isaac Lab."""
import os

VEHICLE_RL_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
ASSETS_DIR = os.path.join(VEHICLE_RL_ROOT, "assets")
USD_DIR = os.path.join(ASSETS_DIR, "usd")
URDF_DIR = os.path.join(ASSETS_DIR, "urdf")
