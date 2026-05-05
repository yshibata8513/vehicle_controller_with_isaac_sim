"""gym registry entry points for `Vehicle-Tracking-Direct-v0` (PR 3).

The legacy entry point shape was `vehicle_rl.envs.tracking_env:TrackingEnvCfg`
(a class), which Isaac Lab's `parse_env_cfg` calls with no args. With YAML
externalization (PR 3) all tunable defaults live in the experiment YAML, so
the entry point is now a 0-arg factory that:

  1. Reads the experiment YAML path from env var `VEHICLE_RL_EXPERIMENT_YAML`
     (set by `train_ppo.py` / `play.py` / future `--config` callers; the
     legacy `--course X` CLI on `train_ppo.py` is mapped to the matching
     `configs/experiments/rl/phase3_<course>.yaml` before this is read).
  2. Falls back to `configs/experiments/rl/phase3_circle_stage0a.yaml`
     when no env var is set, so a bare `gym.make(id)` still produces a
     fully-initialized cfg consistent with the YAML defaults.
  3. Calls `make_tracking_env_cfg(...)` / `make_ppo_runner_cfg(...)` to
     translate the resolved bundle into Isaac Lab cfg objects.

Splitting the factories out of `tracking/__init__.py` keeps the gym
registration import light: `import vehicle_rl.tasks` does not load the
adapter (which imports Isaac Lab); only `parse_env_cfg(task_name)` does.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any

# Default experiment YAML when no env-var is set. Stage 0a circle is the
# canonical "infra baseline" experiment per the YAML refactor plan.
_DEFAULT_EXPERIMENT_YAML = "configs/experiments/rl/phase3_circle_stage0a.yaml"

# Legacy `--course X` CLI mapping (used by train_ppo.py until PR 4 removes
# the legacy CLI). Each course name maps to the experiment YAML that
# reproduces the phase3 baseline for that course.
LEGACY_COURSE_TO_EXPERIMENT = {
    "circle": "configs/experiments/rl/phase3_circle_stage0a.yaml",
    "random_long": "configs/experiments/rl/phase3_random_long.yaml",
    "random_bank": "configs/experiments/rl/phase3_random_bank.yaml",
    # PR 3 round-2 fix F3: dedicated experiment YAMLs for s_curve / dlc /
    # lemniscate so `--course X` correctly selects the matching course
    # bundle (radius / target_speed / course_ds derive from the course
    # YAML the experiment refs). Previously these aliased to
    # phase3_circle_stage0a, which silently trained on circle.
    "s_curve": "configs/experiments/rl/phase3_s_curve.yaml",
    "dlc": "configs/experiments/rl/phase3_dlc.yaml",
    "lemniscate": "configs/experiments/rl/phase3_lemniscate.yaml",
}


def _repo_root() -> Path:
    # vehicle_rl/src/vehicle_rl/tasks/tracking/entry_points.py -> repo root
    return Path(__file__).resolve().parents[4]


def _resolve_experiment_path() -> Path:
    """Pick the experiment YAML for the current factory call.

    Resolution order:
      1. `VEHICLE_RL_EXPERIMENT_YAML` env var (absolute or repo-relative).
      2. `_DEFAULT_EXPERIMENT_YAML` (repo-relative).
    """
    repo_root = _repo_root()
    raw = os.environ.get("VEHICLE_RL_EXPERIMENT_YAML")
    if raw is None:
        raw = _DEFAULT_EXPERIMENT_YAML
    candidate = Path(raw)
    if not candidate.is_absolute():
        candidate = repo_root / candidate
    if not candidate.is_file():
        raise FileNotFoundError(
            f"experiment YAML not found: {candidate} "
            f"(VEHICLE_RL_EXPERIMENT_YAML={os.environ.get('VEHICLE_RL_EXPERIMENT_YAML')!r})"
        )
    return candidate


def _load_resolved_experiment() -> dict[str, Any]:
    from vehicle_rl.config.loader import load_experiment

    repo_root = _repo_root()
    return load_experiment(_resolve_experiment_path(), repo_root=repo_root)


def tracking_env_cfg_factory():
    """0-arg factory for `env_cfg_entry_point`. Returns a `TrackingEnvCfg`."""
    from vehicle_rl.config.isaac_adapter import make_tracking_env_cfg

    bundle = _load_resolved_experiment()
    return make_tracking_env_cfg(
        env_bundle=bundle["env"],
        course_bundle=bundle["course"],
        controller_bundle=bundle["env"]["speed_controller"]["controller"],
        vehicle_bundle=bundle["vehicle"],
        dynamics_bundle=bundle["dynamics"],
    )


def rsl_rl_runner_cfg_factory():
    """0-arg factory for `rsl_rl_cfg_entry_point`. Returns a `RslRlOnPolicyRunnerCfg`."""
    from vehicle_rl.config.isaac_adapter import make_ppo_runner_cfg

    bundle = _load_resolved_experiment()
    return make_ppo_runner_cfg(bundle["agent"])


__all__ = [
    "LEGACY_COURSE_TO_EXPERIMENT",
    "rsl_rl_runner_cfg_factory",
    "tracking_env_cfg_factory",
]
