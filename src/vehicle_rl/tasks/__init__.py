"""Phase 3 RL task registry.

Importing this module registers all `Vehicle-*` gym ids with gymnasium.
Phase 3 training scripts do `import vehicle_rl.tasks` to trigger
registration before `gym.make(...)`.
"""
from . import tracking  # noqa: F401
