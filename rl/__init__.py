"""Yu-Gi-Oh! RL training system with PPO.

Requires: pip install -e ".[train]"
"""

from .config import TrainingConfig

__all__ = [
    "TrainingConfig",
    "TrainingEnv",
    "SubprocVecEnv",
    "YuGiOhNet",
    "PPOTrainer",
]


def __getattr__(name: str):
    if name == "TrainingEnv":
        from .env_wrapper import TrainingEnv

        return TrainingEnv
    if name == "SubprocVecEnv":
        from .env_wrapper import SubprocVecEnv

        return SubprocVecEnv
    if name == "YuGiOhNet":
        from .network import YuGiOhNet

        return YuGiOhNet
    if name == "PPOTrainer":
        from .ppo import PPOTrainer

        return PPOTrainer
    raise AttributeError(f"module 'rl' has no attribute {name!r}")
