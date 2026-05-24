"""Yu-Gi-Oh! RL training system with PPO.

Requires: pip install -e ".[train]"
"""

from yugioh_rl.config import TrainingConfig

__all__ = [
    "TrainingConfig",
    "TrainingEnv",
    "SubprocVecEnv",
    "YuGiOhNet",
    "PPOTrainer",
]


def __getattr__(name: str):
    if name == "TrainingEnv":
        from yugioh_rl.env_wrapper import TrainingEnv

        return TrainingEnv
    if name == "SubprocVecEnv":
        from yugioh_rl.env_wrapper import SubprocVecEnv

        return SubprocVecEnv
    if name == "YuGiOhNet":
        from yugioh_rl.network import YuGiOhNet

        return YuGiOhNet
    if name == "PPOTrainer":
        from yugioh_rl.ppo import PPOTrainer

        return PPOTrainer
    raise AttributeError(f"module 'yugioh_rl' has no attribute {name!r}")
