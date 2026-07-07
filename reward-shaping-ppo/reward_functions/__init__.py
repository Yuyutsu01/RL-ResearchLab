from typing import Any

from reward_functions.base import RewardShaper
from reward_functions.dense import (
    AcrobotDenseRewardShaper,
    CartPoleDenseRewardShaper,
    LunarLanderDenseRewardShaper,
    MountainCarDenseRewardShaper,
)
from reward_functions.identity import IdentityRewardShaper


def get_reward_shaper(
    strategy_name: str,
    params: dict[str, Any] | None = None,
    env_id: str | None = None,
) -> RewardShaper:
    """
    Factory function to retrieve a RewardShaper instance by name and environment.

    Args:
        strategy_name: The identifier for the reward shaping strategy (e.g. "identity").
        params: Optional dictionary of hyperparameters for the strategy.
        env_id: The environment identifier (e.g. "MountainCar-v0").

    Returns:
        An instantiated RewardShaper object.

    Raises:
        ValueError: If an unknown strategy name is provided.
    """
    if params is None:
        params = {}

    name = strategy_name.lower().strip()
    if "tuning/" in name:
        name = "identity"

    if name == "identity":
        return IdentityRewardShaper()
    elif name == "dense":
        if env_id == "MountainCar-v0":
            return MountainCarDenseRewardShaper(**params)
        elif env_id == "Acrobot-v1":
            return AcrobotDenseRewardShaper(**params)
        elif env_id in ["LunarLander-v2", "LunarLander-v3"]:
            return LunarLanderDenseRewardShaper(**params)
        else:
            return CartPoleDenseRewardShaper(**params)
    elif name == "pbrs":
        from reward_functions.pbrs import (
            AcrobotPBRSRewardShaper,
            CartPolePBRSRewardShaper,
            LunarLanderPBRSRewardShaper,
            MountainCarPBRSRewardShaper,
        )

        if env_id == "MountainCar-v0":
            return MountainCarPBRSRewardShaper(**params)
        elif env_id == "Acrobot-v1":
            return AcrobotPBRSRewardShaper(**params)
        elif env_id in ["LunarLander-v2", "LunarLander-v3"]:
            return LunarLanderPBRSRewardShaper(**params)
        else:
            return CartPolePBRSRewardShaper(**params)
    else:
        raise ValueError(f"Unknown reward shaping strategy: '{strategy_name}'")
