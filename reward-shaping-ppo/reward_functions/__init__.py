from typing import Any

from reward_functions.base import RewardShaper
from reward_functions.dense import DenseRewardShaper
from reward_functions.identity import IdentityRewardShaper


def get_reward_shaper(strategy_name: str, params: dict[str, Any] | None = None) -> RewardShaper:
    """
    Factory function to retrieve a RewardShaper instance by name.

    Args:
        strategy_name: The identifier for the reward shaping strategy (e.g. "identity").
        params: Optional dictionary of hyperparameters for the strategy.

    Returns:
        An instantiated RewardShaper object.

    Raises:
        ValueError: If an unknown strategy name is provided.
    """
    if params is None:
        params = {}

    name = strategy_name.lower().strip()

    if name == "identity":
        return IdentityRewardShaper()
    elif name == "dense":
        return DenseRewardShaper(**params)
    else:
        raise ValueError(f"Unknown reward shaping strategy: '{strategy_name}'")
