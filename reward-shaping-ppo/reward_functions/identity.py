from typing import Any

from reward_functions.base import RewardShaper


class IdentityRewardShaper(RewardShaper):
    """
    Identity Reward Shaper baseline.

    Returns the original environment reward without any modification.
    Used as the control/baseline strategy for comparative studies.
    """

    def reset(self, initial_state: Any, info: dict[str, Any]) -> None:
        """No internal state to reset."""
        pass

    def shape_reward(
        self,
        state: Any,
        action: Any,
        reward: float,
        next_state: Any,
        done: bool,
        info: dict[str, Any],
    ) -> float:
        """Passes the original reward through directly."""
        return reward
