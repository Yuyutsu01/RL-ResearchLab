from abc import ABC, abstractmethod
from typing import Any


class RewardShaper(ABC):
    """
    Abstract Base Class for modular reward shaping strategies.

    All custom reward shaping methods must inherit from this class and
    implement the reset and shape_reward methods.
    """

    @abstractmethod
    def reset(self, initial_state: Any, info: dict[str, Any]) -> None:
        """
        Resets the internal state of the reward shaper. Called at the start
        of each new episode during environment reset.

        Args:
            initial_state: The starting observation of the environment.
            info: Diagnostic information from the environment reset.
        """
        pass

    @abstractmethod
    def shape_reward(
        self, state: Any, action: Any, reward: float, next_state: Any, done: bool, info: dict[str, Any]
    ) -> float:
        """
        Computes the shaped reward for a single transitions.

        Args:
            state: The observation before the action was taken.
            action: The action executed by the agent.
            reward: The raw, unshaped reward returned by the environment.
            next_state: The observation after the action was taken.
            done: A boolean indicating whether the episode has ended
                (terminated or truncated).
            info: Diagnostic information dictionary from the environment step.

        Returns:
            The modified (shaped) reward float.
        """
        pass
