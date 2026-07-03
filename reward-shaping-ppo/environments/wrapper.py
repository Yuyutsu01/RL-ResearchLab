from typing import Any

import gymnasium as gym

from reward_functions.base import RewardShaper


class RewardShapingWrapper(gym.Wrapper):
    """
    A Gymnasium environment wrapper that intercepts rewards and shapes them
    using a modular RewardShaper.

    Tracks cumulative original and shaped rewards for evaluation and logs
    these to the 'info' dictionary at the end of each episode.
    """

    def __init__(self, env: gym.Env, reward_shaper: RewardShaper):
        """
        Initializes the RewardShapingWrapper.

        Args:
            env: The raw Gymnasium environment.
            reward_shaper: The RewardShaper instance to apply to transitions.
        """
        super().__init__(env)
        self.reward_shaper = reward_shaper

        # Track the state before the step action is executed
        self._current_obs: Any | None = None

        # Running accumulators for evaluation reporting
        self._episode_original_reward: float = 0.0
        self._episode_shaped_reward: float = 0.0
        self._episode_length: int = 0

    def reset(self, **kwargs) -> tuple[Any, dict[str, Any]]:
        """
        Resets the environment and the reward shaper.

        Args:
            **kwargs: Arguments passed to the underlying reset method.

        Returns:
            The initial observation and info dictionary.
        """
        obs, info = self.env.reset(**kwargs)
        self._current_obs = obs

        # Reset internal reward shaping metrics or baseline states (e.g. potentials)
        self.reward_shaper.reset(obs, info)

        # Reset trackers
        self._episode_original_reward = 0.0
        self._episode_shaped_reward = 0.0
        self._episode_length = 0

        return obs, info

    def step(self, action: Any) -> tuple[Any, float, bool, bool, dict[str, Any]]:
        """
        Steps the environment, shapes the reward, and accumulates statistics.

        Args:
            action: The action selected by the agent.

        Returns:
            A tuple of (next observation, shaped reward, terminated, truncated, info dict).
        """
        next_obs, reward, terminated, truncated, info = self.env.step(action)
        done = terminated or truncated

        # Shape the reward using the transition details
        shaped_reward = self.reward_shaper.shape_reward(
            state=self._current_obs, action=action, reward=reward, next_state=next_obs, done=done, info=info
        )

        # Accumulate metrics
        self._episode_original_reward += reward
        self._episode_shaped_reward += shaped_reward
        self._episode_length += 1

        # Cache step-level metrics in the info dictionary
        info["step_original_reward"] = reward
        info["step_shaped_reward"] = shaped_reward

        # When the episode ends, record the aggregated episode-level values
        if done:
            info["original_reward"] = self._episode_original_reward
            info["shaped_reward"] = self._episode_shaped_reward
            info["episode_length"] = self._episode_length

        # Update current observation for the next step
        self._current_obs = next_obs

        return next_obs, shaped_reward, terminated, truncated, info
