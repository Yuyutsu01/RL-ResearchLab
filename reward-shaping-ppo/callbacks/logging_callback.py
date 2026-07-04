from collections import deque

from stable_baselines3.common.callbacks import BaseCallback


class ResearchLoggingCallback(BaseCallback):
    """
    Custom callback for Stable-Baselines3 that logs unshaped (original) and
    shaped reward metrics to TensorBoard and stdout.

    Tracks rolling averages of episode statistics using a moving window.
    """

    def __init__(self, rolling_window: int = 100, verbose: int = 0):
        """
        Initializes the ResearchLoggingCallback.

        Args:
            rolling_window: Size of the moving window for averaging episode statistics.
            verbose: Verbosity level.
        """
        super().__init__(verbose)
        self.rolling_window = rolling_window
        self.original_rewards: deque[float] = deque(maxlen=rolling_window)
        self.shaped_rewards: deque[float] = deque(maxlen=rolling_window)
        self.episode_lengths: deque[int] = deque(maxlen=rolling_window)
        self.episode_count = 0

    def _on_step(self) -> bool:
        """
        Runs on each step of the environment. Intercepts transition diagnostics
        to check for completed episodes and log metrics.
        """
        # Retrieve the environment infos list from the model local variables
        infos = self.locals.get("infos", [])

        for info in infos:
            # SB3 Monitor adds 'episode' to info upon termination/truncation
            if "episode" in info:
                orig_rew = info.get("original_reward")
                shape_rew = info.get("shaped_reward")
                ep_len = info["episode"].get("l")

                if orig_rew is not None:
                    self.original_rewards.append(orig_rew)
                if shape_rew is not None:
                    self.shaped_rewards.append(shape_rew)
                if ep_len is not None:
                    self.episode_lengths.append(ep_len)

                self.episode_count += 1

        # Record metrics if at least one episode has completed
        if len(self.original_rewards) > 0:
            self.logger.record(
                "rollout/ep_original_rew_mean",
                sum(self.original_rewards) / len(self.original_rewards),
            )
        if len(self.shaped_rewards) > 0:
            self.logger.record(
                "rollout/ep_shaped_rew_mean",
                sum(self.shaped_rewards) / len(self.shaped_rewards),
            )
        if len(self.episode_lengths) > 0:
            self.logger.record(
                "rollout/ep_len_mean",
                sum(self.episode_lengths) / len(self.episode_lengths),
            )

        return True
