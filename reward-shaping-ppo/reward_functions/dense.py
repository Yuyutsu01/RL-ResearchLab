from typing import Any

from reward_functions.base import RewardShaper


class DenseRewardShaper(RewardShaper):
    """
    Dense Reward Shaper for Gymnasium CartPole-v1.

    Modifies the default environment reward by subtracting penalties
    proportional to the cart's displacement from the center of the track
    and the pole's angular tilt from the vertical equilibrium line.

    Mathematical Formulation:
        R_shaped = R_original + max_bonus - (position_weight * |x'| + angle_weight * |theta'|)
    """

    def __init__(self, position_weight: float = 0.1, angle_weight: float = 1.0, max_bonus: float = 0.0):
        """
        Initializes the DenseRewardShaper.

        Args:
            position_weight: Scaling factor for cart displacement penalty.
            angle_weight: Scaling factor for pole angular tilt penalty.
            max_bonus: Optional constant bonus added at every step.
        """
        self.position_weight = position_weight
        self.angle_weight = angle_weight
        self.max_bonus = max_bonus

    def reset(self, initial_state: Any, info: dict[str, Any]) -> None:
        """No state variables are maintained across steps in this strategy."""
        pass

    def shape_reward(
        self, state: Any, action: Any, reward: float, next_state: Any, done: bool, info: dict[str, Any]
    ) -> float:
        """
        Transforms the raw environment reward to a dense signal.

        Args:
            state: The cart-pole state [x, v, theta, omega] at step t.
            action: The action (left/right push) applied.
            reward: The raw step reward (default is 1.0).
            next_state: The state [x', v', theta', omega'] at step t+1.
            done: Termination/truncation flag.
            info: Diagnostic dict.

        Returns:
            The shaped dense reward as a float.
        """
        # Extract next state coordinates:
        # Index 0: Cart Position (x)
        # Index 2: Pole Angle (theta in radians)
        cart_pos = next_state[0]
        pole_angle = next_state[2]

        # Calculate displacement and tilt penalty terms
        pos_penalty = self.position_weight * abs(cart_pos)
        angle_penalty = self.angle_weight * abs(pole_angle)

        # Compute final shaped reward
        shaped_reward = reward + self.max_bonus - (pos_penalty + angle_penalty)

        return float(shaped_reward)
