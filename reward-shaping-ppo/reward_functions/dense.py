import math
from typing import Any

from reward_functions.base import RewardShaper


class CartPoleDenseRewardShaper(RewardShaper):
    """
    Dense Reward Shaper for Gymnasium CartPole-v1.

    Modifies the default environment reward by subtracting penalties
    proportional to the cart's displacement from the center of the track
    and the pole's angular tilt from the vertical equilibrium line.

    Mathematical Formulation:
        R_shaped = R_original + max_bonus - (position_weight * |x'| + angle_weight * |theta'|)
    """

    def __init__(
        self,
        position_weight: float = 0.1,
        angle_weight: float = 1.0,
        max_bonus: float = 0.0,
    ):
        self.position_weight = position_weight
        self.angle_weight = angle_weight
        self.max_bonus = max_bonus

    def reset(self, initial_state: Any, info: dict[str, Any]) -> None:
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
        # Extract next state coordinates:
        cart_pos = next_state[0]
        pole_angle = next_state[2]

        pos_penalty = self.position_weight * abs(cart_pos)
        angle_penalty = self.angle_weight * abs(pole_angle)

        shaped_reward = reward + self.max_bonus - (pos_penalty + angle_penalty)
        return float(shaped_reward)


class MountainCarDenseRewardShaper(RewardShaper):
    """
    Dense Reward Shaper for Gymnasium MountainCar-v0.

    Adds continuous rewards based on the height (potential energy)
    and velocity (kinetic energy) to guide exploration.
    """

    def __init__(self, height_weight: float = 1.0, velocity_weight: float = 10.0):
        self.height_weight = height_weight
        self.velocity_weight = velocity_weight

    def reset(self, initial_state: Any, info: dict[str, Any]) -> None:
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
        position = next_state[0]
        velocity = next_state[1]

        height_bonus = self.height_weight * (math.sin(3.0 * position) + 1.0)
        vel_bonus = self.velocity_weight * (velocity**2)

        shaped_reward = reward + height_bonus + vel_bonus
        return float(shaped_reward)


class AcrobotDenseRewardShaper(RewardShaper):
    """
    Dense Reward Shaper for Gymnasium Acrobot-v1.

    Adds rewards based on the pendulum tip height and link velocities
    to facilitate swing-up dynamics.
    """

    def __init__(self, height_weight: float = 1.0, velocity_weight: float = 0.1):
        self.height_weight = height_weight
        self.velocity_weight = velocity_weight

    def reset(self, initial_state: Any, info: dict[str, Any]) -> None:
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
        cos_theta1 = next_state[0]
        sin_theta1 = next_state[1]
        cos_theta2 = next_state[2]
        sin_theta2 = next_state[3]
        theta1_dot = next_state[4]
        theta2_dot = next_state[5]

        # tip height = -cos(theta1) - cos(theta1 + theta2)
        tip_height = -cos_theta1 - (cos_theta1 * cos_theta2 - sin_theta1 * sin_theta2)
        height_bonus = self.height_weight * (tip_height + 2.0)
        vel_bonus = self.velocity_weight * (theta1_dot**2 + theta2_dot**2)

        shaped_reward = reward + height_bonus + vel_bonus
        return float(shaped_reward)


class LunarLanderDenseRewardShaper(RewardShaper):
    """
    Dense Reward Shaper for Gymnasium LunarLander-v2.

    Adds continuous penalties for horizontal drift, velocity, and tilt angle.
    """

    def __init__(
        self,
        distance_weight: float = 0.5,
        velocity_weight: float = 0.5,
        angle_weight: float = 1.0,
    ):
        self.distance_weight = distance_weight
        self.velocity_weight = velocity_weight
        self.angle_weight = angle_weight

    def reset(self, initial_state: Any, info: dict[str, Any]) -> None:
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
        x = next_state[0]
        vx = next_state[2]
        vy = next_state[3]
        theta = next_state[4]

        dist_penalty = self.distance_weight * abs(x)
        vel_penalty = self.velocity_weight * (abs(vx) + abs(vy))
        angle_penalty = self.angle_weight * abs(theta)

        shaped_reward = reward - dist_penalty - vel_penalty - angle_penalty
        return float(shaped_reward)
