import math
from typing import Any

from reward_functions.base import RewardShaper


class CartPolePBRSRewardShaper(RewardShaper):
    """
    Potential-Based Reward Shaping (PBRS) for Gymnasium CartPole-v1.

    Guarantees policy invariance while guiding exploration.

    Potential Function:
        Phi(s) = - (w_x * |x| + w_v * |v| + w_theta * |theta| + w_omega * |omega|)   [l1 type]
        Phi(s) = - (w_x * x^2 + w_v * v^2 + w_theta * theta^2 + w_omega * omega^2) [l2 type]
    """

    def __init__(
        self,
        position_weight: float = 0.1,
        velocity_weight: float = 0.0,
        angle_weight: float = 1.0,
        angular_velocity_weight: float = 0.0,
        gamma: float = 0.99,
        potential_type: str = "l1",
    ):
        self.position_weight = position_weight
        self.velocity_weight = velocity_weight
        self.angle_weight = angle_weight
        self.angular_velocity_weight = angular_velocity_weight
        self.gamma = gamma

        ptype = potential_type.lower().strip()
        if ptype not in ["l1", "l2"]:
            raise ValueError(
                f"potential_type must be 'l1' or 'l2', got '{potential_type}'"
            )
        self.potential_type = ptype

    def reset(self, initial_state: Any, info: dict[str, Any]) -> None:
        pass

    def _potential(self, state: Any) -> float:
        x = state[0]
        v = state[1]
        theta = state[2]
        omega = state[3]

        if self.potential_type == "l1":
            val = (
                self.position_weight * abs(x)
                + self.velocity_weight * abs(v)
                + self.angle_weight * abs(theta)
                + self.angular_velocity_weight * abs(omega)
            )
        else:
            val = (
                self.position_weight * (x**2)
                + self.velocity_weight * (v**2)
                + self.angle_weight * (theta**2)
                + self.angular_velocity_weight * (omega**2)
            )
        return float(-val)

    def shape_reward(
        self,
        state: Any,
        action: Any,
        reward: float,
        next_state: Any,
        done: bool,
        info: dict[str, Any],
    ) -> float:
        phi_s = self._potential(state)
        phi_s_prime = 0.0 if done else self._potential(next_state)

        shaping_reward = self.gamma * phi_s_prime - phi_s
        shaped_reward = reward + shaping_reward
        return float(shaped_reward)


class MountainCarPBRSRewardShaper(RewardShaper):
    """
    Potential-Based Reward Shaping (PBRS) for Gymnasium MountainCar-v0.

    Potential Function:
        Phi(s) = w_h * (sin(3*p) + 1.0) + w_v * v^2
    """

    def __init__(
        self,
        height_weight: float = 1.0,
        velocity_weight: float = 10.0,
        gamma: float = 0.99,
    ):
        self.height_weight = height_weight
        self.velocity_weight = velocity_weight
        self.gamma = gamma

    def reset(self, initial_state: Any, info: dict[str, Any]) -> None:
        pass

    def _potential(self, state: Any) -> float:
        position = state[0]
        velocity = state[1]
        val = self.height_weight * (
            math.sin(3.0 * position) + 1.0
        ) + self.velocity_weight * (velocity**2)
        return float(val)

    def shape_reward(
        self,
        state: Any,
        action: Any,
        reward: float,
        next_state: Any,
        done: bool,
        info: dict[str, Any],
    ) -> float:
        phi_s = self._potential(state)
        # Boundary condition: Phi(s_terminal) = 0.0
        phi_s_prime = 0.0 if done else self._potential(next_state)

        shaping_reward = self.gamma * phi_s_prime - phi_s
        shaped_reward = reward + shaping_reward
        return float(shaped_reward)


class AcrobotPBRSRewardShaper(RewardShaper):
    """
    Potential-Based Reward Shaping (PBRS) for Gymnasium Acrobot-v1.

    Potential Function:
        Phi(s) = w_h * (tip_height + 2.0) + w_v * (theta1_dot^2 + theta2_dot^2)
    """

    def __init__(
        self,
        height_weight: float = 1.0,
        velocity_weight: float = 0.1,
        gamma: float = 0.99,
    ):
        self.height_weight = height_weight
        self.velocity_weight = velocity_weight
        self.gamma = gamma

    def reset(self, initial_state: Any, info: dict[str, Any]) -> None:
        pass

    def _potential(self, state: Any) -> float:
        cos_theta1 = state[0]
        sin_theta1 = state[1]
        cos_theta2 = state[2]
        sin_theta2 = state[3]
        theta1_dot = state[4]
        theta2_dot = state[5]

        tip_height = -cos_theta1 - (cos_theta1 * cos_theta2 - sin_theta1 * sin_theta2)
        val = self.height_weight * (tip_height + 2.0) + self.velocity_weight * (
            theta1_dot**2 + theta2_dot**2
        )
        return float(val)

    def shape_reward(
        self,
        state: Any,
        action: Any,
        reward: float,
        next_state: Any,
        done: bool,
        info: dict[str, Any],
    ) -> float:
        phi_s = self._potential(state)
        # Boundary condition: Phi(s_terminal) = 0.0
        phi_s_prime = 0.0 if done else self._potential(next_state)

        shaping_reward = self.gamma * phi_s_prime - phi_s
        shaped_reward = reward + shaping_reward
        return float(shaped_reward)


class LunarLanderPBRSRewardShaper(RewardShaper):
    """
    Potential-Based Reward Shaping (PBRS) for Gymnasium LunarLander-v2.

    Potential Function:
        Phi(s) = - w_d * distance_to_pad - w_v * speed - w_a * abs(theta)
    """

    def __init__(
        self,
        distance_weight: float = 0.5,
        velocity_weight: float = 0.5,
        angle_weight: float = 1.0,
        gamma: float = 0.99,
    ):
        self.distance_weight = distance_weight
        self.velocity_weight = velocity_weight
        self.angle_weight = angle_weight
        self.gamma = gamma

    def reset(self, initial_state: Any, info: dict[str, Any]) -> None:
        pass

    def _potential(self, state: Any) -> float:
        x = state[0]
        y = state[1]
        vx = state[2]
        vy = state[3]
        theta = state[4]

        dist = math.sqrt(x**2 + y**2)
        vel = math.sqrt(vx**2 + vy**2)

        val = (
            -self.distance_weight * dist
            - self.velocity_weight * vel
            - self.angle_weight * abs(theta)
        )
        return float(val)

    def shape_reward(
        self,
        state: Any,
        action: Any,
        reward: float,
        next_state: Any,
        done: bool,
        info: dict[str, Any],
    ) -> float:
        phi_s = self._potential(state)
        # Boundary condition: Phi(s_terminal) = 0.0
        phi_s_prime = 0.0 if done else self._potential(next_state)

        shaping_reward = self.gamma * phi_s_prime - phi_s
        shaped_reward = reward + shaping_reward
        return float(shaped_reward)
