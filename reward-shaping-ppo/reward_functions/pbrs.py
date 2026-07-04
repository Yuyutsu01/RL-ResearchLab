from typing import Any

from reward_functions.base import RewardShaper


class PBRSRewardShaper(RewardShaper):
    """
    Potential-Based Reward Shaping (PBRS) for Gymnasium CartPole-v1.

    Guarantees policy invariance while guiding exploration.

    Mathematical Formulation:
        F(s, a, s') = gamma * Phi(s') - Phi(s)
        R_shaped = R_original + F(s, a, s')

    Potential Function:
        Phi(s) = - (w_x * |x| + w_v * |v| + w_theta * |theta| + w_omega * |omega|)   [l1 type]
        Phi(s) = - (w_x * x^2 + w_v * v^2 + w_theta * theta^2 + w_omega * omega^2) [l2 type]

    Special Boundary Condition:
        Phi(s_terminal) = 0.0
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
            raise ValueError(f"potential_type must be 'l1' or 'l2', got '{potential_type}'")
        self.potential_type = ptype

    def reset(self, initial_state: Any, info: dict[str, Any]) -> None:
        """Stateless design doesn't require maintaining history variables across resets."""
        pass

    def _potential(self, state: Any) -> float:
        """Computes the potential value of a given state."""
        # State values: [x, v, theta, omega]
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
        # Boundary Condition: Phi(s_terminal) = 0.0
        phi_s_prime = 0.0 if done else self._potential(next_state)

        shaping_reward = self.gamma * phi_s_prime - phi_s
        shaped_reward = reward + shaping_reward

        return float(shaped_reward)
