# Reward Design Specification: Dense Reward Shaping

This document outlines the design logic, mathematical formulation, expected behavior, and risk profile for the **Dense Reward Shaping** strategy.

---

## 1. Problem Statement: The Default Reward Limitations

The default reward for `CartPole-v1` is:
$$R_{\text{original}} = 1.0 \quad \forall t$$
This is a pure **survival reward**. It represents a coarse, step-wise flat surface. 

From an optimization perspective, this creates two key challenges:
1. **No Directional Gradient**: PPO's value function network (the Critic) cannot distinguish between a state where the pole is perfectly balanced at the track center ($x = 0, \theta = 0$) and a state where the pole is tilted at $11^\circ$ near the track boundary ($x = 2.3, \theta = 0.209$).
2. **Exploration inefficiency**: The agent must explore randomly until it falls over to realize that a state sequence was sub-optimal. This makes early training slow to converge.

---

## 2. Design Intuition & Motivation

Our goal is to reshape the reward landscape into a smooth **potential funnel** centered at the state origin $(x = 0, \theta = 0)$. By adding continuous penalties for cart displacement and pole tilt, we provide PPO with a direct gradient pointing toward the center-upright equilibrium state.

```
Reward
 ^
 |        /\ (Maximum Reward = 1.0 at origin)
 |       /  \
 |      /    \  <--- Smooth Gradient Funnel
 |     /      \
 |    /        \
+----+----------+----> State Displacements (|x|, |\theta|)
```

---

## 3. Mathematical Formulation

The dense shaped reward is defined as:
$$R_{\text{shaped}}(s, a, s') = R_{\text{original}} + F(s, a, s')$$
$$R_{\text{shaped}}(s, a, s') = 1.0 - \left( w_1 \cdot |x'| + w_2 \cdot |\theta'| \right)$$

Where:
* $x'$ is the cart position in the next state (`next_state[0]`).
* $\theta'$ is the pole angle in radians in the next state (`next_state[2]`).
* $w_1$ is the **position weight** (default: `0.1`).
* $w_2$ is the **angle weight** (default: `1.0`).

### Constraints and Bounds:
* At perfect balance ($x'=0, \theta'=0$), the reward is the maximum: $R_{\text{shaped}} = 1.0$.
* At the failure boundaries ($|x'| = 2.4, |\theta'| = 0.209$):
  - Position penalty: $0.1 \times 2.4 = 0.24$
  - Angle penalty: $1.0 \times 0.209 \approx 0.21$
  - Minimum step reward: $1.0 - (0.24 + 0.21) = 0.55$.
* This guarantees that the shaped reward remains positive ($R_{\text{shaped}} > 0.0$), keeping the environment's intrinsic incentive to survive intact.

---

## 4. Alternative Ideas Evaluated

1. **Quadratic Penalties ($-w_1 x^2 - w_2 \theta^2$)**:
   - *Pros*: Smooth gradients near the origin.
   - *Cons*: Penalties near the boundary scale quadratically, which can cause large gradient spikes during early training fall-overs, destabilizing PPO. Linear penalties ($|x|, |\theta|$) keep gradient updates stable.
2. **Velocity Penalties ($-w_3 |\dot{x}| - w_4 |\dot{\theta}|$)**:
   - *Cons*: Discourages the agent from moving. To keep the pole balanced, the cart *must* move. Penalizing velocity can cause the policy to freeze, leading to premature tip-overs. We prioritize position and angle over velocities.

---

## 5. Expected Behavior & Trade-offs

* **Acceleration**: The agent should learn to stabilize the pole much earlier in training (within ~15,000 steps compared to baseline's ~25,000).
* **Variance Reduction**: Shading the state space should guide all seeds toward the same optimal balancing actions, reducing run-to-run standard deviations.
* **Risk (Policy Subversion)**: If the cart position penalty weight $w_1$ is set too high, the agent might refuse to move to save the pole because moving away from the center carries a large penalty. This represents a trade-off between centering stability and survival rate.
