# LunarLander-v2 Potential-Based Reward Shaping Design Specification

This document details the mathematical formulation, design rationale, and properties of the Potential-Based Reward Shaping (PBRS) strategy for the `LunarLander-v2` environment.

---

## 1. Motivation
Heuristic dense rewards for rocket landing run the risk of distorting the optimal policy (e.g. encouraging the lander to crash early to avoid accumulating distance penalties over a long descent, or hovering indefinitely).

Potential-Based Reward Shaping (PBRS) addresses this risk by deriving the shaping reward from a potential difference. This guarantees **policy invariance**—the optimal policy under the shaped reward remains identical to the optimal policy under the default environment reward, while guiding exploration.

---

## 2. Mathematical Formulation
The shaped reward is defined as:
$$R_{\text{shaped}} = R_{\text{original}} + F_{\text{pbrs}}(s, a, s')$$
where the shaping reward is computed as:
$$F_{\text{pbrs}}(s, a, s') = \gamma \cdot \Phi(s') - \Phi(s)$$

- $\gamma$ is the discount factor ($0.99$).
- $\Phi(s)$ is the potential function.

The potential function $\Phi(s)$ is designed as:
$$\Phi(s) = -w_d \cdot \sqrt{x^2 + y^2} - w_v \cdot \sqrt{v_x^2 + v_y^2} - w_a \cdot |\theta|$$

- $x, y$ are the position coordinates.
- $v_x, v_y$ are the velocities.
- $\theta$ is the rotation angle.
- $w_d$ is the position weight.
- $w_v$ is the velocity weight.
- $w_a$ is the angle weight.

---

## 3. Boundary Conditions
For policy invariance to hold strictly in episodic settings, the potential at the terminal absorbing states must equal $0.0$:
$$\Phi(s_{\text{terminal}}) = 0.0$$
- If transition $s \to s'$ results in termination (crash or safe landing), we set $\Phi(s') = 0.0$ explicitly.
- This prevents the agent from retaining "potential" values across episode boundaries and guarantees convergence to the true optimal control policy.

---

## 4. Design Intuition
By defining the potential as the negative weighted distance to the pad, velocity magnitude, and rotation angle, the potential $\Phi(s) \le 0.0$.
As the lander gets closer to $(0,0)$ with low speed and 0 angle, the potential approaches its maximum value $0.0$.
The agent is rewarded when the potential increases ($F_{\text{pbrs}} > 0$), meaning it gets closer to the pad, decelerates, or straightens up. If the lander drifts away, accelerates, or tilts, it is penalized.

Because the total cumulative shaped reward along any trajectory depends only on the start and end potentials:
$$\sum_{t} \gamma^t F_{\text{pbrs}}(s_t, a_t, s_{t+1}) = \gamma^T \Phi(s_T) - \Phi(s_0)$$
any looping cycle results in a net shaping reward of $0.0$. Thus, there is no incentive to hover indefinitely just to collect rewards, preserving the optimal policy.

---

## 5. Expected Improvements
- **Optimal Policy Conservation**: Guaranteed convergence to the optimal landing solution.
- **Enhanced Sample Efficiency**: Guides the PPO agent to descend vertically aligned and slowly, leading to fewer initial crashes.

---

## 6. Parameters
- `distance_weight` ($w_d$): $0.5$
- `velocity_weight` ($w_v$): $0.5$
- `angle_weight` ($w_a$): $1.0$
- `gamma` ($\gamma$): $0.99$
