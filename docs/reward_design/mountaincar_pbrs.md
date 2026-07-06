# MountainCar-v0 Potential-Based Reward Shaping Design Specification

This document details the mathematical formulation, design rationale, and properties of the Potential-Based Reward Shaping (PBRS) strategy for the `MountainCar-v0` environment.

---

## 1. Motivation
While Dense Reward shaping provides a continuous gradient, it runs the risk of policy distortion (e.g. the agent learning to oscillate in the valley indefinitely instead of reaching the goal).

Potential-Based Reward Shaping (PBRS) addresses this limitation by deriving the shaping reward from a potential difference. This guarantees **policy invariance**—the optimal policy under the shaped reward remains identical to the optimal policy under the default environment reward.

---

## 2. Mathematical Formulation
The shaped reward is defined as:
$$R_{\text{shaped}} = R_{\text{original}} + F_{\text{pbrs}}(s, a, s')$$
where the shaping reward is computed as:
$$F_{\text{pbrs}}(s, a, s') = \gamma \cdot \Phi(s') - \Phi(s)$$

- $\gamma$ is the discount factor matching the environment configuration (typically $0.99$).
- $\Phi(s)$ is the potential function.

The potential function $\Phi(s)$ is designed as:
$$\Phi(s) = w_h \cdot (\sin(3 \cdot p) + 1.0) + w_v \cdot v^2$$

- $p$ is the car position.
- $v$ is the velocity.
- $w_h$ is the position height potential weight.
- $w_v$ is the velocity kinetic energy weight.

---

## 3. Boundary Conditions
For policy invariance to hold strictly in episodic settings, the potential at the terminal absorbing states must equal $0.0$:
$$\Phi(s_{\text{terminal}}) = 0.0$$
- If transition $s \to s'$ results in goal termination (i.e. $p' \ge 0.5$ and `done` is true), we set $\Phi(s') = 0.0$ explicitly.
- This prevents the agent from retaining "potential" values across episode boundaries and guarantees convergence to the true optimal control policy.

---

## 4. Design Intuition
By defining the potential as the sum of potential and kinetic energy (height and velocity squared), the agent receives a positive shaping reward $F_{\text{pbrs}} > 0$ when it increases its height or gathers speed. If the agent moves backwards or down the hill, $F_{\text{pbrs}} < 0$, penalizing the waste of energy.

Because the total cumulative shaped reward along any trajectory depends only on the start and end potentials:
$$\sum_{t} \gamma^t F_{\text{pbrs}}(s_t, a_t, s_{t+1}) = \gamma^T \Phi(s_T) - \Phi(s_0)$$
any looping cycle results in a net shaping reward of $0.0$. Thus, there is no incentive to oscillate in the valley indefinitely, resolving the policy distortion risk.

---

## 5. Expected Improvements
- **Optimal Policy Conservation**: Guaranteed convergence to the optimal solution.
- **Enhanced Sample Efficiency**: Guides the PPO agent to climb the hill early by providing step-level feedback, while preserving policy invariants.

---

## 6. Parameters
- `height_weight` ($w_h$): $1.0$
- `velocity_weight` ($w_v$): $10.0$
- `gamma` ($\gamma$): $0.99$
