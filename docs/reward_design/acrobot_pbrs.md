# Acrobot-v1 Potential-Based Reward Shaping Design Specification

This document details the mathematical formulation, design rationale, and properties of the Potential-Based Reward Shaping (PBRS) strategy for the `Acrobot-v1` environment.

---

## 1. Motivation
Heuristic dense rewards for the `Acrobot-v1` swing-up problem run the risk of policy distortion (e.g. the pendulum oscillating below the goal height indefinitely to gather rewards).

Potential-Based Reward Shaping (PBRS) addresses this risk by deriving the shaping reward from a potential difference. This guarantees **policy invariance**—the optimal policy under the shaped reward remains identical to the optimal policy under the default environment reward.

---

## 2. Mathematical Formulation
The shaped reward is defined as:
$$R_{\text{shaped}} = R_{\text{original}} + F_{\text{pbrs}}(s, a, s')$$
where the shaping reward is computed as:
$$F_{\text{pbrs}}(s, a, s') = \gamma \cdot \Phi(s') - \Phi(s)$$

- $\gamma$ is the environment discount factor ($0.99$).
- $\Phi(s)$ is the potential function.

The potential function $\Phi(s)$ is designed as:
$$\Phi(s) = w_h \cdot (\text{tip\_height} + 2.0) + w_v \cdot \left(\dot{\theta}_1^2 + \dot{\theta}_2^2\right)$$

- $\text{tip\_height}$ is $-\cos(\theta_1) - \cos(\theta_1 + \theta_2) = -x[0] - (x[0] \cdot x[2] - x[1] \cdot x[3])$.
- $\dot{\theta}_1, \dot{\theta}_2$ are the angular velocities.
- $w_h$ is the height potential weight.
- $w_v$ is the velocity potential weight.

---

## 3. Boundary Conditions
For policy invariance to hold strictly in episodic settings, the potential at the terminal absorbing states must equal $0.0$:
$$\Phi(s_{\text{terminal}}) = 0.0$$
- If transition $s \to s'$ results in goal termination (i.e. tip height $\ge 1.0$), we set $\Phi(s') = 0.0$ explicitly.
- This prevents the agent from retaining "potential" values across episode boundaries and guarantees convergence to the true optimal control policy.

---

## 4. Design Intuition
By defining the potential as a function of the height of the pendulum tip and its squared velocities, the agent is rewarded when it increases tip height or gains speed. If the pendulum swings down or decelerates, the potential difference is negative, penalizing the agent.

Because the total cumulative shaped reward along any trajectory depends only on the start and end potentials:
$$\sum_{t} \gamma^t F_{\text{pbrs}}(s_t, a_t, s_{t+1}) = \gamma^T \Phi(s_T) - \Phi(s_0)$$
any loop in state space results in a net shaping reward of $0.0$. Thus, there is no incentive to oscillate below the goal line indefinitely, resolving the policy distortion risk.

---

## 5. Expected Improvements
- **Optimal Policy Conservation**: Guaranteed convergence to the optimal solution.
- **Enhanced Sample Efficiency**: Guides the PPO agent to swing the pendulum high early by providing step-level feedback, while preserving policy invariants.

---

## 6. Parameters
- `height_weight` ($w_h$): $1.0$
- `velocity_weight` ($w_v$): $0.1$
- `gamma` ($\gamma$): $0.99$
