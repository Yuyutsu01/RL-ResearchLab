# Reward Design Specification: Potential-Based Reward Shaping (PBRS)

This document outlines the design logic, mathematical formulation, theoretical guarantees, and implementation specifics of **Potential-Based Reward Shaping (PBRS)** for the `CartPole-v1` environment.

---

## 1. Problem Statement: Heuristic vs. Potential-Based Shaping

Heuristic dense rewards (like the one implemented in `DenseRewardShaper`) reshape the reward landscape to guide exploration, but they do so at a major cost: **they alter the original MDP's optimal policy**. 
For example, adding a heuristic penalty:
$$R_{\text{shaped}}(s, a, s') = R(s, a, s') - f(s')$$
can make the agent prefer early termination over survival if the state-space penalties exceed the survival reward, or cause the agent to get stuck in sub-optimal loops (reward hacking).

**Potential-Based Reward Shaping (PBRS)** solves this problem. It introduces a mathematically rigorous formulation that guarantees **policy invariance**—meaning the optimal policy of the shaped MDP is identical to that of the original MDP under all circumstances, while still providing the dense training signal needed for sample efficiency.

---

## 2. Theoretical Foundation: The Ng, Harada & Russell (1999) Theorem

In their landmark paper, *Policy invariance under reward shaping: General theorems and practical results*, Andrew Ng, Daishi Harada, and Stuart Russell proved that a shaping reward $F(s, a, s')$ preserves the optimal policy if and only if it is formulated as the difference in a state potential function $\Phi$:

$$F(s, a, s') = \gamma \Phi(s') - \Phi(s)$$

Where:
* $\Phi(s)$ is a real-valued potential function representing the "goodness" of state $s$.
* $\gamma$ is the discount factor of the Markov Decision Process (MDP).

### Proof of Policy Invariance (Telescoping Returns)
Let $\tau = (s_0, s_1, s_2, \dots, s_T)$ be an episode trajectory. The cumulative discounted return under the shaped reward $R' = R + F$ is:

$$G'_0 = \sum_{t=0}^{T-1} \gamma^t R'(s_t, a_t, s_{t+1})$$
$$G'_0 = \sum_{t=0}^{T-1} \gamma^t \left[ R(s_t, a_t, s_{t+1}) + \gamma \Phi(s_{t+1}) - \Phi(s_t) \right]$$
$$G'_0 = \sum_{t=0}^{T-1} \gamma^t R(s_t, a_t, s_{t+1}) + \sum_{t=0}^{T-1} \left[ \gamma^{t+1} \Phi(s_{t+1}) - \gamma^t \Phi(s_t) \right]$$

Expanding the second summation reveals a telescoping series:
$$\sum_{t=0}^{T-1} \left[ \gamma^{t+1} \Phi(s_{t+1}) - \gamma^t \Phi(s_t) \right] = \gamma^T \Phi(s_T) - \Phi(s_0)$$

Thus:
$$G'_0 = G_0 + \gamma^T \Phi(s_T) - \Phi(s_0)$$

For any policy $\pi$, the return is shifted only by a constant term dependent on the start state $s_0$ and the terminal state $s_T$. Because this shift is independent of the sequence of actions taken along the trajectory, the relative ordering of policy values is preserved:
$$\pi_1 \text{ is better than } \pi_2 \text{ under } R \iff \pi_1 \text{ is better than } \pi_2 \text{ under } R'$$

### The Episodic Boundary Condition
For episodic tasks, to prevent the agent from exploiting terminal potentials (e.g., getting a final reward of $-\Phi(s_{T-1})$ by failing early, which can alter the optimal policy), the potential of the terminal state $s_T$ must be zero:
$$\Phi(s_T) = 0$$
This forces the final transition shaping reward to be:
$$F(s_{T-1}, a, s_T) = 0 - \Phi(s_{T-1}) = - \Phi(s_{T-1})$$

---

## 3. Potential Function Design for CartPole

We define a parameterized, modular potential function $\Phi(s)$ based on the standard four state variables of `CartPole-v1`:
* $x$: Cart position (bounds: $\pm 2.4$)
* $v$: Cart velocity
* $\theta$: Pole angle in radians (bounds: $\pm 0.209$ rad)
* $\omega$: Pole angular velocity

### Mathematical Formulation
For maximum flexibility, the potential function supports both absolute (L1) and quadratic (L2) norms:

$$\Phi(s) = \begin{cases} 
- \left( w_x |x| + w_v |v| + w_\theta |\theta| + w_\omega |\omega| \right) & \text{if potential\_type = "l1"} \\
- \left( w_x x^2 + w_v v^2 + w_\theta \theta^2 + w_\omega \omega^2 \right) & \text{if potential\_type = "l2"} 
\end{cases}$$

### Component Analysis (Inclusion/Exclusion Rationale)
* **Pole Angle ($\theta$)**: Included (default weight $w_\theta = 1.0$). This is the primary driver of failure. Keeping the pole upright maximises the potential.
* **Cart Position ($x$)**: Included (default weight $w_x = 0.1$). Prevents the cart from drifting into the outer boundaries.
* **Cart Velocity ($v$)**: Excluded by default ($w_v = 0.0$). Velocity is necessary to balance the pole; penalizing velocity directly can cause sluggish actions.
* **Pole Angular Velocity ($\omega$)**: Excluded by default ($w_\omega = 0.0$). Standard baselines focus on positions to match heuristic comparison experiments.

---

## 4. Comparison: Heuristic Dense vs. PBRS

| Property | Heuristic Dense Reward | Potential-Based Reward Shaping (PBRS) |
| :--- | :--- | :--- |
| **Mathematical Formulation** | $R_t' = R_t - f(s_t')$ | $R_t' = R_t + \gamma \Phi(s_t') - \Phi(s_t)$ |
| **Policy Invariance** | ❌ None (can change optimal policy) |  Guaranteed (optimal policy is preserved) |
| **Exploration Guidance** | Good (pulls agent to target) | Good (pulls agent to higher potential) |
| **Risk of Reward Hacking** | ⚠️ High (agent may exploit penalty bounds) |  Zero (no cyclic loops can accumulate reward) |
| **Episodic Boundary Safety** | No special treatment | Handled via $\Phi(s_{\text{terminal}}) = 0$ |

---

## 5. Expected Behavior & Risks

* **Expected Behavior**: PBRS should accelerate early learning (reach rewards of 100, 200, and 300 earlier than baseline) due to the potential gradient. It should converge to the exact same ceiling performance of 500.0 as the Identity baseline without policy degradation.
* **Risks**: If the potential scale is too small, the gradient will be weak. If too large, the initial updates might be erratic. However, since the baseline return is a telescoping sum, the final policy remains mathematically optimal.
