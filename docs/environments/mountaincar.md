# MountainCar-v0 Environment Specification

This document details the mechanics, state-action transitions, physics boundaries, reward structures, exploration challenges, and reward shaping opportunities of the `MountainCar-v0` Gymnasium environment.

---

## 1. Overview
The `MountainCar-v0` environment is a classic underpowered physical control problem. A car is positioned in a valley between two hills. The goal is to drive the car up the right hill to reach the target (flag). However, the engine of the car is weaker than gravity, meaning the car cannot climb the right slope from a stationary starting position. Instead, it must build momentum by rocking back and forth up the left hill first.

---

## 2. Observation Space
The observation space is a 2-dimensional continuous space representing the mechanical state of the car:

| Index | Observation | Min | Max | Unit |
| :--- | :--- | :---: | :---: | :--- |
| 0 | Position | -1.2 | 0.6 | meters |
| 1 | Velocity | -0.07 | 0.07 | m/s |

- **Initial State Range**: Position is randomized uniformly in $[-0.6, -0.4]$; Velocity is set to $0.0$.

---

## 3. Action Space
The action space is a discrete space with 3 actions:

| Action Value | Action Description | Applied Force |
| :---: | :--- | :---: |
| 0 | Push Left | Leftward force |
| 1 | Coast / No Engine | Zero force |
| 2 | Push Right | Rightward force |

---

## 4. Reward Function
- **Default Reward**: $-1.0$ for each step taken.
- **Goal Reward**: $0.0$ (when position reaches or exceeds the target threshold of $0.5$).
- **Objective**: Maximize the cumulative reward, which is equivalent to minimizing the number of timesteps to climb the hill.

---

## 5. Termination Conditions
The environment terminates when:
1. **Goal Reached**: The car's position is $\ge 0.5$.
2. **Episode Truncated**: The episode length reaches the time limit of $200$ steps.

---

## 6. Physics and Dynamics
The physics are governed by discretized gravitational and motor forces:
$$v_{t+1} = \text{clip}\left(v_t + (a - 1) \cdot 0.001 - \cos(3 \cdot p_t) \cdot 0.0025, -0.07, 0.07\right)$$
$$p_{t+1} = \text{clip}\left(p_t + v_{t+1}, -1.2, 0.6\right)$$

- $p_t$ is position at step $t$.
- $v_t$ is velocity at step $t$.
- $a \in \{0, 1, 2\}$ is the action choice.
- If $p_{t+1}$ hits the left boundary of $-1.2$, $v_{t+1}$ is reset to $0.0$.

---

## 7. Exploration Challenges & Difficulty
- **Sparse Feedback**: Because the car receives a flat $-1.0$ penalty at every step and the goal is rarely reached by random exploration, the agent has no initial gradient.
- **Underpowered Engine**: Simple greedy policies that attempt to drive directly toward the flag will fail, as they get stuck at the bottom local minimum. The agent must learn to move *away* from the goal initially to climb the left hill and gather mechanical energy.

---

## 8. Expected Role of Reward Shaping
Reward shaping transforms this sparse problem into a dense one:
- **Dense Shaping**: Adding a positive gradient based on height and speed directly guides the agent to climb higher and maintain velocity.
- **PBRS**: Guarantees that the optimal policy is invariant (preventing loops where the car swings indefinitely to collect reward) while adding a potential gradient towards the goal.

---

## 9. Potential Function Design Considerations
For Potential-Based Reward Shaping to be policy invariant, the potential function $\Phi(s)$ must satisfy:
1. $\Phi(s)$ represents the proximity to the goal.
2. $\Phi(s_{\text{terminal}}) = 0.0$ at the goal state.

We formulate the potential based on the car's height:
$$\text{Height}(p) = \sin(3 \cdot p)$$
$$\Phi(s) = w_h \cdot (\sin(3 \cdot p) + 1.0) + w_v \cdot v^2$$
Since $\sin(3 \cdot p)$ is at minimum $-1.0$ (when $p \approx -0.52$), the term $(\sin(3 \cdot p) + 1.0)$ is always $\ge 0$. At the goal $p \ge 0.5$, we enforce $\Phi(s') = 0.0$ when $done$ is true to preserve policy invariance.

---

## 10. Comparison with CartPole-v1
- **State Space**: CartPole has 4 continuous dimensions (position, velocity, angle, angular velocity); MountainCar has 2 continuous dimensions.
- **Dense/Sparse Defaults**: CartPole is a dense reward environment (receives $+1.0$ for keeping the pole up, terminating on failure). MountainCar is sparse (receives $-1.0$ at every step, terminating on success).
- **Exploration Regime**: CartPole is an stabilization task; MountainCar is a hard exploration task.

---

## 11. References
- Sutton, R. S., & Barto, A. G. (2018). *Reinforcement Learning: An Introduction*. MIT Press.
- Ng, A. Y., Harada, D., & Russell, S. (1999). *Policy invariance under reward shaping*. ICML.
