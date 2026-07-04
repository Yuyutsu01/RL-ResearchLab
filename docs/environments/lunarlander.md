# LunarLander-v2 Environment Specification

This document details the mechanics, state-action transitions, physics boundaries, reward structures, exploration challenges, and reward shaping opportunities of the `LunarLander-v2` Gymnasium environment.

---

## 1. Overview
The `LunarLander-v2` environment is a classic rocket-landing simulator. A lander starts at the top center of the screen and must land safely on a designated landing pad at $(0, 0)$ coordinates. It is a multi-objective continuous control problem with a discrete action space.

---

## 2. Observation Space
The observation space is an 8-dimensional continuous vector representing the state of the lander:

| Index | Observation | Min | Max | Unit |
| :--- | :--- | :---: | :---: | :--- |
| 0 | Position X | -1.5 | 1.5 | meters |
| 1 | Position Y | -1.5 | 1.5 | meters |
| 2 | Velocity X ($v_x$) | -inf | inf | m/s |
| 3 | Velocity Y ($v_y$) | -inf | inf | m/s |
| 4 | Angle ($\theta$) | -inf | inf | radians |
| 5 | Angular Velocity ($\dot{\theta}$) | -inf | inf | rad/s |
| 6 | Left Leg Ground Contact | 0 | 1 | Boolean |
| 7 | Right Leg Ground Contact | 0 | 1 | Boolean |

---

## 3. Action Space
The action space is a discrete space with 4 actions:

| Action Value | Action Description | Force Applied |
| :---: | :--- | :--- |
| 0 | Do Nothing | Zero engine force |
| 1 | Fire Left Orientation Engine | Rightward rotation torque |
| 2 | Fire Main Engine | Upward thrust force |
| 3 | Fire Right Orientation Engine | Leftward rotation torque |

---

## 4. Reward Function
- **Default (Built-in) Reward**: Unlike MountainCar and Acrobot, LunarLander's default reward is already highly shaped:
  - Moving closer to the landing pad: $+100$ to $+140$.
  - Lander coming to rest: additional landing points.
  - Main engine firing: $-0.3$ per frame (discourages fuel waste).
  - Side engines firing: $-0.03$ per frame.
  - Each leg ground contact: $+10.0$.
  - Crashing (terminal state): $-100.0$.
  - Landing safely (terminal state): $+100.0$.

---

## 5. Termination Conditions
The environment terminates when:
1. **Lander Crashes**: The lander body touches the terrain or moves out of boundaries.
2. **Lander Rest**: The lander comes to rest on the terrain.
3. **Episode Truncated**: The episode length reaches the time limit of $400$ steps.

---

## 6. Physics and Dynamics
- The terrain is generated procedurally as a set of coordinates. The landing pad is located at $(0, 0)$.
- The lander is modeled as a rigid body subject to gravity, engine forces, and drag.
- Engine thrust modifies the velocity and angular velocity based on the orientation angle $\theta$.

---

## 7. Exploration Challenges & Difficulty
- **Control Instability**: The lander is highly unstable. Applying incorrect torque (actions 1 or 3) can easily cause the lander to spin out of control.
- **Competing Objectives**: The agent must balance moving toward the landing pad rapidly while decelerating enough to prevent a high-velocity impact, all while conserving fuel.

---

## 8. Expected Role of Reward Shaping
- **Dense Shaping**: While the default reward is shaped, adding direct linear distance penalties and alignment penalties can make the trajectory gradient even cleaner, helping the policy converge faster.
- **PBRS**: Modifies the default reward using a potential function, guaranteeing that we don't distort the default optimal policy (e.g. preventing the lander from hovering indefinitely just to collect rewards).

---

## 9. Potential Function Design Considerations
For LunarLander PBRS, we formulate the potential $\Phi(s)$ based on coordinates and stability:
$$\Phi(s) = -w_d \cdot \sqrt{x^2 + y^2} - w_v \cdot \sqrt{v_x^2 + v_y^2} - w_a \cdot |\theta|$$
At the goal/terminal state, we set $\Phi(s_{\text{terminal}}) = 0.0$ to guarantee policy invariance.

---

## 10. Comparison with CartPole-v1
- **Dynamics Complexity**: CartPole has simple linear/angular dynamics; LunarLander has complex 2D spatial rocket dynamics, terrain collisions, and physical leg contacts.
- **Action Space**: CartPole has 2 actions (left/right push); LunarLander has 4 actions (main and side engines).
- **Default Reward**: CartPole has binary dense survival rewards; LunarLander has complex, shaped, multi-term rewards.

---

## 11. References
- Brockman, G., et al. (2016). *OpenAI Gym*. arXiv preprint arXiv:1606.01540.
- Ng, A. Y., Harada, D., & Russell, S. (1999). *Policy invariance under reward shaping*. ICML.
