# Acrobot-v1 Environment Specification

This document details the mechanics, state-action transitions, physics boundaries, reward structures, exploration challenges, and reward shaping opportunities of the `Acrobot-v1` Gymnasium environment.

---

## 1. Overview
The `Acrobot-v1` environment is a two-link pendulum system where only the joint between the links is actuated (the elbow is motorized, but the shoulder is free-hanging). The goal is to swing the tip of the second link above a height of $1.0$ relative to the shoulder joint. It is a classic underactuated robotic control benchmark.

---

## 2. Observation Space
The observation space is a 6-dimensional continuous space representing the angles and angular velocities of the two links:

| Index | Observation | Min | Max | Unit |
| :--- | :--- | :---: | :---: | :--- |
| 0 | Cosine of link 1 angle ($\cos(\theta_1)$) | -1.0 | 1.0 | unitless |
| 1 | Sine of link 1 angle ($\sin(\theta_1)$) | -1.0 | 1.0 | unitless |
| 2 | Cosine of link 2 angle ($\cos(\theta_2)$) | -1.0 | 1.0 | unitless |
| 3 | Sine of link 2 angle ($\sin(\theta_2)$) | -1.0 | 1.0 | unitless |
| 4 | Link 1 angular velocity ($\dot{\theta}_1$) | -12.57 | 12.57 | rad/s |
| 5 | Link 2 angular velocity ($\dot{\theta}_2$) | -28.27 | 28.27 | rad/s |

- **Initial State Range**: Angles $\theta_1$ and $\theta_2$ are initialized uniformly in $[-0.1, 0.1]$; angular velocities are set to $0.0$.

---

## 3. Action Space
The action space is a discrete space with 3 actions:

| Action Value | Action Description | Applied Torque |
| :---: | :--- | :---: |
| 0 | Apply Negative Torque | Counter-clockwise torque |
| 1 | Coast / No Torque | Zero torque |
| 2 | Apply Positive Torque | Clockwise torque |

---

## 4. Reward Function
- **Default Reward**: $-1.0$ for each step taken.
- **Goal Reward**: $0.0$ (when the tip reaches or exceeds the target height of $1.0$).
- **Objective**: Minimize the number of steps required to swing the tip above the threshold line.

---

## 5. Termination Conditions
The environment terminates when:
1. **Goal Reached**: The tip height is $\ge 1.0$.
2. **Episode Truncated**: The episode length reaches the time limit of $500$ steps.

---

## 6. Physics and Dynamics
The shoulder joint is located at $(0, 0)$.
The elbow joint is at $(l_1 \sin(\theta_1), -l_1 \cos(\theta_1))$.
The tip position is:
$$x_{\text{tip}} = l_1 \sin(\theta_1) + l_2 \sin(\theta_1 + \theta_2)$$
$$y_{\text{tip}} = -l_1 \cos(\theta_1) - l_2 \cos(\theta_1 + \theta_2)$$

In Gymnasium, $l_1 = l_2 = 1.0$, which simplifies the tip height equation to:
$$y_{\text{tip}} = -\cos(\theta_1) - \cos(\theta_1 + \theta_2)$$
Using the trigonometric identity $\cos(A + B) = \cos A \cos B - \sin A \sin B$:
$$y_{\text{tip}} = -x[0] - (x[0] \cdot x[2] - x[1] \cdot x[3])$$
where $x[i]$ represents the observation indices.

---

## 7. Exploration Challenges & Difficulty
- **Underactuation**: The shoulder joint is unactuated. Torque applied at the elbow must be transferred dynamically to rotate link 1.
- **Sparse Feedback**: Just like MountainCar, the agent receives $-1.0$ until success. A random agent rarely swings high enough to reach height $1.0$ within 500 steps, meaning the policy gradient is flat initially.

---

## 8. Expected Role of Reward Shaping
- **Dense Shaping**: Adding a positive gradient proportional to $y_{\text{tip}}$ immediately rewards actions that lift the pendulum tip, easing the exploration barrier.
- **PBRS**: Adds a potential representing mechanical energy or height, preserving the optimal policy while ensuring that the agent does not just loop to collect reward.

---

## 9. Potential Function Design Considerations
For PBRS, we want the potential $\Phi(s)$ to reward swing-up height and keep the velocity bounded.
$$\Phi(s) = w_h \cdot (y_{\text{tip}} + 2.0) + w_v \cdot (\dot{\theta}_1^2 + \dot{\theta}_2^2)$$
Since $y_{\text{tip}} \in [-2.0, 2.0]$, the term $(y_{\text{tip}} + 2.0) \ge 0$. We enforce $\Phi(s') = 0.0$ when the state is terminal (i.e., tip reaches height $\ge 1.0$) to guarantee policy invariance.

---

## 10. Comparison with CartPole-v1
- **Task Goal**: CartPole is an stabilization task (keep the pole near $0$ angle); Acrobot is a swing-up task (reach a high target state).
- **Control Interface**: CartPole applies force to a cart (direct translational movement); Acrobot applies torque to the elbow joint (indirect swing-up movement).
- **Reward Density**: CartPole has dense positive rewards; Acrobot has sparse negative penalties.

---

## 11. References
- Spong, M. W. (1995). *The swing up control problem for the Acrobot*. IEEE Control Systems Magazine.
- Ng, A. Y., Harada, D., & Russell, S. (1999). *Policy invariance under reward shaping*. ICML.
