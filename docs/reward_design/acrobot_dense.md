# Acrobot-v1 Dense Reward Shaping Design Specification

This document details the design rationale, mathematical formulation, and properties of the heuristic Dense Reward shaping strategy for the `Acrobot-v1` environment.

---

## 1. Motivation
The default reward function of `Acrobot-v1` provides $-1.0$ at each step, ending with $0.0$ when the tip reaches the target height of $1.0$. Because a random agent rarely swings high enough to reach the goal by chance, the training gradient is flat at the start of learning.

By adding a dense shaping reward based on the height of the pendulum tip and its angular velocities, we provide a continuous gradient that encourages swing-up behavior.

---

## 2. Mathematical Formulation
The shaped reward is defined as:
$$R_{\text{shaped}} = R_{\text{original}} + F_{\text{dense}}(s, a, s')$$
where the dense reward term $F_{\text{dense}}(s, a, s')$ is:
$$F_{\text{dense}}(s, a, s') = w_h \cdot (\text{tip\_height}' + 2.0) + w_v \cdot \left((\dot{\theta}'_1)^2 + (\dot{\theta}'_2)^2\right)$$

- $\text{tip\_height}'$ is the next tip height.
- $\dot{\theta}'_1, \dot{\theta}'_2$ are the next angular velocities of link 1 and link 2.
- $w_h$ is the scaling weight for the tip height.
- $w_v$ is the scaling weight for the velocities.

The tip height is computed as:
$$\text{tip\_height} = -\cos(\theta_1) - \cos(\theta_1 + \theta_2) = -x[0] - (x[0] \cdot x[2] - x[1] \cdot x[3])$$
where $x$ represents the observation vector.

---

## 3. Design Intuition
- **Height Term**: $\text{tip\_height}$ ranges from $-2.0$ (when both links hang down) to $+2.0$ (when both links point straight up). Adding $+2.0$ offsets this range to a non-negative scale $[0.0, 4.0]$.
- **Velocity Term**: The squared velocities $(\dot{\theta}_1)^2 + (\dot{\theta}_2)^2$ encourage the pendulum to keep swinging and accumulate mechanical energy, helping it swing higher.

---

## 4. Expected Improvements
- **Accelerated Learning**: The PPO policy will receive immediate positive reinforcement for lifting the second link, leading to faster swing-up.
- **Continuous Learning Curve**: Prevents the flat "no-learning" period.

---

## 5. Potential Risks & Limitations
- **Policy Distortion (Oscillation Loops)**: Since the shaping is heuristic and not potential-based, the agent might learn to swing the pendulum back and forth just below the goal line to continuously harvest the height and speed bonuses, rather than crossing the goal line (which terminates the episode).
- **Tuning Sensitivity**: Balance between height reward and speed reward is critical. Too much speed reward can lead to uncontrollable spinning.

---

## 6. Parameters
- `height_weight` ($w_h$): $1.0$
- `velocity_weight` ($w_v$): $0.1$
