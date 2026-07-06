# MountainCar-v0 Dense Reward Shaping Design Specification

This document details the design rationale, mathematical formulation, parameter selections, and potential risks of the heuristic Dense Reward shaping strategy for the `MountainCar-v0` environment.

---

## 1. Motivation
The default reward function of `MountainCar-v0` provides $-1.0$ at each step, ending with $0.0$ when the car reaches the flag (position $\ge 0.5$). Since a random agent rarely climbs the slope by chance, the training gradient is initially zero.

By adding a dense shaping reward that rewards height (potential energy) and velocity (kinetic energy), we provide a continuous gradient that encourages the car to climb higher and swing back and forth with speed.

---

## 2. Mathematical Formulation
The shaped reward is defined as:
$$R_{\text{shaped}} = R_{\text{original}} + F_{\text{dense}}(s, a, s')$$
where the dense reward term $F_{\text{dense}}(s, a, s')$ is:
$$F_{\text{dense}}(s, a, s') = w_h \cdot (\sin(3 \cdot p') + 1.0) + w_v \cdot (v')^2$$

- $p'$ is the next position.
- $v'$ is the next velocity.
- $w_h$ is the scaling weight for the height potential energy.
- $w_v$ is the scaling weight for the velocity kinetic energy.

---

## 3. Design Intuition
- **Height Term**: $\sin(3 \cdot p)$ represents the height of the car on the sinusoidal hill. Since position $p \in [-1.2, 0.6]$, the height ranges from $-1.0$ to $1.0$. The offset $+1.0$ translates the height into a strictly non-negative bonus $[0, 2.0]$, preventing it from penalizing position updates.
- **Velocity Term**: $(v')^2$ represents the kinetic energy of the car. Since $v \in [-0.07, 0.07]$, the velocity term is small, so we use a high weight (e.g. $w_v = 10.0$ or $100.0$) to scale this term. This encourages the car to keep moving and gain speed.

---

## 4. Expected Improvements
- **Faster Initial Training**: The agent will immediately receive positive reinforcement for climbing up the sides of the valley, enabling early convergence compared to the flat $-1.0$ baseline.
- **Continuous Learning Curve**: Prevents the flat "no-learning" period associated with sparse reward landscapes.

---

## 5. Potential Risks & Limitations
- **Policy Distortion (Suboptimal Loops)**: Because $F_{\text{dense}}(s, a, s')$ is added directly without being derived from a potential difference, the reward function is **not** policy invariant.
- **Exploitable Cycles**: The agent might discover a closed trajectory loop where it swings back and forth in the valley indefinitely to accumulate the height/velocity bonuses, instead of climbing to the goal. This is a classic risk of heuristic reward shaping.
- **Sensitivity to Weights**: If $w_v$ or $w_h$ are set too high, the motor penalties (step cost of $-1.0$) are overwhelmed, leading to policy failure.

---

## 6. Parameters
- `height_weight` ($w_h$): $1.0$ (standard value to balance the $-1.0$ step cost).
- `velocity_weight` ($w_v$): $10.0$ (compensates for the small scale of the squared velocity term).
