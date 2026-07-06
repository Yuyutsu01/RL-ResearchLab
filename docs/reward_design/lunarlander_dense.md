# LunarLander-v2 Dense Reward Shaping Design Specification

This document details the design rationale, mathematical formulation, and properties of the heuristic Dense Reward shaping strategy for the `LunarLander-v2` environment.

---

## 1. Motivation
The default reward function of `LunarLander-v2` is already shaped (penalizes distance to pad, speed, leg contacts, and engine fire). However, the lander can still suffer from slow training and high variance because the default weights are balance-centric rather than path-centric.

By adding a dense shaping reward that directly penalizes horizontal distance, speed, and tilt, we provide a clean, steep gradient toward the landing pad.

---

## 2. Mathematical Formulation
The shaped reward is defined as:
$$R_{\text{shaped}} = R_{\text{original}} + F_{\text{dense}}(s, a, s')$$
where the dense reward term $F_{\text{dense}}(s, a, s')$ is:
$$F_{\text{dense}}(s, a, s') = -w_d \cdot |x'| - w_v \cdot (|v'_x| + |v'_y|) - w_a \cdot |\theta'|$$

- $x', y'$ are the next position coordinates.
- $v'_x, v'_y$ are the next velocities.
- $\theta'$ is the next lander angle.
- $w_d$ is the position penalty weight.
- $w_v$ is the velocity penalty weight.
- $w_a$ is the angle penalty weight.

---

## 3. Design Intuition
- **Distance Penalty**: Penalizes the horizontal distance from the central axis ($x=0$), encouraging the lander to align horizontally with the landing pad early in the descent.
- **Velocity Penalty**: Penalizes high velocities, encouraging the lander to descend slowly and maintain stable control.
- **Angle Penalty**: Penalizes rotation angle ($\theta$), keeping the lander vertically upright to prevent flip-overs and crashes.

---

## 4. Expected Improvements
- **Faster Convergence**: Helps the lander descend horizontally aligned and slowly, leading to fewer crashes and faster convergence.
- **Path Stabilization**: Guides the agent to form smooth, vertical descent trajectories rather than wild swings.

---

## 5. Potential Risks & Limitations
- **Policy Distortion (Hovering/Fuel Waste)**: Since the dense term directly adds negative penalties, it could distort the policy (e.g. encouraging the lander to crash early to avoid accumulating distance penalties over a long descent, or hovering too high).
- **Interference with Default Shaping**: Because LunarLander already has a complex default reward structure, adding direct dense penalties could conflict with leg contact bonuses or crash/safe landing terms, leading to suboptimal landing behaviors.

---

## 6. Parameters
- `distance_weight` ($w_d$): $0.5$
- `velocity_weight` ($w_v$): $0.5$
- `angle_weight` ($w_a$): $1.0$
