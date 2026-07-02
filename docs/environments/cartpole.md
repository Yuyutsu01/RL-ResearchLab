# CartPole-v1 Gymnasium Environment Specification

This document details the mechanics, state-action transitions, physics boundaries, reward structures, and success criteria of the `CartPole-v1` Gymnasium environment.

---

## 1. Physical Description

The system consists of an inverted pendulum (pole) attached by an unactuated joint to a cart, which moves along a frictionless track. The system is uncontrolled and unstable by default, requiring horizontal forces applied to the cart to keep the pole balanced upright.

```
       | (Pole, mass m, length l)
       | 
       | \  theta (Angle)
       |  \ 
      [Cart] ====> Force F (Discrete: Left or Right)
     (mass M)
---------------------------- (Frictionless Track)
```

---

## 2. State & Observation Space

The observation space is a 4-dimensional continuous vector representing the state of the system at time step $t$:
$$S_t = [x, \dot{x}, \theta, \dot{\theta}]^T$$

| Index | Observation | Unit | Minimum Bounds | Maximum Bounds |
| :---: | :--- | :---: | :---: | :---: |
| **0** | Cart Position ($x$) | m | -4.8 | 4.8 |
| **1** | Cart Velocity ($\dot{x}$) | m/s | $-\infty$ | $\infty$ |
| **2** | Pole Angle ($\theta$) | rad | -0.418 (-24°) | 0.418 (24°) |
| **3** | Pole Angular Velocity ($\dot{\theta}$) | rad/s | $-\infty$ | $\infty$ |

---

## 3. Action Space

The action space is a discrete binary choice ($A_t \in \{0, 1\}$):
* **`0`**: Push the cart to the **left** (apply a force of $-10.0$ N).
* **`1`**: Push the cart to the **right** (apply a force of $+10.0$ N).

*Note: The force applied is constant in magnitude, meaning the system is underactuated; the agent cannot choose the force intensity.*

---

## 4. Physics and Equations of Motion

The system dynamics are simulated using the following equations of motion:

$$\ddot{\theta} = \frac{g \sin\theta + \cos\theta \left[ \frac{-F - m l \dot{\theta}^2 \sin\theta}{M + m} \right]}{l \left[ \frac{4}{3} - \frac{m \cos^2\theta}{M + m} \right]}$$

$$\ddot{x} = \frac{F + m l \left( \dot{\theta}^2 \sin\theta - \ddot{\theta} \cos\theta \right)}{M + m}$$

Where:
* $g = 9.8$ $\text{m/s}^2$ (acceleration due to gravity)
* $M = 1.0$ kg (mass of the cart)
* $m = 0.1$ kg (mass of the pole)
* $l = 0.5$ m (half-pole length; center of mass distance)
* $F = \pm 10.0$ N (applied horizontal force)

Integration is computed using Euler's method with a step size of $\Delta t = 0.02$ seconds.

---

## 5. Termination & Truncation Conditions

An episode concludes immediately when any of the following conditions is met:

1. **Cart Boundary Exceeded (Termination)**: 
   $$|x| > 2.4 \text{ meters}$$
   The cart leaves the central track bounds.
2. **Pole Angle Exceeded (Termination)**:
   $$\theta > 12^\circ \text{ (approx. } 0.2095 \text{ radians)}$$
   The pole tilts too far from the vertical vertical line.
3. **Step Limit Reached (Truncation)**:
   $$t \ge 500 \text{ steps}$$
   The agent successfully balances the pole for 10.0 seconds of simulation time.

---

## 6. Default Reward Structure

The default Gymnasium reward is defined as:
$$R_t = 1.0 \quad \forall t$$
For every step the agent prevents termination, it receives a flat reward of $+1.0$.

### Scientific Challenges with the Default Reward:
* **Sparse EXPLORATION Guidance**: Because the reward is always $+1.0$, PPO receives no information about whether a specific state is "better" than another (e.g., balancing perfectly vertically at the center vs. leaning heavily near the boundary).
* **Sparse CRITIC Signal**: The critic network receives a constant value input, making value estimation slow to refine until the agent starts experiencing failures.

---

## 7. Success Criteria

The environment is officially considered "solved" if the agent achieves an average unshaped evaluation reward of:
$$\bar{R} \ge 475.0 \text{ over 100 consecutive episodes.}$$
For research purposes, we evaluate performance asymptotically, aiming to reach the absolute maximum ceiling score of **500.00**.
