# Research Hypotheses

We formulate the following hypotheses regarding how each reward shaping strategy will perform relative to the unshaped baseline:

---

## Hypothesis 1: Identity Reward (Baseline)
* **Definition**: $R_{shaped}(s, a, s') = R_{original}(s, a, s')$.
* **Expectation**: Slowest convergence speed but guaranteed optimal policy convergence.
* **Rationale**: The agent must rely entirely on sparse/unmodified signals. It will take longer to discover high-reward regions of the state space, but there is zero risk of policy subversion.

## Hypothesis 2: Dense Reward Shaping
* **Definition**: Adding continuous feedback signals based on distance/angle alignment at every step.
* **Expectation**: Significantly faster convergence in early stages, but high risk of policy subversion and sub-optimal asymptotic behavior.
* **Rationale**: The agent is provided immediate gradient guides, reducing random walk search times. However, if the dense reward is not mathematically aligned, the agent will learn to maximize the dense signal rather than the target objective, converging to a sub-optimal local minimum.

## Hypothesis 3: Distance-Based Reward Shaping
* **Definition**: Penaltes proportional to distance from key targets (e.g. horizontal center or vertical upright).
* **Expectation**: Smooth learning curves, faster convergence than baseline on simple tasks, moderate policy subversion risk.
* **Rationale**: Acting as a gravitational pull toward target states, it limits exploration in high-penalty states. However, it can make exploration of complex boundary behaviors difficult.

## Hypothesis 4: Potential-Based Reward Shaping (PBRS)
* **Definition**: Difference potentials of the form $F(s, a, s') = \gamma \Phi(s') - \Phi(s)$.
* **Expectation**: Faster convergence speed with **guaranteed zero policy subversion**.
* **Rationale**: Because PBRS acts as a state-potential transition difference, it mathematically preserves the optimal policy layout. The agent will converge to the exact same optimal policy as the baseline, but the potential function will guide search trajectories to reach that policy with fewer environment interactions.

## Hypothesis 5: Penalty-Based Reward Shaping (Constraint Enforced)
* **Definition**: Adding large negative values when the agent enters unsafe/unstable regions.
* **Expectation**: Reduced variance in late-stage training and safer policy behaviors, but potentially slower initial discovery rates.
* **Rationale**: Penalties enforce safety bounds. By penalizing unstable angles or high cart velocities, the agent avoids terminal boundaries. This stabilizes updates but can cause the policy to become overly conservative.

## Hypothesis 6: Adaptive Reward Shaping
* **Definition**: Scaling factors or potential offsets that update based on variance of historical rewards or state distributions.
* **Expectation**: Highest sample efficiency and robust convergence across different tasks, but higher computational overhead.
* **Rationale**: An adaptive shaper scales feedback to prevent reward signals from dominating or vanishing. Early in training, shaping guides exploration; as the agent converges, the shaping terms are annealed, transitioning the policy gradient strictly to the original objective.
