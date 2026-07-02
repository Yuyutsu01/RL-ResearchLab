# Future Research Directions

As this framework expands, several promising avenues for future research are identified:

---

## 1. Multi-Environment Scaling

While the framework currently benchmarks on `CartPole-v1`, it is designed to easily scale to more complex environments:
* **`MountainCar-v0`**: A classic sparse-reward problem. Moving from default sparse steps to energy-based potential shaping is a standard benchmark.
* **`Acrobot-v1`**: A double-pendulum swing-up task, suitable for testing angular momentum-based potential shapers.
* **`LunarLander-v2`**: A task with vector observation states and continuous/discrete actions, ideal for testing penalty-based shapers (e.g. landing velocity limits) and dense distance shapers.
* **`Pendulum-v1`**: A continuous control task, providing a base for testing continuous PBRS difference equations.

---

## 2. Meta-Learning for Adaptive Shaping

Adaptive reward shaping currently uses simple heuristics (like learning progress or step counts). A potential extension is **Meta-Reward Shaping**, where a separate meta-agent learns to output the shaping potential $\Phi(s)$ or scaling factor $\beta$ using:
* Reinforcement learning (e.g. optimizing the training progress of the base agent).
* Evolutionary algorithms to search the parameter space of the shaper.

---

## 3. Potential Shaping from Human Demonstrations

Instead of manually designing potential functions, we can construct the potential $\Phi(s)$ by fitting a density model (like a Variational Autoencoder or normalizing flow) over a small set of expert human demonstrations:
$$\Phi(s) = \log p_{expert}(s)$$
States that are highly similar to expert paths will output high potentials, providing a natural potential-based gradient guiding the agent toward expert behaviors.

---

## 4. Multi-Agent Cooperative Shaping

In cooperative multi-agent RL (MARL), agents face a coordination challenge. Designing local potential shapers that incentivize individual agents to act in a way that maximizes a global team objective (e.g. difference evaluations) is a major area of active RL research.
