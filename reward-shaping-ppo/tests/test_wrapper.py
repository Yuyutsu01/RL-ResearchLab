import gymnasium as gym

from environments.wrapper import RewardShapingWrapper
from reward_functions.identity import IdentityRewardShaper


def test_reward_shaping_wrapper():
    """
    Verifies that RewardShapingWrapper tracks transition observations,
    populates step metrics, and injects episode summaries upon termination/truncation.
    """
    env = gym.make("CartPole-v1")
    shaper = IdentityRewardShaper()
    wrapped_env = RewardShapingWrapper(env, shaper)

    # 1. Test Reset behavior
    obs, info = wrapped_env.reset(seed=42)
    assert wrapped_env._current_obs is not None
    assert wrapped_env._episode_original_reward == 0.0
    assert wrapped_env._episode_shaped_reward == 0.0
    assert wrapped_env._episode_length == 0

    # 2. Test Step behavior
    action = wrapped_env.action_space.sample()
    next_obs, reward, terminated, truncated, info = wrapped_env.step(action)

    # Step-level metrics should exist
    assert "step_original_reward" in info
    assert "step_shaped_reward" in info
    assert info["step_original_reward"] == reward
    assert info["step_shaped_reward"] == reward

    # 3. Test Terminal behavior
    done = terminated or truncated
    while not done:
        action = wrapped_env.action_space.sample()
        next_obs, reward, terminated, truncated, info = wrapped_env.step(action)
        done = terminated or truncated

    # Terminal step should contain aggregated episode-level metrics
    assert "original_reward" in info
    assert "shaped_reward" in info
    assert "episode_length" in info
    assert info["original_reward"] == wrapped_env._episode_original_reward
    assert info["shaped_reward"] == wrapped_env._episode_shaped_reward
    assert info["episode_length"] == wrapped_env._episode_length

    wrapped_env.close()


def test_wrapper_preserves_physics_and_termination():
    """
    Verifies that RewardShapingWrapper behaves identically to the raw environment
    for observation transitions, termination triggers, and truncation timeouts.
    """
    import numpy as np

    from reward_functions import get_reward_shaper

    raw_env = gym.make("CartPole-v1")
    dense_shaper = get_reward_shaper("dense", {"position_weight": 0.1, "angle_weight": 1.0})
    wrapped_env = RewardShapingWrapper(gym.make("CartPole-v1"), dense_shaper)

    # Reset both with identical seed
    obs_raw, info_raw = raw_env.reset(seed=100)
    obs_wrap, info_wrap = wrapped_env.reset(seed=100)

    assert np.allclose(obs_raw, obs_wrap), "Initial observations must be identical"

    # Step both environments using identical actions
    for step_idx in range(100):
        # Sample action from space using raw env seed state
        action = raw_env.action_space.sample()

        next_obs_raw, reward_raw, term_raw, trunc_raw, info_raw = raw_env.step(action)
        next_obs_wrap, reward_wrap, term_wrap, trunc_wrap, info_wrap = wrapped_env.step(action)

        # Verify state transition equivalence
        assert np.allclose(next_obs_raw, next_obs_wrap), f"States diverged at step {step_idx}"

        # Verify termination/truncation equivalence (Episode termination unchanged)
        assert term_raw == term_wrap, f"Termination mismatch at step {step_idx}"
        assert trunc_raw == trunc_wrap, f"Truncation mismatch at step {step_idx}"

        if term_raw or trunc_raw:
            break

    raw_env.close()
    wrapped_env.close()
