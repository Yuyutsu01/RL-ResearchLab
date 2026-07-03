import pytest
# ci-trigger

from reward_functions import get_reward_shaper
from reward_functions.identity import IdentityRewardShaper


def test_factory_loading():
    """Verifies that the factory method creates correct shaper instances and raises error on invalid ones."""
    # Identity loader
    shaper = get_reward_shaper("identity")
    assert isinstance(shaper, IdentityRewardShaper)

    # Check case-insensitivity
    shaper_case = get_reward_shaper("  IdEnTiTy ")
    assert isinstance(shaper_case, IdentityRewardShaper)

    # Check invalid loader
    with pytest.raises(ValueError) as excinfo:
        get_reward_shaper("non_existent_shaper")
    assert "Unknown reward shaping strategy" in str(excinfo.value)


def test_identity_shaper():
    """Verifies that the IdentityRewardShaper is a perfect pass-through function."""
    shaper = get_reward_shaper("identity")

    # Reset should run without error
    shaper.reset(initial_state=0.0, info={})

    # Test cases for shape_reward
    test_transitions = [
        # (state, action, reward, next_state, done, info)
        (0.5, 1, 10.0, 0.6, False, {}),
        (0.1, 0, -1.0, 0.0, True, {"terminal_obs": 0.0}),
        (0.0, 0, 0.0, 0.0, False, {}),
    ]

    for state, action, reward, next_state, done, info in test_transitions:
        shaped = shaper.shape_reward(state, action, reward, next_state, done, info)
        assert shaped == reward, f"Identity shaper modified reward from {reward} to {shaped}"


def test_dense_shaper():
    """Verifies that DenseRewardShaper correctly implements the gradient direction of reward penalties."""
    # 1. Initialize dense shaper with config parameters
    shaper = get_reward_shaper("dense", {"position_weight": 0.1, "angle_weight": 1.0, "max_bonus": 0.0})

    state = [0.0, 0.0, 0.0, 0.0]
    action = 1
    reward = 1.0
    done = False
    info = {}

    # 2. Perfect state should yield maximum reward (no penalties)
    r_perfect = shaper.shape_reward(state, action, reward, [0.0, 0.0, 0.0, 0.0], done, info)
    assert r_perfect == 1.0

    # 3. Reward decreases as angle tilts
    r_tilt_small = shaper.shape_reward(state, action, reward, [0.0, 0.0, 0.05, 0.0], done, info)
    r_tilt_large = shaper.shape_reward(state, action, reward, [0.0, 0.0, 0.10, 0.0], done, info)
    assert r_tilt_small < r_perfect, "Small tilt should reduce reward"
    assert r_tilt_large < r_tilt_small, "Larger tilt should further reduce reward"

    # 4. Reward increases as pole angle uprights (relative check)
    assert r_tilt_small > r_tilt_large, "More upright pole should increase reward"

    # 5. Reward decreases as cart drifts from track center
    r_disp_small = shaper.shape_reward(state, action, reward, [0.5, 0.0, 0.0, 0.0], done, info)
    r_disp_large = shaper.shape_reward(state, action, reward, [1.0, 0.0, 0.0, 0.0], done, info)
    assert r_disp_small < r_perfect, "Drift should reduce reward"
    assert r_disp_large < r_disp_small, "Further drift should further reduce reward"

    # 6. Check finite values and absence of NaNs across bounds
    import math

    for pos in [0.0, 1.2, 2.4]:
        for angle in [0.0, 0.1, 0.209]:
            r_val = shaper.shape_reward(state, action, reward, [pos, 0.0, angle, 0.0], done, info)
            assert not math.isnan(r_val), "Dense reward should never contain NaN values"
            assert math.isfinite(r_val), "Dense reward must remain finite"
