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
        assert (
            shaped == reward
        ), f"Identity shaper modified reward from {reward} to {shaped}"



def test_dense_shaper():
    """Verifies that DenseRewardShaper correctly implements the gradient direction of reward penalties."""
    # 1. Initialize dense shaper with config parameters
    shaper = get_reward_shaper(
        "dense", {"position_weight": 0.1, "angle_weight": 1.0, "max_bonus": 0.0}
    )

    state = [0.0, 0.0, 0.0, 0.0]
    action = 1
    reward = 1.0
    done = False
    info = {}

    # 2. Perfect state should yield maximum reward (no penalties)
    r_perfect = shaper.shape_reward(
        state, action, reward, [0.0, 0.0, 0.0, 0.0], done, info
    )
    assert r_perfect == 1.0

    # 3. Reward decreases as angle tilts
    r_tilt_small = shaper.shape_reward(
        state, action, reward, [0.0, 0.0, 0.05, 0.0], done, info
    )
    r_tilt_large = shaper.shape_reward(
        state, action, reward, [0.0, 0.0, 0.10, 0.0], done, info
    )
    assert r_tilt_small < r_perfect, "Small tilt should reduce reward"
    assert r_tilt_large < r_tilt_small, "Larger tilt should further reduce reward"

    # 4. Reward increases as pole angle uprights (relative check)
    assert r_tilt_small > r_tilt_large, "More upright pole should increase reward"

    # 5. Reward decreases as cart drifts from track center
    r_disp_small = shaper.shape_reward(
        state, action, reward, [0.5, 0.0, 0.0, 0.0], done, info
    )
    r_disp_large = shaper.shape_reward(
        state, action, reward, [1.0, 0.0, 0.0, 0.0], done, info
    )
    assert r_disp_small < r_perfect, "Drift should reduce reward"
    assert r_disp_large < r_disp_small, "Further drift should further reduce reward"

    # 6. Check finite values and absence of NaNs across bounds
    import math

    for pos in [0.0, 1.2, 2.4]:
        for angle in [0.0, 0.1, 0.209]:
            r_val = shaper.shape_reward(
                state, action, reward, [pos, 0.0, angle, 0.0], done, info
            )
            assert not math.isnan(r_val), "Dense reward should never contain NaN values"
            assert math.isfinite(r_val), "Dense reward must remain finite"


def test_pbrs_shaper():
    """Verifies that PBRSRewardShaper conforms to the policy invariance math and configuration contracts."""
    import math

    # 1. Initialize via factory
    shaper = get_reward_shaper(
        "pbrs",
        {
            "position_weight": 0.1,
            "velocity_weight": 0.0,
            "angle_weight": 1.0,
            "angular_velocity_weight": 0.0,
            "gamma": 0.99,
            "potential_type": "l1",
        },
    )
    from reward_functions.pbrs import PBRSRewardShaper

    assert isinstance(shaper, PBRSRewardShaper)

    # 2. Test potential function values manually
    state_a = [0.0, 0.0, 0.0, 0.0]
    state_b = [
        0.5,
        2.0,
        0.1,
        1.5,
    ]  # velocities should be ignored because weights are 0.0

    phi_a = shaper._potential(state_a)
    phi_b = shaper._potential(state_b)

    assert phi_a == 0.0, "Potential of perfect state should be 0.0"
    # L1: - (0.1 * |0.5| + 1.0 * |0.1|) = - (0.05 + 0.1) = -0.15
    assert (
        abs(phi_b - (-0.15)) < 1e-7
    ), f"Potential of state_b should be -0.15, got {phi_b}"

    # 3. Test determinism
    assert (
        shaper._potential(state_b) == phi_b
    ), "Potential function is not deterministic"

    # 4. Test shape_reward math matching theoretical formulation: R' = R + gamma * Phi(s') - Phi(s)
    # Case: Transition state_a -> state_b, not done
    r_shaped = shaper.shape_reward(state_a, 1, 1.0, state_b, False, {})
    # Expected shaping: 0.99 * (-0.15) - (0.0) = -0.1485
    # Expected R_shaped: 1.0 - 0.1485 = 0.8515
    assert (
        abs(r_shaped - 0.8515) < 1e-7
    ), f"Shaped reward calculation mismatch: expected 0.8515, got {r_shaped}"

    # 5. Test boundary condition: Phi(s_terminal) = 0.0
    # Case: Transition state_b -> state_a, done=True (episode terminated/truncated)
    r_shaped_done = shaper.shape_reward(state_b, 1, 1.0, state_a, True, {})
    # Expected shaping: 0.99 * 0.0 - (-0.15) = +0.15
    # Expected R_shaped: 1.0 + 0.15 = 1.15
    assert (
        abs(r_shaped_done - 1.15) < 1e-7
    ), f"Shaped reward on done mismatch: expected 1.15, got {r_shaped_done}"

    # 6. Test L2 potential formulation
    shaper_l2 = get_reward_shaper(
        "pbrs",
        {
            "position_weight": 0.1,
            "velocity_weight": 0.5,
            "angle_weight": 1.0,
            "angular_velocity_weight": 0.2,
            "gamma": 0.95,
            "potential_type": "l2",
        },
    )
    # L2: - (0.1 * 0.5^2 + 0.5 * 2.0^2 + 1.0 * 0.1^2 + 0.2 * 1.5^2)
    #     - (0.1 * 0.25 + 0.5 * 4.0 + 1.0 * 0.01 + 0.2 * 2.25)
    #     - (0.025 + 2.0 + 0.01 + 0.45) = -2.485
    phi_l2_b = shaper_l2._potential(state_b)
    assert (
        abs(phi_l2_b - (-2.485)) < 1e-7
    ), f"L2 potential calculation mismatch: expected -2.485, got {phi_l2_b}"

    # 7. Check absence of NaNs and check finiteness across bounds
    for pos in [0.0, 1.2, 2.4]:
        for angle in [0.0, 0.1, 0.209]:
            r_val = shaper.shape_reward(
                state_b, 1, 1.0, [pos, 0.0, angle, 0.0], False, {}
            )
            assert not math.isnan(r_val), "PBRS reward should never contain NaN values"
            assert math.isfinite(r_val), "PBRS reward must remain finite"
