import pytest

from reward_functions import get_reward_shaper
from reward_functions.dense import (
    AcrobotDenseRewardShaper,
    LunarLanderDenseRewardShaper,
    MountainCarDenseRewardShaper,
)
from reward_functions.pbrs import (
    AcrobotPBRSRewardShaper,
    LunarLanderPBRSRewardShaper,
    MountainCarPBRSRewardShaper,
)


def test_factory_loading_generalization():
    """Verifies that factory routes strategy configurations correctly based on env_id."""
    # 1. MountainCar dense & pbrs
    shaper_mc_dense = get_reward_shaper("dense", env_id="MountainCar-v0")
    assert isinstance(shaper_mc_dense, MountainCarDenseRewardShaper)
    shaper_mc_pbrs = get_reward_shaper("pbrs", env_id="MountainCar-v0")
    assert isinstance(shaper_mc_pbrs, MountainCarPBRSRewardShaper)

    # 2. Acrobot dense & pbrs
    shaper_ac_dense = get_reward_shaper("dense", env_id="Acrobot-v1")
    assert isinstance(shaper_ac_dense, AcrobotDenseRewardShaper)
    shaper_ac_pbrs = get_reward_shaper("pbrs", env_id="Acrobot-v1")
    assert isinstance(shaper_ac_pbrs, AcrobotPBRSRewardShaper)

    # 3. LunarLander dense & pbrs
    shaper_ll_dense = get_reward_shaper("dense", env_id="LunarLander-v2")
    assert isinstance(shaper_ll_dense, LunarLanderDenseRewardShaper)
    shaper_ll_pbrs = get_reward_shaper("pbrs", env_id="LunarLander-v2")
    assert isinstance(shaper_ll_pbrs, LunarLanderPBRSRewardShaper)


def test_mountaincar_dense_shaper():
    """Verifies MountainCar dense shaper height and velocity calculations."""
    shaper = MountainCarDenseRewardShaper(height_weight=2.0, velocity_weight=50.0)

    # State: [position, velocity]
    # Pos = 0.0, height_bonus = 2.0 * (sin(0.0) + 1.0) = 2.0
    # Vel = 0.02, vel_bonus = 50.0 * (0.0004) = 0.02
    # reward = -1.0
    # expected shaped reward = -1.0 + 2.0 + 0.02 = 1.02
    shaped = shaper.shape_reward(
        state=[0.0, 0.0],
        action=2,
        reward=-1.0,
        next_state=[0.0, 0.02],
        done=False,
        info={},
    )
    assert pytest.approx(shaped) == 1.02


def test_mountaincar_pbrs_shaper():
    """Verifies MountainCar PBRS potential differences and boundary conditions."""
    shaper = MountainCarPBRSRewardShaper(
        height_weight=1.0, velocity_weight=10.0, gamma=0.99
    )

    state = [-0.5, 0.0]
    next_state = [0.0, 0.02]

    # potentials:
    # phi(state) = 1.0 * (sin(-1.5) + 1.0) + 10.0 * 0.0 = (approx -0.99749 + 1.0) = 0.00251
    # phi(next_state) = 1.0 * (sin(0.0) + 1.0) + 10.0 * 0.0004 = 1.0 + 0.004 = 1.004
    # shaping = gamma * phi(next_state) - phi(state) = 0.99 * 1.004 - 0.00251 = 0.99396 - 0.00251 = 0.99145
    # expected = -1.0 + 0.99145 = -0.00855
    shaped = shaper.shape_reward(
        state=state,
        action=2,
        reward=-1.0,
        next_state=next_state,
        done=False,
        info={},
    )
    assert pytest.approx(shaped, rel=1e-3) == -0.00855

    # Boundary Condition: done=True should set next potential to 0
    # shaping = 0.99 * 0.0 - phi(state) = -0.00251
    # expected = -1.0 - 0.00251 = -1.00251
    shaped_done = shaper.shape_reward(
        state=state,
        action=2,
        reward=-1.0,
        next_state=next_state,
        done=True,
        info={},
    )
    assert pytest.approx(shaped_done, rel=1e-3) == -1.00251


def test_acrobot_dense_shaper():
    """Verifies Acrobot dense shaper tip height calculations."""
    shaper = AcrobotDenseRewardShaper(height_weight=1.0, velocity_weight=0.5)

    # next_state is [cos(theta1), sin(theta1), cos(theta2), sin(theta2), theta1_dot, theta2_dot]
    # If straight down: cos(theta1)=1, sin(theta1)=0, cos(theta2)=1, sin(theta2)=0, dots = 0
    # tip height = -1 - (1 * 1 - 0 * 0) = -2
    # expected = -1.0 + 1.0 * (-2 + 2.0) + 0.5 * 0 = -1.0
    shaped_down = shaper.shape_reward(
        state=[1.0] * 6,
        action=1,
        reward=-1.0,
        next_state=[1.0, 0.0, 1.0, 0.0, 0.0, 0.0],
        done=False,
        info={},
    )
    assert pytest.approx(shaped_down) == -1.0

    # If straight up: cos(theta1)=-1, sin(theta1)=0, cos(theta2)=1, sin(theta2)=0, dots = 0
    # tip height = -(-1) - (-1 * 1 - 0) = 1 + 1 = 2
    # expected = -1.0 + 1.0 * (2 + 2.0) = 3.0
    shaped_up = shaper.shape_reward(
        state=[1.0] * 6,
        action=1,
        reward=-1.0,
        next_state=[-1.0, 0.0, 1.0, 0.0, 0.0, 0.0],
        done=False,
        info={},
    )
    assert pytest.approx(shaped_up) == 3.0


def test_acrobot_pbrs_shaper():
    """Verifies Acrobot PBRS potential differences and boundary conditions."""
    shaper = AcrobotPBRSRewardShaper(height_weight=1.0, velocity_weight=0.1, gamma=0.99)

    # State: straight down (tip_height = -2.0, phi = 1.0 * 0.0 + 0 = 0)
    state = [1.0, 0.0, 1.0, 0.0, 0.0, 0.0]
    # Next State: straight up (tip_height = 2.0, phi = 1.0 * 4.0 + 0 = 4.0)
    next_state = [-1.0, 0.0, 1.0, 0.0, 0.0, 0.0]

    # done=False: shaping = 0.99 * 4.0 - 0.0 = 3.96, expected = -1.0 + 3.96 = 2.96
    shaped = shaper.shape_reward(
        state=state,
        action=1,
        reward=-1.0,
        next_state=next_state,
        done=False,
        info={},
    )
    assert pytest.approx(shaped) == 2.96

    # done=True: shaping = 0.99 * 0.0 - 0.0 = 0.0, expected = -1.0 + 0.0 = -1.0
    shaped_done = shaper.shape_reward(
        state=state,
        action=1,
        reward=-1.0,
        next_state=next_state,
        done=True,
        info={},
    )
    assert pytest.approx(shaped_done) == -1.0


def test_lunarlander_dense_shaper():
    """Verifies LunarLander dense shaper coordinate and stabilization penalties."""
    shaper = LunarLanderDenseRewardShaper(
        distance_weight=0.5, velocity_weight=0.5, angle_weight=1.0
    )

    # next_state: [x, y, vx, vy, theta, vtheta, left_leg, right_leg]
    # next_state = [0.2, 0.5, -0.1, -0.4, 0.1, 0.0, 0.0, 0.0]
    # penalties:
    # dist_penalty = 0.5 * 0.2 = 0.1
    # vel_penalty = 0.5 * (0.1 + 0.4) = 0.25
    # angle_penalty = 1.0 * 0.1 = 0.1
    # total penalty = 0.45
    # expected shaped reward = 10.0 - 0.45 = 9.55
    shaped = shaper.shape_reward(
        state=[0.0] * 8,
        action=0,
        reward=10.0,
        next_state=[0.2, 0.5, -0.1, -0.4, 0.1, 0.0, 0.0, 0.0],
        done=False,
        info={},
    )
    assert pytest.approx(shaped) == 9.55


def test_lunarlander_pbrs_shaper():
    """Verifies LunarLander PBRS potential differences and boundary conditions."""
    shaper = LunarLanderPBRSRewardShaper(
        distance_weight=1.0, velocity_weight=1.0, angle_weight=1.0, gamma=0.99
    )

    # state: [0.0] * 8 (phi = -1.0 * 0.0 - 1.0 * 0.0 - 1.0 * 0 = 0.0)
    state = [0.0] * 8
    # next_state: [0.3, 0.4, -0.1, -0.1, 0.1, 0.0, 0.0, 0.0]
    # dist = sqrt(0.09 + 0.16) = 0.5
    # vel = sqrt(0.01 + 0.01) = sqrt(0.02) approx 0.14142
    # angle = 0.1
    # phi(next_state) = -1.0 * 0.5 - 1.0 * 0.14142 - 1.0 * 0.1 = -0.74142
    # done=False: shaping = 0.99 * -0.74142 - 0.0 = -0.734006
    # expected shaped = 5.0 - 0.734006 = 4.26599
    shaped = shaper.shape_reward(
        state=state,
        action=0,
        reward=5.0,
        next_state=[0.3, 0.4, -0.1, -0.1, 0.1, 0.0, 0.0, 0.0],
        done=False,
        info={},
    )
    assert pytest.approx(shaped, abs=1e-3) == 4.266
