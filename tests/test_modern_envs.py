from copy import deepcopy

import gymnasium as gym
from gymnasium.utils.env_checker import check_env
import numpy as np
import pytest

from mjmpc.envs import GymEnvWrapper, LQREnv

BUNDLED = [
    "SimplePendulum-v0",
    "Swimmer-v0",
    "HalfCheetah-v0",
    "reacher_7dof-v0",
    "continual_reacher-v0",
]


@pytest.mark.parametrize("name", BUNDLED)
def test_gymnasium_contract(name):
    env = gym.make(name)
    try:
        check_env(env.unwrapped, skip_render_check=True)
    finally:
        env.close()


@pytest.mark.parametrize("name", BUNDLED + ["HalfCheetah-v5", "Pendulum-v1"])
def test_snapshot_reproduces_transition_and_rollouts(name):
    env = GymEnvWrapper(gym.make(name))
    try:
        env.reset(seed=7)
        action = np.full(env.d_action, 0.1)
        for _ in range(3):
            env.step(action)
        state = env.get_env_state()
        expected = env.step(action)
        env.set_env_state(state)
        actual = env.step(action)
        np.testing.assert_allclose(actual[0], expected[0], atol=1e-12)
        assert actual[1:3] == expected[1:3]
        env.set_env_state(state)
        mean = np.tile(action, (4, 1))
        obs, rewards, actions, dones, _, next_obs = env.rollout(2, 4, mean)
        np.testing.assert_array_equal(obs[0], obs[1])
        np.testing.assert_array_equal(rewards[0], rewards[1])
        np.testing.assert_allclose(env.step(action)[0], expected[0], atol=1e-12)
        assert actions.shape == (2, 4, env.d_action)
        assert not dones.any()
        assert next_obs.shape == obs.shape
    finally:
        env.close()


def test_time_limit_and_failed_rollout_restore_state():
    env = GymEnvWrapper(gym.make("SimplePendulum-v0", max_episode_steps=2))
    try:
        env.reset(seed=0)
        env.step(np.zeros(1))
        state = env.get_env_state()
        obs, rewards, _, dones, _, _ = env.rollout(2, 4, np.zeros((4, 1)))
        np.testing.assert_array_equal(rewards[0], rewards[1])
        assert np.all(rewards[:, 1:] == 0)
        assert dones.all()
        assert env.get_env_state()["elapsed_steps"] == state["elapsed_steps"]
        with pytest.raises(IndexError):
            env.rollout(1, 4, np.empty((0, 1)))
        np.testing.assert_array_equal(env.get_obs(), obs[0, 0])
        _, _, done, info = env.step(np.zeros(1))
        assert done and info["truncated"] and not info["terminated"]
    finally:
        env.close()


def test_continual_reacher_restores_rng_and_target():
    env = GymEnvWrapper(gym.make("continual_reacher-v0"))
    try:
        env.reset(seed=9)
        env.unwrapped.env_timestep = 49
        state = env.get_env_state()
        env.step(np.zeros(7))
        expected = env.unwrapped.model.site("target").pos.copy()
        env.set_env_state(state)
        env.step(np.zeros(7))
        np.testing.assert_array_equal(env.unwrapped.model.site("target").pos, expected)
    finally:
        env.close()


def test_dynamics_randomization_is_seeded_and_relative_to_defaults():
    env = GymEnvWrapper(gym.make("reacher_7dof-v0"))
    try:
        env.reset(seed=0)
        params = {
            "body_mass": {"r_shoulder_pan_link": [0.2, 0.1]},
            "dof_damping": {"r_shoulder_pan_joint": [0.2, 0.1]},
        }
        state = deepcopy(env.get_obs())
        env.seed(42)
        defaults, first = env.randomize_dynamics(params)
        env.seed(42)
        _, second = env.randomize_dynamics(params)
        for field in params:
            for name in params[field]:
                np.testing.assert_array_equal(first[field][name], second[field][name])
                assert np.all(np.asarray(defaults[field][name]) > 0)
        np.testing.assert_array_equal(env.get_obs(), state)
    finally:
        env.close()


def test_lqr_contract_and_cost():
    env = LQREnv(np.eye(2), np.ones((2, 1)), np.eye(2), np.eye(1))
    check_env(env, skip_render_check=True)
    env.set_env_state({"state": np.array([1.0, 2.0])})
    obs, reward, terminated, truncated, _ = env.step(np.array([0.5]))
    np.testing.assert_array_equal(obs, [1.5, 2.5])
    assert reward == -5.25
    assert not terminated and not truncated
