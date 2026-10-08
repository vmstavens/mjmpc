import gymnasium as gym
import numpy as np
import pytest
import torch

from mjmpc.envs import GymEnvWrapper
from mjmpc.policies import MPCPolicy
from mjmpc.value_functions import (
    LinearVF,
    LinearTimeVaryingVF,
    QuadraticVF,
    QuadraticTimeVaryingVF,
)


@pytest.mark.parametrize("name", ["mppi", "cem", "dmd", "random_shooting", "pfmpc", "reinforce"])
def test_controller_optimizes_real_rollouts(name):
    env = GymEnvWrapper(gym.make("SimplePendulum-v0"))
    try:
        env.reset(seed=5)
        params = dict(
            d_state=env.d_state,
            d_obs=env.d_obs,
            d_action=env.d_action,
            action_lows=env.action_space.low,
            action_highs=env.action_space.high,
            horizon=8,
            num_particles=16,
            gamma=0.99,
            n_iters=2,
            base_action="null",
            seed=11,
        )
        if name == "pfmpc":
            params.update(cov_shift=0.5, cov_resample=1.0, lam=1.0)
        else:
            params.update(init_cov=1.0, step_size=1.0)
        if name == "mppi":
            params.update(lam=1.0, alpha=0.0)
        elif name == "cem":
            params.update(elite_frac=0.25)
        elif name == "dmd":
            params.update(lam=1.0, beta=0.01)
        if name == "reinforce":
            params.pop("base_action")
            params.pop("step_size")
            params.update(lr=0.01, beta=0.0, loss_thresh=0.001, delta_kl=1.0)
        policy = MPCPolicy(name, params)

        def rollout(*args, **kwargs):
            obs, rewards, actions, dones, info, next_obs = env.rollout(*args, **kwargs)
            return dict(
                observations=obs,
                costs=-rewards,
                actions=actions,
                dones=dones,
                infos=info,
                next_observations=next_obs,
            )

        policy.controller.set_sim_state_fn = env.set_env_state
        policy.controller.rollout_fn = rollout
        for _ in range(3):
            action, value = policy.get_action(env.get_env_state())
            assert action.shape == (1,)
            assert np.isfinite(action).all()
            assert np.isfinite(env.step(action)[1])
    finally:
        env.close()


@pytest.mark.parametrize(
    "cls", [LinearVF, QuadraticVF, LinearTimeVaryingVF, QuadraticTimeVaryingVF]
)
def test_value_function_fit_uses_current_torch(cls):
    torch.manual_seed(12)
    obs = torch.randn(32, 5, 2)
    returns = 2 * obs[..., 0] - obs[..., 1] + 0.5
    model = cls(2, 5) if "TimeVarying" in cls.__name__ else cls(2)
    before, after = model.fit(obs, returns, delta_reg=1e-4, return_errors=True)
    assert after < before * 0.01
    assert model(obs).shape == returns.shape


def test_sac_training_update():
    from mjmpc.control.softqmpc.algs.sac import SAC, ReplayMemory

    env = GymEnvWrapper(gym.make("SimplePendulum-v0"))
    try:
        args = dict(
            gamma=0.99,
            tau=0.005,
            alpha=0.2,
            policy="Gaussian",
            target_update_interval=1,
            automatic_entropy_tuning=True,
            cuda=False,
            hidden_size=16,
            lr=0.001,
        )
        agent = SAC(3, env.action_space, args)
        memory = ReplayMemory(32)
        obs = env.reset(seed=8)
        for _ in range(16):
            action = agent.get_action(obs)
            next_obs, reward, done, info = env.step(action)
            memory.push(obs, action, reward, next_obs, float(not info["terminated"]))
            obs = next_obs
        losses = agent.update_parameters(memory, 8, 0)
        assert np.isfinite(losses).all()
    finally:
        env.close()
