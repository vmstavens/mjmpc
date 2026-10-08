"""Adapt Gymnasium environments to the MPC controller rollout contract.

The controller-facing interface retains observation-only reset and four-value
step; termination and truncation are also retained separately in step info.
"""

from collections import defaultdict
from copy import deepcopy
import time

from gymnasium import spaces
import mujoco
import numpy as np
import torch

from .basic.mujoco_state import get_mujoco_state, set_mujoco_state


class GymEnvWrapper:
    def __init__(self, env):
        self.env = env
        self.metadata = env.metadata
        self.observation_space = env.observation_space
        self.action_space = env.action_space
        self.d_obs = spaces.flatdim(self.observation_space)
        self.d_action = spaces.flatdim(self.action_space)
        self._max_episode_steps = env.spec.max_episode_steps if env.spec else None
        self.default_dyn_params = defaultdict(dict)
        self.randomized_dyn_params = defaultdict(dict)
        self._renderer = None
        self.reset()
        physical = self.get_env_state()["env_state"]
        self.d_state = sum(np.size(v) for v in physical.values() if v is not None)

    @property
    def unwrapped(self):
        return self.env.unwrapped

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        info = dict(info, terminated=terminated, truncated=truncated)
        return deepcopy(obs), float(reward), bool(terminated or truncated), deepcopy(info)

    def reset(self, seed=None):
        obs, self.reset_info = self.env.reset(seed=seed)
        if seed is not None:
            self.action_space.seed(seed)
        return deepcopy(obs)

    def seed(self, seed=None):
        """Seed rollout randomness without resetting the physical state."""
        from gymnasium.utils.seeding import np_random

        self.unwrapped.np_random, seed = np_random(seed)
        self.action_space.seed(seed)
        return [seed]

    def _wrappers(self):
        env = self.env
        while env is not self.unwrapped:
            yield env
            env = env.env

    def get_env_state(self):
        env = self.unwrapped
        if hasattr(env, "get_env_state"):
            physical = env.get_env_state()
        elif hasattr(env, "model") and hasattr(env, "data"):
            physical = {"integration_state": get_mujoco_state(env.model, env.data)}
        elif hasattr(env, "state"):
            physical = {"state": deepcopy(env.state)}
        else:
            raise TypeError(
                "Environment must expose get_env_state/set_env_state or MuJoCo model/data"
            )
        return deepcopy(
            {
                "env_state": physical,
                "rng_state": env.np_random.bit_generator.state,
                "elapsed_steps": [
                    w._elapsed_steps for w in self._wrappers() if "_elapsed_steps" in vars(w)
                ],
            }
        )

    def set_env_state(self, state):
        env = self.unwrapped
        physical = state["env_state"]
        if hasattr(env, "set_env_state"):
            env.set_env_state(deepcopy(physical))
        elif "integration_state" in physical:
            set_mujoco_state(env.model, env.data, physical["integration_state"])
        else:
            env.state = deepcopy(physical["state"])
        env.np_random.bit_generator.state = deepcopy(state["rng_state"])
        elapsed = iter(state["elapsed_steps"])
        for wrapper in self._wrappers():
            if "_elapsed_steps" in vars(wrapper):
                wrapper._elapsed_steps = next(elapsed)

    def get_obs(self):
        env = self.unwrapped
        if hasattr(env, "get_obs"):
            return deepcopy(env.get_obs())
        if hasattr(env, "_get_obs"):
            return deepcopy(env._get_obs())
        if hasattr(env, "state"):
            return np.asarray(env.state, dtype=self.observation_space.dtype).copy()
        raise TypeError("Environment must expose get_obs or _get_obs")

    def real_env_step(self, value):
        self.unwrapped.real_step = bool(value)

    def evaluate_success(self, trajectories):
        evaluate = getattr(self.unwrapped, "evaluate_success", None)
        return evaluate(trajectories) if evaluate is not None else None

    def close(self):
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None
        self.env.close()

    def render(self):
        return self.env.render()

    def get_curr_frame(self, frame_size=(640, 480), camera_name=None, device_id=0):
        env = self.unwrapped
        if hasattr(env, "model"):
            width, height = frame_size
            if (
                self._renderer is not None
                and (self._renderer.width, self._renderer.height) != frame_size
            ):
                self._renderer.close()
                self._renderer = None
            if self._renderer is None:
                env.model.vis.global_.offwidth = max(width, env.model.vis.global_.offwidth)
                env.model.vis.global_.offheight = max(height, env.model.vis.global_.offheight)
                self._renderer = mujoco.Renderer(env.model, height=height, width=width)
            self._renderer.update_scene(env.data, camera=camera_name if camera_name else -1)
            return self._renderer.render().copy()
        frame = self.env.render()
        if frame is None:
            raise ValueError("Create the environment with render_mode='rgb_array' to record video")
        return frame

    def randomize_dynamics(self, param_dict=None):
        env = self.unwrapped
        model = env.model
        kinds = {
            "body_mass": "body",
            "body_inertia": "body",
            "dof_damping": "joint",
            "dof_frictionloss": "joint",
            "geom_size": "geom",
            "geom_friction": "geom",
        }
        for field_name, parameters in (param_dict or {}).items():
            if field_name not in kinds:
                raise ValueError(f"Unsupported dynamics field: {field_name}")
            field = getattr(model, field_name)
            for name, (noise_scale, bias_scale) in parameters.items():
                obj = getattr(model, kinds[field_name])(name)
                index = obj.id
                if field_name.startswith("dof_"):
                    start = model.jnt_dofadr[index]
                    end = model.jnt_dofadr[index + 1] if index + 1 < model.njnt else model.nv
                    index = slice(start, end)
                defaults = self.default_dyn_params[field_name]
                if name not in defaults:
                    defaults[name] = field[index].copy()
                mean = (1 + bias_scale) * defaults[name]
                radius = np.abs(mean) * noise_scale
                value = env.np_random.uniform(mean - radius, mean + radius)
                field[index] = value
                self.randomized_dyn_params[field_name][name] = deepcopy(value)
        # Recompute mass/inertia constants while retaining the current trajectory state.
        state = get_mujoco_state(model, env.data)
        mujoco.mj_setConst(model, env.data)
        set_mujoco_state(model, env.data, state)
        return deepcopy(self.default_dyn_params), deepcopy(self.randomized_dyn_params)

    def _rollout(self, batch_size, horizon, action_fn):
        if batch_size < 1 or horizon < 1:
            raise ValueError("batch_size and horizon must be positive")
        shape = (batch_size, horizon)
        observations = np.zeros((*shape, self.d_obs))
        next_observations = np.zeros_like(observations)
        rewards = np.zeros(shape)
        actions = np.zeros((*shape, self.d_action))
        dones = np.zeros(shape, dtype=bool)
        action_infos = []
        state = self.get_env_state()
        started = time.perf_counter()
        inference_time = 0.0
        try:
            for b in range(batch_size):
                self.set_env_state(state)
                obs = self.get_obs()
                done = False
                for t in range(horizon):
                    flat_obs = spaces.flatten(self.observation_space, obs)
                    observations[b, t] = flat_obs
                    if done:
                        next_observations[b, t] = flat_obs
                        dones[b, t] = True
                        action_infos.append({})
                        continue
                    before = time.perf_counter()
                    action, action_info = action_fn(b, t, flat_obs)
                    inference_time += time.perf_counter() - before
                    obs, reward, done, _ = self.step(action)
                    actions[b, t] = action
                    rewards[b, t] = reward
                    dones[b, t] = done
                    next_observations[b, t] = spaces.flatten(self.observation_space, obs)
                    action_infos.append(action_info)
        finally:
            self.set_env_state(state)
        info = {"total_time": time.perf_counter() - started, "inference_time": inference_time}
        return observations, rewards, actions, dones, info, next_observations, action_infos

    def rollout(self, batch_size, horizon, mean, noise=None, mode="open_loop"):
        """Roll out candidates; zero-pad rewards after episode end and restore state."""
        if mode not in {"open_loop", "closed_loop_linear"}:
            raise ValueError(f"Unknown rollout mode: {mode}")

        def action_fn(b, t, obs):
            action = mean[t] if mode == "open_loop" else mean.T @ np.append(obs, 1.0)
            if noise is not None:
                action = action + noise[b, t]
            return action, {}

        return self._rollout(batch_size, horizon, action_fn)[:6]

    def rollout_cl(self, policy, batch_size, horizon, mode="mean", noise=None):
        def action_fn(b, t, obs):
            tensor = torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0)
            sample = noise[b, t] if noise is not None else None
            action, info = policy.get_action(tensor, mode, sample)
            return action.detach().cpu().numpy().reshape(self.d_action), info

        with torch.no_grad():
            obs, rewards, actions, dones, info, next_obs, action_infos = self._rollout(
                batch_size, horizon, action_fn
            )
        return obs, actions, action_infos, rewards, dones, next_obs, info
