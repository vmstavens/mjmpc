import gymnasium as gym
from gymnasium import spaces
import numpy as np


class LQREnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, A, B, Q, R):
        self.A = A
        self.B = B
        self.Q = Q
        self.R = R
        self.d_state = A.shape[0]
        self.d_action = B.shape[-1]
        self.viewer = None

        high = np.array([100] * self.d_state)
        self.action_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(self.d_action,), dtype=np.float64
        )
        self.observation_space = spaces.Box(low=-high, high=high, dtype=np.float64)
        self.reset()

    def step(self, u):
        u = np.asarray(u).reshape(self.d_action)
        cost = self.state.T.dot(self.Q).dot(self.state) + u.T.dot(self.R).dot(u)
        self.state = self.A.dot(self.state) + self.B.dot(u)
        return self._get_obs(), -float(cost), False, False, {}

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        high = np.array([100] * self.d_state)
        self.state = self.np_random.uniform(low=-high, high=high).reshape(self.d_state)
        return self.state.copy(), {}

    def _get_obs(self):
        return self.state.copy()

    def get_env_state(self):
        return {"state": self.state.copy()}

    def set_env_state(self, state_dict):
        self.state = np.asarray(state_dict["state"]).reshape(self.d_state).copy()
