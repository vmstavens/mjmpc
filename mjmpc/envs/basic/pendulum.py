"""Original MJMPC pendulum dynamics with Gymnasium rendering and seeding."""

import numpy as np
from gymnasium.envs.classic_control.pendulum import PendulumEnv as GymnasiumPendulum


class PendulumEnv(GymnasiumPendulum):
    def step(self, u):
        th, thdot = self.state  # th := theta

        g = self.g
        m = self.m
        l = self.l
        dt = self.dt

        u = np.clip(u, -self.max_torque, self.max_torque)[0]
        self.last_u = u  # for rendering
        costs = angle_normalize(th) ** 2 + 0.1 * thdot**2 + 0.001 * (u**2)

        newthdot = thdot + (-3 * g / (2 * l) * np.sin(th + np.pi) + 3.0 / (m * l**2) * u) * dt
        newth = th + newthdot * dt
        newthdot = np.clip(newthdot, -self.max_speed, self.max_speed)  # pylint: disable=E1111

        self.state = np.array([newth, newthdot])
        if self.render_mode == "human":
            self.render()
        return self._get_obs(), -float(costs), False, False, {}

    def get_obs(self):
        return self._get_obs()

    def get_env_state(self):
        return {"state": self.state.copy(), "last_u": self.last_u}

    def set_env_state(self, state):
        self.state = state["state"].copy()
        self.last_u = state.get("last_u")

    def evaluate_success(self, trajectories):
        return 0.0


def angle_normalize(x):
    return (x + np.pi) % (2 * np.pi) - np.pi
