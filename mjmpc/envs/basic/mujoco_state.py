"""Full native MuJoCo integration state for repeatable MPC rollouts."""

import mujoco
import numpy as np

STATE_SPEC = mujoco.mjtState.mjSTATE_INTEGRATION


def get_mujoco_state(model, data):
    state = np.empty(mujoco.mj_stateSize(model, STATE_SPEC))
    mujoco.mj_getState(model, data, state, STATE_SPEC)
    return state


def set_mujoco_state(model, data, state):
    mujoco.mj_setState(model, data, state, STATE_SPEC)
    mujoco.mj_forward(model, data)
    # Forward computes accelerations; retain the saved solver warm start.
    mujoco.mj_setState(model, data, state, STATE_SPEC)


class MujocoStateMixin:
    def get_env_state(self):
        return {"integration_state": get_mujoco_state(self.model, self.data)}

    def set_env_state(self, state):
        set_mujoco_state(self.model, self.data, state["integration_state"])

    def get_obs(self):
        return self._get_obs().copy()
