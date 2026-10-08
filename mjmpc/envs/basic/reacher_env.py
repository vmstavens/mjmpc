import numpy as np
from gymnasium import utils
from gymnasium.envs.mujoco import mujoco_env
from gymnasium import spaces
from .mujoco_state import MujocoStateMixin
import mujoco
import os


class Reacher7DOFEnv(MujocoStateMixin, mujoco_env.MujocoEnv, utils.EzPickle):
    metadata = {"render_modes": ["human", "rgb_array", "depth_array"], "render_fps": 50}

    def __init__(self, **kwargs):

        # trajopt specific attributes
        self.seeding = False
        self.real_step = True
        self.env_timestep = 0

        # placeholder
        self.hand_sid = -2
        self.target_sid = -1

        curr_dir = os.path.dirname(os.path.abspath(__file__))
        mujoco_env.MujocoEnv.__init__(
            self,
            curr_dir + "/../assets/xml/sawyer.xml",
            2,
            observation_space=spaces.Box(-np.inf, np.inf, (20,), dtype=np.float64),
            **kwargs,
        )
        utils.EzPickle.__init__(self, **kwargs)
        self.observation_dim = 20
        self.action_dim = 7

        self.hand_sid = self.model.site("finger").id
        self.target_sid = self.model.site("target").id

    def step(self, a):
        self.do_simulation(a, self.frame_skip)
        hand_pos = self.data.site_xpos[self.hand_sid]
        target_pos = self.data.site_xpos[self.target_sid]
        l1_dist = np.sum(np.abs(hand_pos - target_pos))
        l2_dist = np.linalg.norm(hand_pos - target_pos)
        reward = -l1_dist - 5.0 * l2_dist
        ob = self.get_obs()
        self.env_timestep += 1  # keep track of env timestep for timed events
        self.trigger_timed_events()
        if self.render_mode == "human":
            self.render()
        return ob, reward, False, False, self.get_env_infos()

    def get_obs(self):
        return np.concatenate(
            [
                self.data.qpos.flat,
                self.data.qvel.flat,
                self.data.site_xpos[self.hand_sid],
                self.data.site_xpos[self.hand_sid] - self.data.site_xpos[self.target_sid],
            ]
        )

    # --------------------------------
    # resets and randomization
    # --------------------------------

    def robot_reset(self):
        self.set_state(self.init_qpos, self.init_qvel)

    def target_reset(self):
        target_pos = np.array([0.1, 0.1, 0.1])
        target_pos[0] = self.np_random.uniform(low=-0.3, high=0.3)
        target_pos[1] = self.np_random.uniform(low=-0.2, high=0.2)
        target_pos[2] = self.np_random.uniform(low=-0.25, high=0.25)
        self.model.site_pos[self.target_sid] = target_pos
        mujoco.mj_forward(self.model, self.data)

    def reset_model(self):
        self.robot_reset()
        self.target_reset()
        self.env_timestep = 0
        return self.get_obs()

    def trigger_timed_events(self):
        # will be used in the continual version
        pass

    # --------------------------------
    # get and set states
    # --------------------------------

    def get_env_state(self):
        state = super().get_env_state()
        state.update(
            target_pos=self.model.site_pos[self.target_sid].copy(), timestep=self.env_timestep
        )
        return state

    def set_env_state(self, state):
        self.model.site_pos[self.target_sid] = state["target_pos"]
        self.env_timestep = state["timestep"]
        super().set_env_state(state)

    def get_env_infos(self):
        l2_dist = np.linalg.norm(
            self.data.site_xpos[self.hand_sid] - self.data.site_xpos[self.target_sid]
        )
        goal_achieved = l2_dist < 0.025
        return dict(state=self.get_env_state(), goal_achieved=goal_achieved)

    def evaluate_success(self, paths):
        num_success = 0
        num_paths = len(paths)
        # success if hand close to target for at least 10 steps
        for path in paths:
            if np.sum(path["env_infos"]["goal_achieved"]) > 10:
                num_success += 1
        success_percentage = num_success * 100.0 / num_paths
        return success_percentage


class ContinualReacher7DOFEnv(Reacher7DOFEnv):
    def trigger_timed_events(self):
        if self.env_timestep % 50 == 0 and self.env_timestep > 0 and self.real_step is True:
            self.target_reset()
