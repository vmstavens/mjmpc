from .gym_env_wrapper import GymEnvWrapper
from gymnasium.envs.registration import register

register(
  id='SimplePendulum-v0',
  entry_point='mjmpc.envs.basic.pendulum:PendulumEnv',
  max_episode_steps=200
)

register(
  id='Swimmer-v0',
  entry_point='mjmpc.envs.basic.swimmer:SwimmerEnv',
)

register(
    id='HalfCheetah-v0',
    entry_point='mjmpc.envs.basic.half_cheetah:HalfCheetahEnv',
)

register(
    id='reacher_7dof-v0',
    entry_point='mjmpc.envs.basic.reacher_env:Reacher7DOFEnv',
    max_episode_steps=75,
)

register(
    id='continual_reacher-v0',
    entry_point='mjmpc.envs.basic.reacher_env:ContinualReacher7DOFEnv',
    max_episode_steps=250,
)

from mjmpc.envs.basic.reacher_env import Reacher7DOFEnv, ContinualReacher7DOFEnv
from mjmpc.envs.basic.lqr import LQREnv
