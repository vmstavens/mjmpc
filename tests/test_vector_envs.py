import gymnasium as gym
import numpy as np

from mjmpc.envs import GymEnvWrapper
from mjmpc.envs.vec_env import DummyVecEnv, SubprocVecEnv, TorchModelVecEnv
from mjmpc.policies import LinearGaussianPolicy


def make_env():
    return GymEnvWrapper(gym.make("SimplePendulum-v0"))


def test_subprocess_rollouts_match_serial():
    serial = make_env()
    parallel = SubprocVecEnv([make_env, make_env], start_method="spawn")
    try:
        serial.reset(seed=4)
        parallel.set_env_state(serial.get_env_state())
        mean = np.zeros((4, 1))
        noise = np.random.default_rng(7).normal(size=(4, 4, 1))
        expected = serial.rollout(4, 4, mean, noise)
        actual = parallel.rollout(4, 4, mean, noise)
        for index in (0, 1, 2, 3, 5):
            np.testing.assert_array_equal(actual[index], expected[index])
        assert parallel.reset().shape == (2, 3)
        assert parallel.step(np.zeros((2, 1)))[0].shape == (2, 3)
    finally:
        serial.close()
        parallel.close()


def test_dummy_env_reset_and_step():
    env = DummyVecEnv([make_env, make_env])
    try:
        assert env.reset().shape == (2, 3)
        states = env.get_env_state()
        expected = env.step(np.zeros((2, 1)))[0]
        env.set_env_state(states)
        np.testing.assert_array_equal(env.step(np.zeros((2, 1)))[0], expected)
    finally:
        env.close()


def test_neural_subprocess_rollouts_match_serial():
    policy = LinearGaussianPolicy(3, 1, seed=4)
    serial = make_env()
    parallel = TorchModelVecEnv([make_env, make_env], policy, start_method="spawn")
    try:
        serial.reset(seed=4)
        parallel.set_env_state(serial.get_env_state())
        noise = np.random.default_rng(4).normal(size=(4, 3, 1))
        expected = serial.rollout_cl(policy, 4, 3, mode="sample", noise=noise)
        actual = parallel.rollout(4, 3, mode="sample", noise=noise)
        for index in (0, 1, 3, 4, 5):
            np.testing.assert_allclose(actual[index], expected[index], atol=1e-6)
    finally:
        serial.close()
        parallel.close()


def test_vector_video_stream(tmp_path, monkeypatch):
    from mjmpc.envs.vec_env import VecVideoRecorder
    import imageio.v2 as imageio

    monkeypatch.setenv("SDL_VIDEODRIVER", "dummy")
    env = DummyVecEnv(
        [
            lambda: GymEnvWrapper(gym.make("SimplePendulum-v0", render_mode="rgb_array"))
            for _ in range(2)
        ]
    )
    recorder = VecVideoRecorder(env, tmp_path, lambda step: step == 0, video_length=2)
    try:
        recorder.reset()
        recorder.step(np.zeros((2, 1)))
        recorder.step(np.zeros((2, 1)))
    finally:
        recorder.close()
    path = next(tmp_path.glob("*.mp4"))
    with imageio.get_reader(path) as video:
        assert video.get_data(0).shape[-1] == 3
        assert video.count_frames() == 3
