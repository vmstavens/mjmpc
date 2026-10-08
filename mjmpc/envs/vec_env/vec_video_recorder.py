"""Streaming video recording for controller vector environments."""

from pathlib import Path
import imageio.v2 as imageio
from .base_vec_env import VecEnvWrapper


class VecVideoRecorder(VecEnvWrapper):
    def __init__(
        self, venv, video_folder, record_video_trigger, video_length=200, name_prefix="rl-video"
    ):
        super().__init__(venv)
        self.video_folder = Path(video_folder)
        self.video_folder.mkdir(parents=True, exist_ok=True)
        self.record_video_trigger = record_video_trigger
        self.video_length = video_length
        self.name_prefix = name_prefix
        self.step_id = 0
        self.recorded_frames = 0
        self.video_recorder = None

    def reset(self):
        obs = self.venv.reset()
        if self.record_video_trigger(self.step_id):
            self.start_video_recorder()
        return obs

    def start_video_recorder(self):
        self.close_video_recorder()
        path = self.video_folder / f"{self.name_prefix}-step-{self.step_id}.mp4"
        metadata = self.venv.get_attr("metadata")[0]
        self.video_recorder = imageio.get_writer(path, fps=metadata.get("render_fps", 30))
        self.recorded_frames = 0
        self._capture_frame()

    def _capture_frame(self):
        self.video_recorder.append_data(self.venv.render(mode="rgb_array"))
        self.recorded_frames += 1

    def step_wait(self):
        result = self.venv.step_wait()
        self.step_id += 1
        if self.video_recorder is not None:
            self._capture_frame()
            if self.recorded_frames > self.video_length:
                self.close_video_recorder()
        elif self.record_video_trigger(self.step_id):
            self.start_video_recorder()
        return result

    def close_video_recorder(self):
        if self.video_recorder is not None:
            self.video_recorder.close()
            self.video_recorder = None

    def close(self):
        self.close_video_recorder()
        self.venv.close()
