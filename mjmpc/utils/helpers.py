#!/usr/bin/env python
import numpy as np
import os

from .logger import LoggerClass


def set_qpos_qvel(model, data, qpos, qvel):
    import mujoco
    data.qpos[:] = qpos
    data.qvel[:] = qvel
    mujoco.mj_forward(model, data)

def render_trajs(env, trajectories, n_times=1):
    try:
        # for _ in range(n_times):
        #     for traj in trajectories:
        #         env.reset()
        #         state = traj['states'][0]
        #         env.set_env_state(state)
        #         for action in traj['actions']:
        #             env.render()
        #             obs, rew, done, info = env.step(action)
        for _ in range(n_times):
            for traj in trajectories:
                env.reset()
                for state in traj['states']:
                    env.set_env_state(state)
                    env.render()
    except KeyboardInterrupt:
        print('Exiting ...')


def dump_videos(env, trajectories, frame_size=(640, 480), folder="/tmp/",
                filename="newvid", camera_name=None, device_id=0):
    import imageio.v2 as imageio
    os.makedirs(folder, exist_ok=True)
    for episode, trajectory in enumerate(trajectories):
        env.reset()
        env.set_env_state(trajectory["states"][0])
        output = os.path.join(folder, f"{filename}{episode}.mp4")
        with imageio.get_writer(output, fps=env.metadata.get("render_fps", 30)) as writer:
            for action in trajectory["actions"]:
                env.step(action)
                writer.append_data(env.get_curr_frame(frame_size, camera_name, device_id))


def get_logger(display_name, log_dir, mode):
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    logger = LoggerClass()
    logger.setup(display_name, os.path.join(log_dir, 'log.txt'), 'debug')
    return logger

def stack_tensor_list(tensor_list):
    return np.array(tensor_list)
    # tensor_shape = np.array(tensor_list[0]).shape
    # if tensor_shape is tuple():
    #     return np.array(tensor_list)
    # return np.vstack(tensor_list)

def stack_tensor_dict_list(tensor_dict_list):
    """
    Stack a list of dictionaries of {tensors or dictionary of tensors}.
    :param tensor_dict_list: a list of dictionaries of {tensors or dictionary of tensors}.
    :return: a dictionary of {stacked tensors or dictionary of stacked tensors}
    """
    keys = list(tensor_dict_list[0].keys())
    ret = dict()
    for k in keys:
        example = tensor_dict_list[0][k]
        if isinstance(example, dict):
            v = stack_tensor_dict_list([x[k] for x in tensor_dict_list])
        else:
            v = stack_tensor_list([x[k] for x in tensor_dict_list])
        ret[k] = v
    return ret

def tensor_dict_list_to_array(tensor_dict_list):
    """
    Stack a list of dictionaries into a numpy array
    :param tensor_dict_list: a list of dictionaries of tensors
    :return numpy array
    """
    ret = []
    for d in tensor_dict_list:
        # curr_vals = []
        # for k in d.keys():
            # curr_vals.append(d[k])
        # curr_vals = np.array(curr_vals)
        curr_vals = np.concatenate([d[k] for k in d.keys()])

        ret.append(curr_vals.copy())
    return np.array(ret)

def tensor_dict_to_array(tensor_dict):
    vals = np.concatenate([tensor_dict[k] for k in tensor_dict.keys()])
    return np.array(vals)
