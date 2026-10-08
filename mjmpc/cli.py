#!/usr/bin/env python
import os
import argparse
from copy import deepcopy
from datetime import datetime
import gymnasium as gym
import numpy as np
import pickle
import tqdm
import yaml
from mjmpc.envs import GymEnvWrapper
from mjmpc.envs.vec_env import SubprocVecEnv
from mjmpc.utils import timeit, helpers
from mjmpc.policies import MPCPolicy


def main(default_controller="mppi"):
    parser = argparse.ArgumentParser(description="Run MPC algorithm on given environment")
    parser.add_argument(
        "--config",
        "--config_file",
        required=True,
        type=str,
        help="yaml file with experiment parameters",
    )
    parser.add_argument(
        "--dyn_randomize_config", type=str, help="yaml file with dynamics randomization parameters"
    )
    parser.add_argument("--save_dir", type=str, default="/tmp", help="folder to save data in")
    parser.add_argument(
        "--controller",
        "--controller_type",
        type=str,
        default=default_controller,
        help="controller to run",
    )
    parser.add_argument("--dump_vids", action="store_true", help="flag to dump video of episodes")
    args = parser.parse_args()

    # Load experiment parameters from config file
    with open(args.config) as file:
        exp_params = yaml.safe_load(file)
    if args.dyn_randomize_config is not None:
        with open(args.dyn_randomize_config) as file:
            dynamics_rand_params = yaml.safe_load(file)
    else:
        dynamics_rand_params = None

    controller_name = args.controller
    # Create the main environment
    env_name = exp_params["env_name"]
    sim_env_name = exp_params["sim_env_name"] if "sim_env_name" in exp_params else env_name
    env = gym.make(
        env_name,
        render_mode="rgb_array"
        if args.dump_vids
        else ("human" if exp_params.get("render") else None),
    )
    env = GymEnvWrapper(env)
    env.real_env_step(True)

    # Create logger
    date_time = datetime.now().strftime("%m_%d_%Y_%H_%M_%S")
    log_dir = args.save_dir + "/" + exp_params["env_name"] + "/" + date_time + "/" + controller_name
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    logger = helpers.get_logger(controller_name + "_" + exp_params["env_name"], log_dir, "debug")

    # Function to create vectorized environments for controller simulations
    def make_env():
        gym_env = gym.make(sim_env_name)
        rollout_env = GymEnvWrapper(gym_env)
        rollout_env.real_env_step(False)
        # if dynamics_rand_params is not None:
        #     default_params, randomized_params = rollout_env.randomize_dynamics(dynamics_rand_params)
        #     # print('Default params = {}'.format(default_params))
        #     # print('Randomized params = {}'.format(randomized_params))

        return rollout_env

    # unpack params and create policy params
    policy_params = deepcopy(exp_params[controller_name])
    if "base_action" in exp_params:
        policy_params.setdefault("base_action", exp_params["base_action"])
    policy_params["d_obs"] = env.d_obs
    policy_params["d_state"] = env.d_state
    policy_params["d_action"] = env.d_action
    policy_params["action_lows"] = env.action_space.low
    policy_params["action_highs"] = env.action_space.high
    print(policy_params)
    if "num_cpu" in policy_params and "particles_per_cpu" in policy_params:
        policy_params["num_particles"] = (
            policy_params["num_cpu"] * policy_params["particles_per_cpu"]
        )

    num_cpu = policy_params["num_cpu"]
    n_episodes = exp_params["n_episodes"]
    base_seed = exp_params["seed"]
    ep_length = exp_params["max_ep_length"]

    # Create vectorized environments for MPC simulations
    sim_env = SubprocVecEnv([make_env for i in range(num_cpu)])
    try:
        if dynamics_rand_params is not None:
            default_params, randomized_params = sim_env.randomize_dynamics(
                dynamics_rand_params, base_seed=exp_params["seed"]
            )
            logger.info("Default params = {}".format(default_params))
            logger.info("Randomized params = {}".format(randomized_params))

        def rollout_fn(num_particles, horizon, mean, noise, mode):
            """
            Given a batch of sequences of actions, rollout
            in sim envs and return sequence of costs. The controller is
            agnostic of how the rollouts are generated.
            """
            obs_vec, rew_vec, act_vec, done_vec, info_vec, next_obs_vec = sim_env.rollout(
                num_particles, horizon, mean.copy(), noise, mode
            )
            # we assume environment returns rewards, but controller needs costs
            sim_trajs = dict(
                observations=obs_vec.copy(),
                actions=act_vec.copy(),
                costs=-1.0 * rew_vec.copy(),
                dones=done_vec.copy(),
                next_observations=next_obs_vec.copy(),
                infos=helpers.stack_tensor_dict_list(info_vec.copy()),
            )

            return sim_trajs

        policy_params.pop("particles_per_cpu", None)
        policy_params.pop("num_cpu", None)

        ep_rewards = np.array([0.0] * n_episodes)
        trajectories = []
        logger.info(exp_params[controller_name])

        # Main data collection loop
        timeit.start("start_" + controller_name)
        for i in tqdm.tqdm(range(n_episodes)):
            # seeding to enforce consistent episodes
            episode_seed = base_seed + i * 12345
            policy_params["seed"] = episode_seed
            obs = env.reset(seed=episode_seed)
            sim_env.reset()

            # create MPC policy and set appropriate functions
            policy = MPCPolicy(
                controller_type=controller_name, param_dict=policy_params, batch_size=1
            )  # Only batch_size=1 is supported for now
            policy.controller.set_sim_state_fn = sim_env.set_env_state
            policy.controller.rollout_fn = rollout_fn
            if controller_name in ["ilqr", "softq", "random_shooting_nn"]:
                policy.controller.get_sim_state_fn = sim_env.get_env_state
                policy.controller.sim_step_fn = sim_env.step
                policy.controller.sim_reset_fn = sim_env.reset
                policy.controller.get_sim_obs_fn = sim_env.get_obs

            # Collect data from interactions with environment
            observations = []
            actions = []
            rewards = []
            dones = []
            infos = []
            states = []
            next_states = []
            for _ in tqdm.tqdm(range(ep_length)):
                curr_state = deepcopy(env.get_env_state())
                action, value = policy.get_action(curr_state, calc_val=False)
                obs, reward, done, info = env.step(action)
                observations.append(obs)
                actions.append(action)
                rewards.append(reward)
                dones.append(done)
                infos.append(info)
                states.append(curr_state)
                ep_rewards[i] += reward
                if done:
                    break

            traj = dict(
                observations=np.array(observations),
                actions=np.array(actions),
                rewards=np.array(rewards),
                dones=np.array(dones),
                env_infos=helpers.stack_tensor_dict_list(infos),
                states=states,
            )
            trajectories.append(traj)
        sim_env.close()  # Free up memory

        timeit.stop("start_" + controller_name)  # stop timer after trajectory collection
        with open(os.path.join(log_dir, "trajectories.pkl"), "wb") as file:
            pickle.dump(trajectories, file)
        success_metric = env.evaluate_success(trajectories)
        average_reward = np.average(ep_rewards)
        reward_std = np.std(ep_rewards)

        # Display logs on screen and save in txt file
        logger.info(
            "Avg. reward = {0}, Std. Reward = {1}, Success Metric = {2}".format(
                average_reward, reward_std, success_metric
            )
        )

        # Can also dump data to csv once done
        # for i in range(n_episodes):
        logger.record_tabular("EpisodeReward", ep_rewards)
        logger.record_tabular("Horizon", policy_params["horizon"])
        logger.record_tabular("AverageReward", average_reward)
        logger.record_tabular("StdReward", reward_std)
        logger.record_tabular("SuccessMetric", success_metric)
        logger.record_tabular("NumEpisodes", exp_params["n_episodes"])
        if "num_particles" in policy_params:
            logger.record_tabular("NumParticles", policy_params["num_particles"])

        logger.dump_tabular()

        if args.dump_vids:
            print("Dumping videos")
            helpers.dump_videos(
                env=env,
                trajectories=trajectories,
                frame_size=(1280, 720),
                folder=log_dir,
                filename="vid_traj_",
                camera_name=None,
                device_id=1,
            )

            if "sim_env_name" in exp_params:
                sim_env = gym.make(sim_env_name, render_mode="rgb_array")
                sim_env = GymEnvWrapper(sim_env)
                helpers.dump_videos(
                    env=sim_env,
                    trajectories=trajectories,
                    frame_size=(1280, 720),
                    folder=log_dir,
                    filename="vid_sim_traj_",
                    camera_name=None,
                    device_id=1,
                )

        if exp_params["render"]:
            _ = input("Press enter to display optimized trajectories (will be played 3 times) : ")
            helpers.render_trajs(env, trajectories, n_times=3)

        env.close()
    finally:
        sim_env.close()
        env.close()


if __name__ == "__main__":
    main()
