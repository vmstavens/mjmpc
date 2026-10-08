"""Run a random policy on a modern Gymnasium environment."""

import argparse
import gymnasium as gym
import mjmpc.envs  # register bundled tasks


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", default="SimplePendulum-v0")
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--render", action="store_true")
    args = parser.parse_args()
    env = gym.make(args.env, render_mode="human" if args.render else None)
    try:
        env.reset(seed=args.seed)
        env.action_space.seed(args.seed)
        total_reward = 0.0
        for _ in range(args.steps):
            _, reward, terminated, truncated, _ = env.step(env.action_space.sample())
            total_reward += reward
            if terminated or truncated:
                env.reset()
        print(f"Total reward: {total_reward:.3f}")
    finally:
        env.close()


if __name__ == "__main__":
    main()
