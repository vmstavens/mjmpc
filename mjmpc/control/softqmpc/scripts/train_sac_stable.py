"""Train or evaluate Stable Baselines3 SAC with the optional training extra."""
import argparse
import gymnasium as gym
from stable_baselines3 import SAC
import mjmpc.envs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", default="SimplePendulum-v0")
    parser.add_argument("--steps", type=int, default=100000)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--model", default="sac_pendulum")
    parser.add_argument("--evaluate", action="store_true")
    args = parser.parse_args()
    env = gym.make(args.env)
    try:
        if args.evaluate:
            model = SAC.load(args.model, env=env)
            rewards = []
            for episode in range(10):
                obs, _ = env.reset(seed=args.seed + episode)
                total = 0.0
                while True:
                    action, _ = model.predict(obs, deterministic=True)
                    obs, reward, terminated, truncated, _ = env.step(action)
                    total += reward
                    if terminated or truncated:
                        break
                rewards.append(total)
            print(f"Average reward: {sum(rewards) / len(rewards):.3f}")
        else:
            model = SAC("MlpPolicy", env, verbose=1, seed=args.seed)
            model.learn(total_timesteps=args.steps)
            model.save(args.model)
    finally:
        env.close()


if __name__ == "__main__":
    main()
