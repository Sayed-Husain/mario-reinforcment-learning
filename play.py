"""Watch a trained agent play Super Mario Bros.

Usage:
    uv run play.py --config configs/dqn.yaml --model checkpoints/dqn_final.pt
    uv run play.py --config configs/ppo.yaml --model checkpoints/ppo_final.pt
"""

import argparse
import time

import numpy as np

from mario_rl.config import load_config
from mario_rl.env import make_env


def play(config_path: str, model_path: str, episodes: int = 3):
    """Load a trained agent and watch it play."""
    config = load_config(config_path)

    # Force rendering on so we can see the game
    config.env.render = True
    env = make_env(config.env)

    # Load the trained agent
    if config.algorithm == "dqn":
        from mario_rl.agents.dqn import DQNAgent

        agent = DQNAgent(config)
    elif config.algorithm == "ppo":
        from mario_rl.agents.ppo import PPOAgent

        agent = PPOAgent(config)
    else:
        raise ValueError(f"Unknown algorithm: {config.algorithm}")

    agent.load(model_path)
    print(f"Loaded {config.algorithm.upper()} agent from {model_path}")

    for ep in range(1, episodes + 1):
        state, info = env.reset()
        state = np.array(state)
        total_reward = 0.0
        terminated = False
        truncated = False

        while not (terminated or truncated):
            action = agent.act(state)
            next_state, reward, terminated, truncated, info = env.step(action)
            next_state = np.array(next_state)

            total_reward += reward
            state = next_state
            time.sleep(0.01)  # Slow down slightly for watchability

        status = "CLEARED!" if info.get("flag_get", False) else f"Died at x={info.get('x_pos', 0)}"
        print(f"Episode {ep}: reward={total_reward:.0f}, distance={info.get('x_pos', 0)}, {status}")

    env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Watch a trained Mario agent play")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config file")
    parser.add_argument("--model", type=str, required=True, help="Path to trained model checkpoint")
    parser.add_argument("--episodes", type=int, default=3, help="Number of episodes to play")
    args = parser.parse_args()

    play(args.config, args.model, args.episodes)
