"""Record gameplay of a trained agent as GIF or MP4.

Captures raw game frames (not the preprocessed 84x84 ones) so the output
looks like actual Mario gameplay.

Usage:
    uv run record_video.py --config configs/dqn.yaml --model checkpoints/dqn_final.pt
    uv run record_video.py --config configs/ppo.yaml --model checkpoints/ppo_final --format mp4
    uv run record_video.py --config configs/dqn.yaml --model checkpoints/dqn_final.pt --episodes 5
"""

import argparse
from pathlib import Path

import imageio
import numpy as np
from gym_super_mario_bros import SuperMarioBrosEnv
from nes_py.wrappers import JoypadSpace

from mario_rl.config import load_config
from mario_rl.env import ACTION_SPACES, make_env


def make_recording_env(env_config):
    """Create a raw (unprocessed) Mario env for capturing full-color frames.

    The agent uses a separate preprocessed env (via make_env) for decision-making.
    This env provides the original 240x256 RGB frames for the video output.
    Both envs are stepped in sync with the same actions.
    """
    ROM_MODES = {0: "vanilla", 1: "downsample", 2: "pixel", 3: "rectangle"}
    rom_mode = ROM_MODES.get(env_config.version, "vanilla")

    # Raw env without preprocessing wrappers (outputs full-color frames)
    base_env = SuperMarioBrosEnv(rom_mode=rom_mode, target=(env_config.world, env_config.stage))
    actions = ACTION_SPACES[env_config.action_space]
    base_env = JoypadSpace(base_env, actions)

    return base_env


def record(config_path: str, model_path: str, output: str = "assets/gameplay",
           fmt: str = "gif", episodes: int = 1, fps: int = 30):
    config = load_config(config_path)

    # Agent env (preprocessed) for decision-making
    agent_env = make_env(config.env)

    # Raw env for capturing full-color frames
    raw_env = make_recording_env(config.env)

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

    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    for ep in range(1, episodes + 1):
        frames = []

        # Reset both envs
        agent_obs, _ = agent_env.reset()
        raw_obs = raw_env.reset()
        agent_state = np.array(agent_obs)

        # Capture first frame
        frames.append(raw_obs.copy())

        done = False
        total_reward = 0
        steps = 0

        while not done:
            action = agent.act(agent_state)

            # Step both envs with the same action
            agent_obs, reward, terminated, truncated, info = agent_env.step(action)
            # Raw env uses old gym API (4-tuple)
            raw_obs, _, raw_done, _ = raw_env.step(action)

            agent_state = np.array(agent_obs)
            done = terminated or truncated
            total_reward += reward
            steps += 1

            # Capture frame (skip some for smaller file size)
            if steps % 2 == 0:
                frames.append(raw_obs.copy())

        status = "CLEARED!" if info.get("flag_get", False) else f"x={info.get('x_pos', 0)}"
        print(f"Episode {ep}: reward={total_reward:.0f}, distance={info.get('x_pos', 0)}, {status}")

        # Save video
        suffix = f"_ep{ep}" if episodes > 1 else ""
        filepath = f"{output}{suffix}.{fmt}"

        if fmt == "gif":
            # Lower fps for GIF to keep file size reasonable
            imageio.mimsave(filepath, frames, fps=fps // 2, loop=0)
        else:
            writer = imageio.get_writer(filepath, fps=fps)
            for frame in frames:
                writer.append_data(frame)
            writer.close()

        size_mb = Path(filepath).stat().st_size / (1024 * 1024)
        print(f"Saved {filepath} ({len(frames)} frames, {size_mb:.1f} MB)")

    agent_env.close()
    raw_env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Record gameplay of a trained agent")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config file")
    parser.add_argument("--model", type=str, required=True, help="Path to trained model")
    parser.add_argument("--output", type=str, default="assets/gameplay", help="Output path (without extension)")
    parser.add_argument("--format", type=str, default="gif", choices=["gif", "mp4"], help="Output format")
    parser.add_argument("--episodes", type=int, default=1, help="Number of episodes to record")
    parser.add_argument("--fps", type=int, default=30, help="Frames per second")
    args = parser.parse_args()

    record(args.config, args.model, args.output, args.format, args.episodes, args.fps)
