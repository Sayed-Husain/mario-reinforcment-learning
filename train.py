"""Train a reinforcement learning agent to play Super Mario Bros.

Usage:
    uv run train.py --config configs/dqn.yaml
    uv run train.py --config configs/ppo.yaml
    uv run train.py --config configs/dqn.yaml --resume checkpoints/dqn_latest.pt
"""

import argparse
from pathlib import Path

import numpy as np
from tqdm import tqdm

from mario_rl.config import load_config
from mario_rl.env import make_env
from mario_rl.metrics import MetricsLogger


def _dqn_run_name(config) -> str:
    """Derive a descriptive run name from DQN config flags."""
    if config.dqn.prioritized_replay and config.dqn.dueling:
        base = "rainbow"
    elif config.dqn.prioritized_replay or config.dqn.dueling:
        parts = ["dqn"]
        if config.dqn.dueling:
            parts.append("dueling")
        if config.dqn.prioritized_replay:
            parts.append("per")
        base = "_".join(parts)
    else:
        base = "dqn"

    if config.curriculum.enabled:
        base += "_curriculum"
    return base


def train_dqn(config, resume_path=None, reset_epsilon=None):
    """DQN training loop (manual step-by-step).

    Supports optional curriculum learning: when config.curriculum.enabled is True,
    training starts on easier level versions and advances to harder ones as the
    agent improves. The agent's weights carry over between stages.
    """
    from mario_rl.agents.dqn import DQNAgent
    from mario_rl.curriculum import CurriculumScheduler

    run_name = _dqn_run_name(config)

    # Set up curriculum if enabled
    scheduler = None
    if config.curriculum.enabled and config.curriculum.stages:
        scheduler = CurriculumScheduler(config.curriculum)
        env_config = scheduler.get_current_env_config(config.env)
        stage = scheduler.current_stage
        print(f"Curriculum stage 1/{scheduler.total_stages}: "
              f"World {stage.world}-{stage.stage} v{stage.version}")
    else:
        env_config = config.env

    env = make_env(env_config)
    agent = DQNAgent(config)

    if resume_path:
        print(f"Resuming from {resume_path}")
        agent.load(resume_path)

    if reset_epsilon is not None:
        agent._epsilon = reset_epsilon
        print(f"Epsilon reset to {reset_epsilon}")

    logger = MetricsLogger(config.training.metrics_dir, run_name)
    print(f"Logging metrics to {logger.path}")

    Path(config.training.checkpoint_dir).mkdir(parents=True, exist_ok=True)

    pbar = tqdm(total=config.training.total_timesteps, desc="Training")
    total_steps = 0

    while total_steps < config.training.total_timesteps:
        state, info = env.reset()
        state = np.array(state)
        episode_reward = 0.0
        terminated = False
        truncated = False
        logger.begin_episode()

        while not (terminated or truncated):
            action = agent.act(state)
            next_state, reward, terminated, truncated, info = env.step(action)
            next_state = np.array(next_state)

            agent.learn(state, action, reward, next_state, terminated or truncated)

            logger.log_step(action=action, reward=reward, info=info)
            episode_reward += reward
            state = next_state
            total_steps += 1
            pbar.update(1)

            if total_steps % config.training.save_freq == 0:
                save_path = Path(config.training.checkpoint_dir) / f"{run_name}_{total_steps}.pt"
                agent.save(save_path)
                tqdm.write(f"Checkpoint saved: {save_path}")

            if total_steps >= config.training.total_timesteps:
                break

        logger.end_episode(total_reward=episode_reward, info=info)

        # Curriculum: check if ready to advance to next stage
        if scheduler and not scheduler.is_complete:
            completed = info.get("flag_get", False)
            scheduler.record_episode(completed)

            if scheduler.should_advance(total_steps):
                scheduler.advance(total_steps)
                env.close()
                env_config = scheduler.get_current_env_config(config.env)
                env = make_env(env_config)
                stage = scheduler.current_stage
                tqdm.write(
                    f"=== Curriculum: advanced to stage "
                    f"{scheduler.current_stage_index + 1}/{scheduler.total_stages} "
                    f"(World {stage.world}-{stage.stage} v{stage.version}) "
                    f"at step {total_steps:,} ==="
                )

        if logger.episode_count % 10 == 0:
            stage_info = ""
            if scheduler:
                stage_info = f"Stage {scheduler.current_stage_index + 1} | "
                stage_info += f"Clear rate: {scheduler.completion_rate:.0%} | "
            tqdm.write(
                f"Episode {logger.episode_count} | "
                f"{stage_info}"
                f"Reward: {episode_reward:.0f} | "
                f"Distance: {info.get('x_pos', 0)} | "
                f"{'CLEARED!' if info.get('flag_get', False) else ''}"
            )

    final_path = Path(config.training.checkpoint_dir) / f"{run_name}_final.pt"
    agent.save(final_path)
    print(f"Training complete! Final model saved to {final_path}")

    logger.close()
    env.close()
    pbar.close()


def train_ppo(config, resume_path=None):
    """PPO training (SB3 manages the loop internally).

    SB3's PPO collects rollouts across parallel envs, computes advantages,
    and updates the policy in batches. We pass a callback to capture
    per-episode metrics in the same CSV format as DQN.
    """
    from mario_rl.agents.ppo import PPOAgent

    agent = PPOAgent(config)

    if resume_path:
        print(f"Resuming from {resume_path}")
        agent.load(resume_path)

    logger = MetricsLogger(config.training.metrics_dir, config.algorithm)
    print(f"Logging metrics to {logger.path}")

    Path(config.training.checkpoint_dir).mkdir(parents=True, exist_ok=True)

    agent.train_sb3(metrics_logger=logger)

    final_path = Path(config.training.checkpoint_dir) / "ppo_final"
    agent.save(final_path)
    print(f"Training complete! Final model saved to {final_path}.zip")

    logger.close()


def train(config_path: str, resume_path: str | None = None, reset_epsilon: float | None = None):
    config = load_config(config_path)
    algo_name = _dqn_run_name(config) if config.algorithm == "dqn" else config.algorithm
    print(f"Training {algo_name.upper()} on SuperMarioBros-{config.env.world}-{config.env.stage}")
    print(f"Total timesteps: {config.training.total_timesteps:,}")

    if config.algorithm == "dqn":
        train_dqn(config, resume_path, reset_epsilon)
    elif config.algorithm == "ppo":
        train_ppo(config, resume_path)
    else:
        raise ValueError(f"Unknown algorithm: {config.algorithm}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Mario RL agent")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config file")
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume from")
    parser.add_argument("--reset-epsilon", type=float, default=None, help="Override epsilon on resume (e.g. 0.5)")
    args = parser.parse_args()

    train(args.config, args.resume, args.reset_epsilon)
