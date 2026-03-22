"""Configuration loading and validation.

Loads YAML config files into typed Python dataclasses with sensible defaults.
"""

from dataclasses import dataclass, field
from pathlib import Path

import yaml


@dataclass
class EnvConfig:
    """Environment configuration: level selection and frame preprocessing."""

    world: int = 1  # Mario world (1-8)
    stage: int = 1  # Stage within the world (1-4)
    version: int = 0  # 0=standard, 1=random enemies, 2=no enemies, 3=no enemies+blocks
    action_space: str = "SIMPLE_MOVEMENT"  # SIMPLE_MOVEMENT (7 actions) or COMPLEX_MOVEMENT
    frame_skip: int = 4  # Repeat each action for N frames (speeds up training)
    frame_stack: int = 4  # Stack N consecutive frames (gives agent sense of motion)
    frame_size: int = 84  # Resize frames to NxN pixels
    render: bool = False  # Whether to display the game window


@dataclass
class TrainingConfig:
    """Training hyperparameters."""

    total_timesteps: int = 2_000_000
    learning_rate: float = 0.00025
    batch_size: int = 32
    gamma: float = 0.99  # Discount factor: how much the agent values future vs immediate reward
    save_freq: int = 100_000  # Save a checkpoint every N steps
    log_freq: int = 1_000  # Log metrics every N steps
    checkpoint_dir: str = "checkpoints"
    metrics_dir: str = "metrics"


@dataclass
class DQNConfig:
    """DQN-specific hyperparameters."""

    replay_buffer_size: int = 100_000  # How many past experiences to remember
    epsilon_start: float = 1.0  # Start fully random (explore everything)
    epsilon_end: float = 0.02  # Eventually be 98% greedy, 2% random
    epsilon_decay: int = 100_000  # Steps over which epsilon decays
    target_update_freq: int = 10_000  # Sync target network every N steps
    double_dqn: bool = True  # Use Double DQN (reduces overestimation)
    prioritized_replay: bool = False  # Prioritized experience replay (Schaul et al. 2016)
    priority_alpha: float = 0.6  # How much prioritization to use (0=uniform, 1=full)
    priority_beta_start: float = 0.4  # Importance sampling correction (annealed to 1.0)
    dueling: bool = False  # Dueling network architecture (Wang et al. 2016)


@dataclass
class PPOConfig:
    """PPO-specific hyperparameters (passed to Stable-Baselines3)."""

    n_steps: int = 2048  # Steps per rollout before updating
    n_epochs: int = 10  # Number of passes over collected data per update
    clip_range: float = 0.2  # PPO clipping parameter (limits policy change per update)
    gae_lambda: float = 0.95  # Generalized Advantage Estimation smoothing
    ent_coef: float = 0.01  # Entropy bonus (encourages exploration)
    vf_coef: float = 0.5  # Value function loss weight
    n_envs: int = 4  # Number of parallel environments (PPO benefits from parallelism)


@dataclass
class CurriculumStage:
    """One stage in a curriculum training schedule."""

    world: int = 1
    stage: int = 1
    version: int = 0  # 0=standard, 2=no enemies, 3=no enemies+blocks
    min_timesteps: int = 500_000  # Minimum steps before considering advancement
    advance_threshold: float = 0.8  # Completion rate to trigger advancement


@dataclass
class CurriculumConfig:
    """Curriculum learning settings."""

    enabled: bool = False
    stages: list = field(default_factory=list)  # List of stage dicts
    eval_window: int = 50  # Episodes to average for completion rate check


@dataclass
class Config:
    """Top-level config that wraps everything together."""

    algorithm: str = "dqn"  # "dqn" or "ppo"
    env: EnvConfig = field(default_factory=EnvConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    dqn: DQNConfig = field(default_factory=DQNConfig)
    ppo: PPOConfig = field(default_factory=PPOConfig)
    curriculum: CurriculumConfig = field(default_factory=CurriculumConfig)


def load_config(path: str | Path) -> Config:
    """Load a YAML config file and return a typed Config object.

    Any fields not specified in the YAML file will use their defaults from
    the dataclass definitions above.
    """
    path = Path(path)
    with open(path) as f:
        raw = yaml.safe_load(f)

    # Build curriculum config (stages are a list of dicts → list of dataclasses)
    curriculum_raw = raw.get("curriculum", {})
    stages_raw = curriculum_raw.pop("stages", [])
    curriculum = CurriculumConfig(
        **curriculum_raw,
        stages=[CurriculumStage(**s) for s in stages_raw],
    )

    config = Config(
        algorithm=raw.get("algorithm", "dqn"),
        env=EnvConfig(**raw.get("env", {})),
        training=TrainingConfig(**raw.get("training", {})),
        dqn=DQNConfig(**raw.get("dqn", {})),
        ppo=PPOConfig(**raw.get("ppo", {})),
        curriculum=curriculum,
    )

    return config
