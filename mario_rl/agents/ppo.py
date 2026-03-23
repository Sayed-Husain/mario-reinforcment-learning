"""PPO agent via Stable-Baselines3.

SB3 handles the full training loop internally (rollout collection, GAE
advantage estimation, policy/value updates). This wrapper adapts it to
our BaseAgent interface for consistent usage in train.py and play.py.

Because SB3 manages its own training loop, this agent works differently
from DQN:
    - act() works the same (forward pass through the policy network)
    - learn() is a no-op (SB3 handles updates internally)
    - train_sb3() is the actual training entry point

Reference: Schulman et al. (2017) "Proximal Policy Optimization Algorithms"
"""

from pathlib import Path

import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback, EvalCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv

from mario_rl.agents.base import BaseAgent
from mario_rl.config import Config
from mario_rl.env import make_env


class MetricsCallback(BaseCallback):
    """SB3 callback that logs per-episode metrics to our MetricsLogger.

    SB3 tracks episode stats in info dicts when episodes end. This callback
    captures them and writes to CSV in the same format as DQN, so
    generate_report.py works identically for both algorithms.
    """

    def __init__(self, metrics_logger, verbose=0):
        super().__init__(verbose)
        self._logger = metrics_logger

    def _on_step(self) -> bool:
        for info in self.locals.get("infos", []):
            if "episode" in info:
                self._logger.begin_episode()
                self._logger.log_step(action=0, reward=0, info=info)
                self._logger.end_episode(
                    total_reward=info["episode"]["r"],
                    info=info,
                )
        return True


class PPOAgent(BaseAgent):
    """SB3 PPO wrapped to match BaseAgent interface."""

    def __init__(self, config: Config):
        self._config = config
        self._ppo_config = config.ppo
        self._training_config = config.training
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._model = None

    def _make_sb3_env(self, n_envs: int = 1):
        """Create a vectorized env for SB3.

        Reuses make_env() so PPO sees the exact same preprocessing as DQN
        (same wrappers, same frame stacking, same normalization). The only
        difference is DummyVecEnv wrapping for SB3's parallel rollout API.
        """
        env_config = self._config.env

        # Monitor tracks episode reward/length and adds "episode" to info on done.
        # This is how SB3 knows when episodes end, and our MetricsCallback reads it.
        def _make_monitored():
            return Monitor(make_env(env_config))

        vec_env = DummyVecEnv([_make_monitored for _ in range(n_envs)])

        return vec_env

    def _create_model(self, env=None):
        if env is None:
            env = self._make_sb3_env(n_envs=self._ppo_config.n_envs)

        # Linear LR decay: starts at config value, decays to 0 over training.
        # Prevents late-training policy collapse by reducing update magnitude.
        initial_lr = self._training_config.learning_rate

        def lr_schedule(progress_remaining: float) -> float:
            return initial_lr * progress_remaining

        self._model = PPO(
            policy="CnnPolicy",
            env=env,
            learning_rate=lr_schedule,
            n_steps=self._ppo_config.n_steps,
            batch_size=self._training_config.batch_size,
            n_epochs=self._ppo_config.n_epochs,
            gamma=self._training_config.gamma,
            clip_range=self._ppo_config.clip_range,
            gae_lambda=self._ppo_config.gae_lambda,
            ent_coef=self._ppo_config.ent_coef,
            vf_coef=self._ppo_config.vf_coef,
            verbose=1,
            device=self._device,
            # Our env already normalizes pixels to [0,1], so tell SB3 not to re-normalize
            policy_kwargs={"normalize_images": False},
        )

    def train_sb3(self, metrics_logger=None):
        """Run SB3's internal training loop."""
        env = self._make_sb3_env(n_envs=self._ppo_config.n_envs)
        self._create_model(env)

        callbacks = []
        if metrics_logger:
            callbacks.append(MetricsCallback(metrics_logger))

        # Save checkpoints periodically
        callbacks.append(CheckpointCallback(
            save_freq=max(1, self._training_config.save_freq // self._ppo_config.n_envs),
            save_path=self._training_config.checkpoint_dir,
            name_prefix="ppo",
        ))

        # Evaluate periodically and save the best model.
        # Protects against policy collapse: even if the final model
        # is degraded, the best-performing checkpoint is preserved.
        eval_env = self._make_sb3_env(n_envs=1)
        callbacks.append(EvalCallback(
            eval_env,
            best_model_save_path=self._training_config.checkpoint_dir,
            eval_freq=max(1, self._training_config.save_freq // self._ppo_config.n_envs),
            n_eval_episodes=5,
            deterministic=True,
            verbose=1,
        ))

        self._model.learn(
            total_timesteps=self._training_config.total_timesteps,
            callback=callbacks,
        )

        eval_env.close()
        env.close()

    def act(self, state: np.ndarray) -> int:
        """Select action using the learned policy (deterministic for evaluation)."""
        if self._model is None:
            raise RuntimeError("Model not loaded. Call load() or train_sb3() first.")

        if state.ndim == 3:
            state = state[np.newaxis, ...]

        action, _ = self._model.predict(state, deterministic=True)
        return int(action[0]) if hasattr(action, "__len__") else int(action)

    def learn(self, state, action, reward, next_state, done) -> dict | None:
        # No-op: SB3 handles updates internally via train_sb3()
        return None

    def save(self, path: str | Path) -> None:
        if self._model is None:
            raise RuntimeError("No model to save. Call train_sb3() first.")
        self._model.save(str(path))

    def load(self, path: str | Path) -> None:
        path_str = str(path)
        # SB3 adds .zip extension automatically
        if not path_str.endswith(".zip"):
            path_str = path_str.replace(".pt", "")
        self._model = PPO.load(path_str, device=self._device)
