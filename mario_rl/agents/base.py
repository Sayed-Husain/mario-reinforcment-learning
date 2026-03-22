"""Abstract base agent interface.

All RL agents (DQN, PPO, etc.) implement this interface so that training
and playback scripts are algorithm-agnostic.
"""

from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np


class BaseAgent(ABC):
    """Interface that all RL agents must implement."""

    @abstractmethod
    def act(self, state: np.ndarray) -> int:
        """Choose an action given the current game state.

        Args:
            state: Preprocessed observation, shape (4, 84, 84)

        Returns:
            Action index (0 to n_actions-1)
        """

    @abstractmethod
    def learn(self, state, action, reward, next_state, done) -> dict | None:
        """Update the agent's knowledge from one step of experience.

        Args:
            state: State before action
            action: Action taken
            reward: Reward received
            next_state: State after action
            done: Whether the episode ended

        Returns:
            Optional dict of training metrics (loss, q_value, etc.)
        """

    @abstractmethod
    def save(self, path: str | Path) -> None:
        """Save the agent's learned parameters to disk."""

    @abstractmethod
    def load(self, path: str | Path) -> None:
        """Load previously saved parameters from disk."""
