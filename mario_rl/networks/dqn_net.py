"""DQN neural network architectures.

Standard DQN:
    Conv layers → Flatten → Linear(512) → Linear(n_actions)

Dueling DQN (Wang et al. 2016):
    Conv layers → Flatten → split into two streams:
        Value stream:     Linear(512) → Linear(1)      = V(s)
        Advantage stream: Linear(512) → Linear(n_actions) = A(s,a)
    Combined: Q(s,a) = V(s) + A(s,a) - mean(A)

Both share the same convolutional backbone from DeepMind's 2015 paper.
"""

import torch
from torch import nn


class DQNNetwork(nn.Module):
    """Convolutional Q-network with optional dueling architecture.

    Args:
        n_actions: Number of possible actions (7 for SIMPLE_MOVEMENT)
        in_channels: Number of stacked frames (default 4)
        dueling: If True, use dueling architecture (separate V and A streams)
    """

    def __init__(self, n_actions: int, in_channels: int = 4, dueling: bool = False):
        super().__init__()
        self._dueling = dueling
        self._n_actions = n_actions

        # Shared convolutional backbone
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=8, stride=4),  # 84x84 → 20x20
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),  # 20x20 → 9x9
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),  # 9x9 → 7x7
            nn.ReLU(),
            nn.Flatten(),
        )

        flat_size = 64 * 7 * 7  # 3136

        if dueling:
            # Value stream: "how good is this state overall?"
            self.value_stream = nn.Sequential(
                nn.Linear(flat_size, 512),
                nn.ReLU(),
                nn.Linear(512, 1),
            )
            # Advantage stream: "how much better is each action than average?"
            self.advantage_stream = nn.Sequential(
                nn.Linear(flat_size, 512),
                nn.ReLU(),
                nn.Linear(512, n_actions),
            )
        else:
            self.head = nn.Sequential(
                nn.Linear(flat_size, 512),
                nn.ReLU(),
                nn.Linear(512, n_actions),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass: frames in, Q-values out.

        Args:
            x: Batch of observations, shape (batch, 4, 84, 84), values in [0, 1]

        Returns:
            Q-values for each action, shape (batch, n_actions)
        """
        features = self.features(x)

        if self._dueling:
            value = self.value_stream(features)  # (batch, 1)
            advantage = self.advantage_stream(features)  # (batch, n_actions)
            # Q = V + (A - mean(A))
            # Subtracting mean ensures identifiability: V and A don't drift
            q_values = value + advantage - advantage.mean(dim=1, keepdim=True)
        else:
            q_values = self.head(features)

        return q_values
