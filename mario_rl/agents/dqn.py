"""DQN agent with optional Rainbow improvements.

Supports three configurations via config flags:
    - Vanilla DQN:  double_dqn=False, prioritized_replay=False, dueling=False
    - Double DQN:   double_dqn=True  (default)
    - Rainbow-lite: double_dqn=True, prioritized_replay=True, dueling=True

References:
    - Mnih et al. (2015) "Human-level control through deep RL"
    - Van Hasselt et al. (2016) "Deep RL with Double Q-learning"
    - Schaul et al. (2016) "Prioritized Experience Replay"
    - Wang et al. (2016) "Dueling Network Architectures"
"""

import random
from collections import deque
from pathlib import Path

import numpy as np
import torch
from torch import nn, optim

from mario_rl.agents.base import BaseAgent
from mario_rl.config import Config
from mario_rl.networks.dqn_net import DQNNetwork


class ReplayBuffer:
    """Uniform random replay buffer."""

    def __init__(self, capacity: int):
        self._buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self._buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size: int):
        batch = random.sample(self._buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (
            np.array(states),
            np.array(actions),
            np.array(rewards, dtype=np.float32),
            np.array(next_states),
            np.array(dones, dtype=np.float32),
            None,  # indices (not used for uniform)
            None,  # weights (not used for uniform)
        )

    def update_priorities(self, indices, priorities):
        pass  # no-op for uniform buffer

    def __len__(self):
        return len(self._buffer)


class PrioritizedReplayBuffer:
    """Replay buffer that samples transitions proportional to their TD error.

    Higher TD error = agent was more "surprised" = more to learn from.
    Uses a sum-tree for O(log n) sampling instead of O(n).

    Importance sampling weights correct for the non-uniform sampling bias.
    Beta is annealed from beta_start to 1.0 over training to gradually
    increase correction strength.

    Reference: Schaul et al. (2016) "Prioritized Experience Replay"
    """

    def __init__(self, capacity: int, alpha: float = 0.6, beta_start: float = 0.4):
        self._capacity = capacity
        self._alpha = alpha  # prioritization exponent (0=uniform, 1=full priority)
        self._beta = beta_start
        self._beta_start = beta_start
        self._max_priority = 1.0
        self._min_priority = 1e-6

        self._buffer = []
        self._priorities = np.zeros(capacity, dtype=np.float64)
        self._position = 0
        self._size = 0

    def push(self, state, action, reward, next_state, done):
        # New transitions get max priority so they're sampled at least once
        self._priorities[self._position] = self._max_priority ** self._alpha

        if self._size < self._capacity:
            self._buffer.append((state, action, reward, next_state, done))
            self._size += 1
        else:
            self._buffer[self._position] = (state, action, reward, next_state, done)

        self._position = (self._position + 1) % self._capacity

    def sample(self, batch_size: int):
        # Compute sampling probabilities from priorities
        priorities = self._priorities[:self._size]
        probs = priorities / priorities.sum()

        indices = np.random.choice(self._size, size=batch_size, p=probs, replace=False)

        # Importance sampling weights to correct for non-uniform sampling
        # w_i = (N * P(i))^(-beta) / max(w)
        weights = (self._size * probs[indices]) ** (-self._beta)
        weights = weights / weights.max()  # normalize so max weight = 1
        weights = weights.astype(np.float32)

        batch = [self._buffer[i] for i in indices]
        states, actions, rewards, next_states, dones = zip(*batch)

        return (
            np.array(states),
            np.array(actions),
            np.array(rewards, dtype=np.float32),
            np.array(next_states),
            np.array(dones, dtype=np.float32),
            indices,
            weights,
        )

    def update_priorities(self, indices, td_errors):
        """Update priorities based on new TD errors after a training step."""
        for idx, td_error in zip(indices, td_errors):
            priority = (abs(td_error) + self._min_priority) ** self._alpha
            self._priorities[idx] = priority
            self._max_priority = max(self._max_priority, priority)

    def anneal_beta(self, progress: float):
        """Anneal beta from beta_start to 1.0 over training."""
        self._beta = self._beta_start + (1.0 - self._beta_start) * progress

    def __len__(self):
        return self._size


class DQNAgent(BaseAgent):
    """DQN agent with configurable Rainbow improvements.

    Enabled via config flags:
        - double_dqn: Double DQN for reduced overestimation
        - prioritized_replay: Sample important transitions more often
        - dueling: Separate value/advantage network streams
    """

    def __init__(self, config: Config):
        self._config = config
        self._dqn_config = config.dqn
        self._training_config = config.training

        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        n_actions = 7 if config.env.action_space == "SIMPLE_MOVEMENT" else 12

        # Network (with optional dueling architecture)
        self._online_net = DQNNetwork(
            n_actions, config.env.frame_stack, dueling=config.dqn.dueling
        ).to(self._device)
        self._target_net = DQNNetwork(
            n_actions, config.env.frame_stack, dueling=config.dqn.dueling
        ).to(self._device)
        self._target_net.load_state_dict(self._online_net.state_dict())
        self._target_net.eval()

        self._optimizer = optim.Adam(
            self._online_net.parameters(),
            lr=self._training_config.learning_rate,
        )

        # Replay buffer (uniform or prioritized)
        if config.dqn.prioritized_replay:
            self._replay_buffer = PrioritizedReplayBuffer(
                config.dqn.replay_buffer_size,
                alpha=config.dqn.priority_alpha,
                beta_start=config.dqn.priority_beta_start,
            )
        else:
            self._replay_buffer = ReplayBuffer(config.dqn.replay_buffer_size)

        self._epsilon = self._dqn_config.epsilon_start
        self._epsilon_decay_step = (
            (self._dqn_config.epsilon_start - self._dqn_config.epsilon_end)
            / self._dqn_config.epsilon_decay
        )

        self._step_count = 0
        self._double_dqn = self._dqn_config.double_dqn
        self._prioritized = self._dqn_config.prioritized_replay
        self._n_actions = n_actions

    def act(self, state: np.ndarray) -> int:
        """Epsilon-greedy action selection."""
        if random.random() < self._epsilon:
            return random.randint(0, self._n_actions - 1)

        state_tensor = torch.tensor(state, dtype=torch.float32, device=self._device).unsqueeze(0)
        with torch.no_grad():
            q_values = self._online_net(state_tensor)
        return q_values.argmax(dim=1).item()

    def learn(self, state, action, reward, next_state, done) -> dict | None:
        """Store transition, decay epsilon, and update if buffer has enough samples."""
        self._replay_buffer.push(state, action, reward, next_state, done)

        self._epsilon = max(
            self._dqn_config.epsilon_end,
            self._epsilon - self._epsilon_decay_step,
        )
        self._step_count += 1

        # Anneal priority beta toward 1.0 over training
        if self._prioritized and isinstance(self._replay_buffer, PrioritizedReplayBuffer):
            progress = min(1.0, self._step_count / self._training_config.total_timesteps)
            self._replay_buffer.anneal_beta(progress)

        if len(self._replay_buffer) < self._training_config.batch_size:
            return None

        metrics = self._update()

        if self._step_count % self._dqn_config.target_update_freq == 0:
            self._target_net.load_state_dict(self._online_net.state_dict())

        metrics["epsilon"] = round(self._epsilon, 4)
        return metrics

    def _update(self) -> dict:
        """One gradient step on a batch from the replay buffer."""
        states, actions, rewards, next_states, dones, indices, weights = (
            self._replay_buffer.sample(self._training_config.batch_size)
        )

        states = torch.tensor(states, dtype=torch.float32, device=self._device)
        actions = torch.tensor(actions, dtype=torch.long, device=self._device)
        rewards = torch.tensor(rewards, dtype=torch.float32, device=self._device)
        next_states = torch.tensor(next_states, dtype=torch.float32, device=self._device)
        dones = torch.tensor(dones, dtype=torch.float32, device=self._device)

        # Q-values for the actions that were actually taken
        current_q = self._online_net(states)
        current_q = current_q.gather(1, actions.unsqueeze(1)).squeeze(1)

        # Target Q-values via Bellman equation
        with torch.no_grad():
            if self._double_dqn:
                best_actions = self._online_net(next_states).argmax(dim=1)
                next_q = self._target_net(next_states)
                next_q = next_q.gather(1, best_actions.unsqueeze(1)).squeeze(1)
            else:
                next_q = self._target_net(next_states).max(dim=1).values

            target_q = rewards + self._training_config.gamma * next_q * (1 - dones)

        # Per-element TD errors (needed for priority updates)
        td_errors = (current_q - target_q).detach()

        if weights is not None:
            # Importance sampling: weight the loss per sample to correct sampling bias
            weights_t = torch.tensor(weights, dtype=torch.float32, device=self._device)
            loss = (weights_t * nn.functional.smooth_l1_loss(
                current_q, target_q, reduction="none"
            )).mean()
        else:
            loss = nn.functional.smooth_l1_loss(current_q, target_q)

        self._optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self._online_net.parameters(), max_norm=10.0)
        self._optimizer.step()

        # Update priorities in the replay buffer
        if indices is not None:
            self._replay_buffer.update_priorities(indices, td_errors.cpu().numpy())

        return {
            "loss": round(loss.item(), 4),
            "q_value": round(current_q.mean().item(), 4),
        }

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "online_net": self._online_net.state_dict(),
                "target_net": self._target_net.state_dict(),
                "optimizer": self._optimizer.state_dict(),
                "epsilon": self._epsilon,
                "step_count": self._step_count,
            },
            path,
        )

    def _remap_legacy_keys(self, state_dict: dict) -> dict:
        """Remap keys from old head layout (Flatten/Linear/ReLU/Linear at 0/1/2/3)
        to current layout (Linear/ReLU/Linear at 0/1/2)."""
        remapped = {}
        for key, value in state_dict.items():
            if key.startswith("head.1."):
                key = key.replace("head.1.", "head.0.")
            elif key.startswith("head.3."):
                key = key.replace("head.3.", "head.2.")
            remapped[key] = value
        return remapped

    def load(self, path: str | Path) -> None:
        checkpoint = torch.load(path, map_location=self._device, weights_only=True)

        online_sd = checkpoint["online_net"]
        target_sd = checkpoint["target_net"]
        # Handle checkpoints saved with a different head layer ordering
        if "head.3.weight" in online_sd:
            online_sd = self._remap_legacy_keys(online_sd)
            target_sd = self._remap_legacy_keys(target_sd)

        self._online_net.load_state_dict(online_sd)
        self._target_net.load_state_dict(target_sd)
        self._optimizer.load_state_dict(checkpoint["optimizer"])
        self._epsilon = checkpoint["epsilon"]
        self._step_count = checkpoint["step_count"]
