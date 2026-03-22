"""Custom Gymnasium wrappers for preprocessing Mario observations.

Standard preprocessing pipeline (Mnih et al. 2015):
    Raw game (240x256 RGB @ 60fps)
    -> SkipFrame (repeat action for 4 frames, effective 15fps)
    -> GrayScaleObservation (RGB -> grayscale, 3 channels -> 1)
    -> ResizeObservation (240x256 -> 84x84)
    -> FrameStack (stack 4 frames for motion perception)

Final observation shape: (4, 84, 84) grayscale float32.

Uses gymnasium (maintained fork). The Mario environment is created with
old gym and converted via GymToGymnasium adapter in env.py.
"""

import gymnasium
import numpy as np
from gymnasium.spaces import Box


class SkipFrame(gymnasium.Wrapper):
    """Repeat each action for `skip` frames and accumulate rewards.

    Reduces decision frequency by 4x with negligible control loss, since
    most consecutive frames are nearly identical.
    """

    def __init__(self, env: gymnasium.Env, skip: int = 4):
        super().__init__(env)
        self._skip = skip

    def step(self, action):
        total_reward = 0.0
        for _ in range(self._skip):
            obs, reward, terminated, truncated, info = self.env.step(action)
            total_reward += reward
            if terminated or truncated:
                break
        return obs, total_reward, terminated, truncated, info


class GrayScaleObservation(gymnasium.ObservationWrapper):
    """Convert RGB observations to grayscale. Reduces input size by 3x."""

    def __init__(self, env: gymnasium.Env):
        super().__init__(env)
        obs_shape = self.observation_space.shape[:2]
        self.observation_space = Box(
            low=0, high=255, shape=obs_shape, dtype=np.uint8
        )

    def observation(self, obs):
        import cv2

        gray = cv2.cvtColor(obs, cv2.COLOR_RGB2GRAY)
        return gray


class ResizeObservation(gymnasium.ObservationWrapper):
    """Resize observations to a square shape (default 84x84).

    Standard size since the original DQN paper. Reduces pixel count
    from ~61k to ~7k.
    """

    def __init__(self, env: gymnasium.Env, size: int = 84):
        super().__init__(env)
        self._size = size
        obs_shape = (size, size)
        self.observation_space = Box(
            low=0, high=255, shape=obs_shape, dtype=np.uint8
        )

    def observation(self, obs):
        import cv2

        resized = cv2.resize(obs, (self._size, self._size), interpolation=cv2.INTER_AREA)
        return resized
