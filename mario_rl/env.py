"""Environment factory for creating preprocessed Mario environments.

Wraps gym-super-mario-bros (old gym API) with a gymnasium adapter,
then applies the standard Atari preprocessing pipeline.

Usage:
    config = load_config("configs/dqn.yaml")
    env = make_env(config.env)
    obs, info = env.reset()  # obs shape: (4, 84, 84)
"""

import gymnasium
import numpy as np
from gym_super_mario_bros import SuperMarioBrosEnv
from gymnasium.spaces import Box
from gymnasium.wrappers import FrameStackObservation
from nes_py.wrappers import JoypadSpace

from mario_rl.config import EnvConfig
from mario_rl.wrappers import GrayScaleObservation, ResizeObservation, SkipFrame

# Action spaces from gym-super-mario-bros
# SIMPLE_MOVEMENT: right, right+jump, right+sprint, right+sprint+jump, jump, left, noop
# These 7 actions cover everything Mario needs for basic play.
ACTION_SPACES = {
    "SIMPLE_MOVEMENT": [
        ["right"],
        ["right", "A"],
        ["right", "B"],
        ["right", "A", "B"],
        ["A"],
        ["left"],
        ["NOOP"],
    ],
    "COMPLEX_MOVEMENT": [
        ["right"],
        ["right", "A"],
        ["right", "B"],
        ["right", "A", "B"],
        ["A"],
        ["left"],
        ["left", "A"],
        ["left", "B"],
        ["left", "A", "B"],
        ["down"],
        ["up"],
        ["NOOP"],
    ],
}


class GymToGymnasium(gymnasium.Env):
    """Adapter that wraps an old gym env to expose the modern gymnasium API.

    The old gym API:
        reset() → obs
        step()  → (obs, reward, done, info)

    The modern gymnasium API:
        reset() → (obs, info)
        step()  → (obs, reward, terminated, truncated, info)

    Why not use shimmy? Shimmy's GymV26CompatibilityV0 tries to pass `seed`
    and `options` kwargs to the old env's reset(), which nes-py doesn't accept.
    This thin adapter handles the conversion correctly for our specific case.
    """

    def __init__(self, env):
        self._env = env
        # Copy the spaces from the old env (they're compatible)
        self.observation_space = gymnasium.spaces.Box(
            low=env.observation_space.low,
            high=env.observation_space.high,
            shape=env.observation_space.shape,
            dtype=env.observation_space.dtype,
        )
        self.action_space = gymnasium.spaces.Discrete(env.action_space.n)

    def reset(self, *, seed=None, options=None):
        # Old gym reset() returns just the observation
        obs = self._env.reset()
        return obs, {}  # Gymnasium expects (obs, info)

    def step(self, action):
        obs, reward, done, info = self._env.step(action)
        # In old gym, `done` combines both "terminated" and "truncated".
        # For Mario, done=True typically means Mario died (terminated)
        # or the time ran out (also terminated in this env).
        return obs, reward, done, False, info

    def render(self):
        return self._env.render("rgb_array")

    def close(self):
        self._env.close()


class NormalizeObservation(gymnasium.ObservationWrapper):
    """Scale pixel values from [0, 255] to [0.0, 1.0].

    Keeps input magnitudes small to prevent large gradients during training.
    """

    def __init__(self, env: gymnasium.Env):
        super().__init__(env)
        self.observation_space = Box(
            low=0.0,
            high=1.0,
            shape=self.observation_space.shape,
            dtype=np.float32,
        )

    def observation(self, obs):
        return np.array(obs, dtype=np.float32) / 255.0


def make_env(env_config: EnvConfig) -> gymnasium.Env:
    """Create a preprocessed Mario environment ready for training.

    The wrapper pipeline:
        SuperMarioBros (old gym env)
        → JoypadSpace (reduce to 7 or 12 meaningful button combos)
        → GymToGymnasium (convert old gym → gymnasium API)
        → SkipFrame (repeat action for 4 frames)
        → GrayScale (RGB → single channel)
        → Resize (→ 84x84)
        → Normalize (pixels to [0, 1])
        → FrameStack (stack 4 frames for motion perception)

    Returns an env with observation shape: (4, 84, 84), values in [0, 1]
    """
    # Step 1: Create the base Mario environment (old gym API)
    # We instantiate directly instead of using gym.make() to avoid gym 0.26's
    # broken TimeLimit wrapper, which tries to use new-style step() on an
    # old-style env.
    ROM_MODES = {0: "vanilla", 1: "downsample", 2: "pixel", 3: "rectangle"}
    rom_mode = ROM_MODES.get(env_config.version, "vanilla")
    env = SuperMarioBrosEnv(rom_mode=rom_mode, target=(env_config.world, env_config.stage))

    # Step 2: Reduce action space from 256 button combos to meaningful ones
    actions = ACTION_SPACES[env_config.action_space]
    env = JoypadSpace(env, actions)

    # Step 3: Convert old gym → gymnasium API
    env = GymToGymnasium(env)

    # Step 4: Apply preprocessing wrappers (order matters!)
    env = SkipFrame(env, skip=env_config.frame_skip)
    env = GrayScaleObservation(env)
    env = ResizeObservation(env, size=env_config.frame_size)
    env = NormalizeObservation(env)
    env = FrameStackObservation(env, stack_size=env_config.frame_stack)

    return env
