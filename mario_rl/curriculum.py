"""Curriculum learning scheduler.

Manages staged training where the environment difficulty increases
as the agent improves. Tracks completion rate over a rolling window
and signals when to advance to the next stage.

Typical schedule:
    Stage 1: World 1-1 v2 (no enemies), learn level geometry
    Stage 2: World 1-1 v0 (standard), learn enemy avoidance
"""

from collections import deque
from dataclasses import replace

from mario_rl.config import CurriculumConfig, EnvConfig


class CurriculumScheduler:
    """Tracks agent performance and controls stage transitions.

    Args:
        curriculum_config: CurriculumConfig with stages and eval_window
    """

    def __init__(self, curriculum_config: CurriculumConfig):
        self._stages = curriculum_config.stages
        self._eval_window = curriculum_config.eval_window
        self._current = 0
        self._completions = deque(maxlen=self._eval_window)
        self._stage_start_step = 0

    def get_current_env_config(self, base_config: EnvConfig) -> EnvConfig:
        """Return a modified EnvConfig with the current stage's world/stage/version."""
        stage = self._stages[self._current]
        return replace(
            base_config,
            world=stage.world,
            stage=stage.stage,
            version=stage.version,
        )

    def record_episode(self, completed: bool):
        """Record whether the agent completed the level this episode."""
        self._completions.append(1 if completed else 0)

    @property
    def completion_rate(self) -> float:
        """Rolling completion rate over the eval window."""
        if not self._completions:
            return 0.0
        return sum(self._completions) / len(self._completions)

    def should_advance(self, total_steps: int) -> bool:
        """Check if the agent is ready to move to the next stage."""
        if self._current >= len(self._stages) - 1:
            return False  # Already on the last stage

        stage = self._stages[self._current]
        steps_in_stage = total_steps - self._stage_start_step

        # Must have trained for minimum steps and filled the eval window
        if steps_in_stage < stage.min_timesteps:
            return False
        if len(self._completions) < self._eval_window:
            return False

        return self.completion_rate >= stage.advance_threshold

    def advance(self, total_steps: int):
        """Move to the next curriculum stage."""
        self._current += 1
        self._completions.clear()
        self._stage_start_step = total_steps

    @property
    def current_stage_index(self) -> int:
        return self._current

    @property
    def current_stage(self):
        return self._stages[self._current]

    @property
    def is_complete(self) -> bool:
        """True if we've advanced past all stages (on the last one)."""
        return self._current >= len(self._stages) - 1

    @property
    def total_stages(self) -> int:
        return len(self._stages)
