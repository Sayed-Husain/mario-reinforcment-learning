"""Per-episode metrics logging to CSV.

Writes one row per episode with columns: episode, timestep, reward,
distance, time_alive, death_x, death_y, completed, score, coins,
elapsed_seconds. These CSVs are consumed by generate_report.py to
produce comparison charts.
"""

import csv
from collections import Counter
from datetime import datetime
from pathlib import Path


class MetricsLogger:
    """Logs training metrics to CSV, one row per episode.

    Usage:
        logger = MetricsLogger("metrics", "dqn")
        logger.begin_episode()

        for step in training_loop:
            logger.log_step(action=action, reward=reward, info=info)

        logger.end_episode(total_reward=ep_reward, info=final_info)

        logger.close()
    """

    FIELDNAMES = [
        "episode",
        "timestep",
        "reward",
        "distance",
        "time_alive",
        "death_x",
        "death_y",
        "completed",
        "score",
        "coins",
        "elapsed_seconds",
    ]

    def __init__(self, output_dir: str | Path, run_name: str):
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self._path = output_dir / f"{run_name}_{timestamp}.csv"

        self._file = open(self._path, "w", newline="")
        self._writer = csv.DictWriter(self._file, fieldnames=self.FIELDNAMES)
        self._writer.writeheader()

        self._episode = 0
        self._global_step = 0
        self._episode_start = None
        self._action_counts: Counter = Counter()
        self._steps_this_episode = 0
        self._last_info: dict = {}

    def begin_episode(self):
        """Call at the start of each episode."""
        self._episode += 1
        self._episode_start = datetime.now()
        self._action_counts.clear()
        self._steps_this_episode = 0
        self._last_info = {}

    def log_step(self, action: int, reward: float, info: dict):
        """Call after each environment step."""
        self._global_step += 1
        self._steps_this_episode += 1
        self._action_counts[action] += 1
        self._last_info = info

    def end_episode(self, total_reward: float, info: dict):
        """Call at the end of each episode. Writes one row to the CSV."""
        elapsed = (datetime.now() - self._episode_start).total_seconds()

        # The info dict from gym-super-mario-bros contains:
        # x_pos, y_pos, score, coins, time, life, stage, world, flag_get
        row = {
            "episode": self._episode,
            "timestep": self._global_step,
            "reward": round(total_reward, 2),
            "distance": info.get("x_pos", 0),
            "time_alive": self._steps_this_episode,
            "death_x": info.get("x_pos", 0) if info.get("life", 3) < 3 else -1,
            "death_y": info.get("y_pos", 0) if info.get("life", 3) < 3 else -1,
            "completed": 1 if info.get("flag_get", False) else 0,
            "score": info.get("score", 0),
            "coins": info.get("coins", 0),
            "elapsed_seconds": round(elapsed, 2),
        }

        self._writer.writerow(row)
        self._file.flush()  # Write immediately so we don't lose data on crash

    def close(self):
        """Close the CSV file."""
        self._file.close()

    @property
    def path(self) -> Path:
        """Path to the CSV file being written."""
        return self._path

    @property
    def episode_count(self) -> int:
        return self._episode

    @property
    def global_step(self) -> int:
        return self._global_step
