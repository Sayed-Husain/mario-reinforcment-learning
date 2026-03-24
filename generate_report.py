"""Generate comparison charts from training metrics.

Reads CSV files from metrics/, produces PNG charts in assets/charts/.
Automatically detects all available runs and generates every chart it can.

Usage:
    uv run generate_report.py
    uv run generate_report.py --metrics-dir metrics --output-dir assets/charts
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

# Consistent colors per algorithm
ALGO_COLORS = {
    "dqn": "#6b8cff",
    "rainbow": "#e7413c",
    "ppo": "#ffa726",
}

ALGO_ORDER = ["dqn", "rainbow", "ppo"]
LEVEL_LENGTH = 3161


def setup_style():
    """Dark theme matching the project aesthetic."""
    plt.rcParams.update({
        "figure.facecolor": "#141414",
        "axes.facecolor": "#1a1a1a",
        "axes.edgecolor": "#333",
        "axes.labelcolor": "#ccc",
        "axes.titlecolor": "#fff",
        "text.color": "#ccc",
        "xtick.color": "#888",
        "ytick.color": "#888",
        "grid.color": "#2a2a2a",
        "grid.alpha": 0.8,
        "legend.facecolor": "#1a1a1a",
        "legend.edgecolor": "#333",
        "legend.labelcolor": "#ccc",
        "font.size": 11,
        "axes.titlesize": 14,
        "axes.titleweight": "bold",
        "figure.titlesize": 16,
        "figure.titleweight": "bold",
    })


def algo_color(name):
    return ALGO_COLORS.get(name.lower(), "#aaa")


def load_all_runs(metrics_dir: Path) -> dict[str, pd.DataFrame]:
    """Load all CSV files from metrics dir, keyed by algorithm name."""
    runs = {}
    for csv_path in sorted(metrics_dir.glob("*.csv")):
        df = pd.read_csv(csv_path)
        if df.empty:
            continue
        algo = csv_path.stem.split("_")[0]
        runs[algo] = df
    return {k: runs[k] for k in ALGO_ORDER if k in runs}


def smooth(values: pd.Series, window: int = 50) -> pd.Series:
    return values.rolling(window=window, min_periods=1).mean()


def plot_distance_progress(runs: dict[str, pd.DataFrame], output_dir: Path):
    fig, ax = plt.subplots(figsize=(12, 6))

    for algo, df in runs.items():
        color = algo_color(algo)
        ax.plot(df["episode"], df["distance"], alpha=0.08, color=color)
        ax.plot(df["episode"], smooth(df["distance"]), label=algo.upper(),
                alpha=0.95, color=color, linewidth=2)

    # Reference line for level completion
    ax.axhline(LEVEL_LENGTH, color="#4caf50", linestyle="--", linewidth=1, alpha=0.5)
    ax.text(50, LEVEL_LENGTH + 60, "Level clear (3,161)", color="#4caf50",
            fontsize=9, alpha=0.7)

    # Mark key level landmarks
    landmarks = [(450, "First pipes"), (1300, "Gap"), (2800, "Stairs")]
    for y, label in landmarks:
        ax.axhline(y, color="#444", linestyle=":", linewidth=0.8, alpha=0.4)
        ax.text(50, y + 40, label, color="#666", fontsize=8)

    ax.set_xlabel("Episode")
    ax.set_ylabel("Distance (pixels)")
    ax.set_title("Distance Progress Over Training")
    ax.legend(framealpha=0.9, loc="lower right")
    ax.grid(True)
    ax.set_ylim(bottom=0, top=LEVEL_LENGTH + 300)
    fig.tight_layout()
    fig.savefig(output_dir / "distance_progress.png", dpi=150)
    plt.close(fig)
    print("  Saved distance_progress.png")


def plot_reward_curves(runs: dict[str, pd.DataFrame], output_dir: Path):
    fig, ax = plt.subplots(figsize=(12, 6))

    for algo, df in runs.items():
        color = algo_color(algo)
        ax.plot(df["episode"], df["reward"], alpha=0.08, color=color)
        ax.plot(df["episode"], smooth(df["reward"]), label=algo.upper(),
                alpha=0.95, color=color, linewidth=2)

    # Annotate reward meaning
    ax.axhline(0, color="#666", linestyle="-", linewidth=0.5, alpha=0.5)
    ax.text(50, 50, "Reward > 0: net forward progress", color="#4caf50", fontsize=9, alpha=0.7)
    ax.text(50, -100, "Reward < 0: dying early / moving left", color="#e7413c", fontsize=9, alpha=0.7)

    ax.set_xlabel("Episode")
    ax.set_ylabel("Total Reward")
    ax.set_title("Reward Over Training")
    ax.legend(framealpha=0.9, loc="lower right")
    ax.grid(True)
    fig.tight_layout()
    fig.savefig(output_dir / "reward_curves.png", dpi=150)
    plt.close(fig)
    print("  Saved reward_curves.png")


def plot_algorithm_comparison(runs: dict[str, pd.DataFrame], output_dir: Path):
    if len(runs) < 2:
        print("  Skipping algorithm comparison (need 2+ algorithms)")
        return

    stats = []
    for algo, df in runs.items():
        tail = df.tail(max(1, len(df) // 5))
        stats.append({
            "algo": algo,
            "Algorithm": algo.upper(),
            "Avg Distance": tail["distance"].mean(),
            "Avg Reward": tail["reward"].mean(),
            "Completion Rate (%)": tail["completed"].mean() * 100,
            "Avg Score": tail["score"].mean(),
        })

    stats_df = pd.DataFrame(stats)

    metrics_info = [
        ("Avg Distance", f"out of {LEVEL_LENGTH:,}"),
        ("Avg Reward", "higher = better"),
        ("Completion Rate (%)", "World 1-1"),
        ("Avg Score", "in-game score"),
    ]

    fig, axes = plt.subplots(1, len(metrics_info), figsize=(4.5 * len(metrics_info), 5.5))

    for ax, (metric, subtitle) in zip(axes, metrics_info):
        colors = [algo_color(a) for a in stats_df["algo"]]
        bars = ax.bar(stats_df["Algorithm"], stats_df[metric], color=colors, width=0.6)
        ax.set_title(f"{metric}\n", fontsize=12)
        ax.text(0.5, 1.0, subtitle, transform=ax.transAxes, ha="center",
                fontsize=9, color="#888", style="italic")
        ax.set_ylabel("")
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2., height,
                    f"{height:.0f}", ha="center", va="bottom", fontsize=10, color="#ccc")
        ax.grid(True, axis="y")

    fig.suptitle("Converged Performance (last 20% of training)", y=1.02)
    fig.tight_layout()
    fig.savefig(output_dir / "algorithm_comparison.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  Saved algorithm_comparison.png")


def plot_death_heatmap(runs: dict[str, pd.DataFrame], output_dir: Path):
    fig, ax = plt.subplots(figsize=(12, 5))

    for algo, df in runs.items():
        deaths = df[df["death_x"] > 0]["death_x"]
        if deaths.empty:
            continue
        ax.hist(deaths, bins=60, alpha=0.55, color=algo_color(algo),
                label=f"{algo.upper()} ({len(deaths):,} deaths)", edgecolor="none")

    # Mark known obstacles with shaded regions
    regions = [
        (350, 550, "Pipe\nsection", "#e7413c"),
        (1200, 1400, "Gap", "#ffa726"),
        (2700, 2900, "Staircase", "#6b8cff"),
    ]
    ymax = ax.get_ylim()[1]
    for x1, x2, label, color in regions:
        ax.axvspan(x1, x2, alpha=0.08, color=color)
        ax.text((x1 + x2) / 2, ymax * 0.93, label, color=color, fontsize=9,
                ha="center", va="top", alpha=0.8)

    # Level completion line
    ax.axvline(LEVEL_LENGTH, color="#4caf50", linestyle="--", linewidth=1, alpha=0.5)
    ax.text(LEVEL_LENGTH - 50, ymax * 0.5, "Flag", color="#4caf50", fontsize=9,
            ha="right", alpha=0.7, rotation=90)

    ax.set_xlabel("X Position in Level (pixels)")
    ax.set_ylabel("Death Count")
    ax.set_title("Death Distribution Across World 1-1")
    ax.legend(framealpha=0.9)
    ax.grid(True, axis="y")
    fig.tight_layout()
    fig.savefig(output_dir / "death_heatmap.png", dpi=150)
    plt.close(fig)
    print("  Saved death_heatmap.png")


def plot_action_distribution(runs: dict[str, pd.DataFrame], output_dir: Path):
    fig, axes = plt.subplots(1, len(runs), figsize=(5.5 * len(runs), 5), squeeze=False)

    for idx, (algo, df) in enumerate(runs.items()):
        ax = axes[0][idx]
        n = len(df)
        if n < 20:
            ax.text(0.5, 0.5, "Not enough data", ha="center", va="center",
                    transform=ax.transAxes, color="#888")
            ax.set_title(f"{algo.upper()}")
            continue

        early = df.head(n // 5)["distance"]
        late = df.tail(n // 5)["distance"]

        ax.hist(early, bins=30, alpha=0.5, label=f"First 20% (avg {early.mean():.0f})",
                color="#ffa726", edgecolor="none")
        ax.hist(late, bins=30, alpha=0.5, label=f"Last 20% (avg {late.mean():.0f})",
                color="#4caf50", edgecolor="none")

        # Show the shift
        ax.axvline(early.mean(), color="#ffa726", linestyle="--", linewidth=1.5, alpha=0.7)
        ax.axvline(late.mean(), color="#4caf50", linestyle="--", linewidth=1.5, alpha=0.7)

        improvement = late.mean() - early.mean()
        sign = "+" if improvement > 0 else ""
        ax.set_title(f"{algo.upper()} ({sign}{improvement:.0f} px)")

        ax.set_xlabel("Distance Reached")
        ax.set_ylabel("Episode Count")
        ax.legend(framealpha=0.9, fontsize=9)
        ax.grid(True, axis="y")

    fig.suptitle("How Far Does the Agent Get? Early vs Late Training", y=1.02)
    fig.tight_layout()
    fig.savefig(output_dir / "action_distribution.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  Saved action_distribution.png")


def plot_completion_time(runs: dict[str, pd.DataFrame], output_dir: Path):
    fig, ax = plt.subplots(figsize=(12, 6))
    any_completions = False

    for algo, df in runs.items():
        completed = df[df["completed"] == 1]
        if completed.empty:
            continue
        any_completions = True
        color = algo_color(algo)
        ax.scatter(completed["episode"], completed["time_alive"],
                   label=f"{algo.upper()} ({len(completed):,} clears)",
                   alpha=0.3, s=15, color=color, edgecolor="none")
        if len(completed) > 5:
            smoothed = smooth(completed["time_alive"], window=10)
            ax.plot(completed["episode"], smoothed,
                    linewidth=2.5, color=color, alpha=0.9)

    if not any_completions:
        plt.close(fig)
        print("  Skipping completion time (no completed episodes)")
        return

    # Annotate what faster means
    ax.annotate("Faster clears = more efficient routing",
                xy=(0.98, 0.95), xycoords="axes fraction",
                ha="right", va="top", fontsize=9, color="#888", style="italic")

    # Find and annotate first completion for each algo
    for algo, df in runs.items():
        completed = df[df["completed"] == 1]
        if completed.empty:
            continue
        first = completed.iloc[0]
        ax.annotate(f"First {algo.upper()} clear",
                    xy=(first["episode"], first["time_alive"]),
                    xytext=(first["episode"] + 200, first["time_alive"] + 50),
                    fontsize=8, color=algo_color(algo),
                    arrowprops=dict(arrowstyle="->", color=algo_color(algo), lw=1),
                    alpha=0.8)

    ax.set_xlabel("Episode")
    ax.set_ylabel("Steps to Clear Level")
    ax.set_title("Completion Speed Over Training")
    ax.legend(framealpha=0.9)
    ax.grid(True)
    fig.tight_layout()
    fig.savefig(output_dir / "completion_time.png", dpi=150)
    plt.close(fig)
    print("  Saved completion_time.png")


def generate_report(metrics_dir: str = "metrics", output_dir: str = "assets/charts"):
    metrics_path = Path(metrics_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    setup_style()

    print(f"Loading runs from {metrics_path}/")
    runs = load_all_runs(metrics_path)

    if not runs:
        print("No metrics files found. Run training first.")
        return

    print(f"Found {len(runs)} algorithm(s): {', '.join(r.upper() for r in runs)}")
    total_episodes = sum(len(df) for df in runs.values())
    print(f"Total episodes across all runs: {total_episodes}")
    print()

    print("Generating charts:")
    plot_distance_progress(runs, output_path)
    plot_reward_curves(runs, output_path)
    plot_algorithm_comparison(runs, output_path)
    plot_death_heatmap(runs, output_path)
    plot_action_distribution(runs, output_path)
    plot_completion_time(runs, output_path)

    print(f"\nAll charts saved to {output_path}/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate training report charts")
    parser.add_argument("--metrics-dir", default="metrics", help="Directory with CSV files")
    parser.add_argument("--output-dir", default="assets/charts", help="Output directory for PNGs")
    args = parser.parse_args()

    generate_report(args.metrics_dir, args.output_dir)
