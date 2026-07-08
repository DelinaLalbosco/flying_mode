#!/usr/bin/env python3
"""Visualize a waypoint track as XY, XZ, and YZ projections."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


TRACK_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TRACK = TRACK_ROOT / "data" / "track.json"


def load_waypoints(path: Path) -> list[dict[str, float | int | str]]:
    if path.suffix.lower() == ".json":
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        rows = payload["waypoints"] if isinstance(payload, dict) else payload
    elif path.suffix.lower() == ".csv":
        with path.open("r", encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))
    else:
        raise ValueError(f"Unsupported track format: {path.suffix}")

    waypoints = []
    for index, row in enumerate(rows):
        waypoints.append(
            {
                "id": int(row.get("id", index)),
                "x": float(row["x"]),
                "y": float(row["y"]),
                "z": float(row["z"]),
                "role": row.get("role", "waypoint"),
            }
        )
    return waypoints


def padded_limits(values: np.ndarray, pad_fraction: float = 0.08) -> tuple[float, float]:
    low = float(np.min(values))
    high = float(np.max(values))
    span = max(high - low, 1.0)
    return low - span * pad_fraction, high + span * pad_fraction


def add_labels(ax: plt.Axes, xs: np.ndarray, ys: np.ndarray, ids: np.ndarray) -> None:
    for x_value, y_value, waypoint_id in zip(xs, ys, ids):
        ax.annotate(
            str(int(waypoint_id)),
            (x_value, y_value),
            xytext=(0, 5),
            textcoords="offset points",
            ha="center",
            fontsize=8,
            color="navy",
        )


def draw_projection(
    ax: plt.Axes,
    x_values: np.ndarray,
    y_values: np.ndarray,
    ids: np.ndarray,
    xlabel: str,
    ylabel: str,
    title: str,
    labels: bool,
) -> None:
    ax.plot(x_values, y_values, color="#43a047", linewidth=2.5, zorder=1)
    ax.scatter(
        x_values[1:],
        y_values[1:],
        s=35,
        color="red",
        edgecolor="black",
        linewidth=0.8,
        zorder=3,
    )
    ax.scatter(
        x_values[0],
        y_values[0],
        marker="^",
        s=130,
        color="blue",
        label="start",
        zorder=4,
    )
    ax.scatter(
        x_values[-1],
        y_values[-1],
        marker="*",
        s=130,
        color="red",
        label="end",
        zorder=5,
    )
    if labels:
        add_labels(ax, x_values[1:], y_values[1:], ids[1:])
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.28)
    ax.legend(loc="best")


def visualize(waypoints: list[dict[str, float | int | str]], labels: bool) -> plt.Figure:
    ids = np.array([point["id"] for point in waypoints], dtype=float)
    x = np.array([point["x"] for point in waypoints], dtype=float)
    y = np.array([point["y"] for point in waypoints], dtype=float)
    z = np.array([point["z"] for point in waypoints], dtype=float)

    fig, axes = plt.subplots(1, 3, figsize=(20, 6), constrained_layout=True)
    fig.suptitle(f"Trajectory Visualization ({len(waypoints)} waypoints)", fontsize=16, fontweight="bold")

    axes[0].plot(x, y, color="#43a047", linewidth=2.5, zorder=1)
    scatter = axes[0].scatter(
        x,
        y,
        c=z,
        cmap="plasma",
        s=39,
        edgecolor="black",
        linewidth=0.8,
        zorder=3,
    )
    axes[0].scatter(x[0], y[0], marker="^", s=130, color="blue", label="start", zorder=4)
    axes[0].scatter(x[-1], y[-1], marker="*", s=130, color="red", label="end", zorder=5)
    if labels:
        add_labels(axes[0], x[1:], y[1:], ids[1:])
    axes[0].set_title("Top-Down (XY) View")
    axes[0].set_xlabel("X (m)")
    axes[0].set_ylabel("Y (m)")
    axes[0].grid(True, alpha=0.28)
    axes[0].legend(loc="best")
    axes[0].set_xlim(*padded_limits(x))
    axes[0].set_ylim(*padded_limits(y))
    fig.colorbar(scatter, ax=axes[0], label="Z height (m)")

    draw_projection(axes[1], x, z, ids, "X (m)", "Z (m)", "Side View (XZ)", labels)
    draw_projection(axes[2], y, z, ids, "Y (m)", "Z (m)", "Side View (YZ)", labels)

    axes[1].set_xlim(*padded_limits(x))
    axes[2].set_xlim(*padded_limits(y))
    z_limits = padded_limits(z)
    axes[1].set_ylim(min(0.0, z_limits[0]), z_limits[1])
    axes[2].set_ylim(min(0.0, z_limits[0]), z_limits[1])
    return fig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_TRACK, help="Track JSON or CSV file.")
    parser.add_argument("--save", type=Path, help="Optional output image path.")
    parser.add_argument("--no-labels", action="store_true", help="Hide waypoint id labels.")
    parser.add_argument("--show", action="store_true", help="Open an interactive plot window.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    waypoints = load_waypoints(args.input)
    fig = visualize(waypoints, labels=not args.no_labels)
    if args.save:
        args.save.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(args.save, dpi=160)
        print(f"Saved visualization to {args.save}")
    if args.show or not args.save:
        plt.show()
    plt.close(fig)


if __name__ == "__main__":
    main()
