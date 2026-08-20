#!/usr/bin/env python3

import re
import xml.etree.ElementTree as ET
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


# ============================================================
# S10 TRACK XML
# ============================================================

XML_PATH = Path(
    "src/S10_sdk_deploy/S10_description/s10_mjcf/mjcf/S10_track.xml"
)

# Waypoint reach radius
REACH_RADIUS = 0.20


# ============================================================
# Extract all waypoints from XML
# ============================================================

def load_waypoints(xml_path):

    tree = ET.parse(xml_path)
    root = tree.getroot()

    waypoints = {}

    for geom in root.iter("geom"):

        name = geom.get("name", "")

        match = re.match(
            r"track_waypoint_(\d+)",
            name,
        )

        if match is None:
            continue

        index = int(match.group(1))

        pos = geom.get("pos")

        if pos is None:
            continue

        values = [
            float(v)
            for v in pos.split()
        ]

        if len(values) < 3:
            continue

        waypoints[index] = values[:3]

    if not waypoints:
        raise RuntimeError(
            "No track_waypoint_* elements found."
        )

    indices = sorted(waypoints.keys())

    points = np.array(
        [waypoints[i] for i in indices],
        dtype=float,
    )

    return indices, points


# ============================================================
# Main
# ============================================================

def main():

    if not XML_PATH.exists():

        raise FileNotFoundError(
            f"\nXML file not found:\n{XML_PATH}"
        )

    indices, points = load_waypoints(
        XML_PATH
    )

    print()
    print("=" * 75)
    print("S10 COMPLETE WAYPOINT PATH")
    print("=" * 75)

    print(
        f"{'WP':>5} "
        f"{'X (m)':>12} "
        f"{'Y (m)':>12} "
        f"{'Z (m)':>12}"
    )

    print("-" * 75)

    for i, p in zip(indices, points):

        print(
            f"{i:5d} "
            f"{p[0]:12.4f} "
            f"{p[1]:12.4f} "
            f"{p[2]:12.4f}"
        )

    print("-" * 75)

    print(
        f"Total waypoints: {len(points)}"
    )

    # ========================================================
    # Calculate total path length
    # ========================================================

    path_length = 0.0

    for i in range(1, len(points)):

        segment = np.linalg.norm(
            points[i, :2]
            - points[i - 1, :2]
        )

        path_length += segment

    print(
        f"Approx. X-Y path length: "
        f"{path_length:.3f} m"
    )

    print(
        f"Waypoint reach radius: "
        f"{REACH_RADIUS:.2f} m"
    )

    print("=" * 75)

    # ========================================================
    # X-Y coordinates
    # ========================================================

    x = points[:, 0]
    y = points[:, 1]

    # ========================================================
    # Create top-down plot
    # ========================================================

    fig, ax = plt.subplots(
        figsize=(12, 9)
    )

    # Complete path
    ax.plot(
        x,
        y,
        linewidth=2,
        marker="o",
        markersize=5,
        label="Waypoint path",
        zorder=2,
    )

    # ========================================================
    # Draw waypoint reach radius
    # ========================================================

    for xi, yi in zip(x, y):

        circle = plt.Circle(
            (xi, yi),
            REACH_RADIUS,
            fill=False,
            alpha=0.25,
            linewidth=1,
        )

        ax.add_patch(circle)

    # ========================================================
    # Waypoint numbers
    # ========================================================

    for i, xi, yi in zip(
        indices,
        x,
        y,
    ):

        ax.annotate(
            f"{i}",
            (xi, yi),
            xytext=(6, 6),
            textcoords="offset points",
            fontsize=8,
        )

    # ========================================================
    # Start point
    # ========================================================

    ax.scatter(
        x[0],
        y[0],
        s=180,
        marker="s",
        label="Start",
        zorder=5,
    )

    # ========================================================
    # Final point
    # ========================================================

    ax.scatter(
        x[-1],
        y[-1],
        s=180,
        marker="X",
        label="Final waypoint",
        zorder=5,
    )

    # ========================================================
    # Direction arrows
    # ========================================================

    for i in range(
        len(points) - 1
    ):

        dx = x[i + 1] - x[i]
        dy = y[i + 1] - y[i]

        ax.annotate(
            "",
            xy=(x[i + 1], y[i + 1]),
            xytext=(x[i], y[i]),
            arrowprops=dict(
                arrowstyle="->",
                linewidth=1,
            ),
        )

    # ========================================================
    # Labels
    # ========================================================

    ax.set_xlabel(
        "X position (m)",
        fontsize=13,
    )

    ax.set_ylabel(
        "Y position (m)",
        fontsize=13,
    )

    ax.set_title(
        "Deep Robotics S10 — Complete Waypoint Track",
        fontsize=15,
    )

    ax.grid(
        True,
        linestyle="--",
        alpha=0.5,
    )

    ax.set_aspect(
        "equal",
        adjustable="box",
    )

    ax.legend()

    plt.tight_layout()

    # ========================================================
    # Save
    # ========================================================

    output = Path(
        "S10_complete_waypoint_track.png"
    )

    plt.savefig(
        output,
        dpi=300,
        bbox_inches="tight",
    )

    print()
    print(
        f"Plot saved to:"
    )

    print(
        output.resolve()
    )

    print()

    plt.show()


if __name__ == "__main__":
    main()
