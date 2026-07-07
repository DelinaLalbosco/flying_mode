#!/usr/bin/env python3
"""Open the waypoint track in an interactive MuJoCo viewer."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from xml.sax.saxutils import escape

import mujoco
import mujoco.viewer
import numpy as np


TRACK_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = TRACK_ROOT.parent
DEFAULT_TRACK = TRACK_ROOT / "data" / "track.json"
DEFAULT_BASE_SCENE = (
    REPO_ROOT
    / "src"
    / "S10_sdk_deploy"
    / "S10_description"
    / "s10_mjcf"
    / "mjcf"
    / "scene.xml"
)
DEFAULT_XML = TRACK_ROOT / "scenes" / "track_on_scene.xml"
PROBE_BODY = "track_probe"


def load_waypoints(path: Path) -> list[dict[str, float | int | str]]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    rows = payload["waypoints"] if isinstance(payload, dict) else payload
    return [
        {
            "id": int(row.get("id", index)),
            "x": float(row["x"]),
            "y": float(row["y"]),
            "z": float(row["z"]),
            "role": row.get("role", "waypoint"),
        }
        for index, row in enumerate(rows)
    ]


def rgba(values: tuple[float, float, float, float]) -> str:
    return " ".join(f"{value:.3f}" for value in values)


def waypoint_color(z_value: float, z_min: float, z_max: float) -> tuple[float, float, float, float]:
    span = max(z_max - z_min, 1e-6)
    t = max(0.0, min(1.0, (z_value - z_min) / span))
    return (
        0.12 + 0.88 * t,
        0.05 + 0.85 * max(0.0, t - 0.30) / 0.70,
        0.95 * (1.0 - t),
        1.0,
    )


def waypoint_array(waypoints: list[dict[str, float | int | str]]) -> np.ndarray:
    return np.array([[point["x"], point["y"], point["z"]] for point in waypoints], dtype=float)


def format_vec(values: np.ndarray | tuple[float, ...]) -> str:
    return " ".join(f"{float(value):.4f}" for value in values)


def build_scene_xml(
    waypoints: list[dict[str, float | int | str]],
    *,
    base_scene: Path,
    line_radius: float,
    marker_radius: float,
    post_radius: float,
    show_posts: bool,
) -> str:
    points = waypoint_array(waypoints)
    mins = points.min(axis=0)
    maxs = points.max(axis=0)
    center = (mins + maxs) * 0.5

    lines = [
        '<mujoco model="track_on_scene">',
        f'  <include file="{escape(str(base_scene.resolve()))}"/>',
        '  <compiler angle="radian"/>',
        '  <option timestep="0.01" gravity="0 0 0"/>',
        '  <visual>',
        '    <headlight ambient="0.38 0.38 0.38" diffuse="0.75 0.75 0.75" specular="0.12 0.12 0.12"/>',
        '    <global offwidth="1920" offheight="1080"/>',
        '    <map znear="0.02" zfar="500"/>',
        '  </visual>',
        '  <worldbody>',
        f'    <light name="track_light" pos="{center[0]:.3f} {center[1]:.3f} 85" dir="0 0 -1" diffuse="0.9 0.9 0.9"/>',
        f'    <camera name="track_overview" pos="{center[0]:.3f} {center[1] - 72:.3f} {max(maxs[2] + 38.0, 48.0):.3f}" xyaxes="1 0 0 0 0.5 0.866"/>',
        '    <body name="track_overlay">',
    ]

    for index, (a, b) in enumerate(zip(points[:-1], points[1:])):
        if np.linalg.norm(b - a) < 1e-6:
            continue
        lines.append(
            f'      <geom name="track_segment_{index:03d}" type="capsule" fromto="{format_vec(a)} {format_vec(b)}" '
            f'size="{line_radius:.4f}" rgba="0.05 0.95 0.08 1" contype="0" conaffinity="0"/>'
        )

    z_min = float(points[:, 2].min())
    z_max = float(points[:, 2].max())
    for point, xyz in zip(waypoints, points):
        role = str(point["role"])
        point_id = int(point["id"])
        if role == "start":
            color = (0.05, 0.18, 1.0, 1.0)
            radius = marker_radius * 1.35
        elif role == "end":
            color = (1.0, 0.02, 0.02, 1.0)
            radius = marker_radius * 1.45
        else:
            color = waypoint_color(float(xyz[2]), z_min, z_max)
            radius = marker_radius

        if show_posts and xyz[2] > 0.08:
            lines.append(
                f'      <geom name="track_height_post_{point_id:03d}" type="capsule" fromto="{xyz[0]:.4f} {xyz[1]:.4f} 0 {xyz[0]:.4f} {xyz[1]:.4f} {xyz[2]:.4f}" '
                f'size="{post_radius:.4f}" rgba="0.45 0.45 0.45 0.28" contype="0" conaffinity="0"/>'
            )

        safe_role = escape(role)
        lines.append(
            f'      <geom name="track_waypoint_{point_id:03d}_{safe_role}" type="sphere" pos="{format_vec(xyz)}" '
            f'size="{radius:.4f}" rgba="{rgba(color)}" contype="0" conaffinity="0"/>'
        )

    start = points[0]
    lines.extend(
        [
            '    </body>',
            f'    <body name="{PROBE_BODY}" mocap="true" pos="{format_vec(start)}">',
            f'      <geom name="track_probe_marker" type="sphere" size="{marker_radius * 1.8:.4f}" rgba="0.05 0.25 1 0.82" contype="0" conaffinity="0"/>',
            '    </body>',
            '  </worldbody>',
            '</mujoco>',
        ]
    )
    return "\n".join(lines) + "\n"


def save_scene(xml: str, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(xml, encoding="utf-8")


def cumulative_lengths(points: np.ndarray) -> np.ndarray:
    segment_lengths = np.linalg.norm(points[1:] - points[:-1], axis=1)
    return np.concatenate(([0.0], np.cumsum(segment_lengths)))


def interpolate_path(points: np.ndarray, lengths: np.ndarray, distance: float) -> np.ndarray:
    if distance <= 0.0:
        return points[0].copy()
    if distance >= lengths[-1]:
        return points[-1].copy()
    index = int(np.searchsorted(lengths, distance, side="right") - 1)
    segment_length = max(lengths[index + 1] - lengths[index], 1e-9)
    alpha = (distance - lengths[index]) / segment_length
    return (1.0 - alpha) * points[index] + alpha * points[index + 1]


def configure_camera(viewer: mujoco.viewer.Handle, points: np.ndarray) -> None:
    mins = points.min(axis=0)
    maxs = points.max(axis=0)
    center = (mins + maxs) * 0.5
    span = float(max(maxs[0] - mins[0], maxs[1] - mins[1], 10.0))
    with viewer.lock():
        viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FREE
        viewer.cam.lookat[:] = center
        viewer.cam.distance = span * 1.25
        viewer.cam.azimuth = 135
        viewer.cam.elevation = -35


def print_controls() -> None:
    print(
        "\nMuJoCo track viewer controls:\n"
        "  mouse drag / wheel : rotate, pan, and zoom with the native MuJoCo camera\n"
        "  space              : toggle smooth marker playback\n"
        "  n or right arrow   : move marker to next waypoint\n"
        "  p or left arrow    : move marker to previous waypoint\n"
        "  r                  : reset marker to start\n"
        "  + / -              : increase/decrease playback speed\n"
        "  h                  : print this help\n"
        "  close the window   : quit\n"
    )


def run_viewer(model: mujoco.MjModel, data: mujoco.MjData, points: np.ndarray, speed: float, auto_play: bool) -> None:
    body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, PROBE_BODY)
    if body_id < 0:
        raise RuntimeError(f"Cannot find body '{PROBE_BODY}' in generated model")
    mocap_id = int(model.body_mocapid[body_id])
    if mocap_id < 0:
        raise RuntimeError(f"Body '{PROBE_BODY}' is not a mocap body")

    lengths = cumulative_lengths(points)
    state = {
        "playing": auto_play,
        "distance": 0.0,
        "index": 0,
        "speed": speed,
    }

    def set_probe(position: np.ndarray) -> None:
        data.mocap_pos[mocap_id] = position
        mujoco.mj_forward(model, data)

    def set_waypoint(index: int) -> None:
        state["index"] = int(np.clip(index, 0, len(points) - 1))
        state["distance"] = float(lengths[state["index"]])
        set_probe(points[state["index"]])
        print(f"Waypoint {state['index']}: {points[state['index']]}")

    def on_key(key_code: int) -> None:
        key = chr(key_code).lower() if 0 <= key_code < 256 else ""
        if key == " ":
            state["playing"] = not state["playing"]
            print("Playback", "on" if state["playing"] else "paused")
        elif key == "n" or key_code == 262:
            state["playing"] = False
            set_waypoint(state["index"] + 1)
        elif key == "p" or key_code == 263:
            state["playing"] = False
            set_waypoint(state["index"] - 1)
        elif key == "r":
            state["playing"] = False
            set_waypoint(0)
        elif key in {"+", "="}:
            state["speed"] *= 1.25
            print(f"Playback speed: {state['speed']:.2f} m/s")
        elif key in {"-", "_"}:
            state["speed"] = max(0.05, state["speed"] / 1.25)
            print(f"Playback speed: {state['speed']:.2f} m/s")
        elif key == "h":
            print_controls()

    print_controls()
    set_probe(points[0])
    with mujoco.viewer.launch_passive(model, data, key_callback=on_key) as viewer:
        configure_camera(viewer, points)
        last_time = time.time()
        while viewer.is_running():
            now = time.time()
            dt = now - last_time
            last_time = now
            if state["playing"]:
                state["distance"] += state["speed"] * dt
                if state["distance"] >= lengths[-1]:
                    state["distance"] = lengths[-1]
                    state["playing"] = False
                set_probe(interpolate_path(points, lengths, state["distance"]))
                state["index"] = int(np.searchsorted(lengths, state["distance"], side="right") - 1)
            mujoco.mj_step(model, data)
            viewer.sync()
            time.sleep(1.0 / 60.0)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--track", type=Path, default=DEFAULT_TRACK, help="Track JSON file.")
    parser.add_argument("--base-scene", type=Path, default=DEFAULT_BASE_SCENE, help="Base MJCF scene.xml to load.")
    parser.add_argument("--xml-out", type=Path, default=DEFAULT_XML, help="Generated MJCF output path.")
    parser.add_argument("--line-radius", type=float, default=0.14, help="Rendered track tube radius.")
    parser.add_argument("--marker-radius", type=float, default=0.42, help="Rendered waypoint sphere radius.")
    parser.add_argument("--post-radius", type=float, default=0.045, help="Vertical height guide radius.")
    parser.add_argument("--no-posts", action="store_true", help="Hide vertical height guide posts.")
    parser.add_argument("--speed", type=float, default=6.0, help="Playback speed in meters per second.")
    parser.add_argument("--play", action="store_true", help="Start marker playback immediately.")
    parser.add_argument("--compile-only", action="store_true", help="Generate and load the MJCF, then exit without opening the viewer.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    waypoints = load_waypoints(args.track)
    xml = build_scene_xml(
        waypoints,
        base_scene=args.base_scene,
        line_radius=args.line_radius,
        marker_radius=args.marker_radius,
        post_radius=args.post_radius,
        show_posts=not args.no_posts,
    )
    save_scene(xml, args.xml_out)
    model = mujoco.MjModel.from_xml_path(str(args.xml_out))
    data = mujoco.MjData(model)
    points = waypoint_array(waypoints)
    print(f"Generated MuJoCo scene: {args.xml_out}")
    print(f"Base scene: {args.base_scene}")
    print(f"Loaded {len(points)} waypoints from {args.track}")
    if not args.compile_only:
        run_viewer(model, data, points, speed=args.speed, auto_play=args.play)


if __name__ == "__main__":
    main()
