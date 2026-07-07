#!/usr/bin/env python3
"""Select waypoint dots directly on the real MuJoCo scene geometry."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import glfw
import mujoco
import numpy as np


TRACK_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = TRACK_ROOT.parent
DEFAULT_BASE_SCENE = (
    REPO_ROOT
    / "src"
    / "S10_sdk_deploy"
    / "S10_description"
    / "s10_mjcf"
    / "mjcf"
    / "scene.xml"
)
DEFAULT_JSON = TRACK_ROOT / "data" / "new_track.json"
DEFAULT_CSV = TRACK_ROOT / "data" / "new_track.csv"


def point_color(index: int, count: int) -> np.ndarray:
    if index == 0:
        return np.array([0.05, 0.18, 1.0, 1.0], dtype=np.float32)
    if count > 1 and index == count - 1:
        return np.array([1.0, 0.03, 0.02, 1.0], dtype=np.float32)
    t = index / max(count - 1, 1)
    return np.array([0.25 + 0.70 * t, 0.05 + 0.45 * t, 0.95 * (1.0 - t), 1.0], dtype=np.float32)


def load_json_track(path: Path) -> list[dict[str, float | int | str]]:
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


class MuJoCoPointSelector:
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.model = mujoco.MjModel.from_xml_path(str(args.base_scene))
        self.data = mujoco.MjData(self.model)
        mujoco.mj_forward(self.model, self.data)

        self.points = load_json_track(args.load) if args.load else []
        self._renumber_roles()

        self.cam = mujoco.MjvCamera()
        self.opt = mujoco.MjvOption()
        self.scene = mujoco.MjvScene(self.model, maxgeom=args.maxgeom)
        self.context = None
        self.window = None
        self.last_x = 0.0
        self.last_y = 0.0
        self.button_left = False
        self.button_middle = False
        self.button_right = False
        self.help_visible = True
        self.last_message = "Ctrl+left-click a scene surface to add a waypoint."

        self.reset_camera()

    def reset_camera(self) -> None:
        self.cam.type = mujoco.mjtCamera.mjCAMERA_FREE
        self.cam.lookat[:] = self.model.stat.center
        self.cam.distance = max(float(self.model.stat.extent) * 1.2, 8.0)
        self.cam.azimuth = 135
        self.cam.elevation = -35

    def _renumber_roles(self) -> None:
        for index, point in enumerate(self.points):
            point["id"] = index
            point["role"] = "waypoint"
        if self.points:
            self.points[0]["role"] = "start"
            self.points[-1]["role"] = "end"

    def add_waypoint(self, position: np.ndarray) -> None:
        position = position.astype(float)
        position[2] += self.args.z_offset
        self.points.append(
            {
                "id": len(self.points),
                "x": round(float(position[0]), 4),
                "y": round(float(position[1]), 4),
                "z": round(float(position[2]), 4),
                "role": "waypoint",
            }
        )
        self._renumber_roles()
        self.last_message = f"Added waypoint {len(self.points) - 1}: {position.round(4)}"
        print(self.last_message)

    def save(self) -> None:
        self._renumber_roles()
        payload = {
            "name": self.args.name,
            "units": "meters",
            "source": {
                "method": "selected with select_track_points.py on MuJoCo scene geometry",
                "base_scene": str(self.args.base_scene),
                "z_offset": self.args.z_offset,
            },
            "waypoints": self.points,
        }
        self.args.out_json.parent.mkdir(parents=True, exist_ok=True)
        self.args.out_csv.parent.mkdir(parents=True, exist_ok=True)
        with self.args.out_json.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2)
            handle.write("\n")
        with self.args.out_csv.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=["id", "x", "y", "z", "role"])
            writer.writeheader()
            writer.writerows(self.points)
        self.last_message = f"Saved {len(self.points)} waypoints to {self.args.out_json}"
        print(self.last_message)

    def update_base_scene(self) -> None:
        mujoco.mjv_updateScene(
            self.model,
            self.data,
            self.opt,
            None,
            self.cam,
            mujoco.mjtCatBit.mjCAT_ALL,
            self.scene,
        )

    def add_overlay_geom(self, geom_type, size, pos, rgba) -> None:
        if self.scene.ngeom >= self.scene.maxgeom:
            return
        geom = self.scene.geoms[self.scene.ngeom]
        mujoco.mjv_initGeom(
            geom,
            geom_type,
            np.asarray(size, dtype=np.float64),
            np.asarray(pos, dtype=np.float64),
            np.eye(3, dtype=np.float64).reshape(-1),
            np.asarray(rgba, dtype=np.float32),
        )
        self.scene.ngeom += 1

    def add_overlay_capsule(self, start: np.ndarray, end: np.ndarray, radius: float, rgba: np.ndarray) -> None:
        if self.scene.ngeom >= self.scene.maxgeom:
            return
        geom = self.scene.geoms[self.scene.ngeom]
        mujoco.mjv_initGeom(
            geom,
            mujoco.mjtGeom.mjGEOM_CAPSULE,
            np.zeros(3, dtype=np.float64),
            np.zeros(3, dtype=np.float64),
            np.eye(3, dtype=np.float64).reshape(-1),
            np.asarray(rgba, dtype=np.float32),
        )
        mujoco.mjv_connector(
            geom,
            mujoco.mjtGeom.mjGEOM_CAPSULE,
            radius,
            np.asarray(start, dtype=np.float64),
            np.asarray(end, dtype=np.float64),
        )
        self.scene.ngeom += 1

    def add_track_overlay(self) -> None:
        if not self.points:
            return
        positions = np.array([[p["x"], p["y"], p["z"]] for p in self.points], dtype=np.float64)
        line_rgba = np.array([0.05, 0.95, 0.08, 1.0], dtype=np.float32)
        post_rgba = np.array([0.45, 0.45, 0.45, 0.35], dtype=np.float32)
        for start, end in zip(positions[:-1], positions[1:]):
            if np.linalg.norm(end - start) > 1e-6:
                self.add_overlay_capsule(start, end, self.args.line_radius, line_rgba)
        for index, position in enumerate(positions):
            if self.args.show_posts and position[2] > 0.05:
                floor = np.array([position[0], position[1], 0.0], dtype=np.float64)
                self.add_overlay_capsule(floor, position, self.args.post_radius, post_rgba)
            radius = self.args.marker_radius * (1.35 if index in {0, len(positions) - 1} else 1.0)
            self.add_overlay_geom(
                mujoco.mjtGeom.mjGEOM_SPHERE,
                np.array([radius, 0.0, 0.0], dtype=np.float64),
                position,
                point_color(index, len(positions)),
            )

    def select_scene_point(self, xpos: float, ypos: float) -> None:
        width, height = glfw.get_framebuffer_size(self.window)
        if width <= 0 or height <= 0:
            return
        self.update_base_scene()
        selected = np.zeros(3, dtype=np.float64)
        geom_id = np.zeros(1, dtype=np.int32)
        flex_id = np.zeros(1, dtype=np.int32)
        skin_id = np.zeros(1, dtype=np.int32)
        rel_x = xpos / width
        rel_y = 1.0 - ypos / height
        body_id = mujoco.mjv_select(
            self.model,
            self.data,
            self.opt,
            width / height,
            rel_x,
            rel_y,
            self.scene,
            selected,
            geom_id,
            flex_id,
            skin_id,
        )
        if body_id < 0 and geom_id[0] < 0:
            self.last_message = "No scene geometry selected. Try Ctrl+clicking a visible surface."
            print(self.last_message)
            return
        self.add_waypoint(selected)

    def on_mouse_button(self, window, button, action, mods) -> None:
        self.button_left = glfw.get_mouse_button(window, glfw.MOUSE_BUTTON_LEFT) == glfw.PRESS
        self.button_middle = glfw.get_mouse_button(window, glfw.MOUSE_BUTTON_MIDDLE) == glfw.PRESS
        self.button_right = glfw.get_mouse_button(window, glfw.MOUSE_BUTTON_RIGHT) == glfw.PRESS
        self.last_x, self.last_y = glfw.get_cursor_pos(window)
        ctrl = bool(mods & glfw.MOD_CONTROL)
        if button == glfw.MOUSE_BUTTON_LEFT and action == glfw.PRESS and ctrl:
            self.select_scene_point(self.last_x, self.last_y)
            self.button_left = False

    def on_mouse_move(self, window, xpos, ypos) -> None:
        dx = xpos - self.last_x
        dy = ypos - self.last_y
        self.last_x = xpos
        self.last_y = ypos
        if not (self.button_left or self.button_middle or self.button_right):
            return
        width, height = glfw.get_framebuffer_size(window)
        if width <= 0 or height <= 0:
            return
        shift = (
            glfw.get_key(window, glfw.KEY_LEFT_SHIFT) == glfw.PRESS
            or glfw.get_key(window, glfw.KEY_RIGHT_SHIFT) == glfw.PRESS
        )
        if self.button_right:
            action = mujoco.mjtMouse.mjMOUSE_ZOOM
        elif self.button_middle or (self.button_left and shift):
            action = mujoco.mjtMouse.mjMOUSE_MOVE_H
        else:
            action = mujoco.mjtMouse.mjMOUSE_ROTATE_H
        mujoco.mjv_moveCamera(self.model, action, dx / height, dy / height, self.scene, self.cam)

    def on_scroll(self, window, xoffset, yoffset) -> None:
        mujoco.mjv_moveCamera(self.model, mujoco.mjtMouse.mjMOUSE_ZOOM, 0.0, -0.05 * yoffset, self.scene, self.cam)

    def on_key(self, window, key, scancode, action, mods) -> None:
        if action != glfw.PRESS:
            return
        if key == glfw.KEY_U:
            self.points = self.points[:-1]
            self._renumber_roles()
            self.last_message = "Undid last waypoint."
            print(self.last_message)
        elif key == glfw.KEY_C:
            self.points = []
            self.last_message = "Cleared all waypoints."
            print(self.last_message)
        elif key == glfw.KEY_S:
            self.save()
        elif key == glfw.KEY_R:
            self.reset_camera()
        elif key == glfw.KEY_H:
            self.help_visible = not self.help_visible
        elif key in {glfw.KEY_Q, glfw.KEY_ESCAPE}:
            self.save()
            glfw.set_window_should_close(window, True)

    def draw_overlay_text(self, viewport) -> None:
        if not self.help_visible:
            return
        left = (
            "Ctrl+left-click: add point on scene\n"
            "left drag: rotate | shift+left/middle: pan | right/wheel: zoom\n"
            "u undo | c clear | s save | r reset camera | h hide help | q save+quit"
        )
        right = f"points: {len(self.points)}\n{self.last_message}"
        mujoco.mjr_overlay(
            mujoco.mjtFontScale.mjFONTSCALE_150,
            mujoco.mjtGridPos.mjGRID_TOPLEFT,
            viewport,
            left,
            right,
            self.context,
        )

    def run(self) -> None:
        if not glfw.init():
            raise RuntimeError("Could not initialize GLFW.")
        try:
            self.window = glfw.create_window(self.args.width, self.args.height, "MuJoCo Scene Point Selector", None, None)
            if not self.window:
                raise RuntimeError("Could not create GLFW window.")
            glfw.make_context_current(self.window)
            glfw.swap_interval(1)

            glfw.set_mouse_button_callback(self.window, self.on_mouse_button)
            glfw.set_cursor_pos_callback(self.window, self.on_mouse_move)
            glfw.set_scroll_callback(self.window, self.on_scroll)
            glfw.set_key_callback(self.window, self.on_key)

            self.context = mujoco.MjrContext(self.model, mujoco.mjtFontScale.mjFONTSCALE_150)
            print("Loaded base scene:", self.args.base_scene)
            print("Ctrl+left-click visible scene geometry to select waypoint dots.")

            while not glfw.window_should_close(self.window):
                width, height = glfw.get_framebuffer_size(self.window)
                viewport = mujoco.MjrRect(0, 0, width, height)
                self.update_base_scene()
                self.add_track_overlay()
                mujoco.mjr_render(viewport, self.scene, self.context)
                self.draw_overlay_text(viewport)
                glfw.swap_buffers(self.window)
                glfw.poll_events()
        finally:
            if self.window:
                glfw.destroy_window(self.window)
            glfw.terminate()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-scene", type=Path, default=DEFAULT_BASE_SCENE, help="Base MJCF scene.xml to load.")
    parser.add_argument("--load", type=Path, help="Optional JSON track to preload and edit.")
    parser.add_argument("--out-json", type=Path, default=DEFAULT_JSON, help="Output JSON file.")
    parser.add_argument("--out-csv", type=Path, default=DEFAULT_CSV, help="Output CSV file.")
    parser.add_argument("--name", default="new_scene_track", help="Track name stored in the JSON output.")
    parser.add_argument("--z-offset", type=float, default=0.025, help="Height added to clicked surface points.")
    parser.add_argument("--marker-radius", type=float, default=0.21)
    parser.add_argument("--line-radius", type=float, default=0.07)
    parser.add_argument("--post-radius", type=float, default=0.0225)
    parser.add_argument("--no-posts", dest="show_posts", action="store_false", help="Hide vertical height guide posts.")
    parser.add_argument("--maxgeom", type=int, default=8000, help="Maximum rendered geoms, including scene and overlays.")
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--compile-only", action="store_true", help="Load the MuJoCo scene and optional track, then exit.")
    parser.set_defaults(show_posts=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    selector = MuJoCoPointSelector(args)
    print(f"Loaded base scene: {args.base_scene}")
    print(f"Preloaded waypoints: {len(selector.points)}")
    if not args.compile_only:
        selector.run()


if __name__ == "__main__":
    main()
