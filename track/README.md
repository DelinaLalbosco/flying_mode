# Track

This folder contains the track data and tools. The MuJoCo tools load the original scene from `../src/S10_sdk_deploy/S10_description/s10_mjcf/mjcf/scene.xml` and draw/select waypoints inside that environment.

## Contents

- `images/track.png`: original 3D simulator screenshot.
- `images/track2.png`: original trajectory projection screenshot.
- `data/track.json`: editable waypoint data with 33 waypoints.
- `data/track.csv`: the same waypoint data in spreadsheet-friendly form.
- `scenes/track_on_scene.xml`: generated wrapper MJCF that includes the repository `scene.xml` and adds the track overlay.
- `scripts/mujoco_track_viewer.py`: opens an interactive MuJoCo viewer using the real scene.
- `scripts/visualize_track.py`: recreates the XY, XZ, and YZ trajectory plots as a static reference.
- `scripts/select_track_points.py`: MuJoCo point selector for clicking dots on the real scene geometry.
- `outputs/track_mujoco_preview.png`: offscreen MuJoCo preview of the generated scene.
- `outputs/track_visualization.png`: generated preview from the track data.

## Setup

From this folder, install dependencies if needed:

```bash
python3 -m pip install -r requirements.txt
```

## Open the MuJoCo Track Viewer

This is the main interactive visualizer. It generates `scenes/track_on_scene.xml`, which includes the original `scene.xml`, then opens the native MuJoCo viewer:

```bash
python3 scripts/mujoco_track_viewer.py --play
```

Viewer controls:

- Mouse drag / wheel: rotate, pan, and zoom the MuJoCo camera.
- Space: toggle smooth marker playback along the track.
- `n` or right arrow: move marker to the next waypoint.
- `p` or left arrow: move marker to the previous waypoint.
- `r`: reset marker to the start.
- `+` / `-`: adjust playback speed.
- `h`: print controls in the terminal.

To verify or regenerate the MJCF without opening a window:

```bash
python3 scripts/mujoco_track_viewer.py --compile-only
```

To visualize a newly selected track:

```bash
python3 scripts/mujoco_track_viewer.py --track data/new_track.json --xml-out scenes/new_track_on_scene.xml
```

## Export a Static Plot

```bash
python3 scripts/visualize_track.py \
  --input data/track.json \
  --save outputs/track_visualization.png \
  --show
```

Use `--no-labels` if the waypoint numbers are too dense.

## Select Dots for a New Track

Run the point selector:

```bash
python3 scripts/select_track_points.py \
  --out-json data/new_track.json \
  --out-csv data/new_track.csv
```

Controls:

- Ctrl+left-click: add a waypoint on the clicked MuJoCo scene surface.
- Left drag: rotate the camera.
- Shift+left drag or middle drag: pan the camera.
- Right drag or mouse wheel: zoom.
- `u`: undo the last waypoint.
- `c`: clear all waypoints.
- `s`: save JSON and CSV.
- `r`: reset the camera.
- `h`: hide/show help text.
- `q` or Esc: save and quit.

By default, clicked points are raised by `0.05 m` so the marker remains visible above the surface. Change that with `--z-offset`, for example:

```bash
python3 scripts/select_track_points.py --z-offset 0.10
```

To edit the track instead of starting empty:

```bash
python3 scripts/select_track_points.py --load data/track.json
```

## Data Notes

Tune `data/track.json` if simulator alignment needs adjustment.
