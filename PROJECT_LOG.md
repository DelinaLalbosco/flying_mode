# Project Log — S10 Autonomous Waypoint Patrol

## Phase 0: Baseline Verification — ✅ Complete
- Fresh clone built successfully after fixing numpy header issue (stale `build/` cache; clean rebuild fixed it).
- Verified robot stands and walks via keyboard teleop (`w/a/s/d`, `z`, `c`) in MuJoCo sim.
- Track has 33 waypoints, reach radius 0.2m, distance_mode=xy.

## Phase 1: Control Interface Understanding — ✅ Complete
- `UserCommand` struct fields: `forward_vel_scale`, `side_vel_scale`, `turnning_vel_scale`, `target_mode`, `time_stamp`.
- Interface chain: `main.cpp` → `QwStateMachine::Start()` → selects `KeyboardInterface` / `GamepadInterface` via `RemoteCommandType` enum → feeds `UserCommand` to RL policy.
- `UserCommandInterface` base class requires: `Start()`, `Stop()`, `GetUserCommand()`, `SetMotionStateFeedback()`.
- `RobotMotionState` enum: WaitingForStand=0, StandingUp=1, JointDamping=2, LieDown=4, RLControlMode=6.
- No existing ground-truth pose topic; only `/IMU_DATA` and `/JOINTS_DATA` published by sim.

## Phase 3a: Autonomous Waypoint-Seeking (no obstacle avoidance) — ✅ Complete
### Design
- Kept perception/decision-making in Python (sim side, has ground-truth pose + waypoint coords) and command delivery in C++ (matches existing SDK architecture pattern).
- New ROS 2 topic `/AUTO_NAV_CMD` (`geometry_msgs/msg/Twist`) carries velocity commands:
  - `linear.x` → forward_vel_scale
  - `linear.y` → side_vel_scale
  - `angular.z` → turnning_vel_scale

### Python changes (`mujoco_simulation_ros2.py`)
- Added `from geometry_msgs.msg import Twist` import.
- Added publisher: `self.nav_cmd_pub = self.create_publisher(Twist, '/AUTO_NAV_CMD', 10)`.
- Added `_auto_nav_step()` method: computes heading error to next uncollected waypoint (from `track_waypoint_positions` + ground-truth `data.xpos[track_body_id]`), publishes forward/turn velocity. Simple P-control on yaw error; reduces forward speed when yaw error is large.
- Called `_auto_nav_step()` in main sim loop right after `_update_track_progress()`.

### C++ changes
- New file: `interface/user_command/auto_nav_interface.hpp` — `AutoNavInterface` class implementing `UserCommandInterface`.
  - Creates own `rclcpp::Node("auto_nav_interface")`, subscribes to `/AUTO_NAV_CMD`.
  - Runs own background thread (`spin_loop`) — same pattern as `KeyboardInterface`'s `kb_thread_`.
  - Auto-transitions robot state: WaitingForStand → StandingUp → RLControlMode automatically (replaces manual Z/C keypresses).
- Added `kAutoNav` to `RemoteCommandType` enum (`custom_types.h`).
- Wired `kAutoNav` branch into `QwStateMachine::Start()` (`qw_state_machine.hpp`) to instantiate `AutoNavInterface`.
- Changed `main.cpp` to launch with `RemoteCommandType::kAutoNav`.

### Verification result
- Robot automatically: stands up → enters RL control → walks toward waypoint 0 → timer starts (confirmed in sim log: "[TRACK] Timer started at waypoint 0").
- **Known issue**: robot walks straight into a wall — no obstacle awareness yet (expected; no camera/lidar wired in yet).

## Phase 4: Perception + Obstacle Avoidance — 🔜 Next
- Add depth camera (or lidar) sensor to MJCF.
- Publish depth/lidar data from sim.
- Extend `_auto_nav_step()` (or add a perception node) to detect obstacles/walls/stairs and steer around, or climb stairs where appropriate.
- Must not throttle robot state publishing frequency when adding camera (known gotcha from prior camera integration attempt — state pub must stay at step % 5 / 200Hz).

## Key files touched so far
- `src/S10_sdk_deploy/interface/robot/simulation/mujoco_simulation_ros2.py`
- `src/S10_sdk_deploy/interface/user_command/auto_nav_interface.hpp` (new)
- `src/S10_sdk_deploy/include/types/custom_types.h`
- `src/S10_sdk_deploy/state_machine/quadruped_wheel/qw_state_machine.hpp`
- `src/S10_sdk_deploy/main.cpp`

## Environment notes
- WSL Ubuntu + ROS 2 Jazzy.
- `numpy < 2.0` required; if colcon build fails on `ndarrayobject.h`, do a clean rebuild: `rm -rf build install log` then rebuild.
- Editing convention: prefer `sed`/heredoc/python one-shot scripts over interactive `nano` for precision (nano introduced typos in this session).

## Phase 4: Perception + Obstacle Avoidance — 🔄 In Progress (paused, resume here)

### What's done
- Confirmed MJCF already has a usable camera: `front_camera`, defined on `base_link` in `S10.xml`
  (pos="0.50 0 0.35", euler="0 -20 0" — pitched down 20°). No XML changes needed to use it.
- Added MuJoCo offscreen `Renderer` for depth in `mujoco_simulation_ros2.py` (`self.depth_renderer`).
- Added `_update_depth_perception(step)` method: periodically renders depth from `front_camera`,
  splits image into row-bands (top=wall-height band, bottom=ground/step band), classifies
  `self.wall_ahead` (bool) and `self.wall_steer_bias` (steer direction away from wall).
- Wired into main loop: `_update_depth_perception(step)` called BEFORE `_auto_nav_step()` each sim step,
  right after `mujoco.mj_step()`. Robot state publishing (step % 5, 200Hz) untouched — confirmed no
  regression of the earlier camera/state-pub-frequency bug.
- Updated `_auto_nav_step()`: if `wall_ahead` True, override forward/turn commands to steer around
  using `wall_steer_bias`; otherwise normal beeline-to-waypoint steering. Low/ground-only obstacles
  (like stairs) intentionally do NOT trigger avoidance — left for RL locomotion policy to climb.

### Performance issue found & partially fixed
- Initial depth render every 100 steps (~10Hz) caused visible lag/stutter.
- Root cause measured directly: `renderer.render()` costs ~50ms per call regardless of resolution
  (64x64 vs 24x24 both ~50ms — fixed overhead, not pixel-count-bound).
- Fix applied: reduced `DEPTH_RENDER_STEP_INTERVAL` from 100 to 500 (~2Hz). Lag much improved on
  retest. NOT yet tried: rendering depth in a background thread to fully decouple from physics loop
  (better fix, not yet implemented).

### Bug found via debug logging — NOT YET FIXED
- Added temporary debug log line in `_update_depth_perception` (search "DEPTH-DEBUG" to find/remove
  when done): logs `wall_band_min`, `ground_band_min`, `wall_ahead` each render.
- Observed: `wall_band_min` stayed ~1.47-1.5m constantly throughout approach to the wall, then
  INCREASED to ~2.7m near the end (should decrease as robot nears a wall). `wall_ahead` was False
  the entire time — detection never fired. Robot got stuck pushing against the wall (looked like
  "trying to climb" — that's actually just the RL policy fighting a solid collision with forward
  cmd=0.5 constantly issued since wall_ahead never triggered).
- Hypothesis: WALL_DETECT_MAX_RANGE=1.2m threshold may be wrong relative to real camera readings,
  OR the fixed -20° downward camera pitch is mostly seeing the ground plane at roughly constant
  range rather than the vertical wall face at the row-bands we chose. Needs proper row-by-row depth
  inspection (see "Next step" below) before retuning thresholds blindly.

### New idea surfaced (not yet implemented) — may simplify everything
- `track_overlay.xml` doesn't just place waypoint markers — it draws a COMPLETE pre-built path as
  connected line segments (`track_segment_000`, `_001`, ... `_031`) with exact 3D coordinates
  threading through every waypoint in sequence. This may already route around/through obstacles
  (walls, ramps, stairs) by design, since it's a hand-authored competition track.
- If true, switching `_auto_nav_step()` to follow the segment path (instead of beelining straight
  at the next waypoint) could eliminate most/all wall collisions without needing wall-detection
  logic at all. Depth camera would then mainly serve as backup for genuinely unknown/dynamic
  obstacles (matches competition rule: known waypoints/coords are free to use; only UNKNOWN
  obstacles require perception).
- Was mid-investigation: need to find where actual wall/obstacle collision geometry lives (checked
  S10_track.xml, no "wall" match there — need to check scene.xml, since S10_track.xml likely
  <include>s it) and confirm whether the green line path threads around it.

### Next step (resume here tomorrow)
1. Run: `grep -n "wall\|include" .../scene.xml .../S10_track.xml` to find where wall obstacle
   geometry actually lives, and confirm the include chain (S10_track.xml -> scene.xml -> ?).
2. Compare wall geometry position to the `track_segment_*` line coordinates in track_overlay.xml —
   check if the path already detours around the wall we hit (likely near waypoint 0/1 given how
   quickly we hit it).
3. If path clears the wall: rewrite `_auto_nav_step()` to follow the nearest point on the segment
   polyline (lookahead-point / pure-pursuit style) instead of beelining straight at
   `track_waypoint_positions[track_next_index]`. This becomes the PRIMARY navigation strategy.
4. Keep depth-camera wall detection as a secondary safety net (for genuinely unknown obstacles),
   but first fix its threshold bug via proper row-by-row depth inspection if still needed after
   step 3.
5. Remove temporary `[DEPTH-DEBUG]` log line once thresholds (if still needed) are confirmed correct.

### Known environment reminders
- Always `source /opt/ros/jazzy/setup.bash` + `source install/setup.bash` + `export ROS_DOMAIN_ID=1`
  in EACH new terminal before running rl_deploy or the simulator.
- The Python simulator MUST be run inside the `mujoco_env` conda/venv environment (prompt shows
  `(mujoco_env)`), or `import mujoco` fails.
- No C++ rebuild needed for Phase 4 so far — only Python (`mujoco_simulation_ros2.py`) was touched.
  C++ (`AutoNavInterface`) from Phase 3a is unchanged and working.

### Include chain confirmed
- `S10_track.xml` includes (in order): `S10.xml` (robot), `scene.xml` (environment/terrain —
  likely where wall/obstacle geometry lives), `track_overlay.xml` (waypoints + path line segments).
- Searched scene.xml + S10_track.xml for literal "wall" — no match. Obstacle geometry is named
  something else. NEXT STEP TOMORROW: `grep -n "geom\|body" scene.xml` (or just view the file) to
  find actual obstacle/wall geom names and positions, then compare against track_segment_* coords.

## Session 2 (Day 2): Path-Following Nav + Waypoint Collection Fix — ✅ Complete

### Discovery: entire environment is one unnamed mesh body
- `scene.xml` defines the whole environment (walls, ramps, stairs, everything) as ~2185
  generic mesh geoms (`mesh_1`...`mesh_2185`) under a single body `main_body` — no per-object
  names. Confirmed there is no way to look up "the wall" or "the stairs" by name; only depth
  camera perception can distinguish them at runtime.
- Confirmed `track_overlay.xml` draws a full pre-built green-line path (`track_segment_000`
  through `_031`, 32 segments / 33 points) connecting every waypoint — an officially-provided
  known-safe route. Verified via handbook.txt this is compliant: known waypoint/track coords are
  explicitly permitted by competition rules; only genuinely unknown obstacles require perception.
  Also checked judging criteria (handbook section on preliminary/final evaluation) — perception
  and scene understanding are explicitly scored, so depth camera must remain functionally active
  in the decision loop, not just present as dead code.

### Implemented: hybrid navigation (current working design)
- `_init_path_polyline()`: parses `track_segment_*` fromto coords from track_overlay.xml into
  `self.path_points` (ordered numpy array), loaded once at startup. Confirmed 33 points / 32
  segments load correctly via standalone test.
- `_get_pursuit_target()`: pure-pursuit helper — projects robot onto nearest path segment, walks
  forward by a lookahead distance to find a steering aim point.
- `_on_climbing_segment()`: checks if the robot's nearest path segment has significant elevation
  change (dz > 0.15m), used to detect stairs/ramp segments.
- `_auto_nav_step()` (current logic, in order):
  1. Default: beeline-style steering toward a **pursuit target** on the path (lookahead=1.5m) —
     i.e. follow the green line, not a raw straight line to the waypoint.
  2. **Waypoint-snap override**: when within `WAYPOINT_SNAP_RADIUS = 1.2m` of the next waypoint,
     aim directly at the waypoint instead of the pursuit target, to reliably enter its 0.2m
     collection radius (pure path-following alone was smoothing past waypoints without collecting
     them — this was the main bug fixed this session).
  3. **Climbing stability override**: only when yaw_error is already small (<0.4 rad) AND on a
     climbing segment, clamp turn to ±0.25 and cap forward at 0.35 — prevents fighting steering
     corrections mid-climb, but doesn't block the initial turn needed to face the right direction
     (an earlier version applied this unconditionally and caused a ~75s stall at waypoint 0 by
     over-restricting the initial corrective turn).
  4. **Wall-avoidance override** (from Phase 4 depth camera, unchanged): if `wall_ahead` True,
     reduce forward and add `wall_steer_bias` to turn.

### Verified working (this session's test run)
- Waypoint 0: collected in 8.5s (matches original Phase 3a baseline — confirms the climbing-
  stability stall bug is fixed).
- Waypoints 1-6: all collected via `[TRACK] Reached waypoint N]` log lines, hybrid nav confirmed
  functional end-to-end.
- Wall avoidance still works (confirmed earlier in session, no wall collisions this run).

### Known issue — NOT YET FIXED (stopped here for the day)
- Robot gets physically stuck at a multi-step incline/staircase after waypoint 6 area. Confirmed
  via `ros2 topic echo /AUTO_NAV_CMD`: nav commands are being published correctly and continuously
  (forward=0.35, small turn correction) — so this is NOT a navigation/command-pipeline bug.
  Screenshot showed robot standing upright, stationary, positioned right at the base of what looks
  like a ramp/staircase (diagonal green line, horizontal banded structure in view). This is a pure
  RL-locomotion-policy climbing problem: robot receives valid forward drive commands but doesn't
  make progress up the incline. Likely needs investigation into approach speed/angle/distance
  before the climb, not more nav-layer tuning — this is a different problem class than steering.
- NEXT STEP: investigate this specific incline. Check exact waypoint index / path segment where
  it's stuck (likely one of the big-dz segments identified: 006 dz=0.565, 017 dz=1.725, 022
  dz=1.195, 025 dz=1.030, 027 dz=1.050). Consider: does single-step-at-a-time approach work but
  multi-step doesn't because of approach momentum/angle? May need a dedicated "climbing mode" that
  temporarily changes forward speed/gait cues, or verify if RL policy has a torque/velocity ceiling
  being hit on steep sections.

### Housekeeping
- `[DEPTH-DEBUG]` log line is still present in `_update_depth_perception` — still useful for now
  since wall-avoidance debugging isn't fully done; remove once stairs + wall logic are both fully
  confirmed stable.
- No C++ rebuild needed this session either — all changes remain Python-only
  (`mujoco_simulation_ros2.py`). `AutoNavInterface` (C++) unchanged since Phase 3a.

## Stairs Stall — Root Cause Confirmed (Session 2 continued)

### Diagnosis method
- Added `[NAV-DEBUG]` periodic log (every 200 nav steps) printing wp_idx, robot pos, dist_to_wp,
  yaw_err, climbing flag, forward/turn commands.
- Cross-checked live command output via `ros2 topic echo /AUTO_NAV_CMD --once` while stuck.

### Findings
- Robot got stuck between waypoint 5 and 6, at position ~(-15.45, 20.39, 0.96) — height 0.96 is
  noticeably higher than normal standing height (~0.89-0.90 seen elsewhere in the log), meaning
  the robot had already partially climbed (consistent with "single step OK" observation).
- Position was **completely frozen** for 200+ seconds of sim time — not even minor drift.
  `dist_to_wp` stayed locked at exactly 2.90m throughout.
- Confirmed via live topic echo: `/AUTO_NAV_CMD` was actively publishing `forward=0.5` the whole
  time — nav command pipeline is NOT the problem.
- `wall_ahead` stayed False throughout (wall_band_min ~1.47m, never below the 1.2m threshold) —
  correct behavior, since a stair riser isn't wall-height, so wall-avoidance correctly does not
  trigger.

### Conclusion
This is a **RL locomotion policy limitation**, not a navigation-layer bug. The robot can climb a
single step but stalls partway up a multi-step staircase — receiving continuous valid forward-drive
commands but making zero physical progress. This is a different problem class from anything nav
steering can fix (confirmed: adjusting yaw/forward via `_auto_nav_step()` has no effect since
commands are already correct and the robot simply isn't advancing).

### Next steps to investigate (not yet attempted)
1. Check DeepRobotics SDK docs (`doc/` folder in repo) for any stairs-specific mode, gait
   parameter, or terrain flag exposed via `UserCommand` or elsewhere.
2. Consider "stuck detection + retry" logic: if robot position hasn't changed over N seconds
   while forward command is nonzero, try a brief backward step + re-approach, or momentarily
   increase forward_vel_scale beyond normal cruising speed to give extra momentum.
3. Check if `side_vel_scale` or approach angle relative to the stair edge matters (robot may need
   to be more precisely squared-up to the step edge, not just yaw-aligned to the waypoint).
4. Worth testing manually via keyboard control: manually try to walk the robot up the same
   multi-step staircase with keyboard input, to see if a human-driven approach can clear it (would
   confirm/deny whether this is truly a policy limitation vs. an approach-angle/speed issue we can
   still fix from the nav layer).

### Housekeeping added this sub-session
- Added `[NAV-DEBUG]` temporary log line in `_auto_nav_step()` (prints every 200 calls) — same
  cleanup note as `[DEPTH-DEBUG]`: remove both once stairs handling is resolved and wall-avoidance
  is fully confirmed stable.

## Session 2 (continued): New Stairs Regression Found

### New symptom (different from earlier stall)
- Robot successfully collects waypoints 0, 1, 2. Approaching waypoint 3, gets stuck OSCILLATING
  (not frozen this time) near position (-15.4, 28.7) for a very long stretch -- dist_to_wp stays
  locked around 13.2-13.3, never decreasing, across thousands of NAV-DEBUG log lines.
- Robot height (z) hovers 1.09-1.13 during this stuck period -- well above normal standing height
  (~0.89-0.90 seen elsewhere) -- confirms robot is up on an elevated step/stair structure, not on
  flat ground.
- `climbing=False` reported the ENTIRE time despite z≈1.1 clearly indicating elevated terrain --
  `_on_climbing_segment()` is NOT detecting this location as a climbing segment. Root cause not
  yet found -- possibly the robot has drifted off the nearest-path-segment assumption the check
  relies on, or the dz_threshold (0.15m) doesn't match this particular step's geometry.
- yaw_err oscillates wildly between nearby log lines (e.g. -2.7 to +2.0 rad within a few entries)
  -- consistent with the robot spinning/wobbling in place rather than committing to a direction,
  likely due to pitch instability from being straddled across a step edge (front wheels down,
  back wheels still up -- confirmed via user's visual observation).

### User's visual observation (matches log data)
- When stepping DOWN a single-step stair (not up), the robot sometimes gets stuck with front two
  wheels already on the lower level and back two wheels still on the upper level. This is new --
  did not happen in earlier sessions. Takes a long time to resolve, sometimes recovers, sometimes
  fully stuck.

### Hypothesis for next session
- The wall-avoidance / climbing-stability logic added this session may be inadvertently interfering
  with the RL policy's natural descent gait by keeping `turn` and `forward` commands active/changing
  while the robot is mid-step, when what it may need is near-zero lateral commands and pure forward
  momentum to commit through the step transition.
- `_on_climbing_segment()` false-negative here needs investigation -- add debug print of the actual
  robot_pos vs. nearest path segment distance/dz when climbing=False but z is clearly elevated, to
  see why detection misses this location.
- Possibly needs a NEW check independent of path segments: detect elevated z directly from robot
  pose (compare current z against a rolling baseline) as a more robust "on a step" signal, rather
  than relying solely on proximity to a pre-defined path segment.

### Status: NOT YET FIXED. Stopped here for the day given session length.
### Next session starting point: investigate `_on_climbing_segment()` false negative at the stuck
### location, and consider a z-height-based "on step" detector as a more direct signal than
### path-segment proximity.
