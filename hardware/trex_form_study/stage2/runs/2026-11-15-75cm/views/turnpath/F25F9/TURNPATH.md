# The room a turn needs: F25F9 (the pull request's walk) on 75-M-H-10

A view of the frozen reference and of the committed records. It gives no verdict and adopts nothing.

Two words are kept apart. **Planned** is read from the frozen reference, played kinematically at the bar's commanded clock, with no physics. **Simulated** is read from an episode's trace. No number mixes the two.

Lengths are in metres and in body lengths (BL). One body length is 1.759 m: the model's overall length at its home keyframe, from tail: cover_tail_seg_20 (the tail tip) to head: cover_skull (the snout tip).

Records are read from `stage2/runs/2026-11-15-75cm` and cited by their path under it; it held 0 turn records of this set when this file was written (2026-11-15T23:44:55Z). Set freeze 0, manifest `1a5f606feed7`.

## turn/q90_left: quick 90° turn to the left at 0.40 m/s

![top view](turn_q90_left.png)

Top view of quick 90° turn to the left at 0.40 m/s on F25F9 (the pull request's walk); up is the incoming walking line. Blue, thick: the PLANNED path of the root (the frozen reference, kinematic), from 1 s before the command to the end of the move's file; the grey shapes are the PLANNED outline of the robot at the command and at the end of the move's file. The dashed box is the turn itself: the PLANNED root is 0.63 m along its incoming line and 0.55 m sideways (the ring) when the trunk has turned 90°, 2.16 s after the command. The dotted box is the PLANNED root's corner box to the end of the move's file, 4.87 s after the command: 0.63 m along and 1.69 m sideways (1.14 m of it after the turn); its sideways side is mostly the walk after the turn and depends on the file's length. SIMULATED: no record yet; no simulated path is drawn. Every path is drawn from its own position at the command. The scale bar is 0.5 m.

### Planned

From `transitions/turn90_pi4_left.npz` (sha256 `040740666c16`), played at 1.05 times its own clock.

| reading (planned) | value |
|---|---|
| path speed commanded | 0.40 m/s on the reference's clock; 0.42 m/s on the bar's clock (the reference is played 1.05 times faster) |
| turn rate commanded | 45.0 deg/s on the reference's clock; 47.2 deg/s on the bar's clock |
| path radius commanded (speed over rate) | 0.51 m (0.29 BL) |
| radius fitted to the root's path, steady part | 0.50 m (0.28 BL), RMS 19 mm off the circle (the root sways) |
| **the turn itself**: until the trunk has turned 90° (to within 0.5°) | 2.16 s after the command |
| the turn itself, root: where it is then | 0.63 m (0.36 BL) along the incoming line, 0.55 m (0.31 BL) sideways |
| the turn itself, whole robot: along | -0.96 m (-0.55 BL) to 1.03 m (0.59 BL) |
| the turn itself, whole robot: sideways | -0.59 m (-0.34 BL) to 1.34 m (0.76 BL) |
| corner box to the end of the move's file, 4.87 s after the command; includes 1.14 m of walk after the turn; root: advance along the incoming line | largest 0.63 m (0.36 BL), at the move's end 0.60 m (0.34 BL) |
| corner box to the end of the move's file; root: sideways | largest 1.69 m (0.96 BL), at the move's end 1.69 m (0.96 BL) |
| corner box to the end of the move's file; whole robot: along | -0.96 m (-0.55 BL) to 1.03 m (0.59 BL) |
| corner box to the end of the move's file; whole robot: sideways | -0.59 m (-0.34 BL) to 2.48 m (1.41 BL) |
| swept radius over the steady part of the turn (the turning circle uses this one) | 1.12 m (0.63 BL), set by tail: cover_tail_seg_20 |
| largest distance from the turn's centre, from the command until the trunk has turned 90° | 1.12 m (0.64 BL), set by tail: cover_tail_seg_20, 0.30 s after the command |
| turning circle (twice the swept radius of the steady part) | 2.23 m (1.27 BL) |
| corridor for the 90° corner | 0.81 m (0.46 BL), inner corner set by left leg: cover_left_thigh_upper |
| turned 2 s after the command | 85.2° |
| turned at the end of the move (4.87 s after the command) | 90.0° |

Farthest point from the turn's centre, by part (planned): tail 1.12 m; head 0.96 m; trunk 0.80 m; right foot 0.69 m; right leg 0.67 m; left leg 0.54 m; left foot 0.51 m.

### Simulated

No record yet: the chain had made no record of this turn when this file was written.

## turn/q90_right: quick 90° turn to the right at 0.40 m/s

![top view](turn_q90_right.png)

Top view of quick 90° turn to the right at 0.40 m/s on F25F9 (the pull request's walk); up is the incoming walking line. Blue, thick: the PLANNED path of the root (the frozen reference, kinematic), from 1 s before the command to the end of the move's file; the grey shapes are the PLANNED outline of the robot at the command and at the end of the move's file. The dashed box is the turn itself: the PLANNED root is 0.63 m along its incoming line and 0.55 m sideways (the ring) when the trunk has turned 90°, 2.16 s after the command. The dotted box is the PLANNED root's corner box to the end of the move's file, 4.40 s after the command: 0.63 m along and 1.49 m sideways (0.94 m of it after the turn); its sideways side is mostly the walk after the turn and depends on the file's length. SIMULATED: no record yet; no simulated path is drawn. Every path is drawn from its own position at the command. The scale bar is 0.5 m.

### Planned

From `transitions/turn90_pi4_right.npz` (sha256 `90e6cf660b18`), played at 1.05 times its own clock.

| reading (planned) | value |
|---|---|
| path speed commanded | 0.40 m/s on the reference's clock; 0.42 m/s on the bar's clock (the reference is played 1.05 times faster) |
| turn rate commanded | 45.0 deg/s on the reference's clock; 47.2 deg/s on the bar's clock |
| path radius commanded (speed over rate) | 0.51 m (0.29 BL) |
| radius fitted to the root's path, steady part | 0.50 m (0.28 BL), RMS 19 mm off the circle (the root sways) |
| **the turn itself**: until the trunk has turned 90° (to within 0.5°) | 2.16 s after the command |
| the turn itself, root: where it is then | 0.63 m (0.36 BL) along the incoming line, 0.55 m (0.31 BL) sideways |
| the turn itself, whole robot: along | -0.96 m (-0.55 BL) to 1.03 m (0.59 BL) |
| the turn itself, whole robot: sideways | -0.59 m (-0.34 BL) to 1.34 m (0.76 BL) |
| corner box to the end of the move's file, 4.40 s after the command; includes 0.94 m of walk after the turn; root: advance along the incoming line | largest 0.63 m (0.36 BL), at the move's end 0.58 m (0.33 BL) |
| corner box to the end of the move's file; root: sideways | largest 1.49 m (0.85 BL), at the move's end 1.49 m (0.85 BL) |
| corner box to the end of the move's file; whole robot: along | -0.96 m (-0.55 BL) to 1.03 m (0.59 BL) |
| corner box to the end of the move's file; whole robot: sideways | -0.59 m (-0.34 BL) to 2.28 m (1.30 BL) |
| swept radius over the steady part of the turn (the turning circle uses this one) | 1.12 m (0.63 BL), set by tail: cover_tail_seg_20 |
| largest distance from the turn's centre, from the command until the trunk has turned 90° | 1.12 m (0.64 BL), set by tail: cover_tail_seg_20, 0.30 s after the command |
| turning circle (twice the swept radius of the steady part) | 2.23 m (1.27 BL) |
| corridor for the 90° corner | 0.81 m (0.46 BL), inner corner set by right leg: cover_right_thigh_upper |
| turned 2 s after the command | 85.2° |
| turned at the end of the move (4.40 s after the command) | 90.0° |

Farthest point from the turn's centre, by part (planned): tail 1.12 m; head 0.96 m; trunk 0.80 m; left foot 0.69 m; left leg 0.67 m; right leg 0.54 m; right foot 0.51 m.

### Simulated

No record yet: the chain had made no record of this turn when this file was written.

## turn/t90_0.35_left: 90° turn to the left (0.35 rad/s) at 0.40 m/s

![top view](turn_t90_0.35_left.png)

Top view of 90° turn to the left (0.35 rad/s) at 0.40 m/s on F25F9 (the pull request's walk); up is the incoming walking line. Blue, thick: the PLANNED path of the root (the frozen reference, kinematic), from 1 s before the command to the end of the move's file; the grey shapes are the PLANNED outline of the robot at the command and at the end of the move's file. The dashed box is the turn itself: the PLANNED root is 1.17 m along its incoming line and 1.16 m sideways (the ring) when the trunk has turned 90°, 4.48 s after the command. The dotted box is the PLANNED root's corner box to the end of the move's file, 6.62 s after the command: 1.26 m along and 2.07 m sideways (0.91 m of it after the turn); its sideways side is mostly the walk after the turn and depends on the file's length. SIMULATED: no record yet; no simulated path is drawn. Every path is drawn from its own position at the command. The scale bar is 0.5 m.

### Planned

From `transitions/turn90_0.35_left.npz` (sha256 `7f5ea0c44ace`), played at 1.05 times its own clock.

| reading (planned) | value |
|---|---|
| path speed commanded | 0.40 m/s on the reference's clock; 0.42 m/s on the bar's clock (the reference is played 1.05 times faster) |
| turn rate commanded | 20.1 deg/s on the reference's clock; 21.1 deg/s on the bar's clock |
| path radius commanded (speed over rate) | 1.14 m (0.65 BL) |
| radius fitted to the root's path, steady part | 1.13 m (0.64 BL), RMS 33 mm off the circle (the root sways) |
| **the turn itself**: until the trunk has turned 90° (to within 0.5°) | 4.48 s after the command |
| the turn itself, root: where it is then | 1.17 m (0.67 BL) along the incoming line, 1.16 m (0.66 BL) sideways |
| the turn itself, whole robot: along | -0.96 m (-0.55 BL) to 1.50 m (0.86 BL) |
| the turn itself, whole robot: sideways | -0.41 m (-0.23 BL) to 1.95 m (1.11 BL) |
| corner box to the end of the move's file, 6.62 s after the command; includes 0.91 m of walk after the turn; root: advance along the incoming line | largest 1.26 m (0.72 BL), at the move's end 1.23 m (0.70 BL) |
| corner box to the end of the move's file; root: sideways | largest 2.07 m (1.18 BL), at the move's end 2.07 m (1.18 BL) |
| corner box to the end of the move's file; whole robot: along | -0.96 m (-0.55 BL) to 1.50 m (0.86 BL) |
| corner box to the end of the move's file; whole robot: sideways | -0.41 m (-0.23 BL) to 2.86 m (1.63 BL) |
| swept radius over the steady part of the turn (the turning circle uses this one) | 1.57 m (0.89 BL), set by tail: cover_tail_seg_20 |
| largest distance from the turn's centre, from the command until the trunk has turned 90° | 1.57 m (0.89 BL), set by tail: cover_tail_seg_20, 4.16 s after the command |
| turning circle (twice the swept radius of the steady part) | 3.14 m (1.78 BL) |
| corridor for the 90° corner | 0.80 m (0.46 BL), inner corner set by left foot: hw_left_claw_1 |
| turned 4.48799 s after the command | 89.7° |
| turned at the end of the move (6.62 s after the command) | 90.0° |

Farthest point from the turn's centre, by part (planned): tail 1.57 m; head 1.44 m; trunk 1.37 m; right leg 1.32 m; right foot 1.31 m; left leg 1.19 m; left foot 1.13 m.

### Simulated

No record yet: the chain had made no record of this turn when this file was written.

## turn/t90_0.35_right: 90° turn to the right (0.35 rad/s) at 0.40 m/s

![top view](turn_t90_0.35_right.png)

Top view of 90° turn to the right (0.35 rad/s) at 0.40 m/s on F25F9 (the pull request's walk); up is the incoming walking line. Blue, thick: the PLANNED path of the root (the frozen reference, kinematic), from 1 s before the command to the end of the move's file; the grey shapes are the PLANNED outline of the robot at the command and at the end of the move's file. The dashed box is the turn itself: the PLANNED root is 1.17 m along its incoming line and 1.15 m sideways (the ring) when the trunk has turned 90°, 4.46 s after the command. The dotted box is the PLANNED root's corner box to the end of the move's file, 7.10 s after the command: 1.26 m along and 2.27 m sideways (1.11 m of it after the turn); its sideways side is mostly the walk after the turn and depends on the file's length. SIMULATED: no record yet; no simulated path is drawn. Every path is drawn from its own position at the command. The scale bar is 0.5 m.

### Planned

From `transitions/turn90_0.35_right.npz` (sha256 `e27fd3d822e2`), played at 1.05 times its own clock.

| reading (planned) | value |
|---|---|
| path speed commanded | 0.40 m/s on the reference's clock; 0.42 m/s on the bar's clock (the reference is played 1.05 times faster) |
| turn rate commanded | 20.1 deg/s on the reference's clock; 21.1 deg/s on the bar's clock |
| path radius commanded (speed over rate) | 1.14 m (0.65 BL) |
| radius fitted to the root's path, steady part | 1.13 m (0.64 BL), RMS 34 mm off the circle (the root sways) |
| **the turn itself**: until the trunk has turned 90° (to within 0.5°) | 4.46 s after the command |
| the turn itself, root: where it is then | 1.17 m (0.67 BL) along the incoming line, 1.15 m (0.66 BL) sideways |
| the turn itself, whole robot: along | -0.96 m (-0.55 BL) to 1.50 m (0.86 BL) |
| the turn itself, whole robot: sideways | -0.41 m (-0.23 BL) to 1.95 m (1.11 BL) |
| corner box to the end of the move's file, 7.10 s after the command; includes 1.11 m of walk after the turn; root: advance along the incoming line | largest 1.26 m (0.72 BL), at the move's end 1.20 m (0.68 BL) |
| corner box to the end of the move's file; root: sideways | largest 2.27 m (1.29 BL), at the move's end 2.27 m (1.29 BL) |
| corner box to the end of the move's file; whole robot: along | -0.96 m (-0.55 BL) to 1.50 m (0.86 BL) |
| corner box to the end of the move's file; whole robot: sideways | -0.41 m (-0.23 BL) to 3.06 m (1.74 BL) |
| swept radius over the steady part of the turn (the turning circle uses this one) | 1.57 m (0.89 BL), set by tail: cover_tail_seg_20 |
| largest distance from the turn's centre, from the command until the trunk has turned 90° | 1.57 m (0.89 BL), set by tail: cover_tail_seg_20, 4.16 s after the command |
| turning circle (twice the swept radius of the steady part) | 3.13 m (1.78 BL) |
| corridor for the 90° corner | 0.80 m (0.46 BL), inner corner set by right foot: hw_right_toe_2 |
| turned 4.48799 s after the command | 89.7° |
| turned at the end of the move (7.10 s after the command) | 90.0° |

Farthest point from the turn's centre, by part (planned): tail 1.57 m; head 1.44 m; trunk 1.37 m; left leg 1.32 m; left foot 1.31 m; right leg 1.19 m; right foot 1.13 m.

### Simulated

No record yet: the chain had made no record of this turn when this file was written.

## turn/t90_0.5_left: 90° turn to the left (0.5 rad/s) at 0.40 m/s

![top view](turn_t90_0.5_left.png)

Top view of 90° turn to the left (0.5 rad/s) at 0.40 m/s on F25F9 (the pull request's walk); up is the incoming walking line. Blue, thick: the PLANNED path of the root (the frozen reference, kinematic), from 1 s before the command to the end of the move's file; the grey shapes are the PLANNED outline of the robot at the command and at the end of the move's file. The dashed box is the turn itself: the PLANNED root is 0.90 m along its incoming line and 0.81 m sideways (the ring) when the trunk has turned 90°, 3.20 s after the command. The dotted box is the PLANNED root's corner box to the end of the move's file, 5.68 s after the command: 0.92 m along and 1.87 m sideways (1.06 m of it after the turn); its sideways side is mostly the walk after the turn and depends on the file's length. SIMULATED: no record yet; no simulated path is drawn. Every path is drawn from its own position at the command. The scale bar is 0.5 m.

### Planned

From `transitions/turn90_0.5_left.npz` (sha256 `24ade72302e2`), played at 1.05 times its own clock.

| reading (planned) | value |
|---|---|
| path speed commanded | 0.40 m/s on the reference's clock; 0.42 m/s on the bar's clock (the reference is played 1.05 times faster) |
| turn rate commanded | 28.6 deg/s on the reference's clock; 30.1 deg/s on the bar's clock |
| path radius commanded (speed over rate) | 0.80 m (0.45 BL) |
| radius fitted to the root's path, steady part | 0.78 m (0.44 BL), RMS 34 mm off the circle (the root sways) |
| **the turn itself**: until the trunk has turned 90° (to within 0.5°) | 3.20 s after the command |
| the turn itself, root: where it is then | 0.90 m (0.51 BL) along the incoming line, 0.81 m (0.46 BL) sideways |
| the turn itself, whole robot: along | -0.96 m (-0.55 BL) to 1.22 m (0.69 BL) |
| the turn itself, whole robot: sideways | -0.48 m (-0.27 BL) to 1.60 m (0.91 BL) |
| corner box to the end of the move's file, 5.68 s after the command; includes 1.06 m of walk after the turn; root: advance along the incoming line | largest 0.92 m (0.52 BL), at the move's end 0.88 m (0.50 BL) |
| corner box to the end of the move's file; root: sideways | largest 1.87 m (1.06 BL), at the move's end 1.87 m (1.06 BL) |
| corner box to the end of the move's file; whole robot: along | -0.96 m (-0.55 BL) to 1.22 m (0.69 BL) |
| corner box to the end of the move's file; whole robot: sideways | -0.48 m (-0.27 BL) to 2.66 m (1.51 BL) |
| swept radius over the steady part of the turn (the turning circle uses this one) | 1.31 m (0.75 BL), set by tail: cover_tail_seg_20 |
| largest distance from the turn's centre, from the command until the trunk has turned 90° | 1.32 m (0.75 BL), set by tail: cover_tail_seg_20, 0.38 s after the command |
| turning circle (twice the swept radius of the steady part) | 2.63 m (1.49 BL) |
| corridor for the 90° corner | 0.80 m (0.45 BL), inner corner set by left leg: cover_left_thigh_upper |
| turned 3.14159 s after the command | 88.7° |
| turned at the end of the move (5.68 s after the command) | 90.0° |

Farthest point from the turn's centre, by part (planned): tail 1.31 m; head 1.16 m; trunk 1.06 m; right leg 0.98 m; right foot 0.96 m; left leg 0.84 m; left foot 0.79 m.

### Simulated

No record yet: the chain had made no record of this turn when this file was written.

## turn/t90_0.5_right: 90° turn to the right (0.5 rad/s) at 0.40 m/s

![top view](turn_t90_0.5_right.png)

Top view of 90° turn to the right (0.5 rad/s) at 0.40 m/s on F25F9 (the pull request's walk); up is the incoming walking line. Blue, thick: the PLANNED path of the root (the frozen reference, kinematic), from 1 s before the command to the end of the move's file; the grey shapes are the PLANNED outline of the robot at the command and at the end of the move's file. The dashed box is the turn itself: the PLANNED root is 0.90 m along its incoming line and 0.81 m sideways (the ring) when the trunk has turned 90°, 3.20 s after the command. The dotted box is the PLANNED root's corner box to the end of the move's file, 6.15 s after the command: 0.92 m along and 2.06 m sideways (1.26 m of it after the turn); its sideways side is mostly the walk after the turn and depends on the file's length. SIMULATED: no record yet; no simulated path is drawn. Every path is drawn from its own position at the command. The scale bar is 0.5 m.

### Planned

From `transitions/turn90_0.5_right.npz` (sha256 `564ffa6b2f7f`), played at 1.05 times its own clock.

| reading (planned) | value |
|---|---|
| path speed commanded | 0.40 m/s on the reference's clock; 0.42 m/s on the bar's clock (the reference is played 1.05 times faster) |
| turn rate commanded | 28.6 deg/s on the reference's clock; 30.1 deg/s on the bar's clock |
| path radius commanded (speed over rate) | 0.80 m (0.45 BL) |
| radius fitted to the root's path, steady part | 0.78 m (0.44 BL), RMS 34 mm off the circle (the root sways) |
| **the turn itself**: until the trunk has turned 90° (to within 0.5°) | 3.20 s after the command |
| the turn itself, root: where it is then | 0.90 m (0.51 BL) along the incoming line, 0.81 m (0.46 BL) sideways |
| the turn itself, whole robot: along | -0.96 m (-0.55 BL) to 1.22 m (0.69 BL) |
| the turn itself, whole robot: sideways | -0.48 m (-0.27 BL) to 1.61 m (0.91 BL) |
| corner box to the end of the move's file, 6.15 s after the command; includes 1.26 m of walk after the turn; root: advance along the incoming line | largest 0.92 m (0.52 BL), at the move's end 0.86 m (0.49 BL) |
| corner box to the end of the move's file; root: sideways | largest 2.06 m (1.17 BL), at the move's end 2.06 m (1.17 BL) |
| corner box to the end of the move's file; whole robot: along | -0.96 m (-0.55 BL) to 1.22 m (0.69 BL) |
| corner box to the end of the move's file; whole robot: sideways | -0.48 m (-0.27 BL) to 2.86 m (1.63 BL) |
| swept radius over the steady part of the turn (the turning circle uses this one) | 1.31 m (0.75 BL), set by tail: cover_tail_seg_20 |
| largest distance from the turn's centre, from the command until the trunk has turned 90° | 1.32 m (0.75 BL), set by tail: cover_tail_seg_20, 0.36 s after the command |
| turning circle (twice the swept radius of the steady part) | 2.63 m (1.49 BL) |
| corridor for the 90° corner | 0.80 m (0.45 BL), inner corner set by right leg: cover_right_thigh_upper |
| turned 3.14159 s after the command | 88.7° |
| turned at the end of the move (6.15 s after the command) | 90.0° |

Farthest point from the turn's centre, by part (planned): tail 1.31 m; head 1.16 m; trunk 1.06 m; left leg 0.98 m; left foot 0.96 m; right leg 0.84 m; right foot 0.79 m.

### Simulated

No record yet: the chain had made no record of this turn when this file was written.

## Turns without a reference in this set

- turn/t90_1_left (`transitions/turn90_1_left`): no reference within the thigh twist and the leg drives: no stride factor of the grid passes the kinematic rule (thigh twist at most 0.85 of its range, every other gated joint at least 0.05 rad of range, ik_ok, in both directions)
- turn/t90_1_right (`transitions/turn90_1_right`): no reference within the thigh twist and the leg drives: no stride factor of the grid passes the kinematic rule (thigh twist at most 0.85 of its range, every other gated joint at least 0.05 rad of range, ik_ok, in both directions)

## How each reading is made

- **words**: planned: read from the frozen reference played kinematically at the bar's commanded clock (no physics). simulated: read from an episode's trace. No number mixes the two..
- **clock**: the bar plays a reference 1.05 times faster than the reference's own clock (bar.RULES['clocks']); every time here is on the bar's clock (the seconds of an episode), so a planned speed or rate on the bar's clock is the reference's own times 1.05; lengths and radii do not change with the clock.
- **frame**: the corner frame of a turn: origin at the root (the trunk's origin) at the command instant; 'along' is the commanded heading (the home heading, the bar's psi_cmd), the incoming walking line; 'sideways' is square to it, positive toward the side of the turn.
- **command**: the command instant t_c: planned, the reference clock's time of the move's first frame over the commanded clock rate; simulated, the episode record's command_s (never recomputed).
- **move end**: the end of the move: the time on the bar's clock at which the reference clock reaches the end of the turn's own file (a trace's t_ref column); a trace that ends before it is read to its end and says so. A turn's file goes on after the turn: it holds the walk out of the turn too, and the files of different turns and sets differ in length.
- **ends**: every largest and smallest value over a span of time is taken over the 50 Hz rows in the span and the span's two ends themselves (read between the rows on either side, linearly): a value 'at the end' is then never beyond the largest.
- **turn itself**: the turn itself: from the command to the first row at which the trunk has turned the commanded angle to within 0.5 degrees.  Given: the seconds it took, where the root is then (along its incoming line and sideways), the box of the whole robot (every surface point) over those rows, and how far the root is then from its place at the end of the move (the walk after the turn that the corner box to the move's end includes).  A path whose trunk never turns that far has none of these.
- **heading**: the trunk's heading: the trace's yaw column (control._yaw of the trunk's rotation), unwrapped, as bar's turn reading takes it; 'turned' is its change from the commanded heading, positive toward the side of the turn.
- **steady**: the steady part of the turn: from the first row at which the trunk has turned 0.1667 of the commanded angle to the first row at which it has turned 0.8333 of it (15 to 75 degrees of a 90 degree turn); a path that never gets there 'stops short' and has no fitted radius.
- **radius commanded**: the path radius commanded: the walk's path speed over the turn's rate (v / omega), both from the frozen turn's entry.
- **radius fitted**: the radius fitted: the circle of the root's path over the steady part, with the trunk's heading as the place on the arc: least squares of root = centre + R x n(heading) over the steady rows, n the unit vector square to the heading that points away from the side of the turn; its centre is the turn's centre, and the RMS distance of the rows from that model is given beside it.  A circle fitted to the positions alone (the algebraic fit, recorded as radius_free_fit_m) reads the root's sideways sway of about 3 cm a step as curvature: it gave 0.29 m on V0's planned quick turn, whose commanded radius is 0.51 m, where this fit gives 0.50 m.
- **corner box**: the corner box to the end of the move: from the command to the end of the move, how far the root gets along its incoming line (advance) and sideways, at the largest and at the move's end; the same box for the whole robot (every surface point).  Its sideways side is mostly the walk after the turn and grows with the length of the move's file: read the room of the turn from 'the turn itself', not from this box.
- **surface**: the robot's surface points: per body of the model, the corners of the convex hull of its pads, covers and hardware (geom groups 0, 1 and 2: mesh vertices, and the corners of a primitive's bounding box), carried on each row's pose; the snout tip is the head's point farthest forward at home, the tail tip the tail's farthest back.
- **swept radius**: the swept radius: the largest distance in plan from the turn's centre to any surface point over the steady part of the turn (where the robot is on its circle); the point that sets it is named.  A circle of that radius does not hold the robot through this 90 degree turn: at the command the robot still stands along its incoming line and its tail reaches further from the centre.  That is the second reading, 'the largest distance from the turn's centre from the command until the trunk has turned the commanded angle' (the turn itself), with the point and the instant that set it.
- **turning circle**: the turning circle: twice the swept radius of the steady part, the diameter of the circle a full turn at that rate would sweep about the same centre (a robot that keeps turning; not the larger distance at the command).
- **corridor**: the corridor: the narrowest L-shaped corridor, two legs of one width meeting square, that holds every surface point from the command to the end of the move (both ends included).  Its outer wall along the incoming leg touches the sweep's point farthest to the outside (sideways = Y0), its far wall across the incoming line touches the sweep's point farthest along (along = X); the inner corner is at (X - w, Y0 + w) and the width w is the smallest for which no point lies inside that corner's block: w = the largest, over all points, of min(X - along, sideways - Y0).  It holds the robot up to the end of the move only: a robot that leaves the move off its heading needs more room further on.
- **body length**: lengths are given in metres and in body lengths; a body length is the model's overall length at its home keyframe along the home heading, from the surface's point farthest back to its point farthest forward.
- **simulated**: simulated: every record of the turn under the run directory (each row, delay, floor grip and block) is listed; an episode that stayed up and whose trace is kept and has the record's sha256 is read; an episode that fell is counted and listed, not read; the heading at the bar's judged instant is the record's own (command_s + at_s; the commanded angle + the sign times heading_error_at_deg).
- **pose**: the pose of an episode's trace row is a measured bound, not exact: the attitude and the root's place are exact, the joint angles are a cubic spline through the 50 Hz angles read one physics step earlier.  On a Stage 1 smoke episode run with the plant's own body positions kept (75-M-H-10, turn/walking_0.35_left, R2/42@0ms, seed 3042, 1461 rows) every body origin came back within 0.08 mm on every row but the first and within 0.11 mm on the first (the start's transient lies before the first joint sample).  On the chain's traces there is no such truth, only the trace's own foot and CoM columns: over the 1120 episodes that stayed up in the turn records of V0 and F25F9cl (67 records, 15 Nov) the rebuilt feet lay up to 0.73 mm from them in plan (the median of the records' largest: 0.19 mm) and the CoM up to 0.04 mm, and the two feet's answers for the root's height differed by up to 1.02 mm (1.35 mm in an episode that fell): the largest on the rows next to a touchdown, a slip or a fall, where the joint speeds jump between two rows.  The brief's 0.1 mm is therefore met on a kinematic play and, but for the first row, on the smoke episode, not on every row of the chain's traces (each clip's manifest entry and each turn record's block give their own largest distances: pose_residual_m, pose_check_worst; barviews posecheck prints them for any record).
