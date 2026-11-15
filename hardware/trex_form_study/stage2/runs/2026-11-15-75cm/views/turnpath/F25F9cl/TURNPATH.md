# The room a turn needs: F25F9cl (the candidate) on 75-M-H-10

A view of the frozen reference and of the committed records. It gives no verdict and adopts nothing.

Two words are kept apart. **Planned** is read from the frozen reference, played kinematically at the bar's commanded clock, with no physics. **Simulated** is read from an episode's trace. No number mixes the two.

Lengths are in metres and in body lengths (BL). One body length is 1.759 m: the model's overall length at its home keyframe, from tail: cover_tail_seg_20 (the tail tip) to head: cover_skull (the snout tip).

Records are read from `stage2/runs/2026-11-15-75cm` and cited by their path under it. Set freeze 0, manifest `cebd3d238b37`.

## turn/q90_left: quick 90° turn to the left at 0.40 m/s

![top view](turn_q90_left.png)

Top view of quick 90° turn to the left at 0.40 m/s on F25F9cl (the candidate); up is the incoming walking line. Blue, thick: the PLANNED path of the root (the frozen reference, kinematic), from 1 s before the command to the end of the move; the grey shapes are the PLANNED outline of the robot at the command and at the end of the move; the dashed box is the corner box of the planned root. SIMULATED: no record yet; no simulated path is drawn. Every path is drawn from its own position at the command. The scale bar is 0.5 m.

### Planned

From `transitions/turn90_pi4_left.npz` (sha256 `da01f1bf4ebf`), played at 1.05 times its own clock.

| reading (planned) | value |
|---|---|
| path speed commanded | 0.40 m/s on the reference's clock (0.42 m/s in simulated time) |
| turn rate commanded | 45.0 deg/s on the reference's clock (47.2 deg/s in simulated time) |
| path radius commanded (speed over rate) | 0.51 m (0.29 BL) |
| radius fitted to the root's path, steady part | 0.49 m (0.28 BL), RMS 16 mm off the circle (the root sways) |
| corner box, root: advance along the incoming line | largest 0.63 m (0.36 BL), at the move's end 0.60 m (0.34 BL) |
| corner box, root: sideways | largest 1.60 m (0.91 BL), at the move's end 1.61 m (0.91 BL) |
| corner box, whole robot: along | -0.95 m (-0.54 BL) to 1.02 m (0.58 BL) |
| corner box, whole robot: sideways | -0.59 m (-0.33 BL) to 2.40 m (1.36 BL) |
| swept radius | 1.11 m (0.63 BL), set by tail: cover_tail_seg_20 |
| turning circle (twice the swept radius) | 2.22 m (1.26 BL) |
| corridor for the 90° corner | 0.80 m (0.46 BL), inner corner set by left leg: cover_left_thigh_upper |
| turned 2 s after the command | 85.0° |
| turned at the end of the move (4.67 s after the command) | 90.0° |

Farthest point from the turn's centre, by part (planned): tail 1.11 m; head 0.96 m; trunk 0.80 m; right foot 0.68 m; right leg 0.67 m; left leg 0.53 m; left foot 0.51 m.

### Simulated

No record yet: the chain had made no record of this turn when this file was written.

## turn/q90_right: quick 90° turn to the right at 0.40 m/s

![top view](turn_q90_right.png)

Top view of quick 90° turn to the right at 0.40 m/s on F25F9cl (the candidate); up is the incoming walking line. Blue, thick: the PLANNED path of the root (the frozen reference, kinematic), from 1 s before the command to the end of the move; the grey shapes are the PLANNED outline of the robot at the command and at the end of the move; the dashed box is the corner box of the planned root. SIMULATED: no record yet; no simulated path is drawn. Every path is drawn from its own position at the command. The scale bar is 0.5 m.

### Planned

From `transitions/turn90_pi4_right.npz` (sha256 `ed8bb064d002`), played at 1.05 times its own clock.

| reading (planned) | value |
|---|---|
| path speed commanded | 0.40 m/s on the reference's clock (0.42 m/s in simulated time) |
| turn rate commanded | -45.0 deg/s on the reference's clock (-47.2 deg/s in simulated time) |
| path radius commanded (speed over rate) | -0.51 m (-0.29 BL) |
| radius fitted to the root's path, steady part | 0.49 m (0.28 BL), RMS 16 mm off the circle (the root sways) |
| corner box, root: advance along the incoming line | largest 0.63 m (0.36 BL), at the move's end 0.57 m (0.32 BL) |
| corner box, root: sideways | largest 1.80 m (1.02 BL), at the move's end 1.81 m (1.03 BL) |
| corner box, whole robot: along | -0.96 m (-0.54 BL) to 1.02 m (0.58 BL) |
| corner box, whole robot: sideways | -0.59 m (-0.33 BL) to 2.59 m (1.47 BL) |
| swept radius | 1.11 m (0.63 BL), set by tail: cover_tail_seg_20 |
| turning circle (twice the swept radius) | 2.22 m (1.26 BL) |
| corridor for the 90° corner | 0.80 m (0.46 BL), inner corner set by right leg: cover_right_thigh_upper |
| turned 2 s after the command | 85.0° |
| turned at the end of the move (5.14 s after the command) | 90.0° |

Farthest point from the turn's centre, by part (planned): tail 1.11 m; head 0.96 m; trunk 0.80 m; left foot 0.68 m; left leg 0.67 m; right leg 0.53 m; right foot 0.51 m.

### Simulated

No record yet: the chain had made no record of this turn when this file was written.

## turn/t90_0.35_left: 90° turn to the left (0.35 rad/s) at 0.40 m/s

![top view](turn_t90_0.35_left.png)

Top view of 90° turn to the left (0.35 rad/s) at 0.40 m/s on F25F9cl (the candidate); up is the incoming walking line. Blue, thick: the PLANNED path of the root (the frozen reference, kinematic), from 1 s before the command to the end of the move; the grey shapes are the PLANNED outline of the robot at the command and at the end of the move; the dashed box is the corner box of the planned root. SIMULATED: no record yet; no simulated path is drawn. Every path is drawn from its own position at the command. The scale bar is 0.5 m.

### Planned

From `transitions/turn90_0.35_left.npz` (sha256 `907912da2e6d`), played at 1.05 times its own clock.

| reading (planned) | value |
|---|---|
| path speed commanded | 0.40 m/s on the reference's clock (0.42 m/s in simulated time) |
| turn rate commanded | 20.1 deg/s on the reference's clock (21.1 deg/s in simulated time) |
| path radius commanded (speed over rate) | 1.14 m (0.65 BL) |
| radius fitted to the root's path, steady part | 1.13 m (0.64 BL), RMS 33 mm off the circle (the root sways) |
| corner box, root: advance along the incoming line | largest 1.26 m (0.71 BL), at the move's end 1.23 m (0.70 BL) |
| corner box, root: sideways | largest 2.07 m (1.18 BL), at the move's end 2.08 m (1.18 BL) |
| corner box, whole robot: along | -0.95 m (-0.54 BL) to 1.50 m (0.85 BL) |
| corner box, whole robot: sideways | -0.40 m (-0.23 BL) to 2.87 m (1.63 BL) |
| swept radius | 1.56 m (0.89 BL), set by tail: cover_tail_seg_20 |
| turning circle (twice the swept radius) | 3.13 m (1.78 BL) |
| corridor for the 90° corner | 0.80 m (0.46 BL), inner corner set by left foot: hw_left_claw_1 |
| turned 4.48799 s after the command | 89.7° |
| turned at the end of the move (6.62 s after the command) | 90.0° |

Farthest point from the turn's centre, by part (planned): tail 1.56 m; head 1.44 m; trunk 1.37 m; right leg 1.32 m; right foot 1.31 m; left leg 1.19 m; left foot 1.13 m.

### Simulated

No record yet: the chain had made no record of this turn when this file was written.

## turn/t90_0.35_right: 90° turn to the right (0.35 rad/s) at 0.40 m/s

![top view](turn_t90_0.35_right.png)

Top view of 90° turn to the right (0.35 rad/s) at 0.40 m/s on F25F9cl (the candidate); up is the incoming walking line. Blue, thick: the PLANNED path of the root (the frozen reference, kinematic), from 1 s before the command to the end of the move; the grey shapes are the PLANNED outline of the robot at the command and at the end of the move; the dashed box is the corner box of the planned root. SIMULATED: no record yet; no simulated path is drawn. Every path is drawn from its own position at the command. The scale bar is 0.5 m.

### Planned

From `transitions/turn90_0.35_right.npz` (sha256 `36cdc3f7e13c`), played at 1.05 times its own clock.

| reading (planned) | value |
|---|---|
| path speed commanded | 0.40 m/s on the reference's clock (0.42 m/s in simulated time) |
| turn rate commanded | -20.1 deg/s on the reference's clock (-21.1 deg/s in simulated time) |
| path radius commanded (speed over rate) | -1.14 m (-0.65 BL) |
| radius fitted to the root's path, steady part | 1.13 m (0.64 BL), RMS 33 mm off the circle (the root sways) |
| corner box, root: advance along the incoming line | largest 1.26 m (0.71 BL), at the move's end 1.20 m (0.68 BL) |
| corner box, root: sideways | largest 2.27 m (1.29 BL), at the move's end 2.28 m (1.29 BL) |
| corner box, whole robot: along | -0.96 m (-0.54 BL) to 1.50 m (0.85 BL) |
| corner box, whole robot: sideways | -0.40 m (-0.23 BL) to 3.07 m (1.74 BL) |
| swept radius | 1.56 m (0.89 BL), set by tail: cover_tail_seg_20 |
| turning circle (twice the swept radius) | 3.13 m (1.78 BL) |
| corridor for the 90° corner | 0.80 m (0.46 BL), inner corner set by right foot: hw_right_toe_2 |
| turned 4.48799 s after the command | 89.7° |
| turned at the end of the move (7.10 s after the command) | 90.0° |

Farthest point from the turn's centre, by part (planned): tail 1.56 m; head 1.44 m; trunk 1.37 m; left leg 1.32 m; left foot 1.31 m; right leg 1.19 m; right foot 1.13 m.

### Simulated

No record yet: the chain had made no record of this turn when this file was written.

## turn/t90_0.5_left: 90° turn to the left (0.5 rad/s) at 0.40 m/s

![top view](turn_t90_0.5_left.png)

Top view of 90° turn to the left (0.5 rad/s) at 0.40 m/s on F25F9cl (the candidate); up is the incoming walking line. Blue, thick: the PLANNED path of the root (the frozen reference, kinematic), from 1 s before the command to the end of the move; the grey shapes are the PLANNED outline of the robot at the command and at the end of the move; the dashed box is the corner box of the planned root. SIMULATED: no record yet; no simulated path is drawn. Every path is drawn from its own position at the command. The scale bar is 0.5 m.

### Planned

From `transitions/turn90_0.5_left.npz` (sha256 `770ddc5536c0`), played at 1.05 times its own clock.

| reading (planned) | value |
|---|---|
| path speed commanded | 0.40 m/s on the reference's clock (0.42 m/s in simulated time) |
| turn rate commanded | 28.6 deg/s on the reference's clock (30.1 deg/s in simulated time) |
| path radius commanded (speed over rate) | 0.80 m (0.45 BL) |
| radius fitted to the root's path, steady part | 0.78 m (0.45 BL), RMS 30 mm off the circle (the root sways) |
| corner box, root: advance along the incoming line | largest 0.91 m (0.52 BL), at the move's end 0.89 m (0.50 BL) |
| corner box, root: sideways | largest 1.81 m (1.03 BL), at the move's end 1.81 m (1.03 BL) |
| corner box, whole robot: along | -0.95 m (-0.54 BL) to 1.23 m (0.70 BL) |
| corner box, whole robot: sideways | -0.47 m (-0.27 BL) to 2.60 m (1.48 BL) |
| swept radius | 1.30 m (0.74 BL), set by tail: cover_tail_seg_20 |
| turning circle (twice the swept radius) | 2.61 m (1.48 BL) |
| corridor for the 90° corner | 0.80 m (0.46 BL), inner corner set by left leg: cover_left_thigh_upper |
| turned 3.14159 s after the command | 88.6° |
| turned at the end of the move (5.52 s after the command) | 90.0° |

Farthest point from the turn's centre, by part (planned): tail 1.30 m; head 1.16 m; trunk 1.05 m; right leg 0.97 m; right foot 0.97 m; left leg 0.84 m; left foot 0.79 m.

### Simulated

No record yet: the chain had made no record of this turn when this file was written.

## turn/t90_0.5_right: 90° turn to the right (0.5 rad/s) at 0.40 m/s

![top view](turn_t90_0.5_right.png)

Top view of 90° turn to the right (0.5 rad/s) at 0.40 m/s on F25F9cl (the candidate); up is the incoming walking line. Blue, thick: the PLANNED path of the root (the frozen reference, kinematic), from 1 s before the command to the end of the move; the grey shapes are the PLANNED outline of the robot at the command and at the end of the move; the dashed box is the corner box of the planned root. SIMULATED: no record yet; no simulated path is drawn. Every path is drawn from its own position at the command. The scale bar is 0.5 m.

### Planned

From `transitions/turn90_0.5_right.npz` (sha256 `2a899037256b`), played at 1.05 times its own clock.

| reading (planned) | value |
|---|---|
| path speed commanded | 0.40 m/s on the reference's clock (0.42 m/s in simulated time) |
| turn rate commanded | -28.6 deg/s on the reference's clock (-30.1 deg/s in simulated time) |
| path radius commanded (speed over rate) | -0.80 m (-0.45 BL) |
| radius fitted to the root's path, steady part | 0.78 m (0.45 BL), RMS 31 mm off the circle (the root sways) |
| corner box, root: advance along the incoming line | largest 0.91 m (0.52 BL), at the move's end 0.85 m (0.49 BL) |
| corner box, root: sideways | largest 2.00 m (1.14 BL), at the move's end 2.01 m (1.14 BL) |
| corner box, whole robot: along | -0.96 m (-0.54 BL) to 1.23 m (0.70 BL) |
| corner box, whole robot: sideways | -0.47 m (-0.27 BL) to 2.80 m (1.59 BL) |
| swept radius | 1.30 m (0.74 BL), set by tail: cover_tail_seg_20 |
| turning circle (twice the swept radius) | 2.61 m (1.48 BL) |
| corridor for the 90° corner | 0.80 m (0.46 BL), inner corner set by right leg: cover_right_thigh_upper |
| turned 3.14159 s after the command | 88.6° |
| turned at the end of the move (5.99 s after the command) | 90.0° |

Farthest point from the turn's centre, by part (planned): tail 1.30 m; head 1.16 m; trunk 1.05 m; left leg 0.98 m; left foot 0.97 m; right leg 0.84 m; right foot 0.79 m.

### Simulated

No record yet: the chain had made no record of this turn when this file was written.

## Turns without a reference in this set

- turn/t90_1_left (`transitions/turn90_1_left`): no reference within the thigh twist and the leg drives: no stride factor of the grid passes the kinematic rule (thigh twist at most 0.85 of its range, every other gated joint at least 0.05 rad of range, ik_ok, in both directions)
- turn/t90_1_right (`transitions/turn90_1_right`): no reference within the thigh twist and the leg drives: no stride factor of the grid passes the kinematic rule (thigh twist at most 0.85 of its range, every other gated joint at least 0.05 rad of range, ik_ok, in both directions)

## How each reading is made

- **words**: planned: read from the frozen reference played kinematically at the bar's commanded clock (no physics). simulated: read from an episode's trace. No number mixes the two..
- **clock**: the bar plays a reference at 1.05 times its own clock (bar.RULES['clocks']); every time here is in simulated seconds, so a planned speed or rate in simulated time is the reference's own times 1.05; lengths and radii do not change with the clock.
- **frame**: the corner frame of a turn: origin at the root (the trunk's origin) at the command instant; 'along' is the commanded heading (the home heading, the bar's psi_cmd), the incoming walking line; 'sideways' is square to it, positive toward the side of the turn.
- **command**: the command instant t_c: planned, the reference clock's time of the move's first frame over the commanded clock rate; simulated, the episode record's command_s (never recomputed).
- **move end**: the end of the move: the simulated time at which the reference clock reaches the end of the turn's own file (the trace's t_ref column); a trace that ends before it is read to its end and says so.
- **heading**: the trunk's heading: the trace's yaw column (control._yaw of the trunk's rotation), unwrapped, as bar's turn reading takes it; 'turned' is its change from the commanded heading, positive toward the side of the turn.
- **steady**: the steady part of the turn: from the first row at which the trunk has turned 0.1667 of the commanded angle to the first row at which it has turned 0.8333 of it (15 to 75 degrees of a 90 degree turn); a path that never gets there 'stops short' and has no fitted radius.
- **radius commanded**: the path radius commanded: the walk's path speed over the turn's rate (v / omega), both from the frozen turn's entry.
- **radius fitted**: the radius fitted: the circle of the root's path over the steady part, with the trunk's heading as the place on the arc: least squares of root = centre + R x n(heading) over the steady rows, n the unit vector square to the heading that points away from the side of the turn; its centre is the turn's centre, and the RMS distance of the rows from that model is given beside it.  A circle fitted to the positions alone (the algebraic fit, recorded as radius_free_fit_m) reads the root's sideways sway of about 3 cm a step as curvature: it gave 0.29 m on V0's planned quick turn, whose commanded radius is 0.51 m, where this fit gives 0.50 m.
- **corner box**: the corner box: over the rows from the command to the end of the move, how far the root gets along its incoming line (advance) and sideways, at the largest and at the move's end; the same box for the whole robot (every surface point).
- **surface**: the robot's surface points: per body of the model, the corners of the convex hull of its pads, covers and hardware (geom groups 0, 1 and 2: mesh vertices, and the corners of a primitive's bounding box), carried on each row's pose; the snout tip is the head's point farthest forward at home, the tail tip the tail's farthest back.
- **swept radius**: the swept radius: the largest distance in plan from the turn's centre to any surface point over the steady part of the turn (where the robot is on its circle; the straight entry and exit are in the corner box and the corridor, not here); the point that sets it is named.
- **turning circle**: the turning circle: twice the swept radius, the diameter of the circle a full turn at that rate would sweep about the same centre.
- **corridor**: the corridor: the narrowest L-shaped corridor, two legs of one width meeting square, that holds every surface point from the command to the end of the move.  Its outer wall along the incoming leg touches the sweep's point farthest to the outside (sideways = Y0), its far wall across the incoming line touches the sweep's point farthest along (along = X); the inner corner is at (X - w, Y0 + w) and the width w is the smallest for which no point lies inside that corner's block: w = the largest, over all points, of min(X - along, sideways - Y0).  It holds the robot up to the end of the move only: a robot that leaves the move off its heading needs more room further on.
- **body length**: lengths are given in metres and in body lengths; a body length is the model's overall length at its home keyframe along the home heading, from the surface's point farthest back to its point farthest forward.
- **simulated**: simulated: every record of the turn under the run directory (each row, delay, floor grip and block) is listed; an episode that stayed up and whose trace is kept and has the record's sha256 is read; an episode that fell is counted and listed, not read; the heading at the bar's judged instant is the record's own (command_s + at_s; the commanded angle + the sign times heading_error_at_deg).
