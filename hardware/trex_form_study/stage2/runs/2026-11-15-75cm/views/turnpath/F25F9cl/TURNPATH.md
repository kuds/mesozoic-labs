# The room a turn needs: F25F9cl (the candidate) on 75-M-H-10

A view of the frozen reference and of the committed records. It gives no verdict and adopts nothing.

Two words are kept apart. **Planned** is read from the frozen reference, played kinematically at the bar's commanded clock, with no physics. **Simulated** is read from an episode's trace. No number mixes the two.

Lengths are in metres and in body lengths (BL). One body length is 1.759 m: the model's overall length at its home keyframe, from tail: cover_tail_seg_20 (the tail tip) to head: cover_skull (the snout tip).

Records are read from `stage2/runs/2026-11-15-75cm` and cited by their path under it; it held 60 turn records of this set when this file was written (2026-11-15T23:45:52Z). Set freeze 0, manifest `cebd3d238b37`.

## turn/q90_left: quick 90° turn to the left at 0.40 m/s

![top view](turn_q90_left.png)

Top view of quick 90° turn to the left at 0.40 m/s on F25F9cl (the candidate); up is the incoming walking line. Blue, thick: the PLANNED path of the root (the frozen reference, kinematic), from 1 s before the command to the end of the move's file; the grey shapes are the PLANNED outline of the robot at the command and at the end of the move's file. The dashed box is the turn itself: the PLANNED root is 0.61 m along its incoming line and 0.54 m sideways (the ring) when the trunk has turned 90°, 2.16 s after the command. The dotted box is the PLANNED root's corner box to the end of the move's file, 4.67 s after the command: 0.63 m along and 1.61 m sideways (1.07 m of it after the turn); its sideways side is mostly the walk after the turn and depends on the file's length. Orange, thin: SIMULATED root path of the 1 episode that stayed up in `F25F9cl/75-M-H-10/seeds3042/eval/R2/42@6ms/turn/q90_left.json` (R2/42@6ms, servo delay 6 ms (the judged delay is 10 ms); 1 of 20 episodes stayed up).  Rows before it with no path to draw: R2/42@10ms (0 of 20 episodes stayed up); R2/43@10ms (0 of 20 episodes stayed up); R0@10ms (0 of 20 episodes stayed up). Every path is drawn from its own position at the command. The scale bar is 0.5 m.

### Planned

From `transitions/turn90_pi4_left.npz` (sha256 `da01f1bf4ebf`), played at 1.05 times its own clock.

| reading (planned) | value |
|---|---|
| path speed commanded | 0.40 m/s on the reference's clock; 0.42 m/s on the bar's clock (the reference is played 1.05 times faster) |
| turn rate commanded | 45.0 deg/s on the reference's clock; 47.2 deg/s on the bar's clock |
| path radius commanded (speed over rate) | 0.51 m (0.29 BL) |
| radius fitted to the root's path, steady part | 0.49 m (0.28 BL), RMS 16 mm off the circle (the root sways) |
| **the turn itself**: until the trunk has turned 90° (to within 0.5°) | 2.16 s after the command |
| the turn itself, root: where it is then | 0.61 m (0.35 BL) along the incoming line, 0.54 m (0.30 BL) sideways |
| the turn itself, whole robot: along | -0.96 m (-0.55 BL) to 1.02 m (0.58 BL) |
| the turn itself, whole robot: sideways | -0.59 m (-0.33 BL) to 1.33 m (0.76 BL) |
| corner box to the end of the move's file, 4.67 s after the command; includes 1.07 m of walk after the turn; root: advance along the incoming line | largest 0.63 m (0.36 BL), at the move's end 0.60 m (0.34 BL) |
| corner box to the end of the move's file; root: sideways | largest 1.61 m (0.91 BL), at the move's end 1.61 m (0.91 BL) |
| corner box to the end of the move's file; whole robot: along | -0.96 m (-0.55 BL) to 1.02 m (0.58 BL) |
| corner box to the end of the move's file; whole robot: sideways | -0.59 m (-0.33 BL) to 2.40 m (1.37 BL) |
| swept radius over the steady part of the turn (the turning circle uses this one) | 1.11 m (0.63 BL), set by tail: cover_tail_seg_20 |
| largest distance from the turn's centre, from the command until the trunk has turned 90° | 1.12 m (0.64 BL), set by tail: cover_tail_seg_20, at the command |
| turning circle (twice the swept radius of the steady part) | 2.22 m (1.26 BL) |
| corridor for the 90° corner | 0.80 m (0.46 BL), inner corner set by left leg: cover_left_thigh_upper |
| turned 2 s after the command | 85.0° |
| turned at the end of the move (4.67 s after the command) | 90.0° |

Farthest point from the turn's centre, by part (planned): tail 1.11 m; head 0.96 m; trunk 0.80 m; right foot 0.68 m; right leg 0.67 m; left leg 0.53 m; left foot 0.51 m.

### Simulated

**`F25F9cl/75-M-H-10/seeds3042/eval/R0@0ms/turn/q90_left.json`: controller row R0@0ms, servo delay 0 ms, floor grip 0.9, block seeds3042.**

0 of 20 episodes stayed up; 20 fell: 20 after the command (seeds 3042-3061, between 21.61 and 22.14 s).

No episode of this record stayed up with a trace that can be read: nothing is read.

**`F25F9cl/75-M-H-10/seeds3042/eval/R0@0ms/turn/q90_left_mu0.54.json`: controller row R0@0ms, servo delay 0 ms, floor grip 0.54, block seeds3042.**

0 of 20 episodes stayed up; 20 fell: 20 after the command (seeds 3042-3061, between 21.60 and 22.58 s).

No episode of this record stayed up with a trace that can be read: nothing is read.

**`F25F9cl/75-M-H-10/seeds3042/eval/R0@10ms/turn/q90_left.json`: controller row R0@10ms, servo delay 10 ms, floor grip 0.9, block seeds3042.**

0 of 20 episodes stayed up; 20 fell: 20 after the command (seeds 3042-3061, between 21.59 and 21.92 s).

No episode of this record stayed up with a trace that can be read: nothing is read.

**`F25F9cl/75-M-H-10/seeds3042/eval/R0@10ms/turn/q90_left_mu0.54.json`: controller row R0@10ms, servo delay 10 ms, floor grip 0.54, block seeds3042.**

0 of 20 episodes stayed up; 20 fell: 20 after the command (seeds 3042-3061, between 21.65 and 22.97 s).

No episode of this record stayed up with a trace that can be read: nothing is read.

**`F25F9cl/75-M-H-10/seeds3042/eval/R0@4ms/turn/q90_left.json`: controller row R0@4ms, servo delay 4 ms, floor grip 0.9, block seeds3042.**

0 of 20 episodes stayed up; 20 fell: 20 after the command (seeds 3042-3061, between 21.61 and 21.90 s).

No episode of this record stayed up with a trace that can be read: nothing is read.

**`F25F9cl/75-M-H-10/seeds3042/eval/R0@4ms/turn/q90_left_mu0.54.json`: controller row R0@4ms, servo delay 4 ms, floor grip 0.54, block seeds3042.**

0 of 20 episodes stayed up; 20 fell: 20 after the command (seeds 3042-3061, between 21.70 and 21.91 s).

No episode of this record stayed up with a trace that can be read: nothing is read.

**`F25F9cl/75-M-H-10/seeds3042/eval/R0@6ms/turn/q90_left.json`: controller row R0@6ms, servo delay 6 ms, floor grip 0.9, block seeds3042.**

0 of 20 episodes stayed up; 20 fell: 20 after the command (seeds 3042-3061, between 21.61 and 21.89 s).

No episode of this record stayed up with a trace that can be read: nothing is read.

**`F25F9cl/75-M-H-10/seeds3042/eval/R0@6ms/turn/q90_left_mu0.54.json`: controller row R0@6ms, servo delay 6 ms, floor grip 0.54, block seeds3042.**

0 of 20 episodes stayed up; 20 fell: 20 after the command (seeds 3042-3061, between 21.65 and 21.82 s).

No episode of this record stayed up with a trace that can be read: nothing is read.

**`F25F9cl/75-M-H-10/seeds3042/eval/R2/42@0ms/turn/q90_left.json`: controller row R2/42@0ms, servo delay 0 ms, floor grip 0.9, block seeds3042.**

20 of 20 episodes stayed up; 0 fell.

Of the 20 read (median, then smallest to largest; the last column counts the episodes that have the reading: the turn itself is read only where the trunk turned the commanded angle to within 0.5° before the end of the move):

| reading (simulated) | median (smallest to largest) | episodes |
|---|---|---|
| radius fitted to the root's path, steady part | 0.50 m (0.49 to 0.50); 0.28 BL (0.28 to 0.29) | 20 |
| the turn itself, root: along the incoming line when the trunk has turned 90° | 0.60 m (0.59 to 0.63); 0.34 BL (0.33 to 0.36) | 12 |
| the turn itself, root: sideways when the trunk has turned 90° | 0.58 m (0.51 to 1.36); 0.33 BL (0.29 to 0.78) | 12 |
| the turn itself, whole robot: farthest along | 1.03 m (1.01 to 1.04); 0.59 BL (0.57 to 0.59) | 12 |
| the turn itself, whole robot: farthest sideways | 1.37 m (1.30 to 2.16); 0.78 BL (0.74 to 1.23) | 12 |
| walk after the turn, to the end of the move's file (root) | 1.00 m (0.21 to 1.11); 0.57 BL (0.12 to 0.63) | 12 |
| corner box to the end of the move's file, root: largest advance | 0.67 m (0.60 to 0.71); 0.38 BL (0.34 to 0.41) | 20 |
| corner box to the end of the move's file, root: advance at the end | 0.65 m (0.53 to 0.70); 0.37 BL (0.30 to 0.40) | 20 |
| corner box to the end of the move's file, root: largest sideways | 1.57 m (1.55 to 1.62); 0.89 BL (0.88 to 0.92) | 20 |
| corner box to the end of the move's file, root: sideways at the end | 1.57 m (1.55 to 1.62); 0.89 BL (0.88 to 0.92) | 20 |
| swept radius over the steady part | 1.12 m (1.11 to 1.12); 0.63 BL (0.63 to 0.64) | 20 |
| largest distance from the turn's centre, from the command until the trunk has turned 90° | 1.14 m (1.13 to 1.73); 0.65 BL (0.64 to 0.98) | 12 |
| turning circle (twice the swept radius of the steady part) | 2.23 m (2.22 to 2.25); 1.27 BL (1.26 to 1.28) | 20 |
| corridor for the 90° corner | 0.81 m (0.80 to 0.81); 0.46 BL (0.46 to 0.46) | 20 |
| the turn itself: seconds from the command until the trunk has turned 90° (to within 0.5°) | 2.26 s (2.03 to 4.18) | 12 |
| turn rate over the steady part | 47.7 deg/s (46.6 to 48.4) | 20 |
| path speed over the steady part | 0.41 m/s (0.41 to 0.42) | 20 |
| turned at the bar's judged instant | 84.0° (81.7 to 88.2) | 20 |
| turned at the end of the move | 87.9° (86.1 to 92.0) | 20 |
| turned at the end of the trace | 89.6° (87.3 to 93.6) | 20 |

**`F25F9cl/75-M-H-10/seeds3042/eval/R2/42@0ms/turn/q90_left_mu0.54.json`: controller row R2/42@0ms, servo delay 0 ms, floor grip 0.54, block seeds3042.**

20 of 20 episodes stayed up; 0 fell.

Of the 20 read (median, then smallest to largest; the last column counts the episodes that have the reading: the turn itself is read only where the trunk turned the commanded angle to within 0.5° before the end of the move):

| reading (simulated) | median (smallest to largest) | episodes |
|---|---|---|
| radius fitted to the root's path, steady part | 0.49 m (0.48 to 0.51); 0.28 BL (0.27 to 0.29) | 20 |
| the turn itself, root: along the incoming line when the trunk has turned 90° | 0.62 m (0.60 to 0.64); 0.35 BL (0.34 to 0.36) | 11 |
| the turn itself, root: sideways when the trunk has turned 90° | 0.54 m (0.49 to 1.36); 0.31 BL (0.28 to 0.78) | 11 |
| the turn itself, whole robot: farthest along | 1.04 m (1.01 to 1.05); 0.59 BL (0.57 to 0.60) | 11 |
| the turn itself, whole robot: farthest sideways | 1.34 m (1.29 to 2.16); 0.76 BL (0.73 to 1.23) | 11 |
| walk after the turn, to the end of the move's file (root) | 1.03 m (0.20 to 1.10); 0.58 BL (0.12 to 0.63) | 11 |
| corner box to the end of the move's file, root: largest advance | 0.69 m (0.60 to 0.75); 0.39 BL (0.34 to 0.43) | 20 |
| corner box to the end of the move's file, root: advance at the end | 0.66 m (0.51 to 0.75); 0.38 BL (0.29 to 0.43) | 20 |
| corner box to the end of the move's file, root: largest sideways | 1.57 m (1.54 to 1.60); 0.89 BL (0.87 to 0.91) | 20 |
| corner box to the end of the move's file, root: sideways at the end | 1.57 m (1.54 to 1.60); 0.89 BL (0.87 to 0.91) | 20 |
| swept radius over the steady part | 1.12 m (1.11 to 1.12); 0.63 BL (0.63 to 0.64) | 20 |
| largest distance from the turn's centre, from the command until the trunk has turned 90° | 1.14 m (1.14 to 1.73); 0.65 BL (0.65 to 0.98) | 11 |
| turning circle (twice the swept radius of the steady part) | 2.23 m (2.22 to 2.25); 1.27 BL (1.26 to 1.28) | 20 |
| corridor for the 90° corner | 0.81 m (0.80 to 0.82); 0.46 BL (0.46 to 0.46) | 20 |
| the turn itself: seconds from the command until the trunk has turned 90° (to within 0.5°) | 2.22 s (2.03 to 4.18) | 11 |
| turn rate over the steady part | 47.8 deg/s (46.4 to 49.1) | 20 |
| path speed over the steady part | 0.41 m/s (0.41 to 0.42) | 20 |
| turned at the bar's judged instant | 83.0° (79.4 to 88.9) | 20 |
| turned at the end of the move | 88.1° (83.9 to 94.0) | 20 |
| turned at the end of the trace | 89.6° (85.7 to 95.1) | 20 |

**`F25F9cl/75-M-H-10/seeds3042/eval/R2/42@10ms/turn/q90_left.json`: controller row R2/42@10ms, servo delay 10 ms, floor grip 0.9, block seeds3042.**

0 of 20 episodes stayed up; 20 fell: 20 before the command (seeds 3042-3061, between 4.57 and 5.69 s).

No episode of this record stayed up with a trace that can be read: nothing is read.

**`F25F9cl/75-M-H-10/seeds3042/eval/R2/42@10ms/turn/q90_left_mu0.54.json`: controller row R2/42@10ms, servo delay 10 ms, floor grip 0.54, block seeds3042.**

0 of 20 episodes stayed up; 20 fell: 20 before the command (seeds 3042-3061, between 4.57 and 5.48 s).

No episode of this record stayed up with a trace that can be read: nothing is read.

**`F25F9cl/75-M-H-10/seeds3042/eval/R2/42@4ms/turn/q90_left.json`: controller row R2/42@4ms, servo delay 4 ms, floor grip 0.9, block seeds3042.**

20 of 20 episodes stayed up; 0 fell.

Of the 20 read (median, then smallest to largest; the last column counts the episodes that have the reading: the turn itself is read only where the trunk turned the commanded angle to within 0.5° before the end of the move):

| reading (simulated) | median (smallest to largest) | episodes |
|---|---|---|
| radius fitted to the root's path, steady part | 0.50 m (0.49 to 0.50); 0.28 BL (0.28 to 0.28) | 20 |
| the turn itself, root: along the incoming line when the trunk has turned 90° | 0.61 m (0.60 to 0.62); 0.35 BL (0.34 to 0.36) | 16 |
| the turn itself, root: sideways when the trunk has turned 90° | 0.57 m (0.51 to 0.97); 0.33 BL (0.29 to 0.55) | 16 |
| the turn itself, whole robot: farthest along | 1.03 m (1.01 to 1.04); 0.59 BL (0.57 to 0.59) | 16 |
| the turn itself, whole robot: farthest sideways | 1.37 m (1.31 to 1.77); 0.78 BL (0.74 to 1.00) | 16 |
| walk after the turn, to the end of the move's file (root) | 1.01 m (0.60 to 1.11); 0.57 BL (0.34 to 0.63) | 16 |
| corner box to the end of the move's file, root: largest advance | 0.66 m (0.60 to 0.71); 0.38 BL (0.34 to 0.40) | 20 |
| corner box to the end of the move's file, root: advance at the end | 0.63 m (0.54 to 0.69); 0.36 BL (0.30 to 0.39) | 20 |
| corner box to the end of the move's file, root: largest sideways | 1.58 m (1.56 to 1.62); 0.90 BL (0.89 to 0.92) | 20 |
| corner box to the end of the move's file, root: sideways at the end | 1.58 m (1.56 to 1.62); 0.90 BL (0.89 to 0.92) | 20 |
| swept radius over the steady part | 1.11 m (1.11 to 1.12); 0.63 BL (0.63 to 0.63) | 20 |
| largest distance from the turn's centre, from the command until the trunk has turned 90° | 1.13 m (1.13 to 1.37); 0.64 BL (0.64 to 0.78) | 16 |
| turning circle (twice the swept radius of the steady part) | 2.23 m (2.22 to 2.23); 1.27 BL (1.26 to 1.27) | 20 |
| corridor for the 90° corner | 0.81 m (0.81 to 0.81); 0.46 BL (0.46 to 0.46) | 20 |
| the turn itself: seconds from the command until the trunk has turned 90° (to within 0.5°) | 2.27 s (2.05 to 3.24) | 16 |
| turn rate over the steady part | 47.8 deg/s (47.2 to 48.1) | 20 |
| path speed over the steady part | 0.41 m/s (0.41 to 0.42) | 20 |
| turned at the bar's judged instant | 83.8° (81.2 to 87.4) | 20 |
| turned at the end of the move | 88.7° (86.5 to 92.3) | 20 |
| turned at the end of the trace | 89.4° (87.7 to 92.6) | 20 |

**`F25F9cl/75-M-H-10/seeds3042/eval/R2/42@4ms/turn/q90_left_mu0.54.json`: controller row R2/42@4ms, servo delay 4 ms, floor grip 0.54, block seeds3042.**

20 of 20 episodes stayed up; 0 fell.

Of the 20 read (median, then smallest to largest; the last column counts the episodes that have the reading: the turn itself is read only where the trunk turned the commanded angle to within 0.5° before the end of the move):

| reading (simulated) | median (smallest to largest) | episodes |
|---|---|---|
| radius fitted to the root's path, steady part | 0.49 m (0.49 to 0.50); 0.28 BL (0.28 to 0.28) | 20 |
| the turn itself, root: along the incoming line when the trunk has turned 90° | 0.62 m (0.61 to 0.63); 0.35 BL (0.35 to 0.36) | 12 |
| the turn itself, root: sideways when the trunk has turned 90° | 0.57 m (0.51 to 0.98); 0.33 BL (0.29 to 0.56) | 12 |
| the turn itself, whole robot: farthest along | 1.05 m (1.03 to 1.05); 0.60 BL (0.59 to 0.60) | 12 |
| the turn itself, whole robot: farthest sideways | 1.37 m (1.30 to 1.78); 0.78 BL (0.74 to 1.01) | 12 |
| walk after the turn, to the end of the move's file (root) | 1.00 m (0.58 to 1.09); 0.57 BL (0.33 to 0.62) | 12 |
| corner box to the end of the move's file, root: largest advance | 0.68 m (0.64 to 0.74); 0.39 BL (0.36 to 0.42) | 20 |
| corner box to the end of the move's file, root: advance at the end | 0.66 m (0.59 to 0.73); 0.37 BL (0.34 to 0.41) | 20 |
| corner box to the end of the move's file, root: largest sideways | 1.57 m (1.54 to 1.59); 0.89 BL (0.87 to 0.91) | 20 |
| corner box to the end of the move's file, root: sideways at the end | 1.57 m (1.54 to 1.59); 0.89 BL (0.87 to 0.91) | 20 |
| swept radius over the steady part | 1.11 m (1.11 to 1.12); 0.63 BL (0.63 to 0.63) | 20 |
| largest distance from the turn's centre, from the command until the trunk has turned 90° | 1.15 m (1.14 to 1.37); 0.65 BL (0.65 to 0.78) | 12 |
| turning circle (twice the swept radius of the steady part) | 2.22 m (2.22 to 2.23); 1.26 BL (1.26 to 1.27) | 20 |
| corridor for the 90° corner | 0.81 m (0.80 to 0.81); 0.46 BL (0.46 to 0.46) | 20 |
| the turn itself: seconds from the command until the trunk has turned 90° (to within 0.5°) | 2.29 s (2.11 to 3.29) | 12 |
| turn rate over the steady part | 48.3 deg/s (48.0 to 48.5) | 20 |
| path speed over the steady part | 0.41 m/s (0.41 to 0.42) | 20 |
| turned at the bar's judged instant | 83.0° (80.4 to 85.3) | 20 |
| turned at the end of the move | 87.9° (85.5 to 90.1) | 20 |
| turned at the end of the trace | 88.4° (86.3 to 90.8) | 20 |

**`F25F9cl/75-M-H-10/seeds3042/eval/R2/42@6ms/turn/q90_left.json`: controller row R2/42@6ms, servo delay 6 ms, floor grip 0.9, block seeds3042.**

1 of 20 episodes stayed up; 19 fell: 19 before the command (seeds 3042-3053, 3055-3061, between 14.57 and 16.87 s).

Of the 1 read (median, then smallest to largest; the last column counts the episodes that have the reading: the turn itself is read only where the trunk turned the commanded angle to within 0.5° before the end of the move):

| reading (simulated) | median (smallest to largest) | episodes |
|---|---|---|
| radius fitted to the root's path, steady part | 0.50 m (0.50 to 0.50); 0.28 BL (0.28 to 0.28) | 1 |
| the turn itself, root: along the incoming line when the trunk has turned 90° | - | 0 |
| the turn itself, root: sideways when the trunk has turned 90° | - | 0 |
| the turn itself, whole robot: farthest along | - | 0 |
| the turn itself, whole robot: farthest sideways | - | 0 |
| walk after the turn, to the end of the move's file (root) | - | 0 |
| corner box to the end of the move's file, root: largest advance | 0.68 m (0.68 to 0.68); 0.39 BL (0.39 to 0.39) | 1 |
| corner box to the end of the move's file, root: advance at the end | 0.67 m (0.67 to 0.67); 0.38 BL (0.38 to 0.38) | 1 |
| corner box to the end of the move's file, root: largest sideways | 1.55 m (1.55 to 1.55); 0.88 BL (0.88 to 0.88) | 1 |
| corner box to the end of the move's file, root: sideways at the end | 1.55 m (1.55 to 1.55); 0.88 BL (0.88 to 0.88) | 1 |
| swept radius over the steady part | 1.11 m (1.11 to 1.11); 0.63 BL (0.63 to 0.63) | 1 |
| largest distance from the turn's centre, from the command until the trunk has turned 90° | - | 0 |
| turning circle (twice the swept radius of the steady part) | 2.23 m (2.23 to 2.23); 1.27 BL (1.27 to 1.27) | 1 |
| corridor for the 90° corner | 0.81 m (0.81 to 0.81); 0.46 BL (0.46 to 0.46) | 1 |
| the turn itself: seconds from the command until the trunk has turned 90° (to within 0.5°) | - | 0 |
| turn rate over the steady part | 47.3 deg/s (47.3 to 47.3) | 1 |
| path speed over the steady part | 0.41 m/s (0.41 to 0.41) | 1 |
| turned at the bar's judged instant | 81.6° (81.6 to 81.6) | 1 |
| turned at the end of the move | 86.9° (86.9 to 86.9) | 1 |
| turned at the end of the trace | 85.8° (85.8 to 85.8) | 1 |

**`F25F9cl/75-M-H-10/seeds3042/eval/R2/42@6ms/turn/q90_left_mu0.54.json`: controller row R2/42@6ms, servo delay 6 ms, floor grip 0.54, block seeds3042.**

0 of 20 episodes stayed up; 20 fell: 20 before the command (seeds 3042-3061, between 14.58 and 16.83 s).

No episode of this record stayed up with a trace that can be read: nothing is read.

**`F25F9cl/75-M-H-10/seeds3042/eval/R2/43@0ms/turn/q90_left.json`: controller row R2/43@0ms, servo delay 0 ms, floor grip 0.9, block seeds3042.**

20 of 20 episodes stayed up; 0 fell.

Of the 20 read (median, then smallest to largest; the last column counts the episodes that have the reading: the turn itself is read only where the trunk turned the commanded angle to within 0.5° before the end of the move):

| reading (simulated) | median (smallest to largest) | episodes |
|---|---|---|
| radius fitted to the root's path, steady part | 0.50 m (0.49 to 0.51); 0.28 BL (0.28 to 0.29) | 20 |
| the turn itself, root: along the incoming line when the trunk has turned 90° | 0.61 m (0.60 to 0.62); 0.35 BL (0.34 to 0.35) | 13 |
| the turn itself, root: sideways when the trunk has turned 90° | 0.53 m (0.51 to 1.37); 0.30 BL (0.29 to 0.78) | 13 |
| the turn itself, whole robot: farthest along | 1.03 m (1.01 to 1.04); 0.58 BL (0.58 to 0.59) | 13 |
| the turn itself, whole robot: farthest sideways | 1.33 m (1.30 to 2.16); 0.76 BL (0.74 to 1.23) | 13 |
| walk after the turn, to the end of the move's file (root) | 1.05 m (0.21 to 1.10); 0.60 BL (0.12 to 0.62) | 13 |
| corner box to the end of the move's file, root: largest advance | 0.65 m (0.61 to 0.75); 0.37 BL (0.35 to 0.43) | 20 |
| corner box to the end of the move's file, root: advance at the end | 0.62 m (0.56 to 0.75); 0.35 BL (0.32 to 0.43) | 20 |
| corner box to the end of the move's file, root: largest sideways | 1.58 m (1.54 to 1.61); 0.90 BL (0.87 to 0.91) | 20 |
| corner box to the end of the move's file, root: sideways at the end | 1.58 m (1.54 to 1.61); 0.90 BL (0.87 to 0.91) | 20 |
| swept radius over the steady part | 1.12 m (1.11 to 1.12); 0.63 BL (0.63 to 0.64) | 20 |
| largest distance from the turn's centre, from the command until the trunk has turned 90° | 1.13 m (1.13 to 1.74); 0.64 BL (0.64 to 0.99) | 13 |
| turning circle (twice the swept radius of the steady part) | 2.23 m (2.23 to 2.24); 1.27 BL (1.27 to 1.28) | 20 |
| corridor for the 90° corner | 0.81 m (0.80 to 0.81); 0.46 BL (0.45 to 0.46) | 20 |
| the turn itself: seconds from the command until the trunk has turned 90° (to within 0.5°) | 2.18 s (2.06 to 4.18) | 13 |
| turn rate over the steady part | 47.8 deg/s (46.2 to 48.2) | 20 |
| path speed over the steady part | 0.41 m/s (0.41 to 0.41) | 20 |
| turned at the bar's judged instant | 84.7° (79.6 to 87.1) | 20 |
| turned at the end of the move | 88.9° (83.9 to 91.1) | 20 |
| turned at the end of the trace | 90.2° (85.2 to 92.3) | 20 |

**`F25F9cl/75-M-H-10/seeds3042/eval/R2/43@0ms/turn/q90_left_mu0.54.json`: controller row R2/43@0ms, servo delay 0 ms, floor grip 0.54, block seeds3042.**

20 of 20 episodes stayed up; 0 fell.

Of the 20 read (median, then smallest to largest; the last column counts the episodes that have the reading: the turn itself is read only where the trunk turned the commanded angle to within 0.5° before the end of the move):

| reading (simulated) | median (smallest to largest) | episodes |
|---|---|---|
| radius fitted to the root's path, steady part | 0.49 m (0.48 to 0.50); 0.28 BL (0.27 to 0.29) | 20 |
| the turn itself, root: along the incoming line when the trunk has turned 90° | 0.62 m (0.61 to 0.63); 0.35 BL (0.34 to 0.36) | 13 |
| the turn itself, root: sideways when the trunk has turned 90° | 0.52 m (0.49 to 0.57); 0.30 BL (0.28 to 0.33) | 13 |
| the turn itself, whole robot: farthest along | 1.04 m (1.01 to 1.05); 0.59 BL (0.58 to 0.60) | 13 |
| the turn itself, whole robot: farthest sideways | 1.32 m (1.29 to 1.37); 0.75 BL (0.73 to 0.78) | 13 |
| walk after the turn, to the end of the move's file (root) | 1.05 m (0.99 to 1.10); 0.60 BL (0.56 to 0.63) | 13 |
| corner box to the end of the move's file, root: largest advance | 0.67 m (0.61 to 0.72); 0.38 BL (0.35 to 0.41) | 20 |
| corner box to the end of the move's file, root: advance at the end | 0.65 m (0.55 to 0.71); 0.37 BL (0.31 to 0.40) | 20 |
| corner box to the end of the move's file, root: largest sideways | 1.57 m (1.55 to 1.61); 0.89 BL (0.88 to 0.91) | 20 |
| corner box to the end of the move's file, root: sideways at the end | 1.57 m (1.55 to 1.61); 0.89 BL (0.88 to 0.91) | 20 |
| swept radius over the steady part | 1.11 m (1.11 to 1.12); 0.63 BL (0.63 to 0.64) | 20 |
| largest distance from the turn's centre, from the command until the trunk has turned 90° | 1.14 m (1.14 to 1.15); 0.65 BL (0.65 to 0.65) | 13 |
| turning circle (twice the swept radius of the steady part) | 2.23 m (2.22 to 2.24); 1.27 BL (1.26 to 1.27) | 20 |
| corridor for the 90° corner | 0.81 m (0.80 to 0.82); 0.46 BL (0.46 to 0.46) | 20 |
| the turn itself: seconds from the command until the trunk has turned 90° (to within 0.5°) | 2.18 s (2.06 to 2.28) | 13 |
| turn rate over the steady part | 48.1 deg/s (46.9 to 49.1) | 20 |
| path speed over the steady part | 0.41 m/s (0.41 to 0.41) | 20 |
| turned at the bar's judged instant | 83.6° (81.1 to 87.4) | 20 |
| turned at the end of the move | 88.3° (85.2 to 92.5) | 20 |
| turned at the end of the trace | 89.6° (86.4 to 93.8) | 20 |

**`F25F9cl/75-M-H-10/seeds3042/eval/R2/43@10ms/turn/q90_left.json`: controller row R2/43@10ms, servo delay 10 ms, floor grip 0.9, block seeds3042.**

0 of 20 episodes stayed up; 20 fell: 20 before the command (seeds 3042-3061, between 4.55 and 6.74 s).

No episode of this record stayed up with a trace that can be read: nothing is read.

**`F25F9cl/75-M-H-10/seeds3042/eval/R2/43@10ms/turn/q90_left_mu0.54.json`: controller row R2/43@10ms, servo delay 10 ms, floor grip 0.54, block seeds3042.**

0 of 20 episodes stayed up; 20 fell: 20 before the command (seeds 3042-3061, between 4.56 and 5.95 s).

No episode of this record stayed up with a trace that can be read: nothing is read.

**`F25F9cl/75-M-H-10/seeds3042/eval/R2/43@4ms/turn/q90_left.json`: controller row R2/43@4ms, servo delay 4 ms, floor grip 0.9, block seeds3042.**

19 of 20 episodes stayed up; 1 fell: 1 after the command (seed 3049, at 22.35 s).

Of the 19 read (median, then smallest to largest; the last column counts the episodes that have the reading: the turn itself is read only where the trunk turned the commanded angle to within 0.5° before the end of the move):

| reading (simulated) | median (smallest to largest) | episodes |
|---|---|---|
| radius fitted to the root's path, steady part | 0.50 m (0.49 to 0.50); 0.28 BL (0.28 to 0.29) | 19 |
| the turn itself, root: along the incoming line when the trunk has turned 90° | 0.61 m (0.59 to 0.63); 0.35 BL (0.34 to 0.36) | 12 |
| the turn itself, root: sideways when the trunk has turned 90° | 0.56 m (0.51 to 0.98); 0.32 BL (0.29 to 0.56) | 12 |
| the turn itself, whole robot: farthest along | 1.03 m (1.01 to 1.04); 0.59 BL (0.58 to 0.59) | 12 |
| the turn itself, whole robot: farthest sideways | 1.36 m (1.31 to 1.78); 0.77 BL (0.74 to 1.01) | 12 |
| walk after the turn, to the end of the move's file (root) | 1.02 m (0.59 to 1.10); 0.58 BL (0.34 to 0.62) | 12 |
| corner box to the end of the move's file, root: largest advance | 0.67 m (0.62 to 0.73); 0.38 BL (0.35 to 0.41) | 19 |
| corner box to the end of the move's file, root: advance at the end | 0.64 m (0.57 to 0.72); 0.37 BL (0.32 to 0.41) | 19 |
| corner box to the end of the move's file, root: largest sideways | 1.57 m (1.54 to 1.61); 0.90 BL (0.88 to 0.91) | 19 |
| corner box to the end of the move's file, root: sideways at the end | 1.57 m (1.54 to 1.61); 0.90 BL (0.88 to 0.91) | 19 |
| swept radius over the steady part | 1.11 m (1.11 to 1.12); 0.63 BL (0.63 to 0.64) | 19 |
| largest distance from the turn's centre, from the command until the trunk has turned 90° | 1.13 m (1.13 to 1.37); 0.64 BL (0.64 to 0.78) | 12 |
| turning circle (twice the swept radius of the steady part) | 2.23 m (2.23 to 2.24); 1.27 BL (1.27 to 1.27) | 19 |
| corridor for the 90° corner | 0.81 m (0.80 to 0.82); 0.46 BL (0.46 to 0.47) | 19 |
| the turn itself: seconds from the command until the trunk has turned 90° (to within 0.5°) | 2.24 s (2.07 to 3.25) | 12 |
| turn rate over the steady part | 47.8 deg/s (46.9 to 48.1) | 19 |
| path speed over the steady part | 0.41 m/s (0.41 to 0.42) | 19 |
| turned at the bar's judged instant | 83.5° (81.0 to 86.4) | 19 |
| turned at the end of the move | 88.3° (85.6 to 90.5) | 19 |
| turned at the end of the trace | 89.7° (87.3 to 92.0) | 19 |

**`F25F9cl/75-M-H-10/seeds3042/eval/R2/43@4ms/turn/q90_left_mu0.54.json`: controller row R2/43@4ms, servo delay 4 ms, floor grip 0.54, block seeds3042.**

20 of 20 episodes stayed up; 0 fell.

Of the 20 read (median, then smallest to largest; the last column counts the episodes that have the reading: the turn itself is read only where the trunk turned the commanded angle to within 0.5° before the end of the move):

| reading (simulated) | median (smallest to largest) | episodes |
|---|---|---|
| radius fitted to the root's path, steady part | 0.49 m (0.48 to 0.50); 0.28 BL (0.28 to 0.28) | 20 |
| the turn itself, root: along the incoming line when the trunk has turned 90° | 0.63 m (0.61 to 0.64); 0.36 BL (0.35 to 0.36) | 9 |
| the turn itself, root: sideways when the trunk has turned 90° | 0.53 m (0.50 to 0.59); 0.30 BL (0.28 to 0.34) | 9 |
| the turn itself, whole robot: farthest along | 1.04 m (1.02 to 1.05); 0.59 BL (0.58 to 0.60) | 9 |
| the turn itself, whole robot: farthest sideways | 1.33 m (1.30 to 1.39); 0.76 BL (0.74 to 0.79) | 9 |
| walk after the turn, to the end of the move's file (root) | 1.05 m (0.98 to 1.09); 0.60 BL (0.56 to 0.62) | 9 |
| corner box to the end of the move's file, root: largest advance | 0.70 m (0.63 to 0.77); 0.40 BL (0.36 to 0.44) | 20 |
| corner box to the end of the move's file, root: advance at the end | 0.67 m (0.57 to 0.77); 0.38 BL (0.33 to 0.44) | 20 |
| corner box to the end of the move's file, root: largest sideways | 1.56 m (1.52 to 1.60); 0.89 BL (0.87 to 0.91) | 20 |
| corner box to the end of the move's file, root: sideways at the end | 1.56 m (1.52 to 1.60); 0.89 BL (0.87 to 0.91) | 20 |
| swept radius over the steady part | 1.11 m (1.11 to 1.12); 0.63 BL (0.63 to 0.63) | 20 |
| largest distance from the turn's centre, from the command until the trunk has turned 90° | 1.14 m (1.14 to 1.15); 0.65 BL (0.65 to 0.66) | 9 |
| turning circle (twice the swept radius of the steady part) | 2.22 m (2.22 to 2.23); 1.26 BL (1.26 to 1.27) | 20 |
| corridor for the 90° corner | 0.81 m (0.80 to 0.81); 0.46 BL (0.46 to 0.46) | 20 |
| the turn itself: seconds from the command until the trunk has turned 90° (to within 0.5°) | 2.19 s (2.09 to 2.31) | 9 |
| turn rate over the steady part | 48.4 deg/s (48.0 to 49.0) | 20 |
| path speed over the steady part | 0.41 m/s (0.41 to 0.42) | 20 |
| turned at the bar's judged instant | 82.6° (79.3 to 86.2) | 20 |
| turned at the end of the move | 87.8° (84.1 to 91.3) | 20 |
| turned at the end of the trace | 89.4° (85.7 to 93.0) | 20 |

**`F25F9cl/75-M-H-10/seeds3042/eval/R2/43@6ms/turn/q90_left.json`: controller row R2/43@6ms, servo delay 6 ms, floor grip 0.9, block seeds3042.**

1 of 20 episodes stayed up; 19 fell: 17 before the command (seeds 3042-3045, 3047-3054, 3056-3058, 3060-3061, between 8.45 and 17.68 s); 2 after the command (seeds 3046, 3059, between 18.21 and 18.70 s).

Of the 1 read (median, then smallest to largest; the last column counts the episodes that have the reading: the turn itself is read only where the trunk turned the commanded angle to within 0.5° before the end of the move):

| reading (simulated) | median (smallest to largest) | episodes |
|---|---|---|
| radius fitted to the root's path, steady part | 0.51 m (0.51 to 0.51); 0.29 BL (0.29 to 0.29) | 1 |
| the turn itself, root: along the incoming line when the trunk has turned 90° | - | 0 |
| the turn itself, root: sideways when the trunk has turned 90° | - | 0 |
| the turn itself, whole robot: farthest along | - | 0 |
| the turn itself, whole robot: farthest sideways | - | 0 |
| walk after the turn, to the end of the move's file (root) | - | 0 |
| corner box to the end of the move's file, root: largest advance | 0.81 m (0.81 to 0.81); 0.46 BL (0.46 to 0.46) | 1 |
| corner box to the end of the move's file, root: advance at the end | 0.81 m (0.81 to 0.81); 0.46 BL (0.46 to 0.46) | 1 |
| corner box to the end of the move's file, root: largest sideways | 1.49 m (1.49 to 1.49); 0.85 BL (0.85 to 0.85) | 1 |
| corner box to the end of the move's file, root: sideways at the end | 1.49 m (1.49 to 1.49); 0.85 BL (0.85 to 0.85) | 1 |
| swept radius over the steady part | 1.12 m (1.12 to 1.12); 0.64 BL (0.64 to 0.64) | 1 |
| largest distance from the turn's centre, from the command until the trunk has turned 90° | - | 0 |
| turning circle (twice the swept radius of the steady part) | 2.25 m (2.25 to 2.25); 1.28 BL (1.28 to 1.28) | 1 |
| corridor for the 90° corner | 0.82 m (0.82 to 0.82); 0.47 BL (0.47 to 0.47) | 1 |
| the turn itself: seconds from the command until the trunk has turned 90° (to within 0.5°) | - | 0 |
| turn rate over the steady part | 46.7 deg/s (46.7 to 46.7) | 1 |
| path speed over the steady part | 0.42 m/s (0.42 to 0.42) | 1 |
| turned at the bar's judged instant | 76.9° (76.9 to 76.9) | 1 |
| turned at the end of the move | 81.5° (81.5 to 81.5) | 1 |
| turned at the end of the trace | 83.6° (83.6 to 83.6) | 1 |

**`F25F9cl/75-M-H-10/seeds3042/eval/R2/43@6ms/turn/q90_left_mu0.54.json`: controller row R2/43@6ms, servo delay 6 ms, floor grip 0.54, block seeds3042.**

1 of 20 episodes stayed up; 19 fell: 18 before the command (seeds 3043-3054, 3056-3061, between 14.53 and 17.77 s); 1 after the command (seed 3042, at 22.38 s).

Of the 1 read (median, then smallest to largest; the last column counts the episodes that have the reading: the turn itself is read only where the trunk turned the commanded angle to within 0.5° before the end of the move):

| reading (simulated) | median (smallest to largest) | episodes |
|---|---|---|
| radius fitted to the root's path, steady part | 0.51 m (0.51 to 0.51); 0.29 BL (0.29 to 0.29) | 1 |
| the turn itself, root: along the incoming line when the trunk has turned 90° | - | 0 |
| the turn itself, root: sideways when the trunk has turned 90° | - | 0 |
| the turn itself, whole robot: farthest along | - | 0 |
| the turn itself, whole robot: farthest sideways | - | 0 |
| walk after the turn, to the end of the move's file (root) | - | 0 |
| corner box to the end of the move's file, root: largest advance | 0.93 m (0.93 to 0.93); 0.53 BL (0.53 to 0.53) | 1 |
| corner box to the end of the move's file, root: advance at the end | 0.93 m (0.93 to 0.93); 0.53 BL (0.53 to 0.53) | 1 |
| corner box to the end of the move's file, root: largest sideways | 1.41 m (1.41 to 1.41); 0.80 BL (0.80 to 0.80) | 1 |
| corner box to the end of the move's file, root: sideways at the end | 1.41 m (1.41 to 1.41); 0.80 BL (0.80 to 0.80) | 1 |
| swept radius over the steady part | 1.12 m (1.12 to 1.12); 0.64 BL (0.64 to 0.64) | 1 |
| largest distance from the turn's centre, from the command until the trunk has turned 90° | - | 0 |
| turning circle (twice the swept radius of the steady part) | 2.25 m (2.25 to 2.25); 1.28 BL (1.28 to 1.28) | 1 |
| corridor for the 90° corner | 0.91 m (0.91 to 0.91); 0.52 BL (0.52 to 0.52) | 1 |
| the turn itself: seconds from the command until the trunk has turned 90° (to within 0.5°) | - | 0 |
| turn rate over the steady part | 47.5 deg/s (47.5 to 47.5) | 1 |
| path speed over the steady part | 0.42 m/s (0.42 to 0.42) | 1 |
| turned at the bar's judged instant | 72.2° (72.2 to 72.2) | 1 |
| turned at the end of the move | 77.4° (77.4 to 77.4) | 1 |
| turned at the end of the trace | 80.2° (80.2 to 80.2) | 1 |

## turn/q90_right: quick 90° turn to the right at 0.40 m/s

![top view](turn_q90_right.png)

Top view of quick 90° turn to the right at 0.40 m/s on F25F9cl (the candidate); up is the incoming walking line. Blue, thick: the PLANNED path of the root (the frozen reference, kinematic), from 1 s before the command to the end of the move's file; the grey shapes are the PLANNED outline of the robot at the command and at the end of the move's file. The dashed box is the turn itself: the PLANNED root is 0.61 m along its incoming line and 0.54 m sideways (the ring) when the trunk has turned 90°, 2.17 s after the command. The dotted box is the PLANNED root's corner box to the end of the move's file, 5.14 s after the command: 0.63 m along and 1.81 m sideways (1.27 m of it after the turn); its sideways side is mostly the walk after the turn and depends on the file's length. Orange, thin: SIMULATED root path of the 1 episode that stayed up in `F25F9cl/75-M-H-10/seeds3042/eval/R2/42@6ms/turn/q90_right.json` (R2/42@6ms, servo delay 6 ms (the judged delay is 10 ms); 1 of 20 episodes stayed up).  Rows before it with no path to draw: R2/42@10ms (0 of 20 episodes stayed up); R2/43@10ms (0 of 20 episodes stayed up); R0@10ms (0 of 20 episodes stayed up). Every path is drawn from its own position at the command. The scale bar is 0.5 m.

### Planned

From `transitions/turn90_pi4_right.npz` (sha256 `ed8bb064d002`), played at 1.05 times its own clock.

| reading (planned) | value |
|---|---|
| path speed commanded | 0.40 m/s on the reference's clock; 0.42 m/s on the bar's clock (the reference is played 1.05 times faster) |
| turn rate commanded | 45.0 deg/s on the reference's clock; 47.2 deg/s on the bar's clock |
| path radius commanded (speed over rate) | 0.51 m (0.29 BL) |
| radius fitted to the root's path, steady part | 0.49 m (0.28 BL), RMS 16 mm off the circle (the root sways) |
| **the turn itself**: until the trunk has turned 90° (to within 0.5°) | 2.17 s after the command |
| the turn itself, root: where it is then | 0.61 m (0.35 BL) along the incoming line, 0.54 m (0.31 BL) sideways |
| the turn itself, whole robot: along | -0.96 m (-0.55 BL) to 1.02 m (0.58 BL) |
| the turn itself, whole robot: sideways | -0.59 m (-0.33 BL) to 1.34 m (0.76 BL) |
| corner box to the end of the move's file, 5.14 s after the command; includes 1.27 m of walk after the turn; root: advance along the incoming line | largest 0.63 m (0.36 BL), at the move's end 0.57 m (0.32 BL) |
| corner box to the end of the move's file; root: sideways | largest 1.81 m (1.03 BL), at the move's end 1.81 m (1.03 BL) |
| corner box to the end of the move's file; whole robot: along | -0.96 m (-0.55 BL) to 1.02 m (0.58 BL) |
| corner box to the end of the move's file; whole robot: sideways | -0.59 m (-0.33 BL) to 2.60 m (1.48 BL) |
| swept radius over the steady part of the turn (the turning circle uses this one) | 1.11 m (0.63 BL), set by tail: cover_tail_seg_20 |
| largest distance from the turn's centre, from the command until the trunk has turned 90° | 1.12 m (0.64 BL), set by tail: cover_tail_seg_20, at the command |
| turning circle (twice the swept radius of the steady part) | 2.22 m (1.26 BL) |
| corridor for the 90° corner | 0.80 m (0.46 BL), inner corner set by right leg: cover_right_thigh_upper |
| turned 2 s after the command | 85.0° |
| turned at the end of the move (5.14 s after the command) | 90.0° |

Farthest point from the turn's centre, by part (planned): tail 1.11 m; head 0.96 m; trunk 0.80 m; left foot 0.68 m; left leg 0.67 m; right leg 0.53 m; right foot 0.51 m.

### Simulated

**`F25F9cl/75-M-H-10/seeds3042/eval/R0@0ms/turn/q90_right.json`: controller row R0@0ms, servo delay 0 ms, floor grip 0.9, block seeds3042.**

0 of 20 episodes stayed up; 20 fell: 20 after the command (seeds 3042-3061, between 22.06 and 22.28 s).

No episode of this record stayed up with a trace that can be read: nothing is read.

**`F25F9cl/75-M-H-10/seeds3042/eval/R0@0ms/turn/q90_right_mu0.54.json`: controller row R0@0ms, servo delay 0 ms, floor grip 0.54, block seeds3042.**

0 of 20 episodes stayed up; 20 fell: 20 after the command (seeds 3042-3061, between 22.12 and 22.41 s).

No episode of this record stayed up with a trace that can be read: nothing is read.

**`F25F9cl/75-M-H-10/seeds3042/eval/R0@10ms/turn/q90_right.json`: controller row R0@10ms, servo delay 10 ms, floor grip 0.9, block seeds3042.**

0 of 20 episodes stayed up; 20 fell: 20 after the command (seeds 3042-3061, between 22.08 and 22.34 s).

No episode of this record stayed up with a trace that can be read: nothing is read.

**`F25F9cl/75-M-H-10/seeds3042/eval/R0@10ms/turn/q90_right_mu0.54.json`: controller row R0@10ms, servo delay 10 ms, floor grip 0.54, block seeds3042.**

0 of 20 episodes stayed up; 20 fell: 20 after the command (seeds 3042-3061, between 22.15 and 23.04 s).

No episode of this record stayed up with a trace that can be read: nothing is read.

**`F25F9cl/75-M-H-10/seeds3042/eval/R0@4ms/turn/q90_right.json`: controller row R0@4ms, servo delay 4 ms, floor grip 0.9, block seeds3042.**

0 of 20 episodes stayed up; 20 fell: 20 after the command (seeds 3042-3061, between 22.07 and 22.76 s).

No episode of this record stayed up with a trace that can be read: nothing is read.

**`F25F9cl/75-M-H-10/seeds3042/eval/R0@4ms/turn/q90_right_mu0.54.json`: controller row R0@4ms, servo delay 4 ms, floor grip 0.54, block seeds3042.**

0 of 20 episodes stayed up; 20 fell: 20 after the command (seeds 3042-3061, between 22.15 and 22.79 s).

No episode of this record stayed up with a trace that can be read: nothing is read.

**`F25F9cl/75-M-H-10/seeds3042/eval/R0@6ms/turn/q90_right.json`: controller row R0@6ms, servo delay 6 ms, floor grip 0.9, block seeds3042.**

0 of 20 episodes stayed up; 20 fell: 20 after the command (seeds 3042-3061, between 22.07 and 22.30 s).

No episode of this record stayed up with a trace that can be read: nothing is read.

**`F25F9cl/75-M-H-10/seeds3042/eval/R0@6ms/turn/q90_right_mu0.54.json`: controller row R0@6ms, servo delay 6 ms, floor grip 0.54, block seeds3042.**

0 of 20 episodes stayed up; 20 fell: 20 after the command (seeds 3042-3061, between 22.15 and 22.92 s).

No episode of this record stayed up with a trace that can be read: nothing is read.

**`F25F9cl/75-M-H-10/seeds3042/eval/R2/42@0ms/turn/q90_right.json`: controller row R2/42@0ms, servo delay 0 ms, floor grip 0.9, block seeds3042.**

20 of 20 episodes stayed up; 0 fell.

Of the 20 read (median, then smallest to largest; the last column counts the episodes that have the reading: the turn itself is read only where the trunk turned the commanded angle to within 0.5° before the end of the move):

| reading (simulated) | median (smallest to largest) | episodes |
|---|---|---|
| radius fitted to the root's path, steady part | 0.50 m (0.49 to 0.51); 0.28 BL (0.28 to 0.29) | 20 |
| the turn itself, root: along the incoming line when the trunk has turned 90° | 0.61 m (0.59 to 0.63); 0.35 BL (0.34 to 0.36) | 16 |
| the turn itself, root: sideways when the trunk has turned 90° | 0.57 m (0.51 to 1.37); 0.32 BL (0.29 to 0.78) | 16 |
| the turn itself, whole robot: farthest along | 1.03 m (1.02 to 1.04); 0.59 BL (0.58 to 0.59) | 16 |
| the turn itself, whole robot: farthest sideways | 1.37 m (1.31 to 2.16); 0.78 BL (0.74 to 1.23) | 16 |
| walk after the turn, to the end of the move's file (root) | 1.20 m (0.40 to 1.28); 0.68 BL (0.23 to 0.73) | 16 |
| corner box to the end of the move's file, root: largest advance | 0.67 m (0.63 to 0.82); 0.38 BL (0.36 to 0.47) | 20 |
| corner box to the end of the move's file, root: advance at the end | 0.61 m (0.56 to 0.77); 0.35 BL (0.32 to 0.44) | 20 |
| corner box to the end of the move's file, root: largest sideways | 1.77 m (1.72 to 1.79); 1.01 BL (0.98 to 1.02) | 20 |
| corner box to the end of the move's file, root: sideways at the end | 1.77 m (1.72 to 1.79); 1.01 BL (0.98 to 1.02) | 20 |
| swept radius over the steady part | 1.12 m (1.11 to 1.12); 0.63 BL (0.63 to 0.64) | 20 |
| largest distance from the turn's centre, from the command until the trunk has turned 90° | 1.14 m (1.13 to 1.73); 0.65 BL (0.64 to 0.99) | 16 |
| turning circle (twice the swept radius of the steady part) | 2.23 m (2.22 to 2.24); 1.27 BL (1.26 to 1.28) | 20 |
| corridor for the 90° corner | 0.81 m (0.80 to 0.81); 0.46 BL (0.46 to 0.46) | 20 |
| the turn itself: seconds from the command until the trunk has turned 90° (to within 0.5°) | 2.25 s (2.09 to 4.18) | 16 |
| turn rate over the steady part | 47.8 deg/s (46.5 to 48.2) | 20 |
| path speed over the steady part | 0.41 m/s (0.41 to 0.42) | 20 |
| turned at the bar's judged instant | 84.2° (78.6 to 86.1) | 20 |
| turned at the end of the move | 89.9° (84.6 to 91.6) | 20 |
| turned at the end of the trace | 89.7° (84.6 to 91.2) | 20 |

**`F25F9cl/75-M-H-10/seeds3042/eval/R2/42@0ms/turn/q90_right_mu0.54.json`: controller row R2/42@0ms, servo delay 0 ms, floor grip 0.54, block seeds3042.**

20 of 20 episodes stayed up; 0 fell.

Of the 20 read (median, then smallest to largest; the last column counts the episodes that have the reading: the turn itself is read only where the trunk turned the commanded angle to within 0.5° before the end of the move):

| reading (simulated) | median (smallest to largest) | episodes |
|---|---|---|
| radius fitted to the root's path, steady part | 0.49 m (0.48 to 0.53); 0.28 BL (0.28 to 0.30) | 20 |
| the turn itself, root: along the incoming line when the trunk has turned 90° | 0.62 m (0.60 to 0.63); 0.35 BL (0.34 to 0.36) | 6 |
| the turn itself, root: sideways when the trunk has turned 90° | 0.54 m (0.51 to 0.57); 0.31 BL (0.29 to 0.33) | 6 |
| the turn itself, whole robot: farthest along | 1.04 m (1.03 to 1.04); 0.59 BL (0.59 to 0.59) | 6 |
| the turn itself, whole robot: farthest sideways | 1.34 m (1.30 to 1.37); 0.76 BL (0.74 to 0.78) | 6 |
| walk after the turn, to the end of the move's file (root) | 1.23 m (1.20 to 1.28); 0.70 BL (0.68 to 0.73) | 6 |
| corner box to the end of the move's file, root: largest advance | 0.71 m (0.65 to 0.84); 0.41 BL (0.37 to 0.48) | 20 |
| corner box to the end of the move's file, root: advance at the end | 0.66 m (0.57 to 0.80); 0.38 BL (0.32 to 0.45) | 20 |
| corner box to the end of the move's file, root: largest sideways | 1.76 m (1.71 to 1.78); 1.00 BL (0.97 to 1.01) | 20 |
| corner box to the end of the move's file, root: sideways at the end | 1.76 m (1.71 to 1.78); 1.00 BL (0.97 to 1.01) | 20 |
| swept radius over the steady part | 1.12 m (1.11 to 1.13); 0.63 BL (0.63 to 0.64) | 20 |
| largest distance from the turn's centre, from the command until the trunk has turned 90° | 1.14 m (1.14 to 1.15); 0.65 BL (0.65 to 0.65) | 6 |
| turning circle (twice the swept radius of the steady part) | 2.23 m (2.22 to 2.27); 1.27 BL (1.26 to 1.29) | 20 |
| corridor for the 90° corner | 0.81 m (0.80 to 0.82); 0.46 BL (0.45 to 0.46) | 20 |
| the turn itself: seconds from the command until the trunk has turned 90° (to within 0.5°) | 2.21 s (2.11 to 2.27) | 6 |
| turn rate over the steady part | 48.0 deg/s (45.0 to 48.4) | 20 |
| path speed over the steady part | 0.41 m/s (0.41 to 0.42) | 20 |
| turned at the bar's judged instant | 82.5° (76.9 to 85.5) | 20 |
| turned at the end of the move | 88.4° (83.7 to 92.2) | 20 |
| turned at the end of the trace | 88.2° (84.1 to 91.9) | 20 |

**`F25F9cl/75-M-H-10/seeds3042/eval/R2/42@10ms/turn/q90_right.json`: controller row R2/42@10ms, servo delay 10 ms, floor grip 0.9, block seeds3042.**

0 of 20 episodes stayed up; 20 fell: 20 before the command (seeds 3042-3061, between 4.57 and 5.69 s).

No episode of this record stayed up with a trace that can be read: nothing is read.

**`F25F9cl/75-M-H-10/seeds3042/eval/R2/42@10ms/turn/q90_right_mu0.54.json`: controller row R2/42@10ms, servo delay 10 ms, floor grip 0.54, block seeds3042.**

0 of 20 episodes stayed up; 20 fell: 20 before the command (seeds 3042-3061, between 4.57 and 5.48 s).

No episode of this record stayed up with a trace that can be read: nothing is read.

**`F25F9cl/75-M-H-10/seeds3042/eval/R2/42@4ms/turn/q90_right.json`: controller row R2/42@4ms, servo delay 4 ms, floor grip 0.9, block seeds3042.**

19 of 20 episodes stayed up; 1 fell: 1 after the command (seed 3043, at 22.60 s).

Of the 19 read (median, then smallest to largest; the last column counts the episodes that have the reading: the turn itself is read only where the trunk turned the commanded angle to within 0.5° before the end of the move):

| reading (simulated) | median (smallest to largest) | episodes |
|---|---|---|
| radius fitted to the root's path, steady part | 0.50 m (0.49 to 0.50); 0.28 BL (0.28 to 0.29) | 19 |
| the turn itself, root: along the incoming line when the trunk has turned 90° | 0.61 m (0.60 to 0.63); 0.35 BL (0.34 to 0.36) | 17 |
| the turn itself, root: sideways when the trunk has turned 90° | 0.56 m (0.51 to 0.59); 0.32 BL (0.29 to 0.34) | 17 |
| the turn itself, whole robot: farthest along | 1.03 m (1.02 to 1.04); 0.58 BL (0.58 to 0.59) | 17 |
| the turn itself, whole robot: farthest sideways | 1.36 m (1.30 to 1.39); 0.77 BL (0.74 to 0.79) | 17 |
| walk after the turn, to the end of the move's file (root) | 1.23 m (1.19 to 1.30); 0.70 BL (0.67 to 0.74) | 17 |
| corner box to the end of the move's file, root: largest advance | 0.65 m (0.62 to 0.75); 0.37 BL (0.35 to 0.42) | 19 |
| corner box to the end of the move's file, root: advance at the end | 0.59 m (0.54 to 0.70); 0.34 BL (0.31 to 0.40) | 19 |
| corner box to the end of the move's file, root: largest sideways | 1.78 m (1.74 to 1.80); 1.01 BL (0.99 to 1.02) | 19 |
| corner box to the end of the move's file, root: sideways at the end | 1.78 m (1.74 to 1.80); 1.01 BL (0.99 to 1.02) | 19 |
| swept radius over the steady part | 1.11 m (1.11 to 1.12); 0.63 BL (0.63 to 0.63) | 19 |
| largest distance from the turn's centre, from the command until the trunk has turned 90° | 1.13 m (1.13 to 1.13); 0.64 BL (0.64 to 0.64) | 17 |
| turning circle (twice the swept radius of the steady part) | 2.23 m (2.23 to 2.23); 1.27 BL (1.27 to 1.27) | 19 |
| corridor for the 90° corner | 0.81 m (0.80 to 0.81); 0.46 BL (0.46 to 0.46) | 19 |
| the turn itself: seconds from the command until the trunk has turned 90° (to within 0.5°) | 2.23 s (2.07 to 2.30) | 17 |
| turn rate over the steady part | 47.7 deg/s (47.4 to 47.9) | 19 |
| path speed over the steady part | 0.41 m/s (0.41 to 0.42) | 19 |
| turned at the bar's judged instant | 84.4° (81.0 to 86.2) | 19 |
| turned at the end of the move | 90.8° (87.5 to 92.1) | 19 |
| turned at the end of the trace | 90.1° (86.8 to 91.5) | 19 |

**`F25F9cl/75-M-H-10/seeds3042/eval/R2/42@4ms/turn/q90_right_mu0.54.json`: controller row R2/42@4ms, servo delay 4 ms, floor grip 0.54, block seeds3042.**

20 of 20 episodes stayed up; 0 fell.

Of the 20 read (median, then smallest to largest; the last column counts the episodes that have the reading: the turn itself is read only where the trunk turned the commanded angle to within 0.5° before the end of the move):

| reading (simulated) | median (smallest to largest) | episodes |
|---|---|---|
| radius fitted to the root's path, steady part | 0.49 m (0.49 to 0.50); 0.28 BL (0.28 to 0.28) | 20 |
| the turn itself, root: along the incoming line when the trunk has turned 90° | 0.62 m (0.61 to 0.64); 0.35 BL (0.35 to 0.36) | 16 |
| the turn itself, root: sideways when the trunk has turned 90° | 0.55 m (0.50 to 0.59); 0.31 BL (0.28 to 0.34) | 16 |
| the turn itself, whole robot: farthest along | 1.04 m (1.03 to 1.06); 0.59 BL (0.59 to 0.60) | 16 |
| the turn itself, whole robot: farthest sideways | 1.35 m (1.30 to 1.39); 0.77 BL (0.74 to 0.79) | 16 |
| walk after the turn, to the end of the move's file (root) | 1.23 m (1.17 to 1.29); 0.70 BL (0.66 to 0.73) | 16 |
| corner box to the end of the move's file, root: largest advance | 0.66 m (0.64 to 0.73); 0.38 BL (0.36 to 0.41) | 20 |
| corner box to the end of the move's file, root: advance at the end | 0.61 m (0.57 to 0.68); 0.35 BL (0.32 to 0.39) | 20 |
| corner box to the end of the move's file, root: largest sideways | 1.78 m (1.75 to 1.79); 1.01 BL (0.99 to 1.02) | 20 |
| corner box to the end of the move's file, root: sideways at the end | 1.78 m (1.75 to 1.79); 1.01 BL (0.99 to 1.02) | 20 |
| swept radius over the steady part | 1.11 m (1.11 to 1.12); 0.63 BL (0.63 to 0.63) | 20 |
| largest distance from the turn's centre, from the command until the trunk has turned 90° | 1.14 m (1.14 to 1.15); 0.65 BL (0.65 to 0.65) | 16 |
| turning circle (twice the swept radius of the steady part) | 2.22 m (2.22 to 2.23); 1.26 BL (1.26 to 1.27) | 20 |
| corridor for the 90° corner | 0.81 m (0.80 to 0.81); 0.46 BL (0.46 to 0.46) | 20 |
| the turn itself: seconds from the command until the trunk has turned 90° (to within 0.5°) | 2.23 s (2.10 to 2.33) | 16 |
| turn rate over the steady part | 48.2 deg/s (48.0 to 48.5) | 20 |
| path speed over the steady part | 0.41 m/s (0.41 to 0.42) | 20 |
| turned at the bar's judged instant | 84.0° (81.6 to 85.2) | 20 |
| turned at the end of the move | 90.6° (88.2 to 92.0) | 20 |
| turned at the end of the trace | 89.8° (87.4 to 91.3) | 20 |

**`F25F9cl/75-M-H-10/seeds3042/eval/R2/42@6ms/turn/q90_right.json`: controller row R2/42@6ms, servo delay 6 ms, floor grip 0.9, block seeds3042.**

1 of 20 episodes stayed up; 19 fell: 19 before the command (seeds 3042-3053, 3055-3061, between 14.57 and 16.87 s).

Of the 1 read (median, then smallest to largest; the last column counts the episodes that have the reading: the turn itself is read only where the trunk turned the commanded angle to within 0.5° before the end of the move):

| reading (simulated) | median (smallest to largest) | episodes |
|---|---|---|
| radius fitted to the root's path, steady part | 0.48 m (0.48 to 0.48); 0.27 BL (0.27 to 0.27) | 1 |
| the turn itself, root: along the incoming line when the trunk has turned 90° | 0.61 m (0.61 to 0.61); 0.35 BL (0.35 to 0.35) | 1 |
| the turn itself, root: sideways when the trunk has turned 90° | 0.49 m (0.49 to 0.49); 0.28 BL (0.28 to 0.28) | 1 |
| the turn itself, whole robot: farthest along | 1.02 m (1.02 to 1.02); 0.58 BL (0.58 to 0.58) | 1 |
| the turn itself, whole robot: farthest sideways | 1.28 m (1.28 to 1.28); 0.73 BL (0.73 to 0.73) | 1 |
| walk after the turn, to the end of the move's file (root) | 1.29 m (1.29 to 1.29); 0.73 BL (0.73 to 0.73) | 1 |
| corner box to the end of the move's file, root: largest advance | 0.62 m (0.62 to 0.62); 0.35 BL (0.35 to 0.35) | 1 |
| corner box to the end of the move's file, root: advance at the end | 0.55 m (0.55 to 0.55); 0.31 BL (0.31 to 0.31) | 1 |
| corner box to the end of the move's file, root: largest sideways | 1.77 m (1.77 to 1.77); 1.01 BL (1.01 to 1.01) | 1 |
| corner box to the end of the move's file, root: sideways at the end | 1.77 m (1.77 to 1.77); 1.01 BL (1.01 to 1.01) | 1 |
| swept radius over the steady part | 1.11 m (1.11 to 1.11); 0.63 BL (0.63 to 0.63) | 1 |
| largest distance from the turn's centre, from the command until the trunk has turned 90° | 1.13 m (1.13 to 1.13); 0.64 BL (0.64 to 0.64) | 1 |
| turning circle (twice the swept radius of the steady part) | 2.23 m (2.23 to 2.23); 1.27 BL (1.27 to 1.27) | 1 |
| corridor for the 90° corner | 0.83 m (0.83 to 0.83); 0.47 BL (0.47 to 0.47) | 1 |
| the turn itself: seconds from the command until the trunk has turned 90° (to within 0.5°) | 2.11 s (2.11 to 2.11) | 1 |
| turn rate over the steady part | 47.7 deg/s (47.7 to 47.7) | 1 |
| path speed over the steady part | 0.40 m/s (0.40 to 0.40) | 1 |
| turned at the bar's judged instant | 84.9° (84.9 to 84.9) | 1 |
| turned at the end of the move | 92.7° (92.7 to 92.7) | 1 |
| turned at the end of the trace | 92.8° (92.8 to 92.8) | 1 |

**`F25F9cl/75-M-H-10/seeds3042/eval/R2/42@6ms/turn/q90_right_mu0.54.json`: controller row R2/42@6ms, servo delay 6 ms, floor grip 0.54, block seeds3042.**

0 of 20 episodes stayed up; 20 fell: 20 before the command (seeds 3042-3061, between 14.58 and 16.83 s).

No episode of this record stayed up with a trace that can be read: nothing is read.

**`F25F9cl/75-M-H-10/seeds3042/eval/R2/43@0ms/turn/q90_right.json`: controller row R2/43@0ms, servo delay 0 ms, floor grip 0.9, block seeds3042.**

20 of 20 episodes stayed up; 0 fell.

Of the 20 read (median, then smallest to largest; the last column counts the episodes that have the reading: the turn itself is read only where the trunk turned the commanded angle to within 0.5° before the end of the move):

| reading (simulated) | median (smallest to largest) | episodes |
|---|---|---|
| radius fitted to the root's path, steady part | 0.50 m (0.49 to 0.51); 0.28 BL (0.28 to 0.29) | 20 |
| the turn itself, root: along the incoming line when the trunk has turned 90° | 0.61 m (0.59 to 0.62); 0.35 BL (0.34 to 0.35) | 12 |
| the turn itself, root: sideways when the trunk has turned 90° | 0.57 m (0.51 to 0.97); 0.32 BL (0.29 to 0.55) | 12 |
| the turn itself, whole robot: farthest along | 1.03 m (1.02 to 1.04); 0.58 BL (0.58 to 0.59) | 12 |
| the turn itself, whole robot: farthest sideways | 1.36 m (1.31 to 1.77); 0.78 BL (0.74 to 1.01) | 12 |
| walk after the turn, to the end of the move's file (root) | 1.21 m (0.80 to 1.28); 0.69 BL (0.45 to 0.73) | 12 |
| corner box to the end of the move's file, root: largest advance | 0.68 m (0.62 to 0.78); 0.38 BL (0.35 to 0.44) | 20 |
| corner box to the end of the move's file, root: advance at the end | 0.62 m (0.54 to 0.74); 0.35 BL (0.31 to 0.42) | 20 |
| corner box to the end of the move's file, root: largest sideways | 1.77 m (1.74 to 1.79); 1.01 BL (0.99 to 1.02) | 20 |
| corner box to the end of the move's file, root: sideways at the end | 1.77 m (1.74 to 1.79); 1.01 BL (0.99 to 1.02) | 20 |
| swept radius over the steady part | 1.12 m (1.11 to 1.12); 0.63 BL (0.63 to 0.64) | 20 |
| largest distance from the turn's centre, from the command until the trunk has turned 90° | 1.14 m (1.13 to 1.36); 0.65 BL (0.64 to 0.78) | 12 |
| turning circle (twice the swept radius of the steady part) | 2.23 m (2.23 to 2.25); 1.27 BL (1.26 to 1.28) | 20 |
| corridor for the 90° corner | 0.81 m (0.80 to 0.81); 0.46 BL (0.46 to 0.46) | 20 |
| the turn itself: seconds from the command until the trunk has turned 90° (to within 0.5°) | 2.24 s (2.09 to 3.22) | 12 |
| turn rate over the steady part | 47.7 deg/s (46.1 to 48.0) | 20 |
| path speed over the steady part | 0.41 m/s (0.41 to 0.42) | 20 |
| turned at the bar's judged instant | 83.8° (79.6 to 86.2) | 20 |
| turned at the end of the move | 89.8° (85.3 to 92.2) | 20 |
| turned at the end of the trace | 89.6° (85.0 to 91.8) | 20 |

**`F25F9cl/75-M-H-10/seeds3042/eval/R2/43@0ms/turn/q90_right_mu0.54.json`: controller row R2/43@0ms, servo delay 0 ms, floor grip 0.54, block seeds3042.**

20 of 20 episodes stayed up; 0 fell.

Of the 20 read (median, then smallest to largest; the last column counts the episodes that have the reading: the turn itself is read only where the trunk turned the commanded angle to within 0.5° before the end of the move):

| reading (simulated) | median (smallest to largest) | episodes |
|---|---|---|
| radius fitted to the root's path, steady part | 0.49 m (0.48 to 0.51); 0.28 BL (0.27 to 0.29) | 20 |
| the turn itself, root: along the incoming line when the trunk has turned 90° | 0.63 m (0.60 to 0.64); 0.36 BL (0.34 to 0.36) | 11 |
| the turn itself, root: sideways when the trunk has turned 90° | 0.53 m (0.49 to 0.98); 0.30 BL (0.28 to 0.56) | 11 |
| the turn itself, whole robot: farthest along | 1.04 m (1.03 to 1.05); 0.59 BL (0.58 to 0.60) | 11 |
| the turn itself, whole robot: farthest sideways | 1.33 m (1.29 to 1.78); 0.76 BL (0.73 to 1.01) | 11 |
| walk after the turn, to the end of the move's file (root) | 1.22 m (0.79 to 1.29); 0.69 BL (0.45 to 0.73) | 11 |
| corner box to the end of the move's file, root: largest advance | 0.68 m (0.63 to 0.81); 0.39 BL (0.36 to 0.46) | 20 |
| corner box to the end of the move's file, root: advance at the end | 0.62 m (0.55 to 0.76); 0.36 BL (0.31 to 0.43) | 20 |
| corner box to the end of the move's file, root: largest sideways | 1.76 m (1.72 to 1.78); 1.00 BL (0.98 to 1.01) | 20 |
| corner box to the end of the move's file, root: sideways at the end | 1.76 m (1.72 to 1.78); 1.00 BL (0.98 to 1.01) | 20 |
| swept radius over the steady part | 1.11 m (1.11 to 1.12); 0.63 BL (0.63 to 0.64) | 20 |
| largest distance from the turn's centre, from the command until the trunk has turned 90° | 1.14 m (1.14 to 1.37); 0.65 BL (0.65 to 0.78) | 11 |
| turning circle (twice the swept radius of the steady part) | 2.23 m (2.22 to 2.25); 1.27 BL (1.26 to 1.28) | 20 |
| corridor for the 90° corner | 0.81 m (0.80 to 0.82); 0.46 BL (0.46 to 0.46) | 20 |
| the turn itself: seconds from the command until the trunk has turned 90° (to within 0.5°) | 2.23 s (2.09 to 3.25) | 11 |
| turn rate over the steady part | 47.9 deg/s (46.5 to 48.7) | 20 |
| path speed over the steady part | 0.41 m/s (0.41 to 0.42) | 20 |
| turned at the bar's judged instant | 83.2° (79.1 to 86.1) | 20 |
| turned at the end of the move | 89.3° (85.1 to 92.8) | 20 |
| turned at the end of the trace | 89.3° (85.1 to 92.6) | 20 |

**`F25F9cl/75-M-H-10/seeds3042/eval/R2/43@10ms/turn/q90_right.json`: controller row R2/43@10ms, servo delay 10 ms, floor grip 0.9, block seeds3042.**

0 of 20 episodes stayed up; 20 fell: 20 before the command (seeds 3042-3061, between 4.55 and 6.74 s).

No episode of this record stayed up with a trace that can be read: nothing is read.

**`F25F9cl/75-M-H-10/seeds3042/eval/R2/43@10ms/turn/q90_right_mu0.54.json`: controller row R2/43@10ms, servo delay 10 ms, floor grip 0.54, block seeds3042.**

0 of 20 episodes stayed up; 20 fell: 20 before the command (seeds 3042-3061, between 4.56 and 5.95 s).

No episode of this record stayed up with a trace that can be read: nothing is read.

**`F25F9cl/75-M-H-10/seeds3042/eval/R2/43@4ms/turn/q90_right.json`: controller row R2/43@4ms, servo delay 4 ms, floor grip 0.9, block seeds3042.**

19 of 20 episodes stayed up; 1 fell: 1 after the command (seed 3060, at 22.81 s).

Of the 19 read (median, then smallest to largest; the last column counts the episodes that have the reading: the turn itself is read only where the trunk turned the commanded angle to within 0.5° before the end of the move):

| reading (simulated) | median (smallest to largest) | episodes |
|---|---|---|
| radius fitted to the root's path, steady part | 0.50 m (0.49 to 0.50); 0.28 BL (0.28 to 0.28) | 19 |
| the turn itself, root: along the incoming line when the trunk has turned 90° | 0.61 m (0.60 to 0.63); 0.35 BL (0.34 to 0.36) | 12 |
| the turn itself, root: sideways when the trunk has turned 90° | 0.55 m (0.51 to 0.60); 0.31 BL (0.29 to 0.34) | 12 |
| the turn itself, whole robot: farthest along | 1.03 m (1.02 to 1.05); 0.59 BL (0.58 to 0.59) | 12 |
| the turn itself, whole robot: farthest sideways | 1.35 m (1.30 to 1.40); 0.76 BL (0.74 to 0.80) | 12 |
| walk after the turn, to the end of the move's file (root) | 1.23 m (1.17 to 1.29); 0.70 BL (0.67 to 0.74) | 12 |
| corner box to the end of the move's file, root: largest advance | 0.67 m (0.62 to 0.76); 0.38 BL (0.35 to 0.43) | 19 |
| corner box to the end of the move's file, root: advance at the end | 0.62 m (0.54 to 0.71); 0.35 BL (0.31 to 0.41) | 19 |
| corner box to the end of the move's file, root: largest sideways | 1.78 m (1.74 to 1.80); 1.01 BL (0.99 to 1.02) | 19 |
| corner box to the end of the move's file, root: sideways at the end | 1.78 m (1.74 to 1.80); 1.01 BL (0.99 to 1.02) | 19 |
| swept radius over the steady part | 1.11 m (1.11 to 1.12); 0.63 BL (0.63 to 0.63) | 19 |
| largest distance from the turn's centre, from the command until the trunk has turned 90° | 1.13 m (1.13 to 1.14); 0.64 BL (0.64 to 0.65) | 12 |
| turning circle (twice the swept radius of the steady part) | 2.23 m (2.23 to 2.23); 1.27 BL (1.27 to 1.27) | 19 |
| corridor for the 90° corner | 0.81 m (0.80 to 0.81); 0.46 BL (0.46 to 0.46) | 19 |
| the turn itself: seconds from the command until the trunk has turned 90° (to within 0.5°) | 2.21 s (2.08 to 2.32) | 12 |
| turn rate over the steady part | 47.7 deg/s (47.2 to 48.3) | 19 |
| path speed over the steady part | 0.41 m/s (0.41 to 0.42) | 19 |
| turned at the bar's judged instant | 84.0° (80.7 to 86.4) | 19 |
| turned at the end of the move | 89.7° (86.4 to 92.6) | 19 |
| turned at the end of the trace | 89.6° (86.3 to 92.5) | 19 |

**`F25F9cl/75-M-H-10/seeds3042/eval/R2/43@4ms/turn/q90_right_mu0.54.json`: controller row R2/43@4ms, servo delay 4 ms, floor grip 0.54, block seeds3042.**

20 of 20 episodes stayed up; 0 fell.

Of the 20 read (median, then smallest to largest; the last column counts the episodes that have the reading: the turn itself is read only where the trunk turned the commanded angle to within 0.5° before the end of the move):

| reading (simulated) | median (smallest to largest) | episodes |
|---|---|---|
| radius fitted to the root's path, steady part | 0.49 m (0.49 to 0.49); 0.28 BL (0.28 to 0.28) | 20 |
| the turn itself, root: along the incoming line when the trunk has turned 90° | 0.63 m (0.61 to 0.64); 0.36 BL (0.35 to 0.36) | 11 |
| the turn itself, root: sideways when the trunk has turned 90° | 0.56 m (0.51 to 1.38); 0.32 BL (0.29 to 0.78) | 11 |
| the turn itself, whole robot: farthest along | 1.05 m (1.03 to 1.06); 0.59 BL (0.59 to 0.60) | 11 |
| the turn itself, whole robot: farthest sideways | 1.36 m (1.31 to 2.17); 0.77 BL (0.74 to 1.24) | 11 |
| walk after the turn, to the end of the move's file (root) | 1.21 m (0.38 to 1.28); 0.69 BL (0.22 to 0.73) | 11 |
| corner box to the end of the move's file, root: largest advance | 0.70 m (0.64 to 0.80); 0.40 BL (0.37 to 0.45) | 20 |
| corner box to the end of the move's file, root: advance at the end | 0.64 m (0.57 to 0.76); 0.37 BL (0.33 to 0.43) | 20 |
| corner box to the end of the move's file, root: largest sideways | 1.77 m (1.72 to 1.80); 1.01 BL (0.98 to 1.02) | 20 |
| corner box to the end of the move's file, root: sideways at the end | 1.77 m (1.72 to 1.80); 1.01 BL (0.98 to 1.02) | 20 |
| swept radius over the steady part | 1.11 m (1.11 to 1.11); 0.63 BL (0.63 to 0.63) | 20 |
| largest distance from the turn's centre, from the command until the trunk has turned 90° | 1.15 m (1.14 to 1.75); 0.65 BL (0.65 to 0.99) | 11 |
| turning circle (twice the swept radius of the steady part) | 2.23 m (2.22 to 2.23); 1.26 BL (1.26 to 1.27) | 20 |
| corridor for the 90° corner | 0.81 m (0.80 to 0.81); 0.46 BL (0.46 to 0.46) | 20 |
| the turn itself: seconds from the command until the trunk has turned 90° (to within 0.5°) | 2.26 s (2.12 to 4.22) | 11 |
| turn rate over the steady part | 48.3 deg/s (47.9 to 48.8) | 20 |
| path speed over the steady part | 0.41 m/s (0.41 to 0.42) | 20 |
| turned at the bar's judged instant | 83.1° (79.5 to 85.3) | 20 |
| turned at the end of the move | 89.6° (85.3 to 91.7) | 20 |
| turned at the end of the trace | 89.6° (85.2 to 91.8) | 20 |

**`F25F9cl/75-M-H-10/seeds3042/eval/R2/43@6ms/turn/q90_right.json`: controller row R2/43@6ms, servo delay 6 ms, floor grip 0.9, block seeds3042.**

1 of 20 episodes stayed up; 19 fell: 18 before the command (seeds 3042-3045, 3047-3054, 3056-3061, between 8.45 and 18.18 s); 1 after the command (seed 3046, at 18.70 s).

Of the 1 read (median, then smallest to largest; the last column counts the episodes that have the reading: the turn itself is read only where the trunk turned the commanded angle to within 0.5° before the end of the move):

| reading (simulated) | median (smallest to largest) | episodes |
|---|---|---|
| radius fitted to the root's path, steady part | 0.49 m (0.49 to 0.49); 0.28 BL (0.28 to 0.28) | 1 |
| the turn itself, root: along the incoming line when the trunk has turned 90° | 0.56 m (0.56 to 0.56); 0.32 BL (0.32 to 0.32) | 1 |
| the turn itself, root: sideways when the trunk has turned 90° | 0.53 m (0.53 to 0.53); 0.30 BL (0.30 to 0.30) | 1 |
| the turn itself, whole robot: farthest along | 0.98 m (0.98 to 0.98); 0.56 BL (0.56 to 0.56) | 1 |
| the turn itself, whole robot: farthest sideways | 1.32 m (1.32 to 1.32); 0.75 BL (0.75 to 0.75) | 1 |
| walk after the turn, to the end of the move's file (root) | 1.32 m (1.32 to 1.32); 0.75 BL (0.75 to 0.75) | 1 |
| corner box to the end of the move's file, root: largest advance | 0.57 m (0.57 to 0.57); 0.32 BL (0.32 to 0.32) | 1 |
| corner box to the end of the move's file, root: advance at the end | 0.41 m (0.41 to 0.41); 0.23 BL (0.23 to 0.23) | 1 |
| corner box to the end of the move's file, root: largest sideways | 1.83 m (1.83 to 1.83); 1.04 BL (1.04 to 1.04) | 1 |
| corner box to the end of the move's file, root: sideways at the end | 1.83 m (1.83 to 1.83); 1.04 BL (1.04 to 1.04) | 1 |
| swept radius over the steady part | 1.11 m (1.11 to 1.11); 0.63 BL (0.63 to 0.63) | 1 |
| largest distance from the turn's centre, from the command until the trunk has turned 90° | 1.13 m (1.13 to 1.13); 0.64 BL (0.64 to 0.64) | 1 |
| turning circle (twice the swept radius of the steady part) | 2.23 m (2.23 to 2.23); 1.27 BL (1.27 to 1.27) | 1 |
| corridor for the 90° corner | 0.81 m (0.81 to 0.81); 0.46 BL (0.46 to 0.46) | 1 |
| the turn itself: seconds from the command until the trunk has turned 90° (to within 0.5°) | 1.99 s (1.99 to 1.99) | 1 |
| turn rate over the steady part | 46.7 deg/s (46.7 to 46.7) | 1 |
| path speed over the steady part | 0.40 m/s (0.40 to 0.40) | 1 |
| turned at the bar's judged instant | 90.6° (90.6 to 90.6) | 1 |
| turned at the end of the move | 97.5° (97.5 to 97.5) | 1 |
| turned at the end of the trace | 97.7° (97.7 to 97.7) | 1 |

**`F25F9cl/75-M-H-10/seeds3042/eval/R2/43@6ms/turn/q90_right_mu0.54.json`: controller row R2/43@6ms, servo delay 6 ms, floor grip 0.54, block seeds3042.**

1 of 20 episodes stayed up; 19 fell: 18 before the command (seeds 3043-3054, 3056-3061, between 14.53 and 17.77 s); 1 after the command (seed 3055, at 22.49 s).

Of the 1 read (median, then smallest to largest; the last column counts the episodes that have the reading: the turn itself is read only where the trunk turned the commanded angle to within 0.5° before the end of the move):

| reading (simulated) | median (smallest to largest) | episodes |
|---|---|---|
| radius fitted to the root's path, steady part | 0.50 m (0.50 to 0.50); 0.28 BL (0.28 to 0.28) | 1 |
| the turn itself, root: along the incoming line when the trunk has turned 90° | - | 0 |
| the turn itself, root: sideways when the trunk has turned 90° | - | 0 |
| the turn itself, whole robot: farthest along | - | 0 |
| the turn itself, whole robot: farthest sideways | - | 0 |
| walk after the turn, to the end of the move's file (root) | - | 0 |
| corner box to the end of the move's file, root: largest advance | 0.99 m (0.99 to 0.99); 0.56 BL (0.56 to 0.56) | 1 |
| corner box to the end of the move's file, root: advance at the end | 0.96 m (0.96 to 0.96); 0.55 BL (0.55 to 0.55) | 1 |
| corner box to the end of the move's file, root: largest sideways | 1.60 m (1.60 to 1.60); 0.91 BL (0.91 to 0.91) | 1 |
| corner box to the end of the move's file, root: sideways at the end | 1.60 m (1.60 to 1.60); 0.91 BL (0.91 to 0.91) | 1 |
| swept radius over the steady part | 1.12 m (1.12 to 1.12); 0.64 BL (0.64 to 0.64) | 1 |
| largest distance from the turn's centre, from the command until the trunk has turned 90° | - | 0 |
| turning circle (twice the swept radius of the steady part) | 2.24 m (2.24 to 2.24); 1.27 BL (1.27 to 1.27) | 1 |
| corridor for the 90° corner | 0.88 m (0.88 to 0.88); 0.50 BL (0.50 to 0.50) | 1 |
| the turn itself: seconds from the command until the trunk has turned 90° (to within 0.5°) | - | 0 |
| turn rate over the steady part | 48.2 deg/s (48.2 to 48.2) | 1 |
| path speed over the steady part | 0.42 m/s (0.42 to 0.42) | 1 |
| turned at the bar's judged instant | 71.8° (71.8 to 71.8) | 1 |
| turned at the end of the move | 78.5° (78.5 to 78.5) | 1 |
| turned at the end of the trace | 79.3° (79.3 to 79.3) | 1 |

## turn/t90_0.35_left: 90° turn to the left (0.35 rad/s) at 0.40 m/s

![top view](turn_t90_0.35_left.png)

Top view of 90° turn to the left (0.35 rad/s) at 0.40 m/s on F25F9cl (the candidate); up is the incoming walking line. Blue, thick: the PLANNED path of the root (the frozen reference, kinematic), from 1 s before the command to the end of the move's file; the grey shapes are the PLANNED outline of the robot at the command and at the end of the move's file. The dashed box is the turn itself: the PLANNED root is 1.17 m along its incoming line and 1.16 m sideways (the ring) when the trunk has turned 90°, 4.48 s after the command. The dotted box is the PLANNED root's corner box to the end of the move's file, 6.62 s after the command: 1.26 m along and 2.08 m sideways (0.92 m of it after the turn); its sideways side is mostly the walk after the turn and depends on the file's length. SIMULATED: no episode of `F25F9cl/75-M-H-10/seeds3042/eval/R2/42@10ms/turn/t90_0.35_left.json` stayed up with a trace that can be read; none is drawn. Every path is drawn from its own position at the command. The scale bar is 0.5 m.

### Planned

From `transitions/turn90_0.35_left.npz` (sha256 `907912da2e6d`), played at 1.05 times its own clock.

| reading (planned) | value |
|---|---|
| path speed commanded | 0.40 m/s on the reference's clock; 0.42 m/s on the bar's clock (the reference is played 1.05 times faster) |
| turn rate commanded | 20.1 deg/s on the reference's clock; 21.1 deg/s on the bar's clock |
| path radius commanded (speed over rate) | 1.14 m (0.65 BL) |
| radius fitted to the root's path, steady part | 1.13 m (0.64 BL), RMS 33 mm off the circle (the root sways) |
| **the turn itself**: until the trunk has turned 90° (to within 0.5°) | 4.48 s after the command |
| the turn itself, root: where it is then | 1.17 m (0.66 BL) along the incoming line, 1.16 m (0.66 BL) sideways |
| the turn itself, whole robot: along | -0.96 m (-0.55 BL) to 1.50 m (0.85 BL) |
| the turn itself, whole robot: sideways | -0.40 m (-0.23 BL) to 1.95 m (1.11 BL) |
| corner box to the end of the move's file, 6.62 s after the command; includes 0.92 m of walk after the turn; root: advance along the incoming line | largest 1.26 m (0.71 BL), at the move's end 1.23 m (0.70 BL) |
| corner box to the end of the move's file; root: sideways | largest 2.08 m (1.18 BL), at the move's end 2.08 m (1.18 BL) |
| corner box to the end of the move's file; whole robot: along | -0.96 m (-0.55 BL) to 1.50 m (0.85 BL) |
| corner box to the end of the move's file; whole robot: sideways | -0.40 m (-0.23 BL) to 2.87 m (1.63 BL) |
| swept radius over the steady part of the turn (the turning circle uses this one) | 1.56 m (0.89 BL), set by tail: cover_tail_seg_20 |
| largest distance from the turn's centre, from the command until the trunk has turned 90° | 1.57 m (0.89 BL), set by tail: cover_tail_seg_20, 4.16 s after the command |
| turning circle (twice the swept radius of the steady part) | 3.13 m (1.78 BL) |
| corridor for the 90° corner | 0.80 m (0.46 BL), inner corner set by left foot: hw_left_claw_1 |
| turned 4.48799 s after the command | 89.7° |
| turned at the end of the move (6.62 s after the command) | 90.0° |

Farthest point from the turn's centre, by part (planned): tail 1.56 m; head 1.44 m; trunk 1.37 m; right leg 1.32 m; right foot 1.31 m; left leg 1.19 m; left foot 1.13 m.

### Simulated

**`F25F9cl/75-M-H-10/seeds3042/eval/R2/42@10ms/turn/t90_0.35_left.json`: controller row R2/42@10ms, servo delay 10 ms, floor grip 0.9, block seeds3042.**

0 of 20 episodes stayed up; 20 fell: 20 before the command (seeds 3042-3061, between 4.57 and 5.69 s).

No episode of this record stayed up with a trace that can be read: nothing is read.

**`F25F9cl/75-M-H-10/seeds3042/eval/R2/43@10ms/turn/t90_0.35_left.json`: controller row R2/43@10ms, servo delay 10 ms, floor grip 0.9, block seeds3042.**

0 of 20 episodes stayed up; 20 fell: 20 before the command (seeds 3042-3061, between 4.55 and 6.74 s).

No episode of this record stayed up with a trace that can be read: nothing is read.

## turn/t90_0.35_right: 90° turn to the right (0.35 rad/s) at 0.40 m/s

![top view](turn_t90_0.35_right.png)

Top view of 90° turn to the right (0.35 rad/s) at 0.40 m/s on F25F9cl (the candidate); up is the incoming walking line. Blue, thick: the PLANNED path of the root (the frozen reference, kinematic), from 1 s before the command to the end of the move's file; the grey shapes are the PLANNED outline of the robot at the command and at the end of the move's file. The dashed box is the turn itself: the PLANNED root is 1.17 m along its incoming line and 1.16 m sideways (the ring) when the trunk has turned 90°, 4.47 s after the command. The dotted box is the PLANNED root's corner box to the end of the move's file, 7.10 s after the command: 1.26 m along and 2.28 m sideways (1.12 m of it after the turn); its sideways side is mostly the walk after the turn and depends on the file's length. SIMULATED: no episode of `F25F9cl/75-M-H-10/seeds3042/eval/R2/42@10ms/turn/t90_0.35_right.json` stayed up with a trace that can be read; none is drawn. Every path is drawn from its own position at the command. The scale bar is 0.5 m.

### Planned

From `transitions/turn90_0.35_right.npz` (sha256 `36cdc3f7e13c`), played at 1.05 times its own clock.

| reading (planned) | value |
|---|---|
| path speed commanded | 0.40 m/s on the reference's clock; 0.42 m/s on the bar's clock (the reference is played 1.05 times faster) |
| turn rate commanded | 20.1 deg/s on the reference's clock; 21.1 deg/s on the bar's clock |
| path radius commanded (speed over rate) | 1.14 m (0.65 BL) |
| radius fitted to the root's path, steady part | 1.13 m (0.64 BL), RMS 33 mm off the circle (the root sways) |
| **the turn itself**: until the trunk has turned 90° (to within 0.5°) | 4.47 s after the command |
| the turn itself, root: where it is then | 1.17 m (0.66 BL) along the incoming line, 1.16 m (0.66 BL) sideways |
| the turn itself, whole robot: along | -0.96 m (-0.55 BL) to 1.50 m (0.85 BL) |
| the turn itself, whole robot: sideways | -0.40 m (-0.23 BL) to 1.95 m (1.11 BL) |
| corner box to the end of the move's file, 7.10 s after the command; includes 1.12 m of walk after the turn; root: advance along the incoming line | largest 1.26 m (0.71 BL), at the move's end 1.20 m (0.68 BL) |
| corner box to the end of the move's file; root: sideways | largest 2.28 m (1.29 BL), at the move's end 2.28 m (1.29 BL) |
| corner box to the end of the move's file; whole robot: along | -0.96 m (-0.55 BL) to 1.50 m (0.85 BL) |
| corner box to the end of the move's file; whole robot: sideways | -0.40 m (-0.23 BL) to 3.07 m (1.75 BL) |
| swept radius over the steady part of the turn (the turning circle uses this one) | 1.56 m (0.89 BL), set by tail: cover_tail_seg_20 |
| largest distance from the turn's centre, from the command until the trunk has turned 90° | 1.57 m (0.89 BL), set by tail: cover_tail_seg_20, 4.15 s after the command |
| turning circle (twice the swept radius of the steady part) | 3.13 m (1.78 BL) |
| corridor for the 90° corner | 0.80 m (0.46 BL), inner corner set by right foot: hw_right_toe_2 |
| turned 4.48799 s after the command | 89.7° |
| turned at the end of the move (7.10 s after the command) | 90.0° |

Farthest point from the turn's centre, by part (planned): tail 1.56 m; head 1.44 m; trunk 1.37 m; left leg 1.32 m; left foot 1.31 m; right leg 1.19 m; right foot 1.13 m.

### Simulated

**`F25F9cl/75-M-H-10/seeds3042/eval/R2/42@10ms/turn/t90_0.35_right.json`: controller row R2/42@10ms, servo delay 10 ms, floor grip 0.9, block seeds3042.**

0 of 20 episodes stayed up; 20 fell: 20 before the command (seeds 3042-3061, between 4.57 and 5.69 s).

No episode of this record stayed up with a trace that can be read: nothing is read.

**`F25F9cl/75-M-H-10/seeds3042/eval/R2/43@10ms/turn/t90_0.35_right.json`: controller row R2/43@10ms, servo delay 10 ms, floor grip 0.9, block seeds3042.**

0 of 20 episodes stayed up; 20 fell: 20 before the command (seeds 3042-3061, between 4.55 and 6.74 s).

No episode of this record stayed up with a trace that can be read: nothing is read.

## turn/t90_0.5_left: 90° turn to the left (0.5 rad/s) at 0.40 m/s

![top view](turn_t90_0.5_left.png)

Top view of 90° turn to the left (0.5 rad/s) at 0.40 m/s on F25F9cl (the candidate); up is the incoming walking line. Blue, thick: the PLANNED path of the root (the frozen reference, kinematic), from 1 s before the command to the end of the move's file; the grey shapes are the PLANNED outline of the robot at the command and at the end of the move's file. The dashed box is the turn itself: the PLANNED root is 0.84 m along its incoming line and 0.85 m sideways (the ring) when the trunk has turned 90°, 3.22 s after the command. The dotted box is the PLANNED root's corner box to the end of the move's file, 5.52 s after the command: 0.91 m along and 1.81 m sideways (0.96 m of it after the turn); its sideways side is mostly the walk after the turn and depends on the file's length. SIMULATED: no episode of `F25F9cl/75-M-H-10/seeds3042/eval/R2/42@10ms/turn/t90_0.5_left.json` stayed up with a trace that can be read; none is drawn. Every path is drawn from its own position at the command. The scale bar is 0.5 m.

### Planned

From `transitions/turn90_0.5_left.npz` (sha256 `770ddc5536c0`), played at 1.05 times its own clock.

| reading (planned) | value |
|---|---|
| path speed commanded | 0.40 m/s on the reference's clock; 0.42 m/s on the bar's clock (the reference is played 1.05 times faster) |
| turn rate commanded | 28.6 deg/s on the reference's clock; 30.1 deg/s on the bar's clock |
| path radius commanded (speed over rate) | 0.80 m (0.45 BL) |
| radius fitted to the root's path, steady part | 0.78 m (0.45 BL), RMS 30 mm off the circle (the root sways) |
| **the turn itself**: until the trunk has turned 90° (to within 0.5°) | 3.22 s after the command |
| the turn itself, root: where it is then | 0.84 m (0.48 BL) along the incoming line, 0.85 m (0.48 BL) sideways |
| the turn itself, whole robot: along | -0.96 m (-0.55 BL) to 1.23 m (0.70 BL) |
| the turn itself, whole robot: sideways | -0.47 m (-0.27 BL) to 1.64 m (0.93 BL) |
| corner box to the end of the move's file, 5.52 s after the command; includes 0.96 m of walk after the turn; root: advance along the incoming line | largest 0.91 m (0.52 BL), at the move's end 0.89 m (0.50 BL) |
| corner box to the end of the move's file; root: sideways | largest 1.81 m (1.03 BL), at the move's end 1.81 m (1.03 BL) |
| corner box to the end of the move's file; whole robot: along | -0.96 m (-0.55 BL) to 1.23 m (0.70 BL) |
| corner box to the end of the move's file; whole robot: sideways | -0.47 m (-0.27 BL) to 2.60 m (1.48 BL) |
| swept radius over the steady part of the turn (the turning circle uses this one) | 1.30 m (0.74 BL), set by tail: cover_tail_seg_20 |
| largest distance from the turn's centre, from the command until the trunk has turned 90° | 1.31 m (0.74 BL), set by tail: cover_tail_seg_20, 0.34 s after the command |
| turning circle (twice the swept radius of the steady part) | 2.61 m (1.48 BL) |
| corridor for the 90° corner | 0.80 m (0.46 BL), inner corner set by left leg: cover_left_thigh_upper |
| turned 3.14159 s after the command | 88.6° |
| turned at the end of the move (5.52 s after the command) | 90.0° |

Farthest point from the turn's centre, by part (planned): tail 1.30 m; head 1.16 m; trunk 1.05 m; right leg 0.97 m; right foot 0.97 m; left leg 0.84 m; left foot 0.79 m.

### Simulated

**`F25F9cl/75-M-H-10/seeds3042/eval/R2/42@10ms/turn/t90_0.5_left.json`: controller row R2/42@10ms, servo delay 10 ms, floor grip 0.9, block seeds3042.**

0 of 20 episodes stayed up; 20 fell: 20 before the command (seeds 3042-3061, between 4.57 and 5.69 s).

No episode of this record stayed up with a trace that can be read: nothing is read.

**`F25F9cl/75-M-H-10/seeds3042/eval/R2/43@10ms/turn/t90_0.5_left.json`: controller row R2/43@10ms, servo delay 10 ms, floor grip 0.9, block seeds3042.**

0 of 20 episodes stayed up; 20 fell: 20 before the command (seeds 3042-3061, between 4.55 and 6.74 s).

No episode of this record stayed up with a trace that can be read: nothing is read.

## turn/t90_0.5_right: 90° turn to the right (0.5 rad/s) at 0.40 m/s

![top view](turn_t90_0.5_right.png)

Top view of 90° turn to the right (0.5 rad/s) at 0.40 m/s on F25F9cl (the candidate); up is the incoming walking line. Blue, thick: the PLANNED path of the root (the frozen reference, kinematic), from 1 s before the command to the end of the move's file; the grey shapes are the PLANNED outline of the robot at the command and at the end of the move's file. The dashed box is the turn itself: the PLANNED root is 0.84 m along its incoming line and 0.84 m sideways (the ring) when the trunk has turned 90°, 3.21 s after the command. The dotted box is the PLANNED root's corner box to the end of the move's file, 5.99 s after the command: 0.91 m along and 2.01 m sideways (1.17 m of it after the turn); its sideways side is mostly the walk after the turn and depends on the file's length. SIMULATED: no episode of `F25F9cl/75-M-H-10/seeds3042/eval/R2/42@10ms/turn/t90_0.5_right.json` stayed up with a trace that can be read; none is drawn. Every path is drawn from its own position at the command. The scale bar is 0.5 m.

### Planned

From `transitions/turn90_0.5_right.npz` (sha256 `2a899037256b`), played at 1.05 times its own clock.

| reading (planned) | value |
|---|---|
| path speed commanded | 0.40 m/s on the reference's clock; 0.42 m/s on the bar's clock (the reference is played 1.05 times faster) |
| turn rate commanded | 28.6 deg/s on the reference's clock; 30.1 deg/s on the bar's clock |
| path radius commanded (speed over rate) | 0.80 m (0.45 BL) |
| radius fitted to the root's path, steady part | 0.78 m (0.45 BL), RMS 31 mm off the circle (the root sways) |
| **the turn itself**: until the trunk has turned 90° (to within 0.5°) | 3.21 s after the command |
| the turn itself, root: where it is then | 0.84 m (0.48 BL) along the incoming line, 0.84 m (0.48 BL) sideways |
| the turn itself, whole robot: along | -0.96 m (-0.55 BL) to 1.23 m (0.70 BL) |
| the turn itself, whole robot: sideways | -0.47 m (-0.27 BL) to 1.63 m (0.93 BL) |
| corner box to the end of the move's file, 5.99 s after the command; includes 1.17 m of walk after the turn; root: advance along the incoming line | largest 0.91 m (0.52 BL), at the move's end 0.85 m (0.49 BL) |
| corner box to the end of the move's file; root: sideways | largest 2.01 m (1.14 BL), at the move's end 2.01 m (1.14 BL) |
| corner box to the end of the move's file; whole robot: along | -0.96 m (-0.55 BL) to 1.23 m (0.70 BL) |
| corner box to the end of the move's file; whole robot: sideways | -0.47 m (-0.27 BL) to 2.80 m (1.59 BL) |
| swept radius over the steady part of the turn (the turning circle uses this one) | 1.30 m (0.74 BL), set by tail: cover_tail_seg_20 |
| largest distance from the turn's centre, from the command until the trunk has turned 90° | 1.31 m (0.74 BL), set by tail: cover_tail_seg_20, 3.05 s after the command |
| turning circle (twice the swept radius of the steady part) | 2.61 m (1.48 BL) |
| corridor for the 90° corner | 0.80 m (0.46 BL), inner corner set by right leg: cover_right_thigh_upper |
| turned 3.14159 s after the command | 88.6° |
| turned at the end of the move (5.99 s after the command) | 90.0° |

Farthest point from the turn's centre, by part (planned): tail 1.30 m; head 1.16 m; trunk 1.05 m; left leg 0.98 m; left foot 0.97 m; right leg 0.84 m; right foot 0.79 m.

### Simulated

**`F25F9cl/75-M-H-10/seeds3042/eval/R2/42@10ms/turn/t90_0.5_right.json`: controller row R2/42@10ms, servo delay 10 ms, floor grip 0.9, block seeds3042.**

0 of 20 episodes stayed up; 20 fell: 20 before the command (seeds 3042-3061, between 4.57 and 5.69 s).

No episode of this record stayed up with a trace that can be read: nothing is read.

**`F25F9cl/75-M-H-10/seeds3042/eval/R2/43@10ms/turn/t90_0.5_right.json`: controller row R2/43@10ms, servo delay 10 ms, floor grip 0.9, block seeds3042.**

0 of 20 episodes stayed up; 20 fell: 20 before the command (seeds 3042-3061, between 4.55 and 6.74 s).

No episode of this record stayed up with a trace that can be read: nothing is read.

## Turns without a reference in this set

- turn/t90_1_left (`transitions/turn90_1_left`): no reference within the thigh twist and the leg drives: no stride factor of the grid passes the kinematic rule (thigh twist at most 0.85 of its range, every other gated joint at least 0.05 rad of range, ik_ok, in both directions)
- turn/t90_1_right (`transitions/turn90_1_right`): no reference within the thigh twist and the leg drives: no stride factor of the grid passes the kinematic rule (thigh twist at most 0.85 of its range, every other gated joint at least 0.05 rad of range, ik_ok, in both directions)

The run directory holds these records of them (no episode to read):

- `F25F9cl/75-M-H-10/seeds3042/eval/R2/42@10ms/turn/t90_1_left.json` (row R2/42@10ms, block seeds3042): status 'not run': no reference within the thigh twist and the leg drives: no stride factor of the grid passes the kinematic rule (thigh twist at most 0.85 of its range, every other gated joint at least 0.05 rad of range, ik_ok, in both directions)
- `F25F9cl/75-M-H-10/seeds3042/eval/R2/42@10ms/turn/t90_1_right.json` (row R2/42@10ms, block seeds3042): status 'not run': no reference within the thigh twist and the leg drives: no stride factor of the grid passes the kinematic rule (thigh twist at most 0.85 of its range, every other gated joint at least 0.05 rad of range, ik_ok, in both directions)
- `F25F9cl/75-M-H-10/seeds3042/eval/R2/43@10ms/turn/t90_1_left.json` (row R2/43@10ms, block seeds3042): status 'not run': no reference within the thigh twist and the leg drives: no stride factor of the grid passes the kinematic rule (thigh twist at most 0.85 of its range, every other gated joint at least 0.05 rad of range, ik_ok, in both directions)
- `F25F9cl/75-M-H-10/seeds3042/eval/R2/43@10ms/turn/t90_1_right.json` (row R2/43@10ms, block seeds3042): status 'not run': no reference within the thigh twist and the leg drives: no stride factor of the grid passes the kinematic rule (thigh twist at most 0.85 of its range, every other gated joint at least 0.05 rad of range, ik_ok, in both directions)

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
