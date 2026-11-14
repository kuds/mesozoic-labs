# Stage 2 look: kinematic clips, plots and machine metrics

Kinematic references, not simulations: each rung's reference walk played back with perfect tracking and no physics (`designlab/stage2/clips.py`), and read on its frames (`designlab/stage2/motion.py`); this page is rendered from `metrics.json` and `clips.json` alone (`python -m designlab.stage2.motion md`). The media (MP4, PNG) are outside git; `clips.json` names each file with its size and sha256. Nothing is adopted (P24).

## Clips

42 clips, 10 s each at 30 fps, 854x480 H.264, each under 1.5 MB, every frame labelled "kinematic reference, not a simulation". Cameras: visitor: eye 1.5 m high, 2.5 m from the look-at point along the floor, bearing -45 deg from the walking direction, field of view 26 deg; side: eye 0.6 m high, 2.5 m from the look-at point along the floor, bearing -90 deg from the walking direction, field of view 26 deg; both follow the body's mean forward motion. Point-light: snout, eye, neck_base, hip_l, hip_r, knee_l, knee_r, mtp_l, mtp_r, toe_l, toe_r, tail_mid, tail_tip on black. Render and encode time 41.3 CPU-min (20.9 min wall).

| Body | Speed | Rung | View | Mode | File | MB |
|---|---|---|---|---|---|---|
| 60-N-T-10 | fr05 | V0 | side | covered | `60-N-T-10/60-N-T-10_fr05_V0_side.mp4` | 0.69 |
| 60-N-T-10 | fr05 | V0 | visitor | covered | `60-N-T-10/60-N-T-10_fr05_V0_visitor.mp4` | 1.05 |
| 60-N-T-10 | fr05 | V6 | side | covered | `60-N-T-10/60-N-T-10_fr05_V6_side.mp4` | 0.70 |
| 60-N-T-10 | fr05 | V6 | visitor | covered | `60-N-T-10/60-N-T-10_fr05_V6_visitor.mp4` | 1.04 |
| 60-N-T-10 | nominal | V0 | side | covered | `60-N-T-10/60-N-T-10_nominal_V0_side.mp4` | 0.60 |
| 60-N-T-10 | nominal | V0 | visitor | covered | `60-N-T-10/60-N-T-10_nominal_V0_visitor.mp4` | 0.90 |
| 60-N-T-10 | nominal | V6 | side | covered | `60-N-T-10/60-N-T-10_nominal_V6_side.mp4` | 0.59 |
| 60-N-T-10 | nominal | V6 | visitor | covered | `60-N-T-10/60-N-T-10_nominal_V6_visitor.mp4` | 0.89 |
| 60-N-T-10 | v040 | V0 | side | covered | `60-N-T-10/60-N-T-10_v040_V0_side.mp4` | 0.68 |
| 60-N-T-10 | v040 | V0 | side | pointlight | `60-N-T-10/60-N-T-10_v040_V0_side_pointlight.mp4` | 0.14 |
| 60-N-T-10 | v040 | V0 | visitor | covered | `60-N-T-10/60-N-T-10_v040_V0_visitor.mp4` | 1.03 |
| 60-N-T-10 | v040 | V0 | visitor | pointlight | `60-N-T-10/60-N-T-10_v040_V0_visitor_pointlight.mp4` | 0.14 |
| 60-N-T-10 | v040 | V1 | visitor | covered | `60-N-T-10/60-N-T-10_v040_V1_visitor.mp4` | 1.03 |
| 60-N-T-10 | v040 | V2 | visitor | covered | `60-N-T-10/60-N-T-10_v040_V2_visitor.mp4` | 1.03 |
| 60-N-T-10 | v040 | V3 | visitor | covered | `60-N-T-10/60-N-T-10_v040_V3_visitor.mp4` | 1.03 |
| 60-N-T-10 | v040 | V4 | visitor | covered | `60-N-T-10/60-N-T-10_v040_V4_visitor.mp4` | 1.03 |
| 60-N-T-10 | v040 | V5 | visitor | covered | `60-N-T-10/60-N-T-10_v040_V5_visitor.mp4` | 1.02 |
| 60-N-T-10 | v040 | V6 | side | covered | `60-N-T-10/60-N-T-10_v040_V6_side.mp4` | 0.68 |
| 60-N-T-10 | v040 | V6 | side | pointlight | `60-N-T-10/60-N-T-10_v040_V6_side_pointlight.mp4` | 0.14 |
| 60-N-T-10 | v040 | V6 | visitor | covered | `60-N-T-10/60-N-T-10_v040_V6_visitor.mp4` | 1.02 |
| 60-N-T-10 | v040 | V6 | visitor | pointlight | `60-N-T-10/60-N-T-10_v040_V6_visitor_pointlight.mp4` | 0.14 |
| 75-M-H-10 | fr05 | V0 | side | covered | `75-M-H-10/75-M-H-10_fr05_V0_side.mp4` | 0.92 |
| 75-M-H-10 | fr05 | V0 | visitor | covered | `75-M-H-10/75-M-H-10_fr05_V0_visitor.mp4` | 1.28 |
| 75-M-H-10 | fr05 | V6 | side | covered | `75-M-H-10/75-M-H-10_fr05_V6_side.mp4` | 0.95 |
| 75-M-H-10 | fr05 | V6 | visitor | covered | `75-M-H-10/75-M-H-10_fr05_V6_visitor.mp4` | 1.28 |
| 75-M-H-10 | nominal | V0 | side | covered | `75-M-H-10/75-M-H-10_nominal_V0_side.mp4` | 0.82 |
| 75-M-H-10 | nominal | V0 | visitor | covered | `75-M-H-10/75-M-H-10_nominal_V0_visitor.mp4` | 1.14 |
| 75-M-H-10 | nominal | V6 | side | covered | `75-M-H-10/75-M-H-10_nominal_V6_side.mp4` | 0.82 |
| 75-M-H-10 | nominal | V6 | visitor | covered | `75-M-H-10/75-M-H-10_nominal_V6_visitor.mp4` | 1.13 |
| 75-M-H-10 | v040 | V0 | side | covered | `75-M-H-10/75-M-H-10_v040_V0_side.mp4` | 0.89 |
| 75-M-H-10 | v040 | V0 | side | pointlight | `75-M-H-10/75-M-H-10_v040_V0_side_pointlight.mp4` | 0.14 |
| 75-M-H-10 | v040 | V0 | visitor | covered | `75-M-H-10/75-M-H-10_v040_V0_visitor.mp4` | 1.24 |
| 75-M-H-10 | v040 | V0 | visitor | pointlight | `75-M-H-10/75-M-H-10_v040_V0_visitor_pointlight.mp4` | 0.14 |
| 75-M-H-10 | v040 | V1 | visitor | covered | `75-M-H-10/75-M-H-10_v040_V1_visitor.mp4` | 1.24 |
| 75-M-H-10 | v040 | V2 | visitor | covered | `75-M-H-10/75-M-H-10_v040_V2_visitor.mp4` | 1.24 |
| 75-M-H-10 | v040 | V3 | visitor | covered | `75-M-H-10/75-M-H-10_v040_V3_visitor.mp4` | 1.24 |
| 75-M-H-10 | v040 | V4 | visitor | covered | `75-M-H-10/75-M-H-10_v040_V4_visitor.mp4` | 1.25 |
| 75-M-H-10 | v040 | V5 | visitor | covered | `75-M-H-10/75-M-H-10_v040_V5_visitor.mp4` | 1.24 |
| 75-M-H-10 | v040 | V6 | side | covered | `75-M-H-10/75-M-H-10_v040_V6_side.mp4` | 0.90 |
| 75-M-H-10 | v040 | V6 | side | pointlight | `75-M-H-10/75-M-H-10_v040_V6_side_pointlight.mp4` | 0.14 |
| 75-M-H-10 | v040 | V6 | visitor | covered | `75-M-H-10/75-M-H-10_v040_V6_visitor.mp4` | 1.24 |
| 75-M-H-10 | v040 | V6 | visitor | pointlight | `75-M-H-10/75-M-H-10_v040_V6_visitor_pointlight.mp4` | 0.14 |

## Readings to note

- F2's crest hold: on V1-V6 the hips stay within 5 % of the bob's range of its crest for 0.42 to 0.49 of the window (a sinusoidal bob: 0.144), so the hip and snout paths run level for a long part of each step (the path plots' straight segments, research 4.1 item 3).
- F5's toe-off at lift-off, V3-V6, the nominal walk (Froude 0.012): 0.300 rad (the research's 0.3-0.5).
- F5's toe-off at lift-off, V3-V6, 0.40 m/s (the agility bar's speed): 0.031 to 0.056 rad (the research's 0.3-0.5).
- F5's toe-off at lift-off, V3-V6, the animal pace (Froude 0.05): 0.020 to 0.084 rad (the research's 0.3-0.5).
- The head's stabilisation ratio (head over trunk angular rate) on V5-V6 is 1.36 to 1.52: the trunk's only rotation is F6's roll, which the head follows (no neck roll), and the trunk never pitches (largest pitch RMS 0.00 deg; F8), so F10's pitch stabilisation shows only as the nod removed: head pitch in the world 2.03 deg RMS on V0, 0.24 to 0.28 on V5, 0.53 to 0.55 on V6 (F14's gaze drift).
- Feet's SPARC from V2 to V3 (F4/F5), the nominal walk (Froude 0.012): -0.17 to -0.16 by the published reading, -0.05 to -0.04 over the fixed band (a sidelobe near 9 Hz at about the 5 % threshold moves the published reading by steps).
- Feet's SPARC from V2 to V3 (F4/F5), 0.40 m/s (the agility bar's speed): -0.26 to -0.23 by the published reading, -0.03 to -0.02 over the fixed band (a sidelobe near 9 Hz at about the 5 % threshold moves the published reading by steps).
- Feet's SPARC from V2 to V3 (F4/F5), the animal pace (Froude 0.05): 0.00 by the published reading, -0.02 over the fixed band (a sidelobe near 9 Hz at about the 5 % threshold moves the published reading by steps).
- Head's periodic SPARC, V5 to V6: 60-N-T-10 nominal -4.79 to -5.05; 60-N-T-10 v040 -5.40 to -6.08; 60-N-T-10 fr05 -5.62 to -6.36; 75-M-H-10 nominal -4.78 to -5.08; 75-M-H-10 v040 -5.45 to -5.87; 75-M-H-10 fr05 -5.90 to -6.29 (F14's slow drifts count as spectral complexity).
- 60-N-T-10 at the animal pace: cadence 2.75 steps/s on V0 and 2.75 on V6, against a dynamically similar T. rex's 1.83; stride 0.80 and 0.80 H (T. rex 1.25 H); sway 0.103 and 0.103 H.
- 75-M-H-10 at the animal pace: cadence 2.47 steps/s on V0 and 2.26 on V6, against a dynamically similar T. rex's 1.64; stride 0.80 and 0.88 H (T. rex 1.25 H); sway 0.101 and 0.115 H.
- Over a Stage 0 limit (labelled on its clips): 75-M-H-10 fr05 V6 (knee speed 1.012).

## Machine metrics (research 4.1 item 5)

Each rung's readings on its frames (definitions: `metrics.json` `method`). 0.40 m/s: the gait the brisk_to_fr05 speed change passes through at 0.40 m/s: period, duty and lift mixed with the speed's weight (transitions.py).

### 60-N-T-10 (60 cm Tapered), the nominal walk (Froude 0.012)

| Reading | V0 | V1 | V2 | V3 | V4 | V5 | V6 |
|---|---|---|---|---|---|---|---|
| hip bob, peak to peak (mm) | 0.0 | 4.3 | 4.3 | 4.3 | 4.3 | 4.3 | 4.3 |
| bob crest hold (share; a sinusoid's 0.144) | - | 0.475 | 0.425 | 0.425 | 0.425 | 0.425 | 0.423 |
| trunk roll RMS (deg) | 0.00 | 0.00 | 0.00 | 0.00 | 1.06 | 1.06 | 1.06 |
| trunk pitch RMS (deg) | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| sway, CoM per stride (H; spec 0.12) | 0.168 | 0.167 | 0.161 | 0.162 | 0.162 | 0.162 | 0.163 |
| head pitch in the world, RMS (deg; band 1.5) | 2.03 | 2.03 | 2.03 | 2.03 | 2.03 | 0.25 | 0.53 |
| head angular rate, RMS (deg/s) | 24.9 | 24.9 | 24.9 | 24.9 | 25.8 | 9.6 | 9.6 |
| stabilisation ratio, head / trunk | - | - | - | - | 3.95 | 1.47 | 1.48 |
| tail phase from today's (cycles) | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.479 | 0.479 |
| tail tip toward the stance side in swing (share) | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 1.00 | 1.00 |
| toe-off at lift-off (rad) | -0.000 | -0.000 | 0.000 | 0.300 | 0.300 | 0.300 | 0.300 |
| swing time (s) | 0.408 | 0.408 | 0.320 | 0.318 | 0.318 | 0.318 | 0.318 |
| duty (left) | 0.600 | 0.600 | 0.686 | 0.688 | 0.688 | 0.688 | 0.683 |
| stride (H) | 0.550 | 0.550 | 0.550 | 0.550 | 0.550 | 0.550 | 0.550 |
| cadence (steps/s) | 1.96 | 1.96 | 1.96 | 1.96 | 1.96 | 1.96 | 1.96 |
| step-time CV (%) | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.85 |
| SPARC, toe tip per swing (mean) | -1.52 | -1.52 | -1.52 | -1.67 | -1.67 | -1.67 | -1.67 |
| SPARC, toe tip, fixed band to 10 Hz | -1.71 | -1.71 | -1.65 | -1.69 | -1.69 | -1.69 | -1.68 |
| SPARC, head (periodic) | -4.86 | -4.86 | -5.05 | -5.31 | -5.17 | -4.79 | -5.05 |
| SPARC, head, fixed band to 10 Hz | -5.13 | -5.30 | -5.52 | -5.36 | -5.22 | -4.88 | -5.14 |
| stance-pad peak yaw moment (N m) | 1.77 | 1.70 | 2.85 | 2.06 | 2.06 | 1.72 | 1.81 |
| Stage 0 | within | within | within | within | within | within | within |

### 60-N-T-10 (60 cm Tapered), 0.40 m/s (the agility bar's speed)

| Reading | V0 | V1 | V2 | V3 | V4 | V5 | V6 |
|---|---|---|---|---|---|---|---|
| hip bob, peak to peak (mm) | 0.0 | 6.9 | 6.9 | 6.9 | 6.9 | 6.9 | 6.9 |
| bob crest hold (share; a sinusoid's 0.144) | - | 0.438 | 0.438 | 0.438 | 0.438 | 0.438 | 0.435 |
| trunk roll RMS (deg) | 0.00 | 0.00 | 0.00 | 0.00 | 1.06 | 1.06 | 1.06 |
| trunk pitch RMS (deg) | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| sway, CoM per stride (H; spec 0.12) | 0.108 | 0.107 | 0.107 | 0.107 | 0.107 | 0.107 | 0.108 |
| head pitch in the world, RMS (deg; band 1.5) | 2.03 | 2.03 | 2.03 | 2.03 | 2.03 | 0.28 | 0.55 |
| head angular rate, RMS (deg/s) | 33.9 | 33.9 | 33.9 | 33.9 | 35.0 | 12.1 | 12.2 |
| stabilisation ratio, head / trunk | - | - | - | - | 3.95 | 1.37 | 1.37 |
| tail phase from today's (cycles) | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.467 | 0.467 |
| tail tip toward the stance side in swing (share) | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 1.00 | 1.00 |
| toe-off at lift-off (rad) | 0.000 | -0.000 | -0.000 | 0.032 | 0.032 | 0.032 | 0.031 |
| swing time (s) | 0.325 | 0.325 | 0.320 | 0.319 | 0.319 | 0.319 | 0.318 |
| duty (left) | 0.567 | 0.567 | 0.574 | 0.576 | 0.576 | 0.576 | 0.569 |
| stride (H) | 0.742 | 0.742 | 0.742 | 0.742 | 0.742 | 0.742 | 0.742 |
| cadence (steps/s) | 2.66 | 2.66 | 2.66 | 2.66 | 2.66 | 2.66 | 2.66 |
| step-time CV (%) | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 1.50 |
| SPARC, toe tip per swing (mean) | -1.40 | -1.40 | -1.40 | -1.63 | -1.63 | -1.63 | -1.63 |
| SPARC, toe tip, fixed band to 10 Hz | -1.64 | -1.64 | -1.64 | -1.66 | -1.66 | -1.66 | -1.66 |
| SPARC, head (periodic) | -5.54 | -5.91 | -5.85 | -5.94 | -5.92 | -5.40 | -6.08 |
| SPARC, head, fixed band to 10 Hz | -5.70 | -6.02 | -5.95 | -6.06 | -6.04 | -5.56 | -6.37 |
| stance-pad peak yaw moment (N m) | 3.74 | 3.50 | 3.63 | 4.63 | 4.62 | 4.11 | 4.16 |
| Stage 0 | within | within | within | within | within | within | within |

### 60-N-T-10 (60 cm Tapered), the animal pace (Froude 0.05)

| Reading | V0 | V1 | V2 | V3 | V4 | V5 | V6 |
|---|---|---|---|---|---|---|---|
| hip bob, peak to peak (mm) | 0.0 | 7.9 | 7.9 | 7.9 | 7.8 | 7.8 | 7.8 |
| bob crest hold (share; a sinusoid's 0.144) | - | 0.438 | 0.438 | 0.438 | 0.438 | 0.438 | 0.435 |
| trunk roll RMS (deg) | 0.00 | 0.00 | 0.00 | 0.00 | 1.06 | 1.06 | 1.06 |
| trunk pitch RMS (deg) | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| sway, CoM per stride (H; spec 0.12) | 0.103 | 0.101 | 0.101 | 0.101 | 0.101 | 0.101 | 0.103 |
| head pitch in the world, RMS (deg; band 1.5) | 2.03 | 2.03 | 2.03 | 2.03 | 2.03 | 0.28 | 0.55 |
| head angular rate, RMS (deg/s) | 35.0 | 35.0 | 35.0 | 35.0 | 36.2 | 12.5 | 12.5 |
| stabilisation ratio, head / trunk | - | - | - | - | 3.95 | 1.36 | 1.36 |
| tail phase from today's (cycles) | -0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.467 | 0.467 |
| tail tip toward the stance side in swing (share) | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 1.00 | 1.00 |
| toe-off at lift-off (rad) | 0.000 | -0.000 | -0.000 | 0.020 | 0.020 | 0.020 | 0.020 |
| swing time (s) | 0.320 | 0.320 | 0.320 | 0.319 | 0.319 | 0.319 | 0.318 |
| duty (left) | 0.560 | 0.560 | 0.560 | 0.562 | 0.562 | 0.562 | 0.555 |
| stride (H) | 0.800 | 0.800 | 0.800 | 0.800 | 0.800 | 0.800 | 0.800 |
| cadence (steps/s) | 2.75 | 2.75 | 2.75 | 2.75 | 2.75 | 2.75 | 2.75 |
| step-time CV (%) | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 1.51 |
| SPARC, toe tip per swing (mean) | -1.40 | -1.40 | -1.40 | -1.40 | -1.40 | -1.40 | -1.40 |
| SPARC, toe tip, fixed band to 10 Hz | -1.64 | -1.64 | -1.64 | -1.65 | -1.65 | -1.65 | -1.65 |
| SPARC, head (periodic) | -5.43 | -5.87 | -5.87 | -5.96 | -6.07 | -5.62 | -6.36 |
| SPARC, head, fixed band to 10 Hz | -5.59 | -5.98 | -5.98 | -6.11 | -6.21 | -5.78 | -6.65 |
| stance-pad peak yaw moment (N m) | 4.23 | 3.91 | 3.91 | 4.96 | 4.95 | 4.40 | 4.45 |
| Stage 0 | within | within | within | within | within | within | within |

### 75-M-H-10 (75 cm Hybrid), the nominal walk (Froude 0.012)

| Reading | V0 | V1 | V2 | V3 | V4 | V5 | V6 |
|---|---|---|---|---|---|---|---|
| hip bob, peak to peak (mm) | 0.0 | 5.3 | 5.3 | 5.3 | 5.3 | 5.3 | 5.3 |
| bob crest hold (share; a sinusoid's 0.144) | - | 0.487 | 0.438 | 0.438 | 0.438 | 0.438 | 0.440 |
| trunk roll RMS (deg) | 0.00 | 0.00 | 0.00 | 0.00 | 1.06 | 1.06 | 1.06 |
| trunk pitch RMS (deg) | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| sway, CoM per stride (H; spec 0.12) | 0.164 | 0.163 | 0.158 | 0.159 | 0.158 | 0.158 | 0.160 |
| head pitch in the world, RMS (deg; band 1.5) | 2.03 | 2.03 | 2.03 | 2.03 | 2.03 | 0.24 | 0.53 |
| head angular rate, RMS (deg/s) | 22.4 | 22.4 | 22.4 | 22.4 | 23.1 | 8.9 | 8.9 |
| stabilisation ratio, head / trunk | - | - | - | - | 3.95 | 1.52 | 1.52 |
| tail phase from today's (cycles) | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | -0.397 | -0.397 |
| tail tip toward the stance side in swing (share) | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 1.00 | 1.00 |
| toe-off at lift-off (rad) | -0.000 | -0.000 | 0.000 | 0.300 | 0.300 | 0.300 | 0.300 |
| swing time (s) | 0.454 | 0.454 | 0.356 | 0.354 | 0.354 | 0.354 | 0.354 |
| duty (left) | 0.600 | 0.600 | 0.686 | 0.688 | 0.688 | 0.688 | 0.683 |
| stride (H) | 0.550 | 0.550 | 0.550 | 0.550 | 0.550 | 0.550 | 0.550 |
| cadence (steps/s) | 1.76 | 1.76 | 1.76 | 1.76 | 1.76 | 1.76 | 1.76 |
| step-time CV (%) | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.84 |
| SPARC, toe tip per swing (mean) | -1.52 | -1.52 | -1.52 | -1.69 | -1.69 | -1.69 | -1.68 |
| SPARC, toe tip, fixed band to 10 Hz | -1.74 | -1.74 | -1.68 | -1.73 | -1.73 | -1.73 | -1.72 |
| SPARC, head (periodic) | -4.87 | -4.87 | -5.06 | -5.36 | -5.21 | -4.78 | -5.08 |
| SPARC, head, fixed band to 10 Hz | -5.14 | -5.35 | -5.57 | -5.45 | -5.30 | -4.93 | -5.31 |
| stance-pad peak yaw moment (N m) | 2.48 | 2.39 | 3.94 | 2.74 | 2.73 | 2.26 | 2.50 |
| Stage 0 | within | within | within | within | within | within | within |

### 75-M-H-10 (75 cm Hybrid), 0.40 m/s (the agility bar's speed)

| Reading | V0 | V1 | V2 | V3 | V4 | V5 | V6 |
|---|---|---|---|---|---|---|---|
| hip bob, peak to peak (mm) | 0.0 | 7.5 | 7.5 | 7.5 | 7.5 | 7.5 | 7.5 |
| bob crest hold (share; a sinusoid's 0.144) | - | 0.463 | 0.450 | 0.450 | 0.450 | 0.450 | 0.455 |
| trunk roll RMS (deg) | 0.00 | 0.00 | 0.00 | 0.00 | 1.06 | 1.06 | 1.06 |
| trunk pitch RMS (deg) | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| sway, CoM per stride (H; spec 0.12) | 0.111 | 0.110 | 0.110 | 0.110 | 0.110 | 0.110 | 0.111 |
| head pitch in the world, RMS (deg; band 1.5) | 2.03 | 2.03 | 2.03 | 2.03 | 2.03 | 0.26 | 0.54 |
| head angular rate, RMS (deg/s) | 29.5 | 29.5 | 29.5 | 29.5 | 30.5 | 10.9 | 10.9 |
| stabilisation ratio, head / trunk | - | - | - | - | 3.95 | 1.41 | 1.41 |
| tail phase from today's (cycles) | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.468 | 0.468 |
| tail tip toward the stance side in swing (share) | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 1.00 | 1.00 |
| toe-off at lift-off (rad) | 0.000 | -0.000 | -0.000 | 0.056 | 0.056 | 0.056 | 0.056 |
| swing time (s) | 0.367 | 0.367 | 0.356 | 0.356 | 0.356 | 0.356 | 0.356 |
| duty (left) | 0.573 | 0.573 | 0.586 | 0.587 | 0.587 | 0.587 | 0.579 |
| stride (H) | 0.686 | 0.686 | 0.686 | 0.686 | 0.686 | 0.686 | 0.686 |
| cadence (steps/s) | 2.32 | 2.32 | 2.32 | 2.32 | 2.32 | 2.32 | 2.32 |
| step-time CV (%) | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 1.46 |
| SPARC, toe tip per swing (mean) | -1.41 | -1.41 | -1.40 | -1.66 | -1.66 | -1.66 | -1.66 |
| SPARC, toe tip, fixed band to 10 Hz | -1.68 | -1.68 | -1.67 | -1.71 | -1.71 | -1.71 | -1.71 |
| SPARC, head (periodic) | -5.37 | -6.07 | -5.95 | -6.01 | -5.94 | -5.45 | -5.87 |
| SPARC, head, fixed band to 10 Hz | -5.71 | -6.14 | -6.02 | -6.08 | -6.01 | -5.55 | -6.00 |
| stance-pad peak yaw moment (N m) | 4.63 | 4.39 | 4.70 | 5.86 | 5.85 | 4.99 | 5.09 |
| Stage 0 | within | within | within | within | within | within | within |

### 75-M-H-10 (75 cm Hybrid), the animal pace (Froude 0.05)

| Reading | V0 | V1 | V2 | V3 | V4 | V5 | V6 | V5F13 |
|---|---|---|---|---|---|---|---|---|
| hip bob, peak to peak (mm) | 0.0 | 9.7 | 9.7 | 9.7 | 9.7 | 9.7 | 9.7 | 9.7 |
| bob crest hold (share; a sinusoid's 0.144) | - | 0.463 | 0.463 | 0.463 | 0.463 | 0.463 | 0.452 | 0.450 |
| trunk roll RMS (deg) | 0.00 | 0.00 | 0.00 | 0.00 | 1.06 | 1.06 | 1.06 | 1.06 |
| trunk pitch RMS (deg) | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| sway, CoM per stride (H; spec 0.12) | 0.101 | 0.100 | 0.100 | 0.100 | 0.099 | 0.100 | 0.115 | 0.114 |
| head pitch in the world, RMS (deg; band 1.5) | 2.03 | 2.03 | 2.03 | 2.03 | 2.03 | 0.27 | 0.54 | 0.26 |
| head angular rate, RMS (deg/s) | 31.4 | 31.4 | 31.4 | 31.4 | 32.5 | 11.5 | 10.7 | 10.7 |
| stabilisation ratio, head / trunk | - | - | - | - | 3.95 | 1.39 | 1.42 | 1.42 |
| tail phase from today's (cycles) | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.468 | 0.468 | 0.468 |
| tail tip toward the stance side in swing (share) | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 1.00 | 1.00 | 1.00 |
| toe-off at lift-off (rad) | 0.000 | -0.000 | -0.000 | 0.025 | 0.025 | 0.025 | 0.084 | 0.084 |
| swing time (s) | 0.356 | 0.356 | 0.356 | 0.355 | 0.355 | 0.355 | 0.356 | 0.355 |
| duty (left) | 0.560 | 0.560 | 0.560 | 0.562 | 0.562 | 0.562 | 0.592 | 0.598 |
| stride (H) | 0.800 | 0.800 | 0.800 | 0.800 | 0.800 | 0.800 | 0.875 | 0.875 |
| cadence (steps/s) | 2.47 | 2.47 | 2.47 | 2.47 | 2.47 | 2.47 | 2.26 | 2.26 |
| step-time CV (%) | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 1.42 | 0.00 |
| SPARC, toe tip per swing (mean) | -1.40 | -1.40 | -1.40 | -1.40 | -1.40 | -1.40 | -1.40 | -1.39 |
| SPARC, toe tip, fixed band to 10 Hz | -1.66 | -1.66 | -1.66 | -1.69 | -1.69 | -1.69 | -1.67 | -1.67 |
| SPARC, head (periodic) | -5.33 | -6.04 | -6.04 | -6.14 | -6.26 | -5.90 | -6.29 | -5.79 |
| SPARC, head, fixed band to 10 Hz | -5.52 | -6.13 | -6.13 | -6.23 | -6.36 | -5.96 | -6.47 | -5.92 |
| stance-pad peak yaw moment (N m) | 5.83 | 5.44 | 5.44 | 6.82 | 6.81 | 5.95 | 5.93 | 5.88 |
| Stage 0 | within | within | within | within | within | within | **over** | within |

