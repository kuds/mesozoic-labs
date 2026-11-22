# Renders for the December museum contact (task 9)

18 Nov 2026. **Every image is a simulation render**, labelled "SIMULATION" on the image. Nothing has been built; no
image shows hardware. Made ahead of the 8 Dec 2026 target, so they can be redone if the build decisions change.

## Roadmap line (for the pitch)

> The Triceratops is the near-term live demonstration (Q3 2027); the T. rex is the flagship, coming at end 2027 /
> early 2028.

## At 60 cm (22 Nov 2026; the build chosen on 18 Nov, D2)

**Every image and clip is a simulation render**, labelled "SIMULATION" on the picture itself.

Stills ([stills_60cm_2026-11-22/](stills_60cm_2026-11-22/), JPEG, each under 500 KB):

| File | Shows | Caption |
|---|---|---|
| `robot_three_quarter.jpg` | The 60 cm robot standing, three-quarter view | Triceratops robot, 60 cm to the frill: simulation. MuJoCo model of the proposed 60 cm robot, 12.7 kg design, standing. Not hardware; nothing is built yet. |
| `robot_side.jpg` | The same, side view | as above |
| `robot_head.jpg` | Head, frill and horns, close | as above |
| `robot_trot_frame.jpg` | The model-based trot at 0.25 m/s, 8.0 s into the episode | Triceratops robot, 60 cm: trotting at the museum speed (simulation). Design variant with passive roll ankles; 10 ms servo delay and sensor noise simulated. Not hardware. |

The adult reconstruction stills below are unchanged (the animal does not depend on the robot's size).

Clip, on the data branch (D11): `walk60_trot_0.25_1x.mp4`, 20 s in real time, 1280 × 720, H.264.
- **What it shows:** the model-based trot (`../walk/walk60.py`, frozen, the controller that passed its
  pre-registered rules in `../walk/REPORT_WALK60.md`), on the design variant with passive roll ankles, at the
  0.25 m/s museum command.
- **Conditions:** 10 ms servo delay and sensor noise on, no randomisation, episode seed 101. Labelled on every
  frame.
- **How to show it:** as a trot developed and tested in simulation on a flat floor. Its weak side is ramps
  (`../walk/REPORT_WALK60.md`, robustness sweep). It is not a demonstration of the robot.

Reproduce:

```sh
MUJOCO_GL=osmesa PYOPENGL_PLATFORM=osmesa python render_stills.py --robot ../sim/generated/trike60_C_closed.xml \
  --size 60 --scale 0.8 --out stills_60cm_2026-11-22
MUJOCO_GL=osmesa PYOPENGL_PLATFORM=osmesa python render_walk_clip.py \
  --out stills_60cm_2026-11-22/robot_trot_frame.jpg --still-at 8.0
IMAGEIO_FFMPEG_EXE=/usr/bin/ffmpeg MUJOCO_GL=osmesa PYOPENGL_PLATFORM=osmesa python render_walk_clip.py \
  --out walk60_trot_0.25_1x.mp4 --seconds 20
```

## At 75 cm (18 Nov 2026, before the size decision)

## Stills ([stills_2026-11-18/](stills_2026-11-18/), JPEG, each under 500 KB)

| File | Shows | Caption |
|---|---|---|
| `robot_three_quarter.jpg` | The 75 cm robot standing, three-quarter view | Triceratops robot, 75 cm to the frill: simulation. MuJoCo model of the proposed 20 kg design, standing. Not hardware; nothing is built yet. |
| `robot_side.jpg` | The same, side view | as above |
| `robot_head.jpg` | Head, frill and horns, close | as above |
| `robot_crawl_variant_frame.jpg` | The crawl at 24.75 s of simulated time, the left forefoot in swing | Statically stable crawl, simulation; design variant with passive roll ankles; 0.02 m/s |
| `adult_three_quarter.jpg` | The adult *Triceratops horridus* reconstruction | Adult Triceratops, the animal the robot is modelled on: simulation. 7.3 m long, about 3.2 m at the frill; estimated proportions and masses. |
| `adult_side.jpg` | The same, side view | as above |

- **Robot model:** `../sim/generated/trike_T20_closed.xml` (20 kg build, the parts list's recommendation; the mass
  decision is still open), standing after 2 s of simulation. The crawl frame and clip use the design variant
  `trike_T20_closed_ankleroll.xml`, which adds a passive roll hinge at each ankle (proposed in
  `../sim/CRAWL_REPORT.md`; not adopted).
- **Adult model:** PR #669's `models/anatomical.xml` (not on main until PR #669 merges).
- The robot's body and legs are shown as frame, drives and ribs; its head, frill and tail are shown as shells. The
  painted body and limb shells of the museum finish are not modelled.
- The crawl variant's pass in simulation is thin (it fails under a stiffer friction model; `../sim/CRAWL_REPORT.md`):
  show the crawl as "a gait being developed in simulation", not as a finished capability.

## Clip (on a data branch)

`render_crawl_clip.py` makes a 20 s clip of the crawl at 2× speed (1280 × 720, H.264, 5.7 MB), labelled on every frame
"SIMULATION", "design variant with passive roll ankles", "0.02 m/s", "shown at 2x speed", "Not hardware". Clips are
over the repository's 500 KB limit, so, as the owner chose on 18 Nov 2026 (decision D11), they live on the
never-merged data branch `claude/triceratops-build-clips`: `crawl_T20_ankleroll_2x.mp4` there. Fetch it with
`git fetch origin claude/triceratops-build-clips` and then `git show FETCH_HEAD:crawl_T20_ankleroll_2x.mp4 > clip.mp4`
(works in single-branch clones too).

## Reproduce

```sh
MUJOCO_GL=osmesa PYOPENGL_PLATFORM=osmesa python render_stills.py \
  --robot ../sim/generated/trike_T20_closed.xml --anatomical PATH/TO/anatomical.xml --out stills_2026-11-18
IMAGEIO_FFMPEG_EXE=/usr/bin/ffmpeg MUJOCO_GL=osmesa PYOPENGL_PLATFORM=osmesa python render_crawl_clip.py \
  --model ../sim/generated/trike_T20_closed_ankleroll.xml --out clip.mp4 --speed 2 --seconds 20
```

(`pip install mujoco==3.10.0 numpy pillow imageio imageio-ffmpeg`; on a machine with a display, drop the
`MUJOCO_GL` settings. The crawl still: `render_crawl_clip.py --model ../sim/generated/trike_T20_closed_ankleroll.xml
--out stills_2026-11-18/robot_crawl_variant_frame.jpg --still-at 24.75`.)
