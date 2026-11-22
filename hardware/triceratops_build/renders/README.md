# Renders for the December museum contact (task 9)

18 Nov 2026. **Every image is a simulation render**, labelled "SIMULATION" on the image. Nothing has been built; no
image shows hardware. Made ahead of the 8 Dec 2026 target, so they can be redone if the build decisions change.

## Roadmap line (for the pitch)

> The Triceratops is the near-term live demonstration (Q3 2027); the T. rex is the flagship, coming at end 2027 /
> early 2028.

## At 60 cm (22 Nov 2026; the build chosen on 18 Nov, D2)

**Every image and clip is a simulation render**, labelled "SIMULATION" on the picture itself. Each carries a badge
top left and another bottom right, so a crop around the robot keeps one. The floor's reflection is turned off.

**Stills** ([stills_60cm_2026-11-22/](stills_60cm_2026-11-22/), JPEG, each under 500 KB):

| File | Shows | Caption |
|---|---|---|
| `robot_three_quarter.jpg` | The 60 cm robot standing, as designed (PR #669's pitch-only ankles), three-quarter view | Triceratops robot, 60 cm to the frill: simulation. MuJoCo model of the proposed 60 cm Triceratops robot, 12.7 kg design, standing. Not hardware; nothing is built yet. |
| `robot_side.jpg` | The same, side view | as above |
| `robot_head.jpg` | Head, frill and horns, close | as above |
| `robot_trot_frame.jpg` | The model-based trot, 8.0 s into the clip's episode | Triceratops robot, 60 cm: trotting at its planned walking speed, 0.25 m/s (simulation). MuJoCo model, 12.7 kg, design variant with passive roll ankles. Model-based trot, 8.0 s into the episode. 10 ms servo delay and sensor noise simulated. Not hardware; nothing is built yet. |

- **What the model shows:** `render_stills.py` checks the caption's size against the model (the frill's top
  stands at 60.0 cm after the 2 s settle).
- **What is not modelled:** the body and legs are shown as frame, drives and ribs; the head, frill and tail are
  shown as shells. The painted body and limb shells of the museum finish are not modelled, though the 12.7 kg
  includes their mass.
- **The adult reconstruction stills** in `stills_2026-11-18/` are still valid; the animal does not depend on the
  robot's size.

**Clip, on the data branch (D11):** `walk60_trot_0.25_1x.mp4`, 20 s shown at actual speed, 1280 × 720, H.264,
5.7 MB.
- **What it shows:** the model-based trot (`../walk/walk60.py`, frozen; the controller that passed its
  pre-registered rules in `../walk/REPORT_WALK60.md`), on the design variant with passive roll ankles, at the
  0.25 m/s planned walking speed. Frames are taken from the 1 ms physics steps at each 1/30 s of simulated time.
- **Conditions:** 10 ms servo delay and sensor noise on, no randomisation, episode seed 101. Seed 101 is typical:
  seeds 101-110 under the same conditions all walk without a fall at 0.246-0.248 m/s (the independent check).
- **The episode itself** (the same simulation run without rendering, `walk60.run_episode`, measured from 2 s):
  - no fall; 0.246 m/s;
  - worst drive at 0.67 of its continuous rating;
  - sideways drift 0.07 m and heading change 2.9° over 18 s;
  - slip ratio 0.06-0.10;
  - the report's two weak points are visible here too: the right fore pitch drive is at its torque limit in 2.4 %
    of steps (the flag is 1 %), and the front feet touch down 3.4 times per second against 1.4 strides, so a
    second touch after landing may be seen.
- **Where it is:** on the data branch `claude/triceratops-build-clips` (the latest commit there replaces the first
  render of 22 Nov). Fetch it with `git fetch origin claude/triceratops-build-clips`, then
  `git show FETCH_HEAD:walk60_trot_0.25_1x.mp4 > clip.mp4`.
- **How to show it:** as a trot developed and tested in simulation, not as a demonstration of the robot. What the
  museum conversation will reach:
  - It has only walked straight ahead, on a flat floor: no turns yet. Its weak side is ramps
    (`../walk/REPORT_WALK60.md`, robustness sweep).
  - It has only been run on the design variant with passive roll ankles, which is not adopted: the foot is still
    an open decision (D5). It has never walked on the as-designed foot shown in the standing stills.
  - Holding a line across a museum floor needs an outside reference on the robot (vision or floor markers). In
    simulation its heading drifts with the simulated gyro.

**Reproduce** (mujoco 3.10.0, numpy 2.5.5, gymnasium 1.4.0, Pillow 13.0.0, imageio 2.38.2, ffmpeg 6.1.1;
byte-for-byte output needs the same Pillow and ffmpeg):

```sh
MUJOCO_GL=osmesa PYOPENGL_PLATFORM=osmesa python render_stills.py --robot ../sim/generated/trike60_C_closed.xml \
  --size 60 --scale 0.8 --no-reflection --second-badge --out stills_60cm_2026-11-22
MUJOCO_GL=osmesa PYOPENGL_PLATFORM=osmesa python render_walk_clip.py \
  --out stills_60cm_2026-11-22/robot_trot_frame.jpg --still-at 8.0
IMAGEIO_FFMPEG_EXE=/usr/bin/ffmpeg MUJOCO_GL=osmesa PYOPENGL_PLATFORM=osmesa python render_walk_clip.py \
  --out walk60_trot_0.25_1x.mp4 --seconds 20
```

## At 75 cm (18 Nov 2026, before the size decision)

**The 75 cm robot stills and the crawl clip are superseded by the 60 cm set: do not show them.** The adult
reconstruction stills here are still valid.

### Stills ([stills_2026-11-18/](stills_2026-11-18/), JPEG, each under 500 KB)

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

### Clip (on a data branch)

`render_crawl_clip.py` makes a 20 s clip of the crawl at 2× speed (1280 × 720, H.264, 5.7 MB), labelled on every frame
"SIMULATION", "design variant with passive roll ankles", "0.02 m/s", "shown at 2x speed", "Not hardware". Clips are
over the repository's 500 KB limit, so, as the owner chose on 18 Nov 2026 (decision D11), they live on the
never-merged data branch `claude/triceratops-build-clips`: `crawl_T20_ankleroll_2x.mp4` there. Fetch it with
`git fetch origin claude/triceratops-build-clips` and then `git show FETCH_HEAD:crawl_T20_ankleroll_2x.mp4 > clip.mp4`
(works in single-branch clones too).

### Reproduce

```sh
MUJOCO_GL=osmesa PYOPENGL_PLATFORM=osmesa python render_stills.py \
  --robot ../sim/generated/trike_T20_closed.xml --anatomical PATH/TO/anatomical.xml --out stills_2026-11-18
IMAGEIO_FFMPEG_EXE=/usr/bin/ffmpeg MUJOCO_GL=osmesa PYOPENGL_PLATFORM=osmesa python render_crawl_clip.py \
  --model ../sim/generated/trike_T20_closed_ankleroll.xml --out clip.mp4 --speed 2 --seconds 20
```

(`pip install mujoco==3.10.0 numpy pillow imageio imageio-ffmpeg gymnasium`; on a machine with a display, drop the
`MUJOCO_GL` settings. The crawl still: `render_crawl_clip.py --model ../sim/generated/trike_T20_closed_ankleroll.xml
--out stills_2026-11-18/robot_crawl_variant_frame.jpg --still-at 24.75`.)
