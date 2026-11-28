# Simulation renders, due 8 Dec 2026 (task 9)

18 Nov 2026. **Every image is a simulation render**, labelled "SIMULATION" on the image. Nothing has been built; no
image shows hardware. Made ahead of the 8 Dec 2026 target, so they can be redone if the build decisions change.

## Roadmap line (for the pitch)

> The Triceratops is the near-term live demonstration (Q3 2027); the T. rex is the flagship, coming at end 2027 /
> early 2028.

## At 60 cm with the 128 mm foreleg (26 Nov 2026; the current set: show these)

The owner adopted the 128 mm upper foreleg on 26 Nov 2026 (D25), after the foreleg comparison
(`../rl/analysis/REPORT_FORELEG_D22.md`, F20). The earlier sets show the 188 mm foreleg, which cannot be built as
modelled (F18). This set replaces the capsule set below and uses the same feet (D19), the same scripts and the same
settings; only the model changes, to `../sim/generated/trike60_C_closed_capsule_fore128.xml` (12.751 kg). **Every image
and clip is a simulation render**, labelled "SIMULATION" with a badge top left and another bottom right.

**Stills** ([stills_60cm_fore128_2026-11-26/](stills_60cm_fore128_2026-11-26/), JPEG, each under 150 KB):

| File | Shows | Caption |
|---|---|---|
| `robot_three_quarter.jpg` | The 60 cm robot standing, three-quarter view | Triceratops robot, 60 cm to the frill: simulation. Simulation: MuJoCo model of the proposed 60 cm Triceratops robot (12.8 kg, capsule feet), standing. Not hardware; nothing is built yet. |
| `robot_side.jpg` | The same, side view | as above |
| `robot_head.jpg` | Head, frill and horns, close | as above |
| `robot_trot_frame.jpg` | The model-based trot, 8.0 s into the clip's episode | Triceratops robot, 60 cm: trotting at its planned walking speed, 0.25 m/s (simulation). MuJoCo model, 12.8 kg, capsule feet (the foot chosen on 22 Nov 2026). Model-based trot, 8.0 s into the episode. 10 ms servo delay and sensor noise simulated. Not hardware; nothing is built yet. |

- **What changes in the picture:**
  - The shoulder sits 40 mm lower.
  - Each elbow drive (X6) sits in a pocket at the shoulder on an offset mount.
  - The upper foreleg is 128 mm long (188 mm before), so the elbow stands higher. This is PR #697's repackaged
    fore-assembly.
- **What the model shows:** `render_stills.py` checks that the frill's top stands at 60 cm after the 2 s settle.
- **What is not modelled:**
  - The body and legs are shown as frame, drives and ribs; the head, frill and tail as shells.
  - The capsule is drawn as a full rigid capsule. A real foot would be a moulded rubber rocker under the ankle
    (`../sim/REPORT_FEET60.md`).
  - Two schematic chassis bars pass into leg drives (F19, F21). Neither collides in the simulation, and a viewer does
    not see the overlap:
    - The shoulder crossmember shows on top of the front drives and touches the front roll and pitch drives: 1.1 mm
      standing, up to 5.3 mm over the clip's episode, and up to 29 mm at the front roll joint's +24 deg limit.
    - The pelvic crossmember runs inside the hip drive cylinders, about 40 mm into the hip roll and pitch drives, in
      every 60 cm model.
  - The pushrod that drives the elbow is fixed in the model.
  - Fit: in the model's packaging check the 128 mm fore-assembly adds no overlap the 188 mm model lacks, and its named
    clearances are 4.7-8.7 mm (X6 to upper strut 8.7, X6 to chassis 7.7, X6 to output crank 7.2, motor plate to rail
    4.7). Whether the real X6 mount, pushrod and guards fit is for the bench article.

**Clip, on the data branch (D11):** `walk60_fore128_trot_0.25_1x.mp4`, 20 s shown at actual speed, 1280 × 720,
H.264, 6.8 MB (data-branch commit 9c020943; the capsule-188 clip there is marked superseded).
- **What it shows:** the frozen model-based trot (`../walk/walk60.py`) on the 128 mm foreleg and capsule feet, at
  the 0.25 m/s planned walking speed.
  - It solves its inverse kinematics on the model itself, so it runs on the new foreleg unchanged (largest
    inverse-kinematics error 0.02 mm), but untuned.
  - The learned policies of the foreleg study walk with irregular strides; this clip shows the model-based trot, not
    those policies.
- **Conditions:** 10 ms servo delay and sensor noise on, no randomisation, episode seed 101; shadow range widened
  (`--shadow-clip 6`).
- **Seeds 101-110 under the same conditions, run without rendering** (`trot_seed_check.py`; outputs
  `stills_60cm_fore128_2026-11-26/trot_seeds_101-110_{fore128,188}.json`):

  | | 128 mm foreleg | 188 mm foreleg (the capsule set's model) |
  |---|---|---|
  | Falls | 0 of 10 | 0 of 10 |
  | Speed from 2 s | 0.250-0.252 m/s | 0.248-0.249 m/s |
  | Worst drive, RMS / continuous | 0.66-0.70 | 0.71-0.73 |
  | Front pitch drive at its torque limit | 0.6-0.8 % of steps (the flag is 1 %) | 1.8-2.1 % |
  | Sideways drift over 18 s | up to 0.27 m | up to 0.27 m |
  | Heading change | up to 5.0 deg | up to 5.6 deg |
  | Front-foot touchdowns | 3.3-4.2 per second | 4.2-4.3 per second |

  The 188 mm figures reproduce the capsule set's.
- **The episode itself** (seed 101, measured from 2 s):
  - no fall; 0.250 m/s;
  - worst drive 0.69 of its continuous rating (left hind pitch);
  - right fore pitch at its torque limit in 0.8 % of steps;
  - sideways drift 0.09 m and heading change 2.2 deg over 18 s;
  - front feet touch down 3.9 times per second: each front foot still rocks and touches down more than once per
    stride.
- **How to show it:** as a trot developed and tested in simulation, not as a demonstration of the robot. It walks
  straight ahead on a flat floor only. Turning is the steering run's work (D20).
- **Where it is:** on the data branch `claude/triceratops-build-clips`. Fetch it with
  `git fetch origin claude/triceratops-build-clips`, then
  `git show FETCH_HEAD:walk60_fore128_trot_0.25_1x.mp4 > clip.mp4`.

**Reproduce** (mujoco 3.10.0, numpy 2.5.5, gymnasium 1.4.0, Pillow 13.0.0, imageio 2.38.2, ffmpeg 6.1.1):

```sh
MUJOCO_GL=osmesa PYOPENGL_PLATFORM=osmesa python render_stills.py \
  --robot ../sim/generated/trike60_C_closed_capsule_fore128.xml --size 60 --scale 0.8 --no-reflection \
  --second-badge --out stills_60cm_fore128_2026-11-26
MUJOCO_GL=osmesa PYOPENGL_PLATFORM=osmesa python render_walk_clip.py \
  --model ../sim/generated/trike60_C_closed_capsule_fore128.xml --shadow-clip 6 \
  --out stills_60cm_fore128_2026-11-26/robot_trot_frame.jpg --still-at 8.0
IMAGEIO_FFMPEG_EXE=/usr/bin/ffmpeg MUJOCO_GL=osmesa PYOPENGL_PLATFORM=osmesa python render_walk_clip.py \
  --model ../sim/generated/trike60_C_closed_capsule_fore128.xml --shadow-clip 6 \
  --out walk60_fore128_trot_0.25_1x.mp4 --seconds 20
python trot_seed_check.py trike60_C_closed_capsule_fore128.xml stills_60cm_fore128_2026-11-26/trot_seeds_101-110_fore128.json
python trot_seed_check.py trike60_C_closed_capsule.xml stills_60cm_fore128_2026-11-26/trot_seeds_101-110_188.json
```

## At 60 cm with the capsule feet (22 Nov 2026; superseded by the 128 mm set above: do not show)

**Superseded on 26 Nov 2026** by the 128 mm foreleg set above: these show the 188 mm foreleg, which cannot be built as
modelled (F18). Kept for the record.

The owner chose the capsule foot on 22 Nov 2026 (D19, `../sim/REPORT_FEET60.md`): each sole is an 88 mm pad,
rounded across its width, with no extra joint. This set replaces the earlier 60 cm set below, which shows the
as-designed foot and the roll-ankle variant. **Every image and clip is a simulation render**, labelled "SIMULATION"
on the picture itself, with a badge top left and another bottom right. The floor's reflection is turned off.

**Stills** ([stills_60cm_capsule_2026-11-22/](stills_60cm_capsule_2026-11-22/), JPEG, each under 150 KB):

| File | Shows | Caption |
|---|---|---|
| `robot_three_quarter.jpg` | The 60 cm robot standing on capsule feet, three-quarter view | Triceratops robot, 60 cm to the frill: simulation. Simulation: MuJoCo model of the proposed 60 cm Triceratops robot (12.7 kg, capsule feet), standing. Not hardware; nothing is built yet. |
| `robot_side.jpg` | The same, side view | as above |
| `robot_head.jpg` | Head, frill and horns, close | as above |
| `robot_trot_frame.jpg` | The model-based trot on capsule feet, 8.0 s into the clip's episode | Triceratops robot, 60 cm: trotting at its planned walking speed, 0.25 m/s (simulation). MuJoCo model, 12.7 kg, capsule feet (the foot chosen on 22 Nov 2026). Model-based trot, 8.0 s into the episode. 10 ms servo delay and sensor noise simulated. Not hardware; nothing is built yet. |

- **What the model shows:**
  - `render_stills.py` checks the caption's size against the model: the frill's top stands at 60 cm after the 2 s
    settle, as on the other feet (the capsule keeps the sole's lowest point 36 mm under the ankle).
  - Both scripts now refuse a caption line too wide for the picture rather than clip it.
- **What is not modelled:**
  - The body and legs are shown as frame, drives and ribs; the head, frill and tail as shells.
  - The capsule is drawn as a full rigid capsule. A real foot would be a moulded rubber rocker under the ankle
    (`../sim/REPORT_FEET60.md`).
- **Shadow:** the clip and the trot frame are rendered with the shadow range widened (`--shadow-clip 6`). The
  model's light casts shadows only within about 0.9 m of the start, so the earlier clip lost the robot's ground
  shadow after about 6 s, and the robot looked as if it hovered. Rendering only; the physics is unchanged.

**Clip, on the data branch (D11):** `walk60_capsule_trot_0.25_1x.mp4`, 20 s shown at actual speed, 1280 × 720,
H.264, 6.7 MB (data-branch commit 408d5c59; it replaces the first render, a7efa5ce, which lost the ground shadow).
- **What it shows:** the model-based trot (`../walk/walk60.py`, frozen) on the capsule feet, at the 0.25 m/s
  planned walking speed. Frames are taken from the 1 ms physics steps at each 1/30 s of simulated time.
- **The controller on this foot:** the trot was developed and judged on the roll-ankle variant
  (`../walk/REPORT_WALK60.md`). On the capsule it runs untuned. The foot study judged it there too and found it
  passing (W1-W4 in all four conditions, `../sim/REPORT_FEET60.md`).
- **Conditions:** 10 ms servo delay and sensor noise on, no randomisation, episode seed 101.
- **Seed 101 is typical:** seeds 101-110 under the same conditions, run without rendering, gave:
  - no fall;
  - 0.248-0.249 m/s;
  - worst drive at 0.71-0.73 of its continuous rating;
  - sideways drift up to 0.27 m and heading change up to 5.6° over 18 s.
- **The episode itself** (seed 101, the same simulation run without rendering, measured from 2 s):
  - no fall; 0.248 m/s;
  - worst drive at 0.73 of its continuous rating;
  - sideways drift 0.07 m and heading change 2.1° over 18 s;
  - slip ratio at the contacts 0.04-0.06;
  - **two weak points, visible here too:**
    - the right fore pitch drive is at its torque limit in 2.0 % of steps (1.8-2.1 % over seeds 101-110; the flag
      is 1 %);
    - the front feet touch down 4.2 times per second against 1.4 strides per second. Each front foot rocks and
      touches down about three times per stride, so repeated touches after landing may be seen. That is more than
      on the roll-ankle variant (F17).
- **Where it is:** on the data branch `claude/triceratops-build-clips`. Fetch it with
  `git fetch origin claude/triceratops-build-clips`, then
  `git show FETCH_HEAD:walk60_capsule_trot_0.25_1x.mp4 > clip.mp4`.
- **How to show it:** as a trot developed and tested in simulation, not as a demonstration of the robot.
  - It has only walked straight ahead, on a flat floor: no turns yet. On the roll-ankle variant its weak side was
    ramps; the robustness sweep has not been run on the capsule feet.
  - Holding a line across a demo floor needs an outside reference on the robot (vision or floor markers). In
    simulation its heading drifts with the simulated gyro.
  - The learned walking policy (run 3) trains on this foot: the owner reported starting it on 22 Nov 2026. The clip
    shows the model-based trot, not that policy.

**Reproduce** (mujoco 3.10.0, numpy 2.5.5, gymnasium 1.4.0, Pillow 13.0.0, imageio 2.38.2, ffmpeg 6.1.1;
byte-for-byte output needs the same Pillow and ffmpeg):

```sh
MUJOCO_GL=osmesa PYOPENGL_PLATFORM=osmesa python render_stills.py \
  --robot ../sim/generated/trike60_C_closed_capsule.xml --size 60 --scale 0.8 --no-reflection --second-badge \
  --out stills_60cm_capsule_2026-11-22
MUJOCO_GL=osmesa PYOPENGL_PLATFORM=osmesa python render_walk_clip.py \
  --model ../sim/generated/trike60_C_closed_capsule.xml --shadow-clip 6 \
  --out stills_60cm_capsule_2026-11-22/robot_trot_frame.jpg --still-at 8.0
IMAGEIO_FFMPEG_EXE=/usr/bin/ffmpeg MUJOCO_GL=osmesa PYOPENGL_PLATFORM=osmesa python render_walk_clip.py \
  --model ../sim/generated/trike60_C_closed_capsule.xml --shadow-clip 6 --out walk60_capsule_trot_0.25_1x.mp4 \
  --seconds 20
```

## At 60 cm, earlier feet (22 Nov 2026; superseded: do not show)

**Superseded on 22 Nov 2026 by the capsule-foot set above.** These show the as-designed foot (stills) and the
roll-ankle variant (clip), neither of which was chosen (D19). Kept for the record; the scripts still reproduce them
byte for byte.

### The earlier 60 cm set (the build chosen on 18 Nov, D2)

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
  shown as shells. The painted body and limb shells of the show finish are not modelled, though the 12.7 kg
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
    of steps (the flag is 1 %), and the front feet touch down 3.4 times per second against 1.4 strides per second, so a
    second touch after landing may be seen.
- **Where it is:** on the data branch `claude/triceratops-build-clips` (the latest commit there replaces the first
  render of 22 Nov). Fetch it with `git fetch origin claude/triceratops-build-clips`, then
  `git show FETCH_HEAD:walk60_trot_0.25_1x.mp4 > clip.mp4`.
- **How to show it:** as a trot developed and tested in simulation, not as a demonstration of the robot. What the
  conversation with a host will reach:
  - It has only walked straight ahead, on a flat floor: no turns yet. Its weak side is ramps
    (`../walk/REPORT_WALK60.md`, robustness sweep).
  - It has only been run on the design variant with passive roll ankles, which is not adopted: the foot was then
    an open decision (D5; since decided: the capsule, D19). This clip shows the roll-ankle variant only; the trot has
    since also passed the foot study's walking rules on the as-designed foot and on the capsule
    (`../sim/REPORT_FEET60.md`).
  - Holding a line across a demo floor needs an outside reference on the robot (vision or floor markers). In
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
  painted body and limb shells of the show finish are not modelled.
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
