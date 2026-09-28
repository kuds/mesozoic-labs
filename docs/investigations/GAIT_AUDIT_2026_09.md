# Gait audit of the certified nodes (2026-09-28)

**Status**: dated investigation note, frozen once merged; corrections are appended, never edited in. It records
findings; nothing in it is decided. Written at `c8b66a6` (PR-B's head), whose tree #566's merge `2b9219d` equals, and
landed with PR-G0. Replays and statue rollouts ran on 2026-09-28 on CPU against `main` = `7ae0a19` (the July
brachiosaurus run against a worktree of its own commit `e179198`). Every number below comes from those replays, except
the two scripted checks in §3 findings 2 and 4, run on 2026-09-28 at `2b9219d`; the plan that acts on them is
[../GAIT_QUALITY_PLAN_2026_09.md](../GAIT_QUALITY_PLAN_2026_09.md).

## 1. Question and answer

Do the certified nodes on Drive do what their gates are meant to certify: a real stance, a real walk or run, a real
recovery? Fifteen nodes were replayed (twelve certified, one failed, one still training, one from July).

- **Locomotion.** Two of the five certified walkers are genuine: compsognathus walks and velociraptor runs, both
  alternating their feet. Three hop on both feet at once: trex seed 42 and seed 44, and compsognathus_robot, whose
  right sole rests on its left foot.
- **Stance.** One of the six certified stances is clean (trex seed 42). Trex seed 44 hops to rebalance in 6 of 40
  episodes; compsognathus marches in place; velociraptor chatters and slides; compsognathus_robot stands on one foot
  with the other stacked on it; dibothrosuchus is a statue, three-legged in some episodes.
- **Recovery.** Trex seed 44 keeps its posture after pushes but answers forward pushes with two-footed hops.
- **Not certified.** The dibothrosuchus re-run in training is a three-legged skid that its gate would pass; the July
  brachiosaurus walker was a neck-pumping bound.
- **Why.** No locomotion gate reads a foot contact, the zero-action statue passes every stance gate, and the robot's
  touch sensors count sole-on-sole contact as floor support.

## 2. Method

### 2.1 The probe

One script, `gait_probe.py`, rolled each handoff pair (checkpoint plus VecNormalize sidecar, deterministic actions,
normalisation in evaluation mode) with the repository's own env construction and plant contract; a refused checkpoint
exits with code 3 and was never forced.

- **Seeds.** `panel`: episode *i* resets with seed 3042 + *i*, as the stance and recovery panels do. `vecenv`: repeats
  the evaluation behind `evaluation_selected.csv` and `evaluation_final.csv` exactly, including the extra reset after
  each episode. `stance_quality` and recovery nodes used `panel` (40 episodes); `reward_and_length` nodes used
  `vecenv` (30 episodes; 20 for the in-training re-run), plus a 40-episode `panel` for the dibothrosuchus stance.
- **Feet** come from the env's `_foot_sensor_groups` (2 or 4); trex digits and brachiosaurus pad plus metatarsal are
  summed as the repository sums them.
- **Two contact channels.** *Touch* is the env's own reading: per foot, the summed touch sensors above 0.1 N on every
  physics substep of a control step (the minimum over substeps). *Floor truth* is the real floor normal force on the
  whole leg above 0.1 N on at least half the substeps, recorded through the env's per-substep hook
  (`_substep_probe_hook`). Every contact metric is computed on both.
- **Events.** Contact flicker shorter than 2 control steps is ignored; left and right touchdowns within 2 steps count
  as one simultaneous event.
- **Windows.** Locomotion metrics skip the first 1 s; `stance_quality` stances skip the stage's 200-step settle; the
  velociraptor and dibothrosuchus stances (`reward_and_length`) skip 1 s (100 steps).
- **Scale.** L is the hip-pitch joint height at the home pose (trex 0.877 m, velociraptor 0.50 m, compsognathus
  0.244 m, compsognathus_robot 0.207 m; hind hip for dibothrosuchus 0.2915 m and brachiosaurus 1.0 m). Froude =
  v²/(gL) on the post-settle speed.
- **Downloads.** Every file with a recorded sha256 matched it. The in-training re-run and the July run (checkpoint and
  vecnorm) have no recorded sha256; they were checked by their counts and by reproducing the recorded rewards. Small
  Drive files copied by hand were wrong at least five times; each was repaired and re-hashed.
- **Probe copies.** Four copies add `--video-frame-dt` only (trex locomotion, velociraptor, quadrupeds) or, for trex
  stance and recovery, push-schedule capture with the repository's per-step recovery judge plus the same video fix.
  Every panel ran on the unchanged probe or on a copy that differs only in video timing, except the recovery panel and
  the July brachiosaurus panel. A legacy driver ran the July panel at `e179198`, with each foot's single touch sensor
  read at the end of the control step, as the July env read it, not the minimum over substeps.

### 2.2 Touch against floor truth

| Node | Touch | Floor truth | Cause |
|---|---|---|---|
| compsognathus_robot locomotion | two feet down 0.97, flight 0.002; label "slide" | flight 0.48; label "hop" 30/30 | the feet touch on 96% of substeps; 1.12 body weights of sole-on-sole force in each sensor; phantom support R 0.57, L 0.50 |
| compsognathus_robot stance | two-foot support 0.9988 | 0.62 | feet touch on 99.95% of substeps; sensors total 3.04 body weights against 1.00 on the floor |
| velociraptor run / stance | flight 0.49 / 0.25 | 0.33 / 0.15 | the minimum over 5 substeps; the sensor also covers only `toe_d3` (18% of standing load is on the right metatarsus) |
| compsognathus stance | unsupported 0.013 | 0.000 | foot handoffs inside one 20 ms step |
| compsognathus walk | label "run" 26/30 | label "walk" 29/30 | the same handoffs |
| dibothrosuchus re-run | 0.39 body weights | 1.00 | 10–30 ms contacts lost to the minimum; 100% of floor load inside the touch-site volumes |
| brachiosaurus July (r1) | 0.019 body weights, touch duty 0.0005–0.034 | 1.00 | only 2.9% of floor load inside the r1 touch spheres (repaired in the current plant, plant_versions note 8) |
| trex, all nodes | agrees; misses 5–8% of floor-contact steps at contact edges on the two locomotion nodes, under 2% on the stance and recovery nodes | — | — |

### 2.3 Sanity checks (recorded against replay)

| Node | Recorded | Replay |
|---|---|---|
| trex `20260914_123816` stance panel | 3459.6 ± 17.8; duty 9.375e-5; UCB 2.517e-4 | 3459.8 ± 17.7; duty and UCB identical |
| trex `20260914_123816` locomotion, handoff | 1929.22 ± 35.54, 1.0525 m/s | 1931.50 ± 37.12 (z = +0.25), 1.0554 m/s |
| same, final checkpoint | 1940.81 ± 32.91 | 1937.40 ± 35.33 |
| trex `20260920_010912` stance panel | 3408.3 ± 88.5; duty 0.0069 | 3418.0 ± 66.8; duty 0.0044 |
| trex `20260920_010912` recovery | 28/40 (LCB 0.560); 140/155 pushes | 34/40 (0.725); 148/155 pushes |
| trex `20260925_033501` locomotion | 2316.81 ± 24.36, 1.5696 m/s | 2314.38 ± 24.30 (z = −0.39), 1.5666 m/s |
| velociraptor stance | 1740.03 ± 301.50, length 970.3 | 1739.70 ± 301.47, 970.3 (the fall at step 109 reproduced) |
| velociraptor locomotion, handoff / final | 2591.88 ± 493.74 / 2683.64 ± 8.56 | 2592.00 ± 493.77 / 2683.62 ± 8.54 (the fall at step 59 reproduced) |
| compsognathus stance panel | 2801.62 ± 51.22; duty 0.01306 | 2799.57 ± 48.36; duty 0.01316 |
| compsognathus locomotion, handoff | 3334.73 ± 9.50, 0.3602 m/s | 3307.35 ± 139.5 with one fall (29 full episodes 3333.19 ± 10.09), 0.3600 m/s |
| same, final checkpoint | 3308.62 ± 13.02, 0.3421 m/s | 3306.46 ± 12.81, 0.3398 m/s (−0.7%, unexplained) |
| compsognathus_robot stance panel | 2979.23 ± 3.98; two-foot 0.9984 | 2979.40 ± 2.84; 0.9988 |
| compsognathus_robot locomotion, handoff / final | 2624.38 ± 11.12 / 2605.91 ± 11.91 | 2626.54 ± 11.89 (z = +0.75) / 2607.18 ± 11.50 |
| dibothrosuchus stance / locomotion | 2597.4871 ± 1.9550 / 2249.8594 ± 3.9617 | identical; every episode within 2e-5 / 7e-6 |
| dibothrosuchus re-run 4.3M / 4.8M | in-training (seed 1042): 3686, 77% full / 855, 10% full | 3633 ± 1422, 15/20 full / 734 ± 902, 1/20 full |
| brachiosaurus July | 6493.76 ± 306.93, 1.49 m/s | 6453.9 ± 374.6 (z = −0.50), 1.478 m/s |

The eleven statue references in the TOMLs and on Drive reproduced within 0.05 reward (for example trex stance 3495.2,
brachiosaurus locomotion 2242.7, dibothrosuchus stance 2598.3).

### 2.4 GPU against CPU

Training evaluated on a Colab L4 GPU; the replays ran on CPU. Means agree; individual episodes do not, and the cause
is chaos, not the probe:

- On the replay machine the repository's own evaluation code matches the probe to within 2e-6 reward (1e-11 on the
  stance and recovery panels), including compsognathus's CPU-only fall.
- Nudging one action by 1e-6 moves an episode's reward by 6–8 (robot) or 10–12 (trex locomotion); by 1e-5, 21.
- Largest per-episode reward gaps: robot 41 (59 on the final checkpoint), trex locomotion 26.7 and 25.9, compsognathus
  stance 28, trex seed-42 stance 7.3, velociraptor 6.8 (stance) and 1.9 (locomotion); dibothrosuchus within 2e-5 (4e-5 on
  the final checkpoint).
- trex seed-44 stance, seed 3042: replay 3350 with duty 0.009; a +1e-7 nudge at step 300 gives 3195 and 0.0625;
  recorded 3171 and 0.0675. A quiet seed does not move. The hop bouts fell in the same six seeds recorded and replayed.
- trex seed-44 recovery, five realizations (float64 inference, three 1e-7 noise runs, plain SB3): 28, 30, 34, 31 and 34
  successes; the float64 run reproduces the recorded 28/40 and 34 full-horizon episodes, with 141 pushes recovered
  against 140 recorded.
- compsognathus locomotion: episode 6 falls at step 780 on CPU and survives on the GPU panel; 100 fresh seeds (5000–
  5099) had no fall (1 fall in 131 episodes overall).

### 2.5 Video

Drive replays store one frame per control step and are encoded at 50 fps, so the 0.01 s species play at half speed
and the unchanged probe spaces their contact sheets at twice the requested time (fixed in the copies and in the
committed probe, §6). The 2 s contact sheets alias every gait here (frames 0.18 s apart against 0.10–0.33 s strides),
so consecutive-frame sheets, zoomed on the feet, were added. Every video agreed with the contact data where it could
resolve it; 5 mm hops, 0.5 mm taps and 20 ms flights are below what a video frame shows, and those come from the
contact data alone.

### 2.6 Limits

The 2-step debounce drops single-step contacts (20 ms at 0.01 s steps, 40 ms at 0.02 s), so floor touchdown counts
slightly undercount. Gait labels are heuristics (four genuine compsognathus walk episodes read "irregular" because
their lead-swap rate sat just under the classifier's 1.4). The pelvis-bounce frequency is unreliable (weak FFT
peaks). Toe slip is measured at the centre of the toe's sensor sphere, so velociraptor's includes toe roll. Hildebrand
labels fail when a limb barely participates or re-plants several times per cycle.

## 3. The gates against the statue

All 21 stage gates of the six species were rolled with the zero-action policy (40 episodes, seeds 3042–3081) and
judged by each stage's own gate code; the 66 behavior recipes were checked against the behavior-certificate rules.

| Species | Stage (deliverable) | Gate | Statue | Admits |
|---|---|---|---|---|
| trex | stance (`stance`) | `stance_quality/v1`: full horizon ≥ 0.95, unsupported duty and its 95% bound ≤ 0.02, rail 2100 | PASS (3495.2, duty 0) | one-foot, staggered, stepping, chattering or sliding stance; tibia contact is not terminated |
| trex | recovery | `recovery_quality/v1`: success LCB ≥ 0.30, paired gain ≥ 0.20 | FAIL, 0/40 | recovery by hopping or stepping: the safe set checks posture only (`min_foot_force_n = 0`) |
| trex | locomotion (`walk`) | `reward_and_length/v1`: reward ≥ 100, mean length ≥ 750, mean speed ≥ 1.0 | FAIL on speed (reward 1091.5) | hop, scoot, slide, lunge late |
| trex | behavior (`hunt`) | `task_success/v1`: LCB ≥ 0.5 at n = 30 | FAIL, 0/30 | lunge and fall: contact counted on the falling step |
| velociraptor | stance (`stand`) | `reward_and_length/v1`: reward ≥ 1050, length ≥ 950 | PASS (1745.8) | anything that survives: hop in place, one leg, metatarsal crouch |
| velociraptor | locomotion (`walk`) | as trex, speed ≥ 2.0 | FAIL on speed | hop, bound, scoot, lunge late |
| velociraptor | strike (`hunt`) | reward ≥ 100, success ≥ 0.5 | FAIL, 0/40 | dive and fall |
| compsognathus (and robot) | stance | `stance_quality/v1`, rail 1800 | PASS (2998.7; robot 2997.8) | one-leg stance paid in full (support = either foot); chatter; slide |
| compsognathus (and robot) | recovery | success LCB ≥ 0.5, gain ≥ 0.1 over statue and brace | FAIL, 20/40 (robot 13/40) | the statue already recovers 93% of shoves (robot 86%) |
| compsognathus (and robot) | locomotion | speed ≥ 0.08 (robot 0.04), reward ≥ 500, length ≥ 900 | FAIL on speed (1499.6) | a 1.6 m (robot 0.8 m) shuffle in 20 s; one-leg hopping fully paid |
| compsognathus (and robot) | behavior (`hunt`) | success ≥ 0.7 | FAIL, 0/40 | any approach mode |
| brachiosaurus | stance (`stand`) | reward ≥ 1040, length ≥ 950 | PASS (1739.1) | kneeling (shins 0.067 m up, unsensed, not terminated), three-leg stance |
| brachiosaurus | locomotion (`walk`) | speed ≥ 0.75 | FAIL on speed (2242.7, of which `gait_symmetry` 2200.0) | pronk, bound, pace, knee-crawl (torso may drop 0.336 m) |
| brachiosaurus | food reach (`hunt`) | success ≥ 0.5 | FAIL, but 11/40 reached without moving | neck stretch or topple |
| dibothrosuchus | stance (`stand`) | reward ≥ 1560, length ≥ 950 | PASS (2598.3) | anything that survives; shin support |
| dibothrosuchus | locomotion (`walk`) | speed ≥ 0.9 | FAIL on speed (2196.9, of which `gait_symmetry` 1945.5) | as brachiosaurus |
| dibothrosuchus | snap (`hunt`) | success ≥ 0.5 | FAIL, 0/40 | lunge and fall |
| all six | 66 behavior recipes | certificate rules, no production caller | FAIL (survival 20/20 passes; tracking 0) | hop, scoot or slide: pelvis-velocity tracking only |

Findings:

1. The statue passes every gate that certifies a published stance or stand deliverable. Unsupported duty counts only
   steps with neither foot above 0.1 N, so stepping, chattering, single-support and sliding stances all score 0.
2. Every locomotion gate tests mean root velocity. The reward floors sit 3–22× below the statue (trex 11×,
   brachiosaurus 22×, dibothrosuchus 22×, compsognathus 3×). Length and velocity are panel means of per-episode
   means: three trex episodes at 0.7 m/s for 1,000 steps plus one lunge at 2.0 m/s that falls at step 250 give mean
   length 812.5 and mean speed 1.025 m/s (recorded 1.02), which `evaluate_stage_gate` passes; distance over time is
   0.80 m/s. At the 30-episode panel, 23 walks and 7 lunges pass both the post-training judge and the in-training
   `CurriculumManager` (scripted through the repository's own code on 2026-09-28).
3. The quadruped `gait_symmetry` reward pays a statue (both diagonal pairs touch down at reset and the history never
   decays) and pays a pronk, bound or pace the same (a simultaneous landing records both pairs). The biped version
   scores a synchronous bounce 1.000; its weight is 0 in every biped stage.
4. Hunt success is read from the per-step reward info while `_is_terminated` runs the height/tilt checks (and the
   nosedive, head-tip and skull-height checks where the species has them) before its success check, so a contact on
   a falling step counts in the evaluation panel but not in the terminal `is_success` that training counts (trex,
   velociraptor, brachiosaurus, dibothrosuchus). Scripted on each hunt env on 2026-09-28: all four ended
   `excessive_tilt` with the success flag at 1.0, and trex's `task_success/v1` passed a 30/30 panel of such steps.
   Floor contact is checked after success, so a tail strike on the contact step counts in both. Compsognathus reads
   success from the termination status, after its fall checks, so a falling step counts in neither (its 0.1 m/s speed
   rule is a second barrier).
5. Knee, shin and proximal-tail contact never ends an episode on trex (tibia 0.157 m up, root may drop 0.226 m),
   velociraptor (metatarsus already touching), brachiosaurus (shins 0.067 m) and dibothrosuchus (shins 0.047 m).
6. The dibothrosuchus `foot_contact_*` knobs do nothing on SB3; the velociraptor foot sensor reads 55% of load; the
   stance diagnostic reads only the forefeet on quadrupeds; `zero_action_baseline.py`'s "FAILS" line for
   `stance_quality` stages compares against the reward rail only.

## 4. Results per node

Floor truth after the settle window; support columns give the fraction of steps with 0 / 1 / 2 feet down (bipeds).

| # | Node | Gate, verdict | Handoff | Support (0 / 1 / 2) | Simultaneous / alternation | Rhythm, geometry | Diagnosis | Intent met |
|---|---|---|---|---|---|---|---|---|
| 1 | trex `20260914_123816` stance, seed 42 | `stance_quality/v1`, PASS 2026-09-15 | `b556bdf5…` | 0.0001 / 0.00003 / 0.9999 | — | 0.006 touchdowns/s; 0.34 m wide, left 4.4 cm ahead; drift 5.9 cm; load 0.48/0.52 | Quiet wide stance; seed 3059 makes two small catch-up hops | Yes |
| 2 | trex `20260914_123816` locomotion | `reward_and_length/v1`, PASS 2026-09-15, 1.05 m/s | `e2c53eb7…` | 0.35 / 0.09 / 0.56 | 0.97 / 0.010 | 7.67 Hz; stride 0.17 L; left 14.7 cm ahead 99.9%; 0.001 lead swaps per stride | Two-footed bunny hop from a fixed split stance (final checkpoint the same) | No |
| 3 | trex `20260920_010912` stance, seed 44 | `stance_quality/v1`, PASS 2026-09-20 | `7f4284ad…` | 0.003 / 0.001 / 0.996 (min 0.951) | hop episodes 0.66 / 0.02 | 0.24 touchdowns/s (max 2.4); drift 0.10 ± 0.14 m (max 0.57) | Quiet in 34/40; forward lean and two-footed catch-up hops in 6/40 (seeds 3042, 3051, 3053, 3063, 3066, 3080) | Partly |
| 4 | trex `20260920_010912` recovery | `recovery_quality/v1`, PASS 2026-09-21 | `a8e41b98…` | 0.028 / 0.028 / 0.945 | 0.59 / 0.03 | 2.39 touchdowns/s; drift 0.67 ± 0.46 m | Posture recovered after 148/155 pushes; forward pushes: 0 of 41 in place (27 mixed, 13 two-footed hops, 1 fall); 60 of 155 pushes involve two-footed touchdowns; 37% of touchdowns outside push windows | Balance yes, clean recovery no |
| 5 | trex `20260925_033501` locomotion, seed 44 | `reward_and_length/v1`, PASS 2026-09-25, 1.57 m/s | `e6de9c92…` | 0.43 / 0.13 / 0.44 | 0.99 / 0.003 | 9.21 Hz; stride 0.21 L; left 9.3 cm ahead; 70 cm stance (right hip roll pinned at 25°) | The same hop, faster and wider | No |
| 6 | velociraptor `20260922_125248` stance | `reward_and_length/v1`, PASS 2026-09-22 | `732b5a71…` | 0.15 / 0.29 / 0.56 | 0.43 / 0.22 | 8.7–10.6 lift-offs per foot per s; 5 mm lifts; slide 0.65 ± 0.22 m, mostly sideways | Crouch on the right metatarsus with feet chattering; the statue passes the same gate (1694 ± 277 on the same resets) | No |
| 7 | velociraptor `20260922_125248` locomotion | `reward_and_length/v1`, PASS 2026-09-23, 3.17 m/s | `7c28eda2…` | 0.33 / 0.67 / 0.008 | 0.018 / 0.93 (final 0.006 / 0.985) | 9.8 Hz; stride 0.70 L; Froude 2.41; 93% of flights between opposite feet; 2.04 lead swaps per stride | Genuine alternating run on the toe tips; a scurry; 1 fall in 30 (final 0 in 30) | Yes, a run |
| 8 | compsognathus `20260921_203149` stance | `stance_quality/v1`, PASS 2026-09-21 | `1d46747f…` | 0.000 / 0.900 / 0.100 | 0.00 / 1.00 | 3.05 Hz; drift 0.21 m; left-heavy (duty 0.61 / 0.49) | Marches in place; 199 below the statue | No |
| 9 | compsognathus `20260921_203149` locomotion | `reward_and_length/v1`, PASS 2026-09-22, 0.36 m/s | `fef74662…` | 0.004 / 0.884 / 0.112 | 0.001 / 0.997 | 4.07 Hz; stride 0.37 L; Froude 0.055; 1.53 lead swaps per stride | Genuine alternating walk, short quick steps, left-heavy | Yes |
| 10 | compsognathus_robot `20260924_031815` stance | `stance_quality/v1`, PASS 2026-09-24 | `b4093e69…` | 0.001 / 0.38 / 0.62 | — | right foot taps 4.3 per s, 0.5 mm; left 1.0 cm ahead; drift 1.8 cm | Stands on the left foot (81% of load) with the right sole on it; 18 below the statue | Standing still yes, two-footed no |
| 11 | compsognathus_robot `20260924_031815` locomotion | `reward_and_length/v1`, PASS 2026-09-28, 0.23 m/s | `dbbfb709…` | 0.48 / 0.11 / 0.40 | 0.85 / 0.02 | 9.2 Hz; stride 0.12 L; clearance 7–9 mm; left 1.3 cm ahead 100% | Synchronous micro-hop from the stacked stance (final checkpoint the same) | No |
| 12 | dibothrosuchus `20260923_020654` stance | `reward_and_length/v1`, PASS 2026-09-23 | `1d3527cc…` | four feet 0.96, three 0.04 | — | 0.004 touchdowns/s; drift 7 mm; hind feet 79% of load | Statue; right forefoot off the floor for a whole episode in 1/30, under 2% of load in 3/30 (6/40 on the panel) | Only as a statue |
| 13 | dibothrosuchus `20260923_020654` locomotion | `reward_and_length/v1`, FAIL 2026-09-23 | `bd32442b…` | four feet 0.94 | — | 0.0002 m/s | Statue, paid 1997.3 of 2249.9 by `gait_symmetry`; the 1.45M final checkpoint rears, leaps once and falls after 28 steps on average (20–62) | — |
| 14 | dibothrosuchus `20260928_012318` locomotion (training) | none yet | 4.3M `7abd4194…`; 4.8M `10e2a5a0…` | 0 / 1 / 2 / 3 feet: 0.34 / 0.45 / 0.18 / 0.03 | pair alternation fore 0.07–0.32, hind 0.24–0.53 | 2.20 m/s displacement speed, Froude 1.56; duty LF/RF/LH/RH 0.32/0.09/0.25/0.24; slip 1.2–2.0 m/s | Three-legged skidding scramble; clears its gate as written (1.95 m/s, mean length 854); by 4.8M it rolls and tips (19/20 tilt) | Would not be |
| 15 | brachiosaurus `20260717_162659` locomotion (July, r1) | July rail v ≥ 0.75 passed; no verdict | `414fad74…` | 0–4 feet: 0.03 / 0.32 / 0.46 / 0.18 / 0.01 | hind feet nearly synchronous (phase 0.04–0.15) | 1.60 m/s displacement speed (gate speed 1.48), Froude 0.24; true cycle 1.45–1.5 s | Neck-pumping asymmetric bound; left foreleg swung to 0.92 m; labels pronk 13, half-bound 9, gallop 7, bound 1 | No |

## 5. Per species

**Trex.** Seed 42's stance is the only clean stance on Drive. Its walker, and seed 44's, hop on both feet at 7.7 and
9.2 Hz with a fixed stagger (left foot 9–15 cm ahead, no lead swaps) and are airborne 35–43% of the time; strides are
0.17–0.21 L, where a walking animal of that size and speed would take strides about eight times longer at about 1 Hz.
The walk reward pays forward speed and an unconditional alive bonus (alive 164 and 205, forward 261 and 475 paid while
airborne); nothing pays alternation. Seed 44's stance hops to rebalance in 6/40 episodes, which the panel-mean duty
(0.0069) hides; its recovery node answers forward pushes with two-footed hops and drifts 0.7 m per episode.

**Velociraptor.** The run is genuine: alternating footfalls, 93% of flights between opposite feet, simultaneous
landings under 2%. It is a toe-tip scurry at about 20 steps per second with a stride of 0.7 L (Alexander's scaling
predicts about 3 at this Froude number); the forward reward is capped at 2.5 m/s on 99.7–100% of steps. The stance is
not a stance: the feet chatter at about 10 Hz, 15% of steps are airborne, the body slides 0.65 m, and the policy beats
the statue mainly by avoiding the nose-down penalty. Its alive bonus (1.75) has no contact condition. The touch sensor
covers only the middle toe; on/off contact at 0.1 N matches the floor force read the env's way (above 0.1 N on every
substep) while running; the gap to floor truth (flight 0.49 against 0.33) is the minimum over substeps (§2.2).

**Compsognathus.** The walk is genuine: alternation 0.997, flight 0.004, feet moving forward in the air rather than
sliding. It is left-heavy and short-stepped, a quirk it inherited from the stance node, which marches in place at
3 Hz on one foot 90% of the time. The stance gate's 1.3% "unsupported" steps are foot handoffs inside a 20 ms step,
not flight.

**Compsognathus_robot.** Both nodes stand or move on stacked feet. The soles are 10 × 9 cm boxes that collide with
each other; the policies narrowed their spacing to 6.3–6.9 cm, and the right sole rests on the left foot's inner
edge. The touch sensors read that contact as floor support, so the reward's support flag (touch sum above 4% of body
weight, which one foot satisfies) is on 99.6% of steps by touch and 34.3% by floor: about 838 of about 2627 reward is
paid for support the floor never gives. The walker hops at 9.2 Hz with 5 mm pelvis bounce and 7–9 mm foot clearance,
foot motions of the same order as typical servo latency and backlash errors, so unlikely to transfer to hardware
(latency and backlash are not modelled or measured here). This is the hardware target.

**Dibothrosuchus.** The certified stance and the failed walk are the same statue; `gait_symmetry` pays 89% of the
walk return for standing still. The 2026-09-28 re-run `20260928_012318` (on `7ae0a19`) was found reusing the
statue-level stance `1d3527cc…` as its ancestor, although the session plan names `RETRAIN_FROM = "stance"`; a run
folder created minutes earlier, `20260928_012025`, holds only `provenance.json` and `zero_action_baseline.json`. At
4.3M the re-run moves at 2.2 m/s by skidding on three legs in 10–30 ms contacts, and its own gate would pass it.

**Brachiosaurus.** No current node exists. The July walker bounded asymmetrically, pumping its neck and swinging one
foreleg high, fed by a `gait_symmetry` term whose r1 touch sensors were almost blind; the sensors have since been
repaired. The current walk reward pays the statue 2200 of 2242.7 through the same term, the statue passes the stance
gate, and the statue reaches the food in 11 of 40 hunt episodes.

## 6. Evidence

The pinned per-node JSON files (metadata, sanity, per-episode rows, summaries), one trace per node, footfall and fore-aft
plots, contact sheets and the statue baselines (`statue_baselines.json`, with per-episode numbers) are 153 files,
64.7 MB; the probe, its copies and the gate audit scripts add under 1 MB.

**Evidence files.** [gait_2026_09/](gait_2026_09/README.md) holds what the repository keeps:
[gait_probe.py](gait_2026_09/gait_probe.py), the hand-run probe (imported by nothing; its code, docstring aside,
equals the trex locomotion, velociraptor and quadruped copies, `--video-frame-dt` fix included);
[gait_audit_2026_09.csv](gait_2026_09/gait_audit_2026_09.csv), one row per audited node and checkpoint (21 rows);
[SHA256SUMS](gait_2026_09/SHA256SUMS), the sha256 of each of the 153 files above; and
[README.md](gait_2026_09/README.md), which says how to regenerate them. The audit's full 559 MB working tree is not in
the repository and is not uploaded anywhere. The per-node files can be regenerated with the probe from each node's
checkpoint pair and replay on Drive, at a `7ae0a19` checkout; the means match, the bytes do not (the README names
the few files that helper scripts not kept here made, and the cited probe runs the pin set leaves out).
