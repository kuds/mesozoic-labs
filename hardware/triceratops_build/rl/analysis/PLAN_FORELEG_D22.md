# Plan: the 60 cm upper foreleg, 128 mm against 160 mm under the RL policy (decision D22)

23 Nov 2026. Simulation only. This is the plan; the pass and fail rules come in a separate pre-registration, committed
before any judged run, after the models below have passed an independent review.

## Why

- PR #697 (the owner's, not merged) recommends a 160 mm upper foreleg for the first build, with 128 mm as the fallback.
  Its evidence comes from scripted controllers: a four-beat walk at 14-17 mm/s and a scripted trot for turning.
- Every RL result so far uses the 188 mm upper foreleg of `sim/generated/trike60_C_closed_capsule.xml`, whose elbow
  drive overlaps the upper strut by 44.9 mm (F18): it cannot be built as modelled.
- The owner chose (D22) to compare 128 and 160 mm under the RL policy, the controller the robot will use, before
  deciding.

## The models

- **Source of the geometry:** PR #697's physics models (`refinement60/models/refined/physics.xml`, 128 mm, and
  `models/mid160/physics.xml`, 160 mm, at its commit 65a44fad). They are this build's generated capsule model with:
  - the fore-assembly repackaged: the X6 moved into a shoulder pocket on an offset mount with a standoff and a drop
    bracket, the upper strut, pushrod and limb cover set to the new length, the lower link re-angled so the neutral
    foot stays where it was;
  - small open-rib chassis edits: rib 4 relieved, the shell seam and the two ToF sensors moved, ToF mount arms added;
  - 12.844 and 12.925 kg in the PR's files (12.751 and 12.808 kg here, after the mass basis below) against
    12.694 kg; the same joints, actuators and keyframes.
- **How they are made here:** `sim/build_model60.py` gains a foreleg option that applies those changes to the
  generated model, so both variants come from the generator with a mass ledger, not as hand-copied files. Each
  generated variant is compiled and compared with PR #697's model array by array (every compiled array and option;
  only the recorded mass deviation below may differ), and the change set reproduces the PR's files exactly.
- **Mass basis, a recorded deviation from PR #697** (after the models' review, 23 Nov): the PR's 128 mm model keeps
  the 188 mm upper strut, upper limb cover and pushrod masses, and its 160 mm model scales those by its own ratio
  to 128 mm, so its strut came out heavier than the 188 mm one. Here each is the 188 mm part's mass times its length
  ratio (same section), and so is the elbow axle the PR lengthened from 40 to 45 mm: 12.751 kg (128 mm) and
  12.808 kg (160 mm) in total, against the PR's 12.844 and 12.925 kg. Logged as A-D22a for the owner to confirm.
  The PR's new parts keep its own mass estimates, without the closed variant's reserve growth (about 0.03 kg).
- **Packaging check** (`--packaging`; 9 values over each of roll, pitch and flexion): every geom of the front roll
  yoke, upper link and lower link against every chassis geom, and PR #697's named clearances that the physics model
  carries. 128 / 160 mm: X6 to upper strut 8.7 / 7.1 mm, X6 to chassis 7.7 / 7.7 mm, motor plate to rail 4.7 /
  4.7 mm, X6 to output crank 7.2 / 29.0 mm (the PR: 8.70, 7.07, 7.68, 4.68 and 7.155 mm). Overlaps left: the
  shoulder yoke around its own roll drive (the mount) and the schematic shoulder crossmember through the shoulder
  (F19), both in the 188 mm model too. Not checked here, taken from PR #697's evidence: the moving pushrod (the
  model's rod is fixed; the PR gives 3.50 mm to the X6) and the dressed meshes (rod to guard 3.30 mm, X6 to the
  sculpted rib 4.97 mm).
- **Against run 3's model:** the same chassis in effect (the open-rib edits do not collide and are mass-neutral),
  but a different fore-assembly: the shoulder pivot 40 mm lower, the X6 in the shoulder, 0.055-0.11 kg more
  fore-assembly hardware, and, with run 3's action scales, about 14 % less forefoot travel per unit action (pitch
  +-0.40 rad: +-111 mm at 188 mm, +-95 mm at 128 and 160 mm). 128 against 160 is fair on shoulder-driven foot
  travel, since both keep the same shoulder-to-foot distance; per +-0.4 rad of elbow the foot moves differently
  (+54/-69 mm forward and +49/-24 mm up at 128 mm, +45/-64 and +58/-36 mm at 160 mm), which is part of the design. Run 3 (188 mm) is therefore reported beside them as a reference, not as
  a third arm: its foreleg cannot be built (F18), and its fore-assembly differs in these ways besides the length.

## Training

Identical to run 3 apart from the model: `configs/run3_terrain.json`, 10,000,000 steps, the same notebook and
settings; four seeds per length (D24: two seeds cannot resolve a 15 % foot-load effect), so eight runs on Colab of
3-5 hours each (run 3's took 3.05 and 5.05 h).

## Measures (the pre-registration, PREREG_FORELEG_D22.md, fixes the judged ones and their thresholds)

- Run 3's rules F1, F2, T1-T5, so each length is also judged as run 3 was.
- Foot loads: the 10 ms mean and the 1 ms peak of each foot's normal force, in the gait report's target case and over
  the demo course (`demo_course.py`); the bench plan sizes the fixture from these.
- Drive heat: RMS torque over the continuous rating for the front pitch drives (the hottest in the demo) and the
  elbows, at 0.25 and 0.15 m/s, and the demo course's winding temperatures.
- Pushes, terrain and slippery floor: the sweep, as for run 3.
- Reported, not judged: proportions (upper foreleg over upper hind leg: 0.681 at 128 mm, 0.851 at 160 mm; the
  RGM.1332500 humerus over femur, 0.682) and the posture (elbow position).

## Order of work

1. The generator option and the equivalence check against PR #697's models; the packaging check.
2. An independent review of the models.
3. The pre-registration: rules, seeds, budgets and stop rules, committed before any judged run.
4. Step sheets for the four runs; the owner runs them.
5. Measurement with the tools at the run commit, the report, its independent review.

## What this does not settle

- Whether the real X6 mount, pushrod and guards fit: the bench article measures that.
- The museum renders due 8 Dec show the 188 mm foreleg; whether to redo them depends on the timing of this decision.
