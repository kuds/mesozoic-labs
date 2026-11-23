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
  - 12.844 kg in total against 12.694 kg; the same joints, actuators and keyframes.
- **How they are made here:** `sim/build_model60.py` gains a foreleg option that applies those changes to the
  generated model, so both variants come from the generator with a mass ledger, not as hand-copied files. Each
  generated variant is checked against PR #697's physics model geom by geom (positions, sizes, masses) and reports
  any difference.
- **Packaging check:** the X6-to-upper-strut signed distance and the other clearances the PR reports, at the neutral
  keyframe and over the joint ranges, must be positive in the generated models (F18's check).
- **Chassis:** both lengths use the same open-rib chassis, so 128 against 160 differs only in the foreleg. Run 3
  (188 mm, the closed chassis, 0.15 kg lighter) is reported beside them as a reference, not as a third arm: its
  chassis differs and its foreleg cannot be built.

## Training

Identical to run 3 apart from the model: `configs/run3_terrain.json`, 10,000,000 steps, seeds 0 and 1, the same
notebook and settings. Four runs on Colab, about 3 hours each (run 3's two took 3.0-3.1 h).

## Measures (the pre-registration picks the judged ones and their thresholds)

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
