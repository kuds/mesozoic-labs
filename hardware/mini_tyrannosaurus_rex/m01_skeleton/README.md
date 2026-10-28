# M1 skeleton — hw_r01

Torso A with forward-folding five-axis legs, ten full 20 × 34 × 26 mm motor
envelopes, full computer/battery/interface/regulator reserves, and a fixed tail.
The 0.7541 kg simulation includes 0.6391 kg of chassis allocations and 0.115 kg
of future head/neck, tail and arm inertia allowances. These are not measured
parts. The retained 30 g torso-skin allowance makes this a loaded development
baseline even though the outer shell is hidden.

The head/neck (90 g), tail (15 g) and arms (10 g) are fixed inertia bodies at
the previous study's locations. Their translucent markers are **not physical
modules, collision-qualified shapes or mounting fixtures**. Their inertia is
retained so locomotion is not developed against an unrealistically light body.
Design and weigh a representative ballast fixture before hardware trials;
replace the allowances with its measured properties in a new physics revision.
The battery remains inside the torso A packing envelope. The rejected outer
silhouette is removed; this milestone focuses on the chassis.

The model uses the study's conservative joint travel and ideal PD actuators,
capped at 0.10 N·m. The cap is a simulation assumption, not a continuous motor
rating. The home keyframe is lowered by 10 micrometres to establish contact for
support-polygon calibration. Both sole touch volumes include all contact
corners; their summed force is checked against model weight.

## Files and evidence

- `revision.toml`: milestone/hardware identity, model allocation and open items.
- `bom.csv`: full mass allocation and component-envelope provenance.
- `validation.json`: reproducible model checks and seeded standing results.
- `preview.png`: three views rendered from the actual committed MuJoCo asset.
- `recovery_measurement.json`: raw quiet, selection and held-out push measurements.
- `training_smoke.json`: 2,048-step PPO integration and policy/normalizer reload evidence.
- `../../../examples/mini_tyrannosaurus_rex/skeleton/`: builder and validation adapter.

Model checks are simulation evidence. The inherited finite-pose hardware
clearance study covers motor boxes, installed electronics reserves and soles;
it excludes detailed mounts, moving cables and future modules. There are no
manufacturing CAD files or physical certification results yet.

The recorded 1,050-pose sweep passes 162,805 pair checks at a 2 mm margin.
All eight seeded 30-second home-command standing runs pass. The 14-second
commanded crouch lowers the pelvis 12.57 mm during its hold, with 0.46 degrees
maximum tilt. This validates these sampled motions, not all possible joint
combinations or single-support walking.

The measured recovery multiple is 1.2. On the separate 40-episode diagnostic
panel, zero commands recover in 12 episodes and fixed brace commands in 13;
at the 1.5 stress multiple neither control recovers. These are PD null controls,
not learned policies. Keep this raw evidence separate from the shared profile
at `configs/tyrannosaurus_rex/robot/recovery_calibration.json`.

The shared PPO trainer completed 2,048 steps. Its checkpoint and frozen
normalization state reloaded through the plant-identity checks and produced
finite observations, actions and rewards for 100 further steps. This is an
integration smoke check; no behavior gate was evaluated.

## Run and train

From the repository root:

```bash
python -m examples.mini_tyrannosaurus_rex.skeleton.build
python -m examples.mini_tyrannosaurus_rex.skeleton.validate
python -m examples.mini_tyrannosaurus_rex.skeleton.render
python -m examples.mini_tyrannosaurus_rex.skeleton.calibrate --measure
python -m environments.tyrannosaurus_rex.scripts.view_model --embodiment robot
python -m environments.shared.train --species tyrannosaurus_rex --embodiment robot train --stage stance
```

`calibrate --measure` repeats the shared push experiment before packaging the
profile. Without `--measure`, it only packages existing evidence and refuses
changed source hashes or task settings. The exact smoke-training command and
runtime versions are recorded in `training_smoke.json`.

In the SB3 notebook select `SPECIES = "tyrannosaurus_rex"`,
`EMBODIMENT = "robot"`, and start with `BEHAVIOR = "stance"` and
`QUICK_TEST = True`. Full recipes have experimental budgets; a short integration
run demonstrates the pipeline, not learned balance or walking.

`stages.toml` defines stance; recovery from stance; locomotion from stance; and
target reaching from locomotion. The legacy recipe label `hunt` means an
upright, slow planar target arrival for this skeleton, not biting. Later
body/head behaviors belong to M2. The recovery calibration characterizes
fixed-command PD controls in simulation, not a learned or hardware controller.

The 41 observations include joint state, heading-free gravity/velocity, IMU
signals, summed foot forces, target direction/distance and body-relative
commands. Velocity, target coordinates and foot forces are privileged simulator
signals. No onboard sensing/estimation pipeline is implemented here. The ten
actions are home-relative position commands, filtered at 10 Hz and applied at
50 Hz. Actual actuator and sensor identification precedes sim-to-real work.

## Acceptance record

M1 remains **in progress**. Simulation standing and clearance are only part of
its exit criteria. Record failures alongside passes. Single-support transfer,
stepping, learned locomotion, real actuator identification, wiring, load-bearing
mounts and physical tests remain open. No milestone release tag is issued.
