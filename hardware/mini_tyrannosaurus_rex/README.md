# Mini T. rex hardware milestones

One evolving robot, three development milestones. A milestone may contain
several hardware revisions and many training runs. Milestones do not replace
the stance, recovery, locomotion and target-reaching behavior stages.

| Milestone | Physical scope | Exit evidence |
|---|---|---|
| [M1 skeleton](m01_skeleton/README.md) | Torso A, forward-folding legs, full hardware envelopes and fixed future-load allowances | Identified actuators, measured mass/inertia, clearance, standing, crouching, weight transfer and initial stepping |
| [M2 integrated](m02_integrated/README.md) | Actual head structure, fixed tail, shell, mounts and routed cables | Repeat M1 tests with the integrated body; evaluate walking, turning, cooling and service access |
| [M3 production candidate](m03_production_candidate/README.md) | Selected parts, toleranced drawings, harness and assembly process | Repeatable assembly, endurance, repairability and repeat-build behavior evidence |

## Identity and storage

The mini is `tyrannosaurus_rex / robot`. The active simulation lives at
`environments/tyrannosaurus_rex/robot/assets/tyrannosaurus_rex_robot.xml`, and
its SB3 recipes at `configs/tyrannosaurus_rex/robot/`. Training never reads
hardware folders or imports checkout-only examples. Layout alternatives remain
in `examples/mini_tyrannosaurus_rex/`; the M1 builder promotes a selected study
into a standalone, committed training asset.

Record each hardware revision in its milestone folder with the bill of
materials, CAD/drawings when available, assembly instructions, calibration and
validation evidence. Current dimensions and masses are allocations; no physical
prototype or manufacturing drawings are claimed by this first revision.

An accepted milestone gets a Git tag such as `mini-m1-hw-r01` pointing at its
exact commit, complete MJCF/asset closure and configuration. No release tag is
created until the acceptance evidence exists. Use that commit/worktree to replay
historical runs; there is no runtime `--hardware-revision` selector. Avoid moving
tags or having historical reports resolve a mutable `latest` model.

Hardware revision, physics revision, policy-interface revision, visual revision,
task digest and training run ID are separate records. Bump affected plant layers
through `configs/plant_versions.toml`, regenerate the plant manifest, digest
snapshot and public catalog, and retain the old run's identity. Visual changes
can remain policy-compatible only when physics and the interface are unchanged.
Mass placement, collision, gains or joint changes require new physical-model
validation; old checkpoints and normalization statistics are not relabeled.

Each run retains its model identity, Git commit, resolved task/algorithm settings,
seed, selected checkpoint, normalization state, parent lineage and evaluation
evidence through the existing result-bundle pipeline. The MJCF's custom metadata
records M1/hw_r01 and its engineering source hash. Keep hardware calibration
with the corresponding revision and link measured deployments by device serial,
firmware version and calibration file. Large run artifacts belong in the normal
run store; only curated summaries and review evidence belong in Git.

The 100 cm robot is a separate mechanical development effort. Reuse measurement,
training and behavior interfaces; size-specific motors, dynamics, calibration
and learned policies need their own validation. The current two-embodiment
registry does not yet provide a simultaneous mini/100 cm product selector.
