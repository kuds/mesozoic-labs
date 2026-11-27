# Hardware

High-level notes on the robot versions of the simulated species: the parts and costs considered, the sizes tried, what
happened and what was learned. This repository holds simulation only; hardware implementation will get its own
repository. Nothing has been built or bought: every result is from simulation or a desk estimate. The one robot model
kept here is the Compsognathus revision B ([model](../../environments/compsognathus_longipes/README.md),
[design record](../reviews/COMPSOGNATHUS_ROBOT_DESIGN.md)). Only the adopted design becomes a species' robot model (its
one `robot` embodiment, [species naming](../SPECIES_NAMING.md)): alternative designs studied in parallel, such as the
75 cm and 60 cm bodies below, stay in `examples/` or on hardware branches until one is adopted (the maintainer's
decision of 2026-11-26).

## Where things stand

The first planned build is the 60 cm Triceratops, chosen on 18 Nov 2026: a quadruped 60 cm tall at the frill, 1.2 m long
and about 12 kg (targets), with the 75 cm T. rex's drive types. Its landed estimate (parts plus freight, import duty,
sales tax and contingency) is $20.1k–$23.0k, against a project target of about $20k. In simulation its trot passed every
rule at 10 ms of motor delay. The next step is planned, with no date adopted: a bench test of one foreleg's drives
(delay, heat inside a shell, safe stops) before the full robot is ordered. A review on 21 Nov 2026 found the parts, CAD
and safety design not yet ready to order.

The T. rex study paused the same day with a 75 cm body as its lead size: a biped must balance at every moment, and its
simulated controllers mostly fell under motor delay. The T. rex is kept as a later target, with a planning milestone of
end 2027 or early 2028. Notes from July 2026 had found no proven sub-$10k recipe for a two-legged runner (control was
the limit) and recommended a small four-legged first build, a juvenile Psittacosaurus shell on the Pupper V3. The pause
cited that advice; the Triceratops replaces the July pick with a larger quadruped.

## Robot concepts at a glance

| Concept | Body plan | Size | Status | Last updated |
|---|---|---|---|---|
| Triceratops, 60 cm | 4 legs | 60 cm frill, 1.2 m long | planned first build | 18 Nov 2026 |
| Triceratops, 75 cm (20 kg and 15 kg lists) | 4 legs | 75 cm frill, 1.5 m long | not adopted | 18 Nov 2026 |
| T. rex, 75 cm Hybrid body | 2 legs | 75 cm crown | lead size; study paused | 18 Nov 2026 |
| T. rex, 60 cm Tapered body | 2 legs | 60 cm crown | study paused | 18 Nov 2026 |
| T. rex, October concepts | 2 legs | 27 cm to 1 m tall | pull requests, all four closed unmerged | 23 Nov 2026 |
| Compsognathus revision B | 2 legs | 35 cm tall, 47 cm long | robot model; stands in simulation | Sep 2026 |
| Compsognathus walker | 2 legs | seven powered joints | on paper | 19 Jul 2026 |
| Velociraptor walker | 2 legs | 10 motors | on paper | 19 Jul 2026 |
| Raptor runner, full size | 2 legs | 1:1 | on paper | 24 Jul 2026 |
| Raptor runner, 0.72 scale | 2 legs | 9 motors | on paper, not independently verified | 24 Jul 2026 |
| Psittacosaurus on Pupper V3 | 4 legs | 25 × 20 × 22 cm | on paper | 24 Jul 2026 |
| Dibothrosuchus-inspired | 4 legs | 1.1–1.3 m long | on paper | 24 Jul 2026 |

Costs of the concepts priced so far, with ranges, bases and price dates, are in
[PARTS_AND_COSTS.md](PARTS_AND_COSTS.md); masses, dimensions and the terms *crown height*, *frill height*, *Hybrid* and
*Tapered* are in [SIZES.md](SIZES.md); the dated history of what was tried is in
[LESSONS_LEARNED.md](LESSONS_LEARNED.md). *Last updated* is the date of the latest status change or estimate.

## Documents

| Document | What it holds |
|---|---|
| [PARTS_AND_COSTS.md](PARTS_AND_COSTS.md) | Parts and estimated costs of each concept; the first purchases planned |
| [SIZES.md](SIZES.md) | Sizes tried for the T. rex (27–120 cm), the Triceratops (50–75 cm) and seven other concepts |
| [LESSONS_LEARNED.md](LESSONS_LEARNED.md) | What was tried, what happened, what carries over, open questions |
| [SIM_TO_REAL_PLAN.md](SIM_TO_REAL_PLAN.md) | Simulation guidance (written 19 Jul 2026): six sim-to-real gaps |
| [REFERENCE_MICRODUCK_2026_10.md](REFERENCE_MICRODUCK_2026_10.md) | A small open biped's training stack (31 Oct 2026) |

SIM_TO_REAL_PLAN.md advises against a physical T. rex; that verdict concerns the 85 kg simulated animal, not the
60–75 cm robots studied later.

## Where the detailed records are

The detailed records are not kept in this repository; each file link below is pinned to a commit that holds it, and the
other records stay in the git history. Pull-request states are as read on 23 Nov 2026.

| Record | Files, branches or pull requests |
|---|---|
| T. rex form study | [ROUND1_REVIEW.md][r1]: four heights; [ROUND2_REVIEW.md][r2]: the picks and the size study |
| | [DEFINITIONS.md][defs]: size and mass terms |
| 60 cm Triceratops build | never on `main`: [PARTS_60CM.md][parts60] (parts list) |
| | [RESULTS.md][walk60] (judged walk trials), [summary_walk60.md][sweep60] (delay sweep) |
| Folded in here | [HARDWARE_BOM.md][bom], [RAPTOR_SCALING_AND_ALTERNATIVES.md][rsa], [hw_chassis_study.py][chassis] |
| Branches never merged | `claude/mesozoic-trex-stage2-refine-4jffre`: code, records, reports, decisions |
| | `claude/mesozoic-trex-stage2-bundle-s2-b1`: 75 cm training inputs, never merged |
| | `claude/mesozoic-trex-stage2-archive-a1`: reference walks kept for training, never merged |
| Early T. rex concepts | [#635](https://github.com/kuds/mesozoic-labs/pull/635): mini, medium, 1 m concepts; closed |
| | [#648](https://github.com/kuds/mesozoic-labs/pull/648): mini torso layouts and leg folds; closed |
| | [#649](https://github.com/kuds/mesozoic-labs/pull/649): mini skeleton; closed |
| | [#651](https://github.com/kuds/mesozoic-labs/pull/651): 60 cm skeleton; closed |
| 60 cm T. rex platform | PR #662: platform and chassis layouts; closed |
| | PR #663: its findings (walking results not reproduced); closed |
| 75 cm T. rex, Jetson | PR #668: results unverified; closed |
| Triceratops design study | [#669](https://github.com/kuds/mesozoic-labs/pull/669): the 75 cm concept; open |

[bom]: https://github.com/kuds/mesozoic-labs/blob/87bafc864fefb16b040bd7a997fea8273bfc6649/docs/hardware/HARDWARE_BOM.md
[rsa]: https://github.com/kuds/mesozoic-labs/blob/87bafc864fefb16b040bd7a997fea8273bfc6649/docs/hardware/RAPTOR_SCALING_AND_ALTERNATIVES.md
[chassis]: https://github.com/kuds/mesozoic-labs/blob/87bafc864fefb16b040bd7a997fea8273bfc6649/environments/shared/scripts/hw_chassis_study.py
[r1]: https://github.com/kuds/mesozoic-labs/blob/87bafc864fefb16b040bd7a997fea8273bfc6649/hardware/trex_form_study/ROUND1_REVIEW.md
[r2]: https://github.com/kuds/mesozoic-labs/blob/87bafc864fefb16b040bd7a997fea8273bfc6649/hardware/trex_form_study/ROUND2_REVIEW.md
[defs]: https://github.com/kuds/mesozoic-labs/blob/87bafc864fefb16b040bd7a997fea8273bfc6649/hardware/trex_form_study/DEFINITIONS.md
[parts60]: https://github.com/kuds/mesozoic-labs/blob/7bfeef5334d0d7484e5428f1b77d2d1847d35c8c/hardware/triceratops_build/design/PARTS_60CM.md
[walk60]: https://github.com/kuds/mesozoic-labs/blob/7bfeef5334d0d7484e5428f1b77d2d1847d35c8c/hardware/triceratops_build/walk/results/walk60_2026-11-19/RESULTS.md
[sweep60]: https://github.com/kuds/mesozoic-labs/blob/7bfeef5334d0d7484e5428f1b77d2d1847d35c8c/hardware/triceratops_build/walk/results/sweep_walk60/summary_walk60.md
