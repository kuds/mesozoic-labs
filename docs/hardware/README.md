# Hardware & Sim-to-Real

Planning and feasibility docs for taking the simulation onto **physical robots**.
These are *plans/designs* in the [docs map](../README.md) sense — nothing here
has been built. `hardware_prototype` is still `planned` in
[`species_manifest.toml`](../../configs/species_manifest.toml).

**Compsognathus model update (2026-09-07):** an
[anatomical proxy and Rev B robot prototype](../../environments/compsognathus_longipes/README.md)
now load and stand in MuJoCo. The robot preserves the later twelve-leg-axis,
1.5856 kg design with a fixed passive tail. Use that page for current
geometry and validation limits; the older cost figures below are not an
updated bill of materials for Rev B. Neither prototype has demonstrated walking.
In the repository a robot is the `robot` embodiment of its species
([species naming](../SPECIES_NAMING.md)): the Rev B prototype is the model
`compsognathus_longipes` / `robot`, and a later hardware revision of it would
be a physics revision of that embodiment, not a new one.

> **Headline conclusion (2026): there is no buildable-today, sub-$10k,
> two-legged _running_ recipe — the limit is control, not motors.** The
> recommended solo first build is a juvenile-Psittacosaurus shell on the open
> Pupper V3 quadruped, capped at ~$4.7k. Reliable walking/trotting is the V1
> deliverable; 1 m/s is a measured stretch target, not a promise. At a ~$10k
> parts cap, a clean-sheet, catalog-actuated _Dibothrosuchus
> elaphros_-inspired quadruped can credibly target 2 m/s and treat 3 m/s as a
> short-sprint stretch. A Compsognathus remains a reliable ~$1k **walker**,
> while full-size bipedal running (13.5 kg raptor) is a $26k+ parts program.

## Reading order

The [Mini Tyrannosaurus Rex prototype](MINI_TYRANNOSAURUS_REX.md) adds a 25.8 cm,
800 g assigned-mass forward-knee MJCF study with a lighter head, linked moving
arms, service-panel views, component ledger and preliminary $2,000 allocation. Its
[standalone example](../../examples/mini_tyrannosaurus_rex/README.md) can be viewed and
smoke-checked without registering a new plant. It has not demonstrated walking
or continuous hardware motor duty.

The [T-Rex size trade study](TREX_SIZE_TRADE_STUDY.md) compares the miniature,
50 cm and 80 cm concepts, with sourced actuator options, reproducible load/mass
screens, first-project budgets and demonstration/marketing tradeoffs. It
recommends a 50 cm packaging and actuator bench study before buying a complete
motor set. The larger concepts have not undergone CAD or gait validation.

The [detailed 50 cm parts and compute plan](studies/trex_50cm/PARTS_AND_COMPUTE.md)
updates that size study with a Jetson/MCU architecture, stronger leg motors,
camera/power choices, a sourced line-item BOM and a cartoon character brief.
Its fuller 7.62 kg mass budget and approximately $14,000 first-project funding
plan supersede the earlier 50 cm shared allowances; hardware remains unqualified.

The [robot variants and build plan](ROBOT_VARIANTS_AND_BUILD_PLAN.md) proposes
how concurrent mini, medium and large robots of one species should be identified,
tested and documented, with a staged mini-to-50 cm timeline at 15–20 hours/week.
It is an architecture proposal; the runtime has not implemented variant selection.

1. **[SIM_TO_REAL_PLAN.md](SIM_TO_REAL_PLAN.md)** — start here. The six
   sim↔hardware gaps (privileged observations, actuation, control loop, domain
   randomization, deployment stack, project maturity) and the phased plan to
   close them.
2. **[HARDWARE_BOM.md](HARDWARE_BOM.md)** — parts, cost, and buildability for
   *walking* robots: Compsognathus (~$950) and Velociraptor (~$8.7k).
3. **[RAPTOR_SCALING_AND_ALTERNATIVES.md](RAPTOR_SCALING_AND_ALTERNATIVES.md)** —
   *running* robots: raptor scaling (full ~$36k, scaled ~$6.9k), the
   bipedal-vs-quadruped analysis, and the recommended sub-$5k first physical
   platform (a juvenile _Psittacosaurus_ on Pupper V3, ~$4.7k program cap),
   plus the custom ~$10k _Dibothrosuchus_-inspired speed tier, proximal-actuator
   packaging, and species tradeoffs.

## Conventions for this track

- Prices are **web-sourced 2026 USD, parts only** (engineering labor excluded);
  expect ±15–25%.
- Torque/mass/cost figures were adversarially verified where noted; corrections
  are folded into each doc rather than appended.
- Two design dials recur throughout: **scale and support pattern set torque**
  (torque scales roughly with size⁴, while a biped loads fewer legs) and
  **replication buys a shorter controls path**. A roughly 10–12 kg quadruped can
  use catalog actuators; a full-scale raptor biped cannot.
