# Hardware & Sim-to-Real

Planning and feasibility docs for taking the simulation onto **physical robots**.
These are *plans/designs* in the [docs map](../README.md) sense — nothing here
has been built. `hardware_prototype` is still `planned` in
[`species_manifest.toml`](../../configs/species_manifest.toml).

> **Headline conclusion (2026): there is no buildable-today, sub-$10k,
> two-legged _running_ recipe — the limit is control, not motors.** The
> recommended solo first build is a juvenile-Psittacosaurus shell on the open
> Pupper V3 quadruped, capped at ~$4.7k. Reliable walking/trotting is the V1
> deliverable; 1 m/s is a measured stretch target, not a promise. At a ~$10k
> parts cap, a clean-sheet, catalog-actuated _Terrestrisuchus_-class quadruped can
> credibly target 2 m/s and treat 3 m/s as a stretch. A Compsognathus remains a
> reliable ~$1k **walker**, while full-size bipedal running (13.5 kg raptor) is
> a $26k+ parts program.

## Reading order

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
   plus the custom ~$10k _Terrestrisuchus_ speed tier and species tradeoffs.

## Conventions for this track

- Prices are **web-sourced 2026 USD, parts only** (engineering labor excluded);
  expect ±15–25%.
- Torque/mass/cost figures were adversarially verified where noted; corrections
  are folded into each doc rather than appended.
- Two design dials recur throughout: **scale and support pattern set torque**
  (torque scales roughly with size⁴, while a biped loads fewer legs) and
  **replication buys a shorter controls path**. A roughly 10 kg quadruped can
  use catalog actuators; a full-scale raptor biped cannot.
