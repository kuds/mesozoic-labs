# Hardware & Sim-to-Real

Planning and feasibility docs for taking the simulation onto **physical robots**.
These are *plans/designs* in the [docs map](../README.md) sense — nothing here
has been built. `hardware_prototype` is still `planned` in
[`species_manifest.toml`](../../configs/species_manifest.toml).

> **Headline conclusion (2026): there is no buildable-today, sub-$10k, two-legged
> _running_ robot — the limit is control, not motors.** An agile runner under
> $10k must be a **quadruped** (Scutellosaurus, ~$5–7k). A two-legged build
> (Compsognathus) is a reliable ~$1k **walker**; running on two legs is a
> research bet, not a deliverable. Full-size bipedal running (13.5 kg raptor) is
> a $20k+ research program.

## Reading order

1. **[SIM_TO_REAL_PLAN.md](SIM_TO_REAL_PLAN.md)** — start here. The six
   sim↔hardware gaps (privileged observations, actuation, control loop, domain
   randomization, deployment stack, project maturity) and the phased plan to
   close them.
2. **[HARDWARE_BOM.md](HARDWARE_BOM.md)** — parts, cost, and buildability for
   *walking* robots: Compsognathus (~$950) and Velociraptor (~$8.7k).
3. **[RAPTOR_SCALING_AND_ALTERNATIVES.md](RAPTOR_SCALING_AND_ALTERNATIVES.md)** —
   *running* robots: raptor scaling (full ~$36k, scaled ~$6.9k), the
   bipedal-vs-quadruped analysis, and the recommended sub-$10k agile-runner
   species (a Scutellosaurus quadruped, ~$5–7k / sub-$5k budget variant).

## Conventions for this track

- Prices are **web-sourced 2026 USD, parts only** (engineering labor excluded);
  expect ±15–25%.
- Torque/mass/cost figures were adversarially verified where noted; corrections
  are folded into each doc rather than appended.
- Two design dials recur throughout: **money buys _size_** (torque ~ size⁴, so
  above ~10 kg you need custom actuators) and **legs buy _reliable running_**
  (quadrupeds have a solved cheap gait stack; cheap bipedal running is unsolved).
