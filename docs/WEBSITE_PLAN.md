# Website Improvement Plan

## Completed

The following items from the original plan have been implemented:

- **1a. Features Section** — `FeaturesSection` component added to `index.tsx` with all 4 features
- **1b. Simulation Preview Section** — `SimulationSection` with terminal window, real API code, and `raptor_balance_ppo.gif`
- **1c. Roadmap Section** — `RoadmapSection` with all 6 phases (Phase 0 complete, Phase 1 active)
- **1d. CTA Section** — "GET STARTED" and "VIEW ON GITHUB" buttons
- **1e. Remove "COMING SOON" Badge** — Badge removed, title updated to "Robotic Dinosaur Locomotion"
- **2a. Fix CHANGELOG Version Status** — v0.2.0 dated 2026-02-09
- **2b. Fix `editUrl`** — Both docs and blog editUrl point to `kuds/mesozoic-labs`
- **2c. Fix GitHub Discussions Link** — Points to `kuds/mesozoic-labs/discussions`
- **2d. Update Privacy Policy Date** — Updated to February 2026
- **3a. Custom Models Page** — Fully developed with architecture overview, MuJoCo XML format, requirements, and examples
- **3d. Hyperparameters Page** — Comprehensive per-stage parameters, PPO/SAC tables, curriculum thresholds, CLI overrides
- **4b. Footer Cleanup** — Well-organized footer with Docs, Community, and More sections

---

## Remaining

### Use the Training GIFs
- `static/img/ppo_apex.gif` (1.1 MB) and `static/img/sac_apex.gif` (22 MB) exist but are unused
- `raptor_balance_ppo.gif` is used in the Simulation section, but the apex GIFs show more advanced locomotion
- Consider embedding `ppo_apex.gif` in the hero or a results section; `sac_apex.gif` may be too large (22 MB) for web use without compression

### Clarify "Basic Dinosaur" in Intro Results Table
- `docs/intro.md` lines 26-27 show "Basic Dinosaur" results (PPO: 319.94, SAC: 3091.31) without context
- Clarify what species/stage these refer to, or replace with per-species results

### Add Per-Species Results to Training Docs
- `docs/training/ppo.md` — Has hyperparameter tables and Velociraptor results; missing T-Rex and Brachiosaurus results
- `docs/training/sac.md` — Same: has overview and Velociraptor results; missing other species

### Optimize Logo SVG
- `static/img/logo.svg` is 38 KB — large for an SVG
- Run through SVGO or manually clean up (remove editor metadata, simplify paths) to improve page load performance
