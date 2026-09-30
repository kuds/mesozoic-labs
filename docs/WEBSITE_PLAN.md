# Website Improvement Plan

**Status (2026-09-30): complete.** Everything under Completed holds. The two
items left under Remaining are settled: the two apex GIFs stay, unused, by the
maintainer's decision of 2026-09-30 (they share their blobs with
`results/velociraptor/{ppo,sac}/stage3_strike.gif`, so the cleanup plan allowed
deleting both copies or neither; [CLEANUP_PLAN_2026_09.md](CLEANUP_PLAN_2026_09.md)
§3.2, CU-16 row), and the logo's size is a [KNOWN_ISSUES.md](KNOWN_ISSUES.md)
entry (Configs, docs & website), which corrects the remedy below: the SVG
wraps one PNG and has no paths and no SVG editor metadata, so SVGO would not
shrink it.

## Completed

The following items from the original plan have been implemented:

- **1a. Features Section** — `FeaturesSection` component added to `index.tsx` with all 4 features
- **1b. Simulation Preview Section** — `SimulationSection` with terminal window, real API code, and `raptor_balance_ppo.gif`
- **1c. Roadmap Section** — `RoadmapSection` with six phase cards (Phase 0 complete; Phases 1–3 in progress, as in ROADMAP.md)
- **1d. CTA Section** — "GET STARTED" and "VIEW ON GITHUB" buttons
- **1e. Remove "COMING SOON" Badge** — Badge removed; public title and tagline now describe dinosaur-inspired simulation research
- **2a. Fix CHANGELOG Version Status** — v0.2.0 dated 2026-02-09
- **2b. Fix `editUrl`** — Both docs and blog editUrl point to `kuds/mesozoic-labs`
- **2c. Fix GitHub Discussions Link** — Points to `kuds/mesozoic-labs/discussions`
- **2d. Update Privacy Policy Date** — Updated to February 2026
- **3a. Custom Models Page** — Fully developed with architecture overview, MuJoCo XML format, requirements, and examples
- **3d. Hyperparameters Page** — Comprehensive per-stage parameters, PPO/SAC tables, curriculum thresholds, CLI overrides
- **3e. Generated Species Catalog** — Model dimensions, current stages, artifact provenance, and result summaries render from generated data
- **4b. Footer Cleanup** — Well-organized footer with Docs, Community, and More sections

---

## Remaining

### Use the Training GIFs
- `static/img/ppo_apex.gif` (1.1 MB) and `static/img/sac_apex.gif` (22 MB) exist but are unused
- `raptor_balance_ppo.gif` is used in the Simulation section, but the apex GIFs show more advanced locomotion
- Consider embedding `ppo_apex.gif` in the hero or a results section; `sac_apex.gif` may be too large (22 MB) for web use without compression

### Optimize Logo SVG
- `static/img/logo.svg` is 38 KB — large for an SVG
- Run through SVGO or manually clean up (remove editor metadata, simplify paths) to improve page load performance
