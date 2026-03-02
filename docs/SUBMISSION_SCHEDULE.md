# CoRL 2026 Submission Schedule

**Abstract deadline:** May 25, 2026
**Full paper deadline:** May 28, 2026
**Today:** March 2, 2026 (~12 weeks)

---

## Current Status (Blockers)

| Item | Status | Risk |
|------|--------|------|
| Velociraptor Stage 1 | DONE | - |
| Velociraptor Stage 2 | BLOCKED — config mismatch, solved once but not reproducible | HIGH |
| Velociraptor Stage 3 | 1 attempt — blocked by Stage 2 | HIGH |
| T-Rex Stage 1 | 299.3/300 ep_len — effectively solved | LOW |
| T-Rex Stage 2-3 | In progress, forgetting issues | MEDIUM |
| Brachiosaurus all stages | NOT STARTED (zero runs) | HIGH |
| Domain randomization code | NOT IMPLEMENTED | HIGH |
| Domain randomization experiments | Blocked by above | HIGH |
| Paper draft | Not started | - |

---

## Phase 1: Fix Training (Weeks 1–3, March 3–22)

> Goal: All 3 species complete Stage 1–2 reliably.

### Week 1 (March 3–8)
- [ ] **Fix Velociraptor Stage 2 config** — revert to what worked:
      `alive_bonus=0.5, posture_weight=0.2, nosedive_weight=0.3`
- [ ] **Launch Velociraptor Stage 2** — 3 seeds, 3M steps each
- [ ] **Launch Brachiosaurus Stage 1** — initial runs, 3 seeds
- [ ] **T-Rex Stage 1** — extend to 7M steps OR relax threshold to 295

### Week 2 (March 9–15)
- [ ] **Analyze Velociraptor Stage 2 results** — confirm reproducibility
- [ ] **Launch Velociraptor Stage 3** from best Stage 2 checkpoint
- [ ] **T-Rex Stage 2** — tune config based on training review findings
- [ ] **Brachiosaurus Stage 1** — analyze, tune, re-run if needed

### Week 3 (March 16–22)
- [ ] **Complete remaining Stage 2–3 runs** for T-Rex
- [ ] **Brachiosaurus Stage 2** — begin locomotion training
- [ ] **Checkpoint:** All species have at least Stage 1–2 solved
- [ ] **Document final hyperparameters** for each species/stage

**Milestone: Baseline training complete for all 3 species.**

---

## Phase 2: Domain Randomization (Weeks 4–6, March 23–April 12)

> Goal: Implement DR code and run robustness experiments.

### Week 4 (March 23–29)
- [ ] **Implement domain randomization in BaseDinoEnv:**
  - Body mass variation (±10%)
  - Joint damping randomization
  - Ground friction variation
- [ ] **Implement sensor noise:**
  - Gaussian noise on joint position/velocity
  - Accelerometer bias drift
  - Touch sensor threshold variation
- [ ] **Unit tests** for all randomization components

### Week 5 (March 30–April 5)
- [ ] **Implement action delay** (1–3 step latency)
- [ ] **Implement terrain variation** (heightfields, slopes, friction zones)
- [ ] **Create DR config schema** in TOML (randomization ranges per species)
- [ ] **Launch DR training runs** — Velociraptor first (most mature)
  - Baseline (no DR) vs. light DR vs. heavy DR, 3 seeds each = 9 runs

### Week 6 (April 6–12)
- [ ] **DR runs for T-Rex and Brachiosaurus** (9 runs each)
- [ ] **Robustness evaluation:** test trained policies under unseen perturbations
- [ ] **Collect transfer metrics:** reward degradation, episode survival under DR
- [ ] **Brachiosaurus Stage 3** — wrap up remaining training

**Milestone: Domain randomization experiments complete, transfer data in hand.**

---

## Phase 3: Analysis & Figures (Weeks 7–8, April 13–26)

> Goal: All quantitative results finalized, figures and tables camera-ready.

### Week 7 (April 13–19)
- [ ] **Statistical analysis** across all runs (mean ± std over seeds)
- [ ] **Key figures:**
  - Training curves (reward vs. steps) per species, per stage
  - Curriculum transition visualization (stage advancement points)
  - Cross-morphology comparison table (obs dims, actions, mass, results)
- [ ] **DR figures:**
  - Robustness heatmap (performance vs. randomization strength)
  - Policy degradation under out-of-distribution perturbations

### Week 8 (April 20–26)
- [ ] **Final result tables** with confidence intervals
- [ ] **Ablation figures** (curriculum vs. no curriculum, DR vs. no DR)
- [ ] **Architecture diagram** of the unified framework
- [ ] **MJCF model visualizations** for each species
- [ ] **Record supplementary video** (~3 min, max 250 MB):
  - Side-by-side locomotion of all 3 species
  - Curriculum stage progression
  - DR robustness comparison

**Milestone: All experimental results and figures finalized.**

---

## Phase 4: Paper Writing (Weeks 9–11, April 27–May 17)

> Goal: Complete 8-page paper draft, internally reviewed.

### Week 9 (April 27–May 3) — First Draft
- [ ] **Section 1: Introduction** (~1 page)
  - Problem: multi-morphology locomotion, per-species engineering cost
  - Contribution summary (5 bullets from abstract)
- [ ] **Section 2: Related Work** (~1 page)
  - Legged locomotion RL (ANYmal, Cassie, quadrupeds)
  - Curriculum learning in robotics
  - Sim-to-real transfer and domain randomization
  - Multi-morphology / universal locomotion controllers
- [ ] **Section 3: Method** (~2 pages)
  - Framework architecture (BaseDinoEnv, TOML configs, curriculum manager)
  - Three morphologies and their MJCF models
  - Curriculum design (3 stages, advancement criteria)
  - Domain randomization strategy

### Week 10 (May 4–10) — Results + Polish
- [ ] **Section 4: Experiments** (~2.5 pages)
  - Experimental setup (MuJoCo, SB3, hardware, training budget)
  - Baseline results per species (table + curves)
  - Curriculum ablation (with vs. without)
  - DR robustness analysis (with vs. without, light vs. heavy)
  - Cross-morphology insights (forgetting patterns, reward sensitivity)
- [ ] **Section 5: Limitations** (~0.5 page, MANDATORY)
  - Simulation-only (no physical robot validation yet)
  - 3 morphologies (limited diversity)
  - PPO/SAC only (no model-based methods)
  - Dinosaur-specific — generalization to arbitrary morphologies untested

### Week 11 (May 11–17) — Internal Review
- [ ] **Section 6: Conclusion** (~0.5 page)
- [ ] **Full internal review** — co-authors read and comment
- [ ] **Double-blind audit** — remove all identifying information
  - Anonymize repo references, author names, affiliations
  - Check figures for embedded metadata
- [ ] **Revise based on feedback** — tighten prose, fix notation

**Milestone: Paper draft complete and internally reviewed.**

---

## Phase 5: Finalize & Submit (Weeks 11.5–12, May 18–28)

### May 18–22 (Final Polish)
- [ ] **Finalize abstract text** (may refine from initial draft based on actual results)
- [ ] **Format check** — LaTeX template compliance, 8-page limit
- [ ] **References cleanup** — consistent format, all cited works present
- [ ] **Finalize supplementary ZIP** (video + appendix if any)
- [ ] **Lock author list** (cannot change after abstract deadline)

### May 25 (Abstract Deadline)
- [ ] **Submit abstract on OpenReview**
- [ ] **Verify author list is correct** — no changes allowed after this

### May 26–27 (Final Paper Push)
- [ ] **Final read-through** by all co-authors
- [ ] **Last-minute result updates** if any late runs finished
- [ ] **Proofread** — typos, figure references, table numbering

### May 28 (Paper Deadline)
- [ ] **Submit full paper + supplementary on OpenReview**
- [ ] **Verify submission** — PDF renders correctly, all files uploaded

---

## Weekly Time Estimates

| Phase | Weeks | Primary Focus | Parallel Work |
|-------|-------|---------------|---------------|
| 1. Fix Training | 1–3 | Training runs, tuning | — |
| 2. Domain Randomization | 4–6 | DR code + experiments | Finish lagging training |
| 3. Analysis & Figures | 7–8 | Results, plots, video | Outline paper sections |
| 4. Paper Writing | 9–11 | Writing, review | — |
| 5. Submit | 11.5–12 | Polish, submit | — |

## Risk Mitigation

| Risk | Likelihood | Mitigation |
|------|-----------|------------|
| Velociraptor Stage 2 still not reproducible | MEDIUM | Fall back to reporting single-run result with high variance caveat |
| Brachiosaurus doesn't solve in time | MEDIUM | Include partial results + training curves; state as in-progress |
| Domain randomization hurts all policies | LOW | Report negative result honestly — this is still a contribution |
| Writing takes longer than planned | HIGH | Start outlining in Week 6; co-authors draft sections in parallel |
| OpenReview submission issues | LOW | Do a dry-run submission 1 week early |

## Hard Deadlines (Non-Negotiable)

| Date | Event |
|------|-------|
| **March 22** | All baseline training must be running or complete |
| **April 12** | All domain randomization experiments launched |
| **April 26** | All figures and results finalized — writing begins |
| **May 17** | Full paper draft complete for internal review |
| **May 22** | Author list locked, abstract finalized |
| **May 25** | Abstract submitted on OpenReview |
| **May 28** | Full paper submitted on OpenReview |
