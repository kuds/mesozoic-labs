# Documentation Map

The docs fall into four categories with different lifecycles. Knowing which
category a document belongs to tells you whether to trust it as current,
read it as history, or update it when things change.

| Category | Lifecycle | Where |
|---|---|---|
| **Living reference** | Continuously updated; always current | `docs/` root ([KNOWN_ISSUES.md](KNOWN_ISSUES.md)) |
| **Plans & designs** | Updated until executed, then marked complete | `docs/` root |
| **Investigations & run analyses** | Point-in-time, dated; never rewritten (corrections are appended or cross-linked) | [`investigations/`](investigations/) |
| **Code & repo reviews** | Archived records; findings migrate to KNOWN_ISSUES | [`reviews/`](reviews/) |

## Living reference

- [KNOWN_ISSUES.md](KNOWN_ISSUES.md) — the single list of verified-but-unfixed
  findings and standing recommendations. Fixed items are deleted; full context
  stays in the archived review or investigation they came from.
- [PLANT_CONTRACT.md](PLANT_CONTRACT.md) — versioned policy-interface,
  physics, visual, and source identities for MuJoCo models and checkpoints.
- [RESULT_BUNDLES.md](RESULT_BUNDLES.md) — canonical Colab/Google Drive result
  artifacts, provenance capture, and promotion validation.

## Plans & designs

| Document | Date | Status |
|---|---|---|
| [ROADMAP.md](ROADMAP.md) | 2026-04-18 | Active — phased project timeline |
| [RECOMMENDATIONS.md](RECOMMENDATIONS.md) | 2026-03-25 | Active — codebase-wide improvement backlog |
| [RL_TRAINING_PLAN.md](RL_TRAINING_PLAN.md) | 2026-03-26 | Active — remaining SAC/PPO runs across species |
| [MJX_CONVERSION_PLAN.md](MJX_CONVERSION_PLAN.md) | 2026-07-13 | Implemented design record — current divergences live in KNOWN_ISSUES and the JAX guide |
| [WEBSITE_PLAN.md](WEBSITE_PLAN.md) | — | Active — Docusaurus site improvements |
| [BALANCE_REWARD_METRICS.md](BALANCE_REWARD_METRICS.md) | 2026-03-16 | Proposed — composite ASHA metric for stage-1 sweeps |
| [REFACTORING.md](REFACTORING.md) | 2026-03-19 | **Complete** — v0.3.0 consolidation plan |
| [CODE_CONSOLIDATION.md](CODE_CONSOLIDATION.md) | 2026-03-19 | **Complete** — v0.3.0 implementation record |

## Investigations & run analyses

Dated, evidence-driven analyses of training runs and reward behavior, in
chronological order. Each is frozen at its date; follow-ups cross-link rather
than rewrite.

| Document | Date | One-liner |
|---|---|---|
| [TRAINING_REVIEW.md](investigations/TRAINING_REVIEW.md) | 2026-03-25 | Review of 140+ SB3 runs (Feb–Mar); empirical basis for the low-penalty stage-2 reward recipe |
| [TRAINING_REVIEW_JAX_STAGE1.md](investigations/TRAINING_REVIEW_JAX_STAGE1.md) | 2026-04-01 | T-Rex JAX/MJX stage-1 run review (KL-halt behavior) |
| [REWARD_DISCREPANCY_INVESTIGATION.md](investigations/REWARD_DISCREPANCY_INVESTIGATION.md) | 2026-04-02 | T-Rex stage-1 reward vs episode-length inconsistency root cause |
| [REWARD_SCALE_REDESIGN.md](investigations/REWARD_SCALE_REDESIGN.md) | 2026-04-18 | Stage-3 terminal-bonus rescale analysis (implementation deferred) |
| [STAGE2_INVESTIGATION.md](investigations/STAGE2_INVESTIGATION.md) | 2026-07-09 | Root cause of the velociraptor stage-2 locomotion collapse (bounded-plant actuator clipping) |
| [STAGE2_RECOMMENDATIONS.md](investigations/STAGE2_RECOMMENDATIONS.md) | 2026-07-11 | Replication review of run 20260711_165924, corrected fall-penalty math, ranked plan — validated 2026-07-12 by run 20260711_235303 (stages 1–3 cleared, stage-2 record); §5 has the outcome and cross-species lessons |

## Code & repo reviews

Archived point-in-time reviews in [`reviews/`](reviews/) —
[REPO_REVIEW_2026_06.md](reviews/REPO_REVIEW_2026_06.md),
[REPO_REVIEW_2026_07_RL_GCP.md](reviews/REPO_REVIEW_2026_07_RL_GCP.md),
[CODE_REVIEW.md](reviews/CODE_REVIEW.md) (superseded). Open findings live in
[KNOWN_ISSUES.md](KNOWN_ISSUES.md), not here.

## Conventions

- New run analysis or root-cause doc → `investigations/`, dated in the header,
  linked from the table above.
- New code/repo review → `reviews/`, findings copied into KNOWN_ISSUES.
- Don't rewrite a dated document when conclusions change — append a correction
  note that links to the newer analysis (see the fall-penalty correction in
  STAGE2_INVESTIGATION.md for the pattern).
