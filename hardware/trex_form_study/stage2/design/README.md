# Working papers of the second session (14–15 Nov 2026)

These are the second session's working papers, copied from its scratch directory so that they survive the container.
They are **not records and not rules**. The rules are `STAGE2_PLAN.md` (section 9), `STAGE2_75_PLAN.md` and
`STAGE2_TRACKB_PLAN.md`; the records are the JSON files the tools write. Where a paper here disagrees with a committed
rule or record, the rule or record wins. Paths inside them that begin `/tmp/claude-0/.../scratchpad/` are the scratch
directory they were written in: read `design/` as this directory and `maps/` as `maps/` below it.

| File | What it is |
|---|---|
| `lead_notes.md` | The lead's running notes: the owner's decisions, facts established, rulings, what ran when, workflow ids |
| `ROUND_A_DESIGN.md` | Design of round A (reference sets, the builder of starts, changes, stops and turns, the bar's readings and runner), after adversarial review; its "Review log" lists every finding |
| `WP_M.md` … `WP_T.md`, `WP_L.md` | Round A's builder briefs, one per work package (M moves, R readings, G regression, C set screen, S sets, B bar, T tables, L look) |
| `ROUND_C_DESIGN.md` | Design of round C (R5, R6, R2L, gates G8a and G9, the trainer, the notebook, the Drive bundle), after adversarial review |
| `WPC_*.md` | Round C's builder briefs (CP policy files, CV vec pool, CB bundle, CK look reading, CE env, CT trainer, CW rows, CF R2L fit, CG gates, CN notebook) |
| `PREREG_75_draft.md`, `PREREG_TRACKB_draft.md`, `S2_AMENDMENTS_lead.md` | Drafts from which the committed rule texts were written |
| `PREREG_RULINGS_15NOV.md`, `TRACKB_RULINGS_15NOV.md`, `TRACKB_RULINGS2_15NOV.md` | The lead's rulings on the points the reviews left open; each is written into the committed rule it settles |
| `REVIEW_RULINGS_15NOV.md` | The lead's rulings on the builds' independent reviews, as given to each fixer |
| `checkpoint/` | The state at the pause of 15 Nov: `STATUS.md`, every agent's report and reviewer's findings, and a git bundle of the unmerged branches |
| `proto_*.py`, `exp*.py`, `twist_*.py`, `turn_price*`, `check_traces.py`, `review_bitid_*.py` | Scratch prototypes and kinematic screens the designs cite; prescribed-motion or scratch results, never records |
| `roundc/` | Round C's scratch experiments and its draft before review |
| `maps/` | Code maps of the study written on 14 Nov, with file:line anchors of commit `7e9ea0d`; they go stale as code changes |

The scripts are kept as they were run (they are not linted and no test imports them). Several expect the scratch
directory's layout and will not run from here without their paths being changed.

Not copied: the owner pages' built HTML (30 MB; they regenerate from the study's page tools and are published, see
`lead_notes.md` for the links), the regenerated Stage 1b traces (git-ignored arrays, byte-identical to the committed
sha256) and the Drive test files.
