# Handoff: Option A, Option B, the walking-feasibility study and walking pilot 3 (2026-11-02 to 2026-11-04)

Text-only durable copy. Never merged; no PR. Nothing here is code that runs on main.

**Read first**
- `reports/MORNING_REPORT.md`: the overnight report of 2026-11-04. It covers:
  - the walking feasibility study;
  - the reward, hop and gate findings;
  - the noise-reset scan;
  - the inputs and plant tests.
- `pilot3/frozen/criteria_pilot3.txt`: walking pilot 3's frozen pre-registration. It was approved 2026-11-04 18:57 UTC
  (see `pilot3/frozen/APPROVED.json`; sha256 853f49d6...). Its branch is `claude/pilot-walk-r10-p3` at 31d02e42.
- `run_log.txt`: the full run log. It continues the phase-critic round's log (lines 1-980) with this session's
  events.

**Contents**
- `reports/OPTION_A_REPORT.md`: the CPU checks on the three phase-critic finals. K1-K4 hold in all three, and the
  parent tie s42/s80 was resolved to s42.
- `reports/OPTION_B_WORKFLOW_RESULTS.txt`: the build and review record of PR #658, which is merged.
- `reports/*_RESULTS.txt`: the structured results of every workflow, with each verifier's corrections.
- `pilot3/frozen/`: the frozen text and its supporting notes:
  - BRANCH_PLAN, NOTES, CHANGES and FIXES;
  - `R6_DESIGN.md`;
  - `R6_VALIDATE_RESULT.md`.
- `pilot3/tools/`: the 105 files pinned by `SHA256SUMS.txt`. These are the judge, scorer, habit, V - G, noise, video,
  runner and iso tools, plus their tests and evidence logs. Every file matches its pin.
- `evidence/`: each stage's `RESULT.md`, `SUMMARY.md` or `NOTES.md`, and its verifier's. It covers:
  - the feasibility analyses;
  - the overnight reward, speed, variants, benchmark, observability, log_std, hop, screen and config-only stages;
  - the Option A lanes;
  - the tool ports.

**What is not here**
- Binary run data and checkpoints. These are on Drive, read-only, under the PC round's RUN_IDs.
- Per-step recordings, venvs, code copies and command logs. These were in the session container only.
