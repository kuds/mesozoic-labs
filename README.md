# Critic-phase screen (PS) durable copy (never merged)

The T. rex physics r10 stance research's critic-phase screen (PS): 8 CPU continuations of PPO training from the
stance6 round's s42 checkpoint at step 2,097,152 to 3,538,944 (175 updates), 4 control against 4 critic_phase (the
critic alone sees the episode phase, min(k,200)/200, a settled flag and 0.98^max(0,200-k), in the always-zero command
slots 60-62; the actor unchanged; the critic still sees the tail angle), paired seeds 2901, 2911, 2921, 2931, trained
from the sealed copy of the never-merged run branch (code tree 331fe8fc). Approved by the maintainer on 2026-10-31
('Confirm as built (Recommended)', 15:54:59 UTC; run_log.txt). This orphan branch is the screen's durable text copy.
It holds no repository code and is never merged or opened as a pull request.

The reading (score_bs.py 3.1, STATUS final, 20:19:27 UTC): line 2, REDUCED: giving the critic the episode phase
reduced the on-policy tail_3 command over the last 8 updates before E (control median +0.616, rise +0.490 against the
0.20 gate; phase median +0.057; every control above every phase run; exact one-sided permutation p 1/70).
Secondaries: SEC-CMD, SEC-MARGIN and SEC-MECH SUPPORTED (1/70 each, Bonferroni-surviving); the guard SEC-TAIL2
SUPPORTED (2/70, small: mostly the controls' tail_2 falling). Two independent checks (check/stat, check/proc) agree.
The separate report-only V-G tool (pinned before the reading) found the phase critic nearly unbiased by episode step
and the control critic still mistiming the step-200 gate. Summary: REPORT.md.

- criteria.txt, APPROVED, SCREEN.json, SHA256SUMS.txt: the approved criteria, the approval record, the queue, the sums;
  NOTES.txt: the build log; REPORT.md: the result for the maintainer.
- tools/: the pinned harness, runner, reader and scorer with their tests; check_build/: the independent build check;
  criteria_work/: the design notes and power; review/: the design and process reviews, the fixes and the re-review;
  validate/: the pre-launch validations without their run ledgers.
- runs/<name>/: each COUNTED continuation's run_meta, CHECK record, per-rollout arm evidence and run records (no
  checkpoints, no npz, no training episodes); runs/<name>_crashed<k>/run_meta.json only: the second wave was killed
  twice by container restarts and rerun from the start pair under the runner's rules; reads/<name>/: the markers.
- score/: the score; check/: the two independent checks; logs/: the runner's logs (the crashed attempts' process logs
  left out, never read).
- vg_tool/, vg_run/: the separate report-only V-G and tail sigma tool and its output.
- run_log.txt: the session's run log as it stood when this copy was made.

Built 2026-10-31 20:45 UTC. FILELIST.txt and SHA256SUMS cover every file (sha256sum -c SHA256SUMS).
