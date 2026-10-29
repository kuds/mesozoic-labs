# Entropy continuation screen durable copy (never merged)

The T. rex physics r10 stance research's entropy continuation screen (ECS): 24 CPU continuations of PPO training
from the stance6 round's own checkpoints (s42 from step 3,145,728 to 4,718,592; s80 from 4,194,304 to 5,767,168), arm A
with the recorded ent_coef schedule against arm B with ent_coef 0, 6 seeds per arm per start, trained from the sealed
copy of the never-merged run branch at 8737cd3d. This orphan branch is the screen's durable copy (D9 (a)). It holds no
repository code and is never merged or opened as a pull request.

The reading (score_ecs.py, STATUS final): line 3, NO DIFFERENCE DETECTED IN WHETHER AN ANY-FLAGGED EVALUATION CAME BY
E FROM THESE STATES (not tested: a shift of the first flag within the window, a modest effect, the early phase).
s42: arm A 6 of 6, arm B 6 of 6; s80: arm A 1 of 6, arm B 1 of 6; exact stratified P_UP = P_DOWN = 17/22. All 24
COUNTED. Two independent checks (check/stat, check/proc) agree with the scorer. Full text: score/score.txt.

- criteria.txt, APPROVED, SHA256SUMS.txt: the approved criteria, its approval record and the pinned sums; v1/-v3/,
  final/ and the survey and power files are the drafts and texts the criteria cite.
- tools/: the pinned harness, runner, reader and scorer with their tests; build/ and validate/: the build's fix record,
  pins and the V1-V5 logs.
- runs/<name>/: each continuation's run_meta, CHECK record and run records (the ledger eval_episodes.csv; no
  checkpoints, no npz, no training episodes); traces/<name>/: each end trace's summaries (no per-step traces).
- score/: the score and every continuation's read (R1, R2, R3); check/: the two independent checks; logs/: the
  runner's logs.
- run_log.txt: the session's run log as it stood when this copy was made.

Built 2026-10-29 12:35 UTC. FILELIST.txt and SHA256SUMS cover every file (sha256sum -c SHA256SUMS).
