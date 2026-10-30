# Critic-blind screen (BS2) durable copy (never merged)

The T. rex physics r10 stance research's second critic-blind screen (BS2): 8 CPU continuations of PPO training from
the stance6 round's s42 checkpoint at step 2,097,152 to 3,538,944 (175 updates), 4 control against 4
critic_blind_tail3 (the normalised tail_3 angle, observation 6, set to 0 before the value network only; the actor
unchanged), paired seeds 2501, 2511, 2521, 2531, trained from the sealed copy of the never-merged run branch at
8737cd3d. Approved by the maintainer on 2026-10-30 ("Go with option a", 10:23 UTC; "Launch with 0.20", 12:53 UTC;
run_log.txt). This orphan branch is the screen's durable text copy. It holds no repository code and is never merged
or opened as a pull request.

The reading (score_bs.py 2.1, STATUS final): line 2, REDUCED: blinding the critic to the tail_3 angle reduced the
on-policy tail_3 command over the last 8 updates before E (control median rise +0.464 against the 0.20 gate; every
control above every blinded run; exact one-sided permutation p 1/70). Secondaries: the ledger tail command and the
margin to the rail SUPPORTED (1/70 each), the end loop gain NOT SUPPORTED. Two independent checks (check/stat,
check/proc) agree with the scorer. Full text: score/score.txt; summary: REPORT.md.

- criteria.txt, APPROVED, SCREEN.json, SHA256SUMS.txt: the approved criteria, the approval record, the screen's
  definition and the sums; NOTES.txt: the build log; REPORT.md: the result for the maintainer.
- tools/: the pinned harness, runner, reader and scorer with their tests; check_build/: the independent build check;
  criteria_work/: the design notes and power (synthetic dry-run roots left out); review/: the design and process
  reviews, the fixes and the re-review of the fixed version (synthetic roots left out); validate/: the pre-launch
  validations without their run ledgers.
- runs/<name>/: each continuation's run_meta, CHECK record, per-rollout arm evidence and run records (the ledger
  eval_episodes.csv; no checkpoints, no npz, no training episodes); reads/<name>/: the markers.
- score/: the score; check/: the two independent checks; logs/: the runner's logs.
- run_log.txt: the session's run log as it stood when this copy was made.

Built 2026-10-30 16:26 UTC. FILELIST.txt and SHA256SUMS cover every file (sha256sum -c SHA256SUMS).
