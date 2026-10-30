# Bias screen durable copy (never merged)

The T. rex physics r10 stance research's bias screen: 8 CPU continuations of PPO training from the stance6 round's
s42 checkpoint at step 1,048,576 to 2,490,368 (175 updates), 4 control against 4 critic_blind_tail3 (the normalised
tail_3 angle, observation 6, set to 0 before the value network only; the actor unchanged), paired seeds 2101, 2111,
2121, 2131, trained from the sealed copy of the never-merged run branch at 8737cd3d. Run under the maintainer's
2026-10-30 00:45 UTC overnight authorisation (run_log.txt). This orphan branch is the screen's durable text copy. It
holds no repository code and is never merged or opened as a pull request.

The reading (score_bs.py, STATUS final): line 1, NOT INFORMATIVE: the control arm did not reproduce the early rise of
the on-policy tail_3 command (control median primary +0.0603, a rise of +0.0918 from -0.0316 at the start, against the
pre-registered 0.10), so no reading about the critic is made. Report only: arm means +0.093 (control) and +0.013
(critic blinded); exact one-sided permutation p 13/70. All 8 COUNTED. Two independent checks (check/stat, check/proc)
agree with the scorer. Full text: score/score.txt.

- criteria.txt, APPROVED, SCREEN.json, SHA256SUMS.txt: the approved criteria, its approval record, the screen's
  definition and the pinned sums; NOTES*.txt: the build notes.
- tools/: the pinned harness, runner, reader and scorer with their tests; review/: the independent review of the
  criteria and the fixes; validate/: the pre-launch validation scripts and logs; scratch/: the criteria work (power).
- runs/<name>/: each continuation's run_meta, CHECK record, per-rollout arm evidence and run records (the ledger
  eval_episodes.csv; no checkpoints, no npz, no training episodes).
- reads/<name>/markers.json: each continuation's read (the primary's window column, the loop and critic readings).
- score/: the score; check/: the two independent checks; logs/: the runner's logs.
- FOR_THE_MAINTAINER.md: the overnight report. overnight_probes/: the night's no-training probes (early bias, early
  warning, the r9 held-out test, margin to rail) and their verifiers, top-level text files only (SKIPPED.txt lists
  what was left out).
- run_log.txt: the session's run log as it stood when this copy was made.

Built 2026-10-30 09:16 UTC. FILELIST.txt and SHA256SUMS cover every file (sha256sum -c SHA256SUMS).
