# Prompt for the next session

Paste everything below the line into a new Claude Code session on kuds/mesozoic-labs.

---

You're continuing a long-running research project in kuds/mesozoic-labs. The goal is a robust, anatomically plausible T. rex stance (MuJoCo + Stable-Baselines3 PPO, physics r9, policy interface r15), vetted to research standard, with robotics later. The previous session ran the stance5 band screen's replication (seeds 70 and 80) to a final decision, added report-only K3 readouts, and built a per-run training-history report. Its working files are on the never-merged branch `claude/handoff-stance5r`.

## 1. Restore before anything else

1. `git fetch origin claude/handoff-stance5r`, check it out into a scratch worktree (never into the main checkout), and read `HANDOFF.md` at its root in full.
2. Run `bash restore.sh` from that worktree. It:
   - checks every file against `SHA256SUMS`;
   - restores the files to `/tmp/claude-0/-home-user-mesozoic-labs/0fdf74a4-2989-56c9-a773-92101004d7de/scratchpad`, the path the pinned tools hardcode;
   - rebuilds the four judge clones at their pinned commits.

   Edit no pinned file.
3. Checkpoints are not on the branch. Re-fetch them from Google Drive (read-only) only if something must be re-judged.

## 2. Standing rules (the maintainer's)

- Prompt me on each real decision (AskUserQuestion, recommended option first). Keep updates concise.
- Physics, reward, task, gate, training-loss and policy-interface changes need my explicit approval. Report-only telemetry does not.
- Fails stay fails. Never stop or change a training run because of a readout or an alarm.
- Screen and pilot branches are never merged.
- Google Drive is read-only.
- Work in scratch clones or worktrees, never in the main checkout. Open PRs only when I ask. Put no model names in anything committed.
- Use the real clock (`date -u`) for every log timestamp. Append every event and decision to the run log (`scratchpad/foot/stance4/live/run_log.txt`).
- If a permission classifier refuses an action, don't work around it: ask me.
- Label findings measured or inferred, and have them adversarially checked before you report them.
- Usage is limited (the account hit its spend limit last round). Batch Drive fetches and keep agent counts lean.

## 3. Where things stand

- **The replication is decided: line 2, not adopted (final).** s70 holds; s80 fails K1 only, on the raised-tail saturation habit (`tail_3_pitch_act` at +1) from 4.3M to 8.4M, and recovered by 10.5M. That makes 2 of 5 band runs holding.
- The band keeps fore-aft balance in every run: K2 passes 5/5 in the band arm against 0/3 in the control.
- The tail habit has now appeared in three runs: stance4 s42, stance5 s42 and stance5r s80.
- The K3 readouts show every band final falling to a sharp left-right hip kick, against 0-11/20 for the control finals. They show no repeat of s42's K3 collapse.
- A scratch tail fix (P1, a bound on the policy mean) helped against its paired control but relapsed, and its effect is not established. It was never pushed.
- Walking pilot 2 is not launched (branch `claude/pilot-walk-r9-b`, held).

## 4. First tasks: ask me before acting on each

1. **Next round.** Ask which of stance5 line 4's menu items to pursue: the tail saturation habit (the evidence's lead candidate), the band dose, or the leg exploration-noise floor. Then design the round with fixed criteria before any training.
2. **Report PR scope.** Ask about the training-history report: (a) the standalone report as built (recommended first PR), or (b) also generating it automatically at the end of every stage, per task #47's design. Then open the PR from `patches/run-training-report/`.
3. **Actuator saturation report.** Raise the question I left: should there be an actuator saturation report for final models and/or during training? Today the ledger does not name the saturating actuator.
4. **Records PR**, documentation only. It carries the items listed in `HANDOFF.md` section 3, item 4: #634's landing, run-log lines 396 onward, the replication outcome, the K3 and tail reports, the readouts, the s80 alarm and the certificates.
