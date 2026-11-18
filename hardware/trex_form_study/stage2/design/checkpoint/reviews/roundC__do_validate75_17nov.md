## Result: V3 stopped. G9-run on V6 / 75-M-H-10 fails part (d) for a reason in the code

G8a (V2) passed. R0L's records (V3) ran and were collected. G9-run (V3) then exited 1 with "G9-run V6 75-M-H-10: (d) False (17 companion records, missing []), (e) True" and "G9-run: FAIL". Your rule is to stop at the first failing gate, so **V5, V6, V7 and V4 were not run**. That means no sigma probe, filter, learning-rate probe, gl-start, r6-start, kill-resume or rates. Nothing was listed in `dev_episodes.json` and S2-4 was not started. Simulation only.

**Why (d) fails, from the record.** Part (d) flags only the companion records of the four q90 turn conditions: `turn/q90_left`, `turn/q90_right` and their `_mu0.54` forms. Each reads "its bar record is 'not run', not done". The set holds no q90 reference, so the bar ran 0 episodes there and wrote bar records with status "not run" plus companions for them.
- Those four are not in (d)'s `expected` list (gated conditions the set has a reference for, plus walk/nominal).
- All 13 expected records are present (`missing []`) and each is ok. `problems []`.
- Every 10 ms bar record that reached its window holds both toe-clearance readings.
- So no episode fails. The fail comes from `gates2.part_d`: it reads every companion under `look/R0L@10ms/` (gates2.py line 1335) and marks any bar record whose status is not "done" (line 1352). It does not leave out the companions of conditions that are not expected.
- Rule Z2 (d) says "every episode of the body's R0L@10ms records holds every reading of its item"; these four records have no episodes.
- **Decision for you:** whether to fix `part_d` and re-run G9-run. Until then, rule Z2 means no training run of this body and head starts. I changed no code.

## Per step
**Setup.** `git merge --ff-only claude/mesozoic-trex-stage2-refine-4jffre` said "Already up to date" (HEAD was 73d828e7). The bit-identity check printed `45719d03...528d c0027992...c1e1` before and after. Disk had 15G free.

**V2 G8a: PASS**
- Command: `gates2 g8a --body 75-M-H-10 --set V6 --record stage2/tb/gates/V6/75-M-H-10/g8a.json`, exit 0, 263.5 s.
- Output: "32 of 32 episodes identical, 2 conditions not run: PASS". Not run: q90 left and right (no reference).
- Record: `complete` true (10 conditions × seeds 3042, 3043 × delays 0 and 5), `differences` [], compared steps equal reference steps in all 32 episodes, all ends alike.
- Made at 73d828e7 on freeze 0 (manifest 2eeaebe8...), dirty false. It cites g9_ref.json af627ba9....
- Own sha256 field b1642a4cce6d99ef073087b8352d0b678cd99b3f5dd90c090f3ab9f4f3359a3e; file sha256 a3e7e47dbc788873cd6617bb2e186e8c2ea0543a21add1d042c1c5dcf409413a.
- Commit 2b6e3b2a.

**V3 R0L's records**
- Command: `lookbar run --set V6 --body 75-M-H-10 --rows R0L@10ms --conditions @gated,walk/nominal --out stage2/tb/work/bar/r0l --procs 2`, exit 0, 409.0 s. Block seeds3042, 20 seeds per condition.
- 13 conditions ran; the 4 q90 conditions were not run (no reference).
- Results (read/fell, then succeed):

| Condition | Read / fell | Succeed |
|---|---|---|
| walk/v0.4 | 20 / 0 | 20, pass |
| walk/v0.4_mu0.54 | 20 / 0 | 13 |
| walk/nominal | 20 / 0 | 0, "[reported]" |
| stop/two_step and _mu0.54 | 20 / 0 each | 20 each |
| nudge/forward, backward, left, right (mu0.9) | 12/8, 13/7, 12/8, 9/11 | 5, 9, 8, 2 |
| nudge/forward, backward, left, right (_mu0.54) | 9/11, 10/10, 13/7, 8/12 | 7, 10, 8, 6 |

  No unread and no error episodes.
- `nb collect --src stage2/tb/work/bar/r0l/V6/75-M-H-10 --dst stage2/tb/bar/r0l/V6/75-M-H-10 --kind bar`: exit 0, 0.36 s, "296 files (296 copied, 0 already there), each verified by sha256".
- COLLECTED.json sha256 00c26615a0851cbf2c2aaa3829290acc59c9d315abe8668b9df4d4caf8d5354f. It records `dirty: true`, because the copies it had just made were not yet committed.
- eval/summary.json 6c4ecca72c8d061a4f0af914ea85339f29e569fa6c24975fab9551daec7fbe1a; look/summary.json a77f42adf16e14902ca50ceb6c7b8135c20298bdc204f2a0533599eef420d097.
- Commit aaa042de.

**V3 G9-run: FAIL**
- Command: `gates2 g9-run --body 75-M-H-10 --set V6 --bar-out stage2/tb/bar/r0l --record stage2/tb/gates/V6/75-M-H-10/g9_run.json`, exit 1, 1.87 s.
- parts_ok: d false, e true. Part (e): "only the 60 cm body and head re-read Stage 1b's turn episodes".
- Made at aaa042de on freeze 0, dirty false.
- Own sha256 field 899f7d7f41ed15ee80f29d6d525855809926210dd075082d81ab92e2b866c3ae; file sha256 b9c02766a02a9bec91125f0029bfbfa2d3ddb277725196a498e25a0bb15d4239.
- I committed the failing record as it was made: cf6d0492.

**Not measured:** no rates, RSS or wall times for V4 to V7, and no learning rate or filter was chosen. Those steps did not run.

## Commits and state
- Branch `wt/wp-p42`: 73d828e7 → 2b6e3b2a → aaa042de → cf6d0492. Each message ends with the two required lines.
- The three commits touch only `stage2/tb/gates/V6/75-M-H-10/{g8a,g9_run}.json` and `stage2/tb/bar/r0l/...`: 299 files, no code, nothing under V0, F25F9 or F25F9cl.
- Nothing pushed or merged. The main checkout is still at dd431e6c with a clean status, and wp-l is at 8bc47346. No process of mine is running.

## What I did not do
- I ran the commands by hand, not through `run_trackb.sh`, so the chain has no done-markers.
- I did not start `tools/memguard.sh`: it marks every other python process for the OOM killer and SIGTERMs the largest, which could include the 60 cm rebuild's jobs.
- R0L's live outputs (80M) stay in the git-ignored `stage2/tb/work/bar/r0l`.

**Notes for when V5 to V4 resume:**
- `probe_sigma` rewrites `sigma_probe.json` whole; it does not add bodies to it as `filter` and `probe-lr` do.
- `rates` defaults to 1, 2, 3 and 4 workers, which conflicts with the `--workers 2` cap.

Logs and commit message files are in `/tmp/claude-0/-home-user-mesozoic-labs/2dca1c00-13f3-5853-acf6-8c3e3e04feb6/scratchpad/p42val/`.
