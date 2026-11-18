## V12 acceptance run on C1: FAILED on one check (`resumed_from_drive`), exit 1. The cause is a code fault in the kill condition; I stopped there and changed no code.

`nb acceptance` ran to the end and wrote its record with `ok: false`. 35 of its 36 checks pass. The one that fails comes from `acceptance`'s kill condition, not from the bundle or the jobs. Details are under "Fault" below.

**Setup**
- Bit-identity printed `45719d03…528d c0027992…c1c1`, the expected pair.
- Disk had 14G free before cloning, 13G after.
- The main checkout `/home/user/mesozoic-labs` is at C1, `git status --porcelain` was empty before and after, and I wrote nothing in it. My only change there is a `git fetch` of the data branch, which updated FETCH_HEAD.
- `git ls-remote origin` shows C1 as the head of `refs/heads/claude/mesozoic-trex-stage2-refine-4jffre`, and the data branch at `ecf95a81` (its parent is b914088b; C1 is not its ancestor). The manifest is the same at C1 and at ecf95a81.

**Where the parts came from**
- `nb acceptance` has no source option. It takes `--parts` and always runs the cells with `BUNDLE_SOURCE="drive"` on a stand-in Drive.
- So, as the task allowed, I took the three parts from the data branch with `git cat-file blob ecf95a81:hardware/trex_form_study/stage2/tb/bundle/s2-b1/parts/<name>`. I put them in `/home/user/mesozoic-wt/tb-accept-bundle/stage2/tb/bundle/s2-b1/parts/`, with C1's MANIFEST.json beside them.
- All three sha256 match the manifest:
  - `models-75-M-H-10.tar`: 2e9a55fc…
  - `V6-75-M-H-10-1.tar`: 11d7bd3e…
  - `V6-75-M-H-10-2.tar`: 6739c8b7…
- The `git:` source itself was therefore not exercised by this run.

**Where it ran**
- No stale `tb-accept` existed.
- I did not make a clone by hand. The tool clones origin itself into `--scratch /home/user/mesozoic-wt/tb-accept` (`clone1`, deleted after the kill, then `clone2`), as design 3.10 step 1 describes.
- I ran the tool from the main checkout's study directory, read-only.

**Command** (cwd `/home/user/mesozoic-labs/hardware/trex_form_study`, in the background, log at `/home/user/mesozoic-wt/tb-accept-bundle/acceptance.log`):
```
PYTHONPATH=. MUJOCO_GL=disable PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONHASHSEED=0 bash tools/lowprio.sh /home/user/mesozoic-labs/hardware/trex_form_study/.venv/bin/python -m designlab.stage2.nb acceptance --ref b7527a66528750746edf8f533c7f3b8a066aa1e3 --parts /home/user/mesozoic-wt/tb-accept-bundle/stage2/tb/bundle/s2-b1/parts --scratch /home/user/mesozoic-wt/tb-accept --record /home/user/mesozoic-wt/tb-accept-bundle/stage2/tb/bundle/s2-b1/ACCEPTANCE.json
```
- **Exit:** 1.
- **Wall time:** 1148.9 s (21:54:37Z to 22:13:46Z). The tool's own `seconds` is 1123.7.
- **Final line:** `S2 ACCEPTANCE b7527a665287 bundle s2-b1: FAILED in 1124 s: resumed_from_drive`

**Record**
- Path: `/home/user/mesozoic-wt/tb-accept-bundle/stage2/tb/bundle/s2-b1/ACCEPTANCE.json` (schema s2-acceptance-1, 29,466 bytes, `ok: false`).
- sha256: `2d1578aaf1c85b8ac10e79a96621c46cbe2896acf4fe3dd6c049428f66ac117f`
- Not committed (that is the sheet step's).

**What the record holds**
- `remote` and `clone_source`: https://github.com/kuds/mesozoic-labs.
- `ls_remote_sha` and the HEAD of both clones: C1.
- `content_sha256`: 9227d8a0bf0bef46769404fe0a5a0443cd2512570bf7b307d2d1633b6527f9dc.
- files 19, bodies `["75-M-H-10"]`, G8a for 75-M-H-10 on V6: 2 of 2, 40368 steps, pass.
- versions: python 3.13.18, mujoco 3.10.0, numpy 2.5.4, scipy 1.18.1, gymnasium 1.4.0, stable-baselines3 2.9.2, torch 2.15.0+cpu, cloudpickle 3.1.2, pillow 13.0.0, PyOpenGL 3.1.10.

**Lines printed** (sessions 0 and 1 the same except line 4)
- `S2 LINE 1/4 commit: b7527a66528750746edf8f533c7f3b8a066aa1e3 (REPO_REF is a commit: yes; checkout clean: yes)`
- `S2 LINE 2/4 inputs: bundle s2-b1, manifest 9227d8a0bf0b, 19 of 19 files match their sha256: OK`
- `S2 LINE 3/4 G8a 75-M-H-10 set V6: zero action = R0 at 0 ms and 10 ms, 2 of 2 episodes identical (40368 physics steps): PASS`
- Line 4, session 0: `S2 LINE 4/4 first rate R5-75-M-H-10-s42: 502 env steps/s (8 envs, 2 workers, 4 vCPUs); 0.004096M steps in about 0.0 h`
- Line 4, session 1: R5 467 env steps/s; R6 575 env steps/s.
- Preflight built 4 envs on V6: R5 and R6, each on hard and robot_floor_carpet.

**The tiny jobs** (V6, 75-M-H-10, seed 42)
- R5 and R6 each reached done at 4096 steps, with last checkpoint 4096 copied.
- R5 checks:
  - Baseline R0L@10ms: N 2, S 24.776 s.
  - At 2048 steps: N 1, S 24.549 s, no gain.
  - At 4096 steps: N 1, S 24.789 s, no gain.
  - Each monitor had 6 episodes; the gain records are present.
- G8b: walk/nominal, 1 of 1 needed 1, pass, at 2048 steps.
- Clips at 2048 and 4096 steps: made, and ffprobe reads them.
- R6 checks: at 2048 steps N 0, S 25.557 s; at 4096 steps N 0, S 25.556 s; no gain.
- DONE.json state is done.
- These are the acceptance run's tiny episodes on V6. I did not add them to `stage2/tb/dev_episodes.json` (D23); that is left to the lead.

**Checks**
- 35 true:
  - killed, line1, line2_ok, line3_pass, line4_after_resume, new_run_line
  - refused_live (exit 1, "LIVE.json written 41 s ago")
  - resume_line, session1_done
  - refused_finished (exit 1)
  - session_files, status_schema, jobs_done, checkpoints_copied, no_part_left
  - the remedies: remedies_killed, remedies_faked, source_change_refused_by_the_trainer, source_change_refused_without_lead, version_change_refused_by_the_trainer, allow_source_change_reached_the_trainer, resumed_to_done_with_the_leads_remedies, version_change_recorded_by_the_trainer
  - head_is_ref, fresh_clones_hold_nothing_ignored, killed_during_a_copy, part_swept (`model_1024.zip.part`), content_hash
  - g8a_every_body, monitor_six_judged, gain_recorded, g8b_read, clip_made, r6_done, main_checkout_unchanged
- 1 false: **resumed_from_drive**.

**Fault** (in the code; for the lead)
- The kill fired at 188.7 s on `model_1024.zip.part exists on the stand-in Drive (a copy in progress)`, with progress 1536 steps and `on_drive: []`.
- That `.part` is the first checkpoint's own copy, so no whole checkpoint was on Drive yet.
- Session 1 printed "resuming session 1: R5-75-M-H-10-s42 from 0 steps". Both sessions in R5's run.json have `resumed_from: None`, and session 0's updates went to `updates.discarded.s1.jsonl`.
- `_drill_checks` (`designlab/stage2/nb.py:3614` and `:3637`) requires a resumed_from greater than 0.
- `acceptance`'s kill condition `part_exists` (`nb.py:3916`) fires at the first `.part` in the first job's checkpoints. `local_run`'s condition and `_remedies`' own kill both wait until "checkpoint 1024 … is on the stand-in Drive", and the remedies' kill did that here.
- In a tiny run the first `.part` is always the first checkpoint's copy, so the check fails whenever the poll catches that copy, which it did here. I expect it to fail again on a rerun.
- Design 3.10 steps 4-5 need both a `.part` at the kill and a checkpoint to copy back. A likely fix is to kill only once a checkpoint greater than 0 is whole on the stand-in Drive and a `.part` exists. I did not make or test that change.
- The function's docstring says it "has not been run" before this.

**Not done:** no rerun, no code change, no commit, push or upload. The scratch is left in place for inspection: `/home/user/mesozoic-wt/tb-accept` (1.2G; logs in `/home/user/mesozoic-wt/tb-accept/logs/`, run in `/home/user/mesozoic-wt/tb-accept/drive/runs/acceptance/`).
