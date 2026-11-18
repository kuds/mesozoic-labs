## Acceptance run on C1 cc437b6a (V12; design 3.10; ruling 45): passed, exit 0, all 36 checks true, no faults

**Step 0.** At 23:28:28Z, `git ls-remote origin refs/heads/claude/mesozoic-trex-stage2-refine-4jffre` printed `cc437b6a44d575db63e89c63e2f9dc42c4424947`, which is C1. After the run it still printed C1, and the main checkout's HEAD was C1.

**Step 1.** `/home/user/mesozoic-wt/tb-accept` did not exist before the run. It is not in `git worktree list`. `df -h /` showed 13G free; after the run it shows 12G, and the scratch directory holds 1.2G.

**Step 2. The parts**
- I fetched `origin claude/mesozoic-trex-stage2-bundle-s2-b1`. FETCH_HEAD is `ecf95a81a6ebb43e0331b5c1db74bdea7e74aaed`, the same as that branch's ls-remote head.
- All three parts are byte-identical (`cmp`) to `git cat-file blob FETCH_HEAD:hardware/trex_form_study/stage2/tb/bundle/s2-b1/parts/<name>`. Their sha256 equal C1's MANIFEST.json, so no part was re-taken:
  - `V6-75-M-H-10-1.tar`: `11d7bd3e3749c3a7c519cf0dfeadab228c8b77b7fd586176e73ba696ebea2387`
  - `V6-75-M-H-10-2.tar`: `6739c8b72699b9887fd99f5c4ee84c0df78c246550252661a3258aa6c529ba39`
  - `models-75-M-H-10.tar`: `2e9a55fcad2ff40b29ca1be706a6e2fddb99f6236d75997423de3264013f6917`
- The MANIFEST.json already in the bundle directory equals C1's `git show cc437b6a…:hardware/trex_form_study/stage2/tb/bundle/s2-b1/MANIFEST.json` byte for byte (`cmp`; both sha256 `25ab5e413064b25adcccfc205326a582a9bb74bd4ec3af34323c504b05163480`). I left it in place and did not rewrite it.
- The manifest's `content_sha256` is `9227d8a0bf0bef46769404fe0a5a0443cd2512570bf7b307d2d1633b6527f9dc`: 19 files in 3 parts, bodies `["75-M-H-10"]`.

**Step 3.** There was no `ACCEPTANCE.json` before the run, so nothing was renamed to `ACCEPTANCE.stale.json`. `ACCEPTANCE.v12-b7527a66.json` is untouched.

One thing I did beyond the steps: the refused 23:21 try had left a log at `acceptance2.log` (it says "472e8d31… is not the head…", EXIT 4). Before the run I renamed it to `/home/user/mesozoic-wt/tb-accept-bundle/acceptance2.refused-472e8d31.log`, so the new log does not overwrite it.

**Step 4. The run**
- The options have not changed since the first run (`nb.py` main: `acceptance --notebook (default) --ref --parts --scratch --record`), so the command ran as given. From `/home/user/mesozoic-labs/hardware/trex_form_study`, in the background:
  `PYTHONPATH=. MUJOCO_GL=disable PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONHASHSEED=0 bash tools/lowprio.sh /home/user/mesozoic-labs/hardware/trex_form_study/.venv/bin/python -m designlab.stage2.nb acceptance --ref cc437b6a44d575db63e89c63e2f9dc42c4424947 --parts /home/user/mesozoic-wt/tb-accept-bundle/stage2/tb/bundle/s2-b1/parts --scratch /home/user/mesozoic-wt/tb-accept --record /home/user/mesozoic-wt/tb-accept-bundle/stage2/tb/bundle/s2-b1/ACCEPTANCE.json`
- The log is `/home/user/mesozoic-wt/tb-accept-bundle/acceptance2.log`. I added START, EXIT/WALL_S and END lines around the command:
  - START 2026-11-17T23:29:29Z
  - `S2 ACCEPTANCE cc437b6a44d5 bundle s2-b1: OK in 1092 s`
  - EXIT 0, WALL_S 1119.0
  - END 2026-11-17T23:48:08Z
- The record's `seconds` is 1092.0.

**The record**
- Path: `/home/user/mesozoic-wt/tb-accept-bundle/stage2/tb/bundle/s2-b1/ACCEPTANCE.json` (29740 bytes).
- File sha256: `41d67f7f06f946ebfd18eea5a3c479ef266f2f5412a8ff501f176a1e9df2a2f8`.
- Its own stamped `sha256` field is `fb42991fb3866e2ecea86d0874ffb8c3cd6e04a3c8dc4487eed95c38a5cace71`, and it holds (`nb._stamped` recomputed equal).
- schema `s2-acceptance-1`, `ok: true`, at 2026-11-17T23:48:08Z.
- remote `https://github.com/kuds/mesozoic-labs`; ls_remote `["refs/heads/claude/mesozoic-trex-stage2-refine-4jffre"]`; ls_remote_sha, clone_head and clone_heads {1, 2} are all `cc437b6a44d575db63e89c63e2f9dc42c4424947`; clone_source is the remote URL.
- bundle_id s2-b1; content_sha256 `9227d8a0…f9dc`; files 19.
- g8a: `{"75-M-H-10": {set V6, identical 2, of 2, steps 40368, passes true}}`.

**Checks (36 of 36 true)**
- killed, line1, line2_ok, line3_pass, line4_after_resume, new_run_line
- refused_live, resume_line, session1_done, refused_finished, session_files, status_schema, jobs_done
- **resumed_from_drive**, checkpoints_copied, no_part_left
- remedies_killed, remedies_faked
- source_change_refused_by_the_trainer, source_change_refused_without_lead, version_change_refused_by_the_trainer
- allow_source_change_reached_the_trainer, resumed_to_done_with_the_leads_remedies, version_change_recorded_by_the_trainer
- head_is_ref, fresh_clones_hold_nothing_ignored, **killed_during_a_copy**, part_swept, content_hash, g8a_every_body
- monitor_six_judged, gain_recorded, g8b_read, clip_made, r6_done, main_checkout_unchanged

**The kill (`killed_during_a_copy`, `resumed_from_drive`)**
- The kill came at 23:33:13Z, 197.3 s into session 0 (exit -9). The kill condition read: "model_2048.zip.part exists on the stand-in Drive (a copy in progress), and checkpoint 1024 of R5-75-M-H-10-s42 is whole there".
- So the copy that was cut was the 2048-step checkpoint. Session 0's last trainer line was "check at 2048 steps (attempt 1)". The record's `progress` (the local progress.json) said 1536, which looks like it lagged behind the trainer.
- `on_drive`: [1024], so checkpoint 1024 was the one whole on the stand-in Drive. `parts_on_drive`: `["jobs/R5-75-M-H-10-s42/checkpoints/model_2048.zip.part"]`.
- Session 1 swept that `.part` (SESSION_1.json `swept_parts`; it printed "removed 1 stale .part file") and resumed from 1024 steps.
- R5's run.json sessions show `resumed_from` None, then 1024 (run_uuid a3c62e46…). R5 and R6 both reached done at 4096 steps.
- For comparison, the first run (b7527a66) killed at `model_1024.zip.part` with nothing on Drive and resumed from 0.

**Attempts:** session0 exit -9 (197.3 s); refused_live exit 1 (1.3 s); session1 exit 0 (476.3 s); refused_finished exit 1 (1.3 s). The latter printed "S2 REFUSED session: …/acceptance: the run is finished (DONE.json: the queue is finished); it is not resumed (rule K7)".

**The lines printed**
- Line 1: `S2 LINE 1/4 commit: cc437b6a44d575db63e89c63e2f9dc42c4424947 (REPO_REF is a commit: yes; checkout clean: yes)`
- Line 2: `S2 LINE 2/4 inputs: bundle s2-b1, manifest 9227d8a0bf0b, 19 of 19 files match their sha256: OK`. The content hash is `9227d8a0bf0bef46769404fe0a5a0443cd2512570bf7b307d2d1633b6527f9dc`.
- Line 3: `S2 LINE 3/4 G8a 75-M-H-10 set V6: zero action = R0 at 0 ms and 10 ms, 2 of 2 episodes identical (40368 physics steps): PASS`
- Line 4 (for completeness):
  - session 0, R5: 463 env steps/s
  - session 1, R5: 484 env steps/s
  - session 1, R6: 488 env steps/s
  - each "(8 envs, 2 workers, 4 vCPUs); 0.004096M steps in about 0.0 h"

**Sheet fields from the record**
- `<<LIVE_S>>` = **29** (refused_live: "LIVE.json written 29 s ago").
- `<<RESUME_STEPS>>` = **1024** (session1: "resuming session 1: R5-75-M-H-10-s42 from 1024 steps; 0 jobs done").

**Every run-directory line printed** (each appeared twice in its log unless marked once)
- session0: `Run directory: /home/user/mesozoic-wt/tb-accept/drive/runs/acceptance (new run)`
- refused_live (once): `Run directory: /home/user/mesozoic-wt/tb-accept/drive/runs/acceptance: still running (LIVE.json written 29 s ago); nothing to do`
- session1: `Run directory: /home/user/mesozoic-wt/tb-accept/drive/runs/acceptance (resuming session 1: R5-75-M-H-10-s42 from 1024 steps; 0 jobs done)`
- refused_finished: none
- remedies-0: `Run directory: /home/user/mesozoic-wt/tb-accept/drive/runs/acceptance-remedies (new run)`
- remedies-1, refused by the trainer's source check: `… /acceptance-remedies (resuming session 1: R5-75-M-H-10-s42 from 1024 steps; 0 jobs done)`
- remedies-2, refused_source: none. It printed "S2 REFUSED session: … this checkout is commit cc437b6a…, the run's is 0000…: another commit is accepted only with allow_source_change … (rule K6; LEAD_RULE)".
- remedies-3, refused by the trainer's version check: `… /acceptance-remedies (resuming session 2: R5-75-M-H-10-s42 from 1024 steps; 0 jobs done)`
- remedies-4, done: `… /acceptance-remedies (resuming session 3: R5-75-M-H-10-s42 from 1024 steps; 0 jobs done)`

**Remedies (design 3.10 step 6)**
- Their kill came at 93.6 s, with checkpoint 1024 on the stand-in Drive.
- They faked numpy as `0.0.0-faked-for-design-3.10-step-6` and the `designlab/stage2/r5core.py` hash as 64 zeros.
- The trainer refused "other sources than the run's last session" and then "other library versions than the run's last session".
- With LEAD.json `allow_source_change` and `allow_version_change` set, the run reached done at 4096 steps.

**Afterwards**
- The main checkout's `git status --porcelain` is still empty (0 lines).
- Bit-identity still prints `45719d030359ec7d5d3914af3814d837fdc05d77652ba45c64792b511027528d c0027992de21067525cc4489b74acb157feb89f3a4c7618580addc2b341be1c1`.

**Faults:** none. No refusal and no traceback, apart from the expected refused sessions.

**Not done:** I committed, pushed, tagged and uploaded nothing. I did not run `sheet-check`, fill the sheet, run pytest, or delete the scratch directory (`/home/user/mesozoic-wt/tb-accept`, 1.2G, holds `S2_ACCEPTANCE_SCRATCH`). I did not copy the record into the main checkout.

Logs: `/home/user/mesozoic-wt/tb-accept/logs/`
