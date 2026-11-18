Verdict: sound; record ok: True

## Verdict: sound. The record's own `ok` is true, and all 36 checks are true.

I checked every claim in the do-report from my own tool output, and every one I could check holds. The acceptance run ran as design 3.10 and ruling 45 require: on C1 cc437b6a, from a fresh clone of the remote, with parts byte-equal to the data branch and to C1's manifest. The ruling 45 kill worked: it cut a later copy only after a checkpoint above 0 was whole, and session 1 resumed from that checkpoint.

### Check 1: the acceptance run

**Remote and checkout**
- At 23:49Z, `git ls-remote origin refs/heads/claude/mesozoic-trex-stage2-refine-4jffre` printed cc437b6a44d575db63e89c63e2f9dc42c4424947, which is C1.
- The main checkout's HEAD is C1, and `git status --porcelain` is empty (0 lines).
- The record agrees:
  - `ls_remote` is `[refs/heads/claude/mesozoic-trex-stage2-refine-4jffre]`.
  - `ls_remote_sha`, `clone_head` and `clone_heads` {1, 2} are all C1.
  - `clone_source` is https://github.com/kuds/mesozoic-labs.
- The surviving `clone2` is at C1 with an empty status, and its origin is that URL.

**Parts and manifest**
- I fetched `origin claude/mesozoic-trex-stage2-bundle-s2-b1`. FETCH_HEAD is ecf95a81a6ebb43e0331b5c1db74bdea7e74aaed, the same as that branch's ls-remote head.
- All three parts are `cmp`-equal to `git cat-file blob FETCH_HEAD:.../parts/<name>`. Their sha256 equal the manifest's `content.parts`:
  - `V6-75-M-H-10-1.tar`: 11d7bd3e…2387
  - `V6-75-M-H-10-2.tar`: 6739c8b7…ba39
  - `models-75-M-H-10.tar`: 2e9a55fc…6917
- The copies under the stand-in Drive (`tb-accept/drive/inputs/s2-b1/`) are `cmp`-equal to the parts too.
- `MANIFEST.json` beside the parts is `cmp`-equal to `git show cc437b6a:.../s2-b1/MANIFEST.json`. Its sha256 is 25ab5e41…3480.
- I recomputed the content hash with `nb.sha256_json(content)`: 9227d8a0bf0bef46769404fe0a5a0443cd2512570bf7b307d2d1633b6527f9dc. That equals the manifest's and the record's `content_sha256`. Files: 19 (5 + 12 + 2).
- `ACCEPTANCE.v12-b7527a66.json` is still sha256 2d1578aa…117f, the value the earlier check gave.

**The log** (`/home/user/mesozoic-wt/tb-accept-bundle/acceptance2.log`)
- START 23:29:29Z
- `S2 ACCEPTANCE cc437b6a44d5 bundle s2-b1: OK in 1092 s`
- EXIT 0, WALL_S 1119.0
- END 23:48:08Z
- The refused 23:21 log is kept separately as `acceptance2.refused-472e8d31.log` (EXIT 4).

**The record** (`/home/user/mesozoic-wt/tb-accept-bundle/stage2/tb/bundle/s2-b1/ACCEPTANCE.json`)
- 29740 bytes. File sha256: 41d67f7f06f946ebfd18eea5a3c479ef266f2f5412a8ff501f176a1e9df2a2f8.
- Its stamped `sha256` field is fb42991fb3866e2ecea86d0874ffb8c3cd6e04a3c8dc4487eed95c38a5cace71. I recomputed it with C1's `nb._stamped` and it is equal.
- schema `s2-acceptance-1`, `ok` true, and `ok == all(checks)`.
- I read all 36 checks; every one is `true`:
  - killed, line1, line2_ok, line3_pass, line4_after_resume, new_run_line
  - refused_live, resume_line, session1_done, refused_finished, session_files, status_schema, jobs_done
  - resumed_from_drive, checkpoints_copied, no_part_left
  - remedies_killed, remedies_faked
  - source_change_refused_by_the_trainer, source_change_refused_without_lead, version_change_refused_by_the_trainer
  - allow_source_change_reached_the_trainer, resumed_to_done_with_the_leads_remedies, version_change_recorded_by_the_trainer
  - head_is_ref, fresh_clones_hold_nothing_ignored, killed_during_a_copy, part_swept, content_hash, g8a_every_body
  - monitor_six_judged, gain_recorded, g8b_read, clip_made, r6_done, main_checkout_unchanged
- `g8a`: 75-M-H-10, set V6, 2 of 2 identical, 40368 steps, passes.

**Line 2 against the manifest**
- In both session 0 and session 1, line 2 reads `S2 LINE 2/4 inputs: bundle s2-b1, manifest 9227d8a0bf0b, 19 of 19 files match their sha256: OK`.
- Its fields are: bundle s2-b1, manifest 9227d8a0bf0b (the first 12 hex of the manifest's `content_sha256`), ok 19 = n 19, OK.
- Line 1: `S2 LINE 1/4 commit: cc437b6a44d575db63e89c63e2f9dc42c4424947 (REPO_REF is a commit: yes; checkout clean: yes)`
- Line 3: `S2 LINE 3/4 G8a 75-M-H-10 set V6: zero action = R0 at 0 ms and 10 ms, 2 of 2 episodes identical (40368 physics steps): PASS`

**The kill and the resume**
- `kill.when` reads: "model_2048.zip.part exists on the stand-in Drive (a copy in progress), and checkpoint 1024 of R5-75-M-H-10-s42 is whole there".
  - The kill came at 23:33:13Z, 197.3 s in, exit -9.
  - `on_drive` is [1024], and `parts_on_drive` is `[jobs/R5-75-M-H-10-s42/checkpoints/model_2048.zip.part]`.
  - The kill string comes from `acceptance_kill`. That function counts a checkpoint only if `_Launcher._whole` passes: the record parses, its step is n, and each of the three files has the record's size and sha256. So checkpoint 1024 was whole on the stand-in Drive at the kill.
  - Session 0's log ends at "check at 2048 steps (attempt 1)".
- Session 1:
  - `SESSION_1.json` has resume true, takeover true, `took_over` session 0, and `swept_parts` [model_2048.zip.part].
  - Its log printed "removed 1 stale .part file", "10 files copied back from Drive", and then "resuming session 1: R5-75-M-H-10-s42 from 1024 steps".
- R5's `run.json` sessions show `resumed_from` None, then 1024 (run_uuid a3c62e46…).
- R5 and R6 both finished with "done at 4096 steps".
- `DONE.json` state is done, and `STATUS.json` has schema s2-status-2, state done.
- No `.part` is left on the stand-in Drive.
- So session 1 resumed from 1024 steps, a checkpoint above 0 that was whole at the kill.

**Sheet values and run-directory lines**
- `<<LIVE_S>>` = 29 (refused_live: "LIVE.json written 29 s ago").
- `<<RESUME_STEPS>>` = 1024 (session1's resume line).
- Both come from `attempts.refused_live.lines.live` and `attempts.session1.lines.resume`, which is what `sheet_check` reads.
- The run-directory lines I printed from the record match the do-report's list exactly: session0 "new run" (twice), refused_live "still running" (once), session1 "resuming session 1 … from 1024 steps; 0 jobs done" (twice), refused_finished none, and remedies-0 to remedies-4 as reported.

**The remedies**
- The kill came at 93.6 s, with checkpoint 1024 on the stand-in Drive.
- Faked values: numpy as `0.0.0-faked-for-design-3.10-step-6`; the `r5core.py` hash as 64 zeros; the QUEUE commit as 40 zeros.
- The trainer refused "other sources than the run's last session", and the session refused with K6 / LEAD_RULE.
- With `allow_source_change` set, the trainer refused "other library versions than the run's last session".
- With both flags set, the run reached DONE at 4096 steps.

**Afterwards**
- The bit-identity check prints `45719d030359ec7d5d3914af3814d837fdc05d77652ba45c64792b511027528d c0027992de21067525cc4489b74acb157feb89f3a4c7618580addc2b341be1c1`.
- `tb-accept` is not in `git worktree list`.
- `df -h /` shows 12G free, and the scratch directory holds 1.2G.

### Anything that would fail on Colab

I found nothing that would fail. Three limits, none of them a fault in this step:
1. **The data-branch route is not exercised here.** By design, step 2 of 3.10 uses `BUNDLE_SOURCE="drive"`, so this run does not test P46's `git:claude/mesozoic-trex-stage2-bundle-s2-b1` route.
   - The earlier check (`roundC__check_accept_17nov.md`) did run `nb.inputs` with that `git:` source from a fresh clone, and got 19 of 19 OK.
   - `git diff b7527a66 cc437b6a -- designlab/` touches only the module header, `local_run`'s neighbourhood/`acceptance`, the new sheet-check code and `main`. It does not touch `inputs` or the `git:` parts reader.
   - The data branch head is still ecf95a81, with the parts byte-equal.
   - Setting `BUNDLE_SOURCE` to `git:` in the sheet is C2's job.
2. **The machine differs from Colab.** The run used 4 vCPUs, `in_colab` false, and no real Drive mount. That is inherent to a simulation-only run.
3. **`dev_episodes.json` is unverified.** Ruling 45 wants it updated, and that is not part of this step.

### Check 2

- `git diff 472e8d31 cc437b6a --stat` touches 5 files, all under `hardware/trex_form_study/stage2/design/`:
  - `ROUND_C_DESIGN.md`
  - `ROUND_C_RULINGS_17NOV.md`
  - `checkpoint/STATUS.md`
  - `reviews/roundC__check_accept2_17nov.md`
  - `reviews/roundC__do_accept2_17nov.md`
- A filter for paths outside `stage2/design/` returned none.
- 472e8d31 is an ancestor of C1, and the commits between them are 6abd75da and cc437b6a.
- So the code, notebook, template and manifest are those already checked.

### Small points in the do-report

- I could not verify the "13G free before" figure or the do-agent's HEAD at its start.
- The `MANIFEST.json` mtime is 23:21, from the refused try, which is consistent with "left in place".
- Renaming the refused log was beyond the steps, but it touches only untracked scratch.

### What I did and did not do

- **Done:**
  - ran `ls-remote` and `git fetch` of the data branch (this writes only FETCH_HEAD);
  - ran `cmp` and `sha256` checks;
  - wrote one read-only script, `/tmp/claude-0/-home-user-mesozoic-labs/2dca1c00-13f3-5853-acf6-8c3e3e04feb6/scratchpad/launch2/check_accept/verify_record.py`, run with the study venv and the required environment.
- **Not done:**
  - I changed no tracked file and made no commit, push or tag.
  - I did not re-run the acceptance or pytest.
  - I did not exercise the `git:` source end to end myself.
  - I did not run `sheet-check`.
  - I did not delete `/home/user/mesozoic-wt/tb-accept`.
