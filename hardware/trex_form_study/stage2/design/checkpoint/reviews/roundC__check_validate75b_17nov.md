Verdict: sound

**Verdict: sound.** Every record says what the report says, and every gate's verdict is what the rules give. G9-run passes, and so do V5, V6, V7 and V4, with V4 provisional. The filter is 10 Hz and the learning rate is 1e-5, resting on the 75-M-H-10 head alone. Apart from the one checked merge of the fix, only records were committed. The branch merges cleanly into the working tip. The problems I found are four small ones in the report's wording or in what it leaves the lead to know; none changes a record or a verdict. Simulation only. I changed no tracked file, committed nothing and ran no test. My scripts are in `/tmp/claude-0/-home-user-mesozoic-labs/2dca1c00-13f3-5853-acf6-8c3e3e04feb6/scratchpad/g9fix/check_val/` (`own_sha.py`, `rerun_g9.py`, `probes.py`, `k3_cmp.py`).

**State and merge readiness**
- `/home/user/mesozoic-wt/wp-p42` is at HEAD dae9c3fd79ea204735216e2be99b6743e03b61df and the working tree is clean (0 status lines). The main checkout is clean at ea6c0a94. `wp-l` is clean, and its last commit is from 17:10 UTC, before this step began.
- The reflog shows `merge claude/mesozoic-trex-stage2-refine-4jffre: Fast-forward` from cf6d0492 to 0f5e8ac5.
- Merge 31b36ea6 has parents 0f5e8ac5 and c60ff993. c60ff993 is the tip that `roundC__check_G9D_17nov.md` checked ("Verdict: merge `wt/wpc-cw` at c60ff993..."). The merge changes only gates2.py, lookbar.py and their two test files.
- Since 31b36ea6 only these paths changed: dev_episodes.json, the two g9_run files and the seven validation records. bar.py and moves.py show no diff since cf6d0492.
- `git merge-tree --write-tree claude/mesozoic-trex-stage2-refine-4jffre HEAD` exits 0 with tree c1ef1f65. designlab/ and tests/ do not differ from the working tip ea6c0a94.
- Bit-identity printed `45719d030359ec7d5d3914af3814d837fdc05d77652ba45c64792b511027528d c0027992de21067525cc4489b74acb157feb89f3a4c7618580addc2b341be1c1`.
- All 10 commit messages end with exactly the two attribution lines (raw bytes checked). No other line names a model.

**Hashes:** every file sha256 and every own sha256 in the report matches what I recomputed.
- g9_run.first.json is b9c02766…4239, the same bytes as cf6d0492's g9_run.json; the commit records it as a 100% rename.
- g9_run.json is eb951159…6e2a, own b270b6ff…d53a, made at 4a2f083e with dirty false.

**G9-run, re-derived in-process with no record written.** Parts, parts_ok, pass_, followed, g9_ref, set_freeze and sources all equal the committed record.
- Part (d): 13 expected conditions, all ok; missing [], problems [], not_expected {}.
- The four q90 conditions read "not run" with 0 episodes, 0 bar episodes and own sha256 ok.
- The three 10 ms walks reached their window 20 of 20 with no missing toe readings. Part (e) does not apply.
- The first record fails only on the four q90 companions ("its bar record is 'not run', not done"), as ruling 42 describes.

**V5**
- Sigma probe: re-derived from the runs. The survived counts are 20, 14, 4, 0, 0, 0 and the mean survival times 30.0, 28.392, 17.193, 3.724, 3.184 and 1.432 s. Each sigma has walk_nominal and walk_v0.4 at 10 seeds each (7000-7009).
- The code builds those episodes on hard floor with randomize False, push_prob 0, head_events False, a stance start and k = 5, as O2 requires.
- Filter: 10 Hz mean 3.184 s, 20 Hz mean 2.772 s, ratio 0.87060 < 0.9, so **10 Hz** (O4). The 20 Hz sigma 0.0809361 equals O4's formula. The 10 Hz runs are identical to the sigma probe's 0.1 runs (same z).

**V6 learning-rate probe**
- Re-derived per cell: 1e-4 fails at both sigmas, and 3e-5 fails at sigma 0.03 ([35, 78, 83, 68, 81, 50]). 1e-5 runs 160 of 160 minibatch steps in all 6 updates at both sigmas, so **1e-5**.
- The floors match J5 (envs 3 and 7 on carpet). The filter is 10 Hz, citing filter.json 25a61f97…. The record keeps earlier heads, so the -X head can be added later.

**V7**
- gl-start: all 7 checks true.
- r6-start, run on S1 as X1 and the chain require (design section 6's table says V6; X1 wins): zero, random, mid and training all true. The tiny run reached 1,024 of 1,024 steps with fingerprint difference 0.0.
- K3: settings as ruled. The kill was a whole-group SIGKILL at a progress reading of 81,920; the run resumed from 65,536 with 3 workers. The record holds 96 equal items at 98,304 and at 131,072.
- I also compared the on-disk checkpoints without gates2's code. Their sha256 values match the record. At both checkpoints all 77 .pth leaves (policy, optimizer, variables) are equal, as are obs/ret mean, var and count, and every fingerprint array.

**V4 rates**
- Hard 362.3 env steps/s (RSS 639 MB) and carpet 301.6 (RSS 703 MB) in one process.
- With 8 envs: W=1 299.0, W=2 535.8, W=3 577.1, W=4 728.8 env steps/s. Pool build 114.8, 58.7, 44.2 and 31.5 s. Parent RSS 549.5-557.5 MB. Worker RSS as reported.
- The record's load before is 0.624, so **provisional: true**, which is the rule's result.

**Rule X1, the commands and wall times**
- The five dev_episodes entries carry the chain's arguments, with counts scaled to the one body: 120, 40, 294,912, 278,528 and 98,304. Each listed_at precedes its step's start in the logs, and each is marked done after exit 0.
- The dirty provenance is dev_episodes.json alone, as the chain produces.
- Every logged command is the chain's own, restricted to 75-M-H-10, and went through tools/lowprio.sh with exit 0. The wall times match the logs: 246.0, 23.1, 678.5, 4.6, 19.3, 716.6 and 564.1 s, and 1.44 s for G9-run.
- Nothing past V4 exists: no S2-4 directory, no chain done-markers, and no study process running.

**Points for the lead (none blocks)**
1. **Worker counts are misstated.** The report says "All other steps used at most 2 worker processes", but V4 ran W=3 and W=4. The report says so elsewhere, and the step allowed all counts because the 60 cm rebuild was not running. K3's 3-worker resume also exceeded the two-process cap; the report discloses that.
2. **The idle-load figure is unverifiable.** The "1-minute load 0.13 before launch" is in no log. The gate reads the load after its own imports and the G9-ref check (gates2.py:1573), so a V4 run on an idle machine can still come out provisional.
3. **Ruling 42's label is only in commit 9f6a5e50's subject.** The G9-run record has no label field. That matches earlier practice (37e0695b).
4. **The not-run companions cite nothing.** They still carry `g9_ref: null`, as the report says. Part (d) does not read that field.
5. **Observation only:** the K3 archives' `clip_range` is SB3's ConstantSchedule in the uninterrupted run and the study's own on resume. Both give 0.2, and it is not a K3 item.
6. **Left out as the task asked:** the 60 cm sets, which are not frozen, and the 75-M-H-10-X learning-rate probe. O3 still asks for the -X probe before any Colab job of this body is queued.

**Not done by me:** I did not re-run the probes, K3 or V4. I could not see the bit-identity prints the report says were made before and after the merge, nor the lead's pytest timing.
