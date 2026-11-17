## Acceptance run again on C1 472e8d31: refused before it began, exit 4. No run took place and no record was written.

`nb acceptance` refused to start because C1 is no longer the head of any branch or tag on the remote. This is not a fault in the code. The check implements design 3.10 step 1 ("`git ls-remote origin` must list the ref's commit on the pushed branch"), and the remote moved past C1 while I was preparing. I stopped there and did not work around it. Using another ref or pushing a ref would be your decision, and pushing is outside my rules.

**Why it refused**
- `git ls-remote origin` at 23:21Z showed:
  - `refs/heads/claude/mesozoic-trex-stage2-refine-4jffre` = `6abd75da6a6bed54598633955292c29ff85e86af`
  - `refs/heads/claude/mesozoic-trex-stage2-bundle-s2-b1` = `ecf95a81a6ebb43e0331b5c1db74bdea7e74aaed`
  - No remote ref has `472e8d3150476b9d3db6cd62ec61e7e85c4c8977` as its head.
- `6abd75da` is the commit "Status: ruling 45 merged, C1 472e8d31, the acceptance run again (handover)":
  - author "Claude", dated 23:19:57Z;
  - it changes only `stage2/design/checkpoint/STATUS.md` (+14 lines);
  - C1 is its parent.
- It was committed in the main checkout while I worked. When I started, HEAD there was 472e8d31; it is now 6abd75da. I did not make that commit and pushed nothing.
- Between 472e8d31 and 6abd75da, nothing under `stage2/tb/bundle` differs (an empty diff), so the MANIFEST is the same at both.
- The code refuses here because `ls-remote` succeeded but did not list the ref as any ref's head. It does not then fall back to the local remote-tracking ref.

**Ways forward (your decision; I did none of them)**
- (a) Take the tip `6abd75da` as C1 and run again with `--ref 6abd75da6a6bed54598633955292c29ff85e86af`.
- (b) Make 472e8d31 the head of some pushed ref.
- (c) Change the check (a code change).

One more thing to know: your commits in the main checkout during a 20-minute run could also upset the run's `main_checkout_unchanged` check (git status before vs after).

**Steps 1 to 3 (done)**
1. I removed `/home/user/mesozoic-wt/tb-accept` (1.2G).
   - It was not a worktree: it is not in `git worktree list` and is not a git repository.
   - It held `S2_ACCEPTANCE_SCRATCH` from the b7527a66 run.
   - `df -h /` afterwards: 13G free.
2. The parts:
   - I fetched `origin claude/mesozoic-trex-stage2-bundle-s2-b1`; FETCH_HEAD = ecf95a81.
   - All three parts in `/home/user/mesozoic-wt/tb-accept-bundle/stage2/tb/bundle/s2-b1/parts/` are byte-identical (`cmp`) to the `git cat-file blob FETCH_HEAD:.../parts/<name>` blobs, and their sha256 equal C1's MANIFEST.json. Nothing needed to be re-taken.

     | Part | sha256 | Bytes |
     |---|---|---|
     | `models-75-M-H-10.tar` | `2e9a55fcad2ff40b29ca1be706a6e2fddb99f6236d75997423de3264013f6917` | 972800 |
     | `V6-75-M-H-10-1.tar` | `11d7bd3e3749c3a7c519cf0dfeadab228c8b77b7fd586176e73ba696ebea2387` | 16281600 |
     | `V6-75-M-H-10-2.tar` | `6739c8b72699b9887fd99f5c4ee84c0df78c246550252661a3258aa6c529ba39` | 16558080 |

   - I wrote C1's MANIFEST.json (`git show 472e8d31...:hardware/trex_form_study/stage2/tb/bundle/s2-b1/MANIFEST.json`) beside the parts.
   - It **equals** the one already there: the same bytes (`cmp`), sha256 `25ab5e413064b25adcccfc205326a582a9bb74bd4ec3af34323c504b05163480`, 10878 bytes, content_sha256 `9227d8a0bf0bef46769404fe0a5a0443cd2512570bf7b307d2d1633b6527f9dc`.
   - It also equals the data branch's MANIFEST.json.
3. I renamed the first run's record (kept) to `/home/user/mesozoic-wt/tb-accept-bundle/stage2/tb/bundle/s2-b1/ACCEPTANCE.v12-b7527a66.json`; sha256 `2d1578aaf1c85b8ac10e79a96621c46cbe2896acf4fe3dd6c049428f66ac117f`, the same before and after the rename.

**Step 4: the command**
- Run from `/home/user/mesozoic-labs/hardware/trex_form_study`, in the background, log at `/home/user/mesozoic-wt/tb-accept-bundle/acceptance2.log`.
- The options are unchanged since the first run (`--notebook`, `--ref`, `--parts`, `--scratch`, `--record`), so I ran it as given:
```
PYTHONPATH=. MUJOCO_GL=disable PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONHASHSEED=0 bash tools/lowprio.sh /home/user/mesozoic-labs/hardware/trex_form_study/.venv/bin/python -m designlab.stage2.nb acceptance --ref 472e8d3150476b9d3db6cd62ec61e7e85c4c8977 --parts /home/user/mesozoic-wt/tb-accept-bundle/stage2/tb/bundle/s2-b1/parts --scratch /home/user/mesozoic-wt/tb-accept --record /home/user/mesozoic-wt/tb-accept-bundle/stage2/tb/bundle/s2-b1/ACCEPTANCE.json
```
- **Exit:** 4.
- **Wall time:** 0.685 s (from 23:21:29Z to 23:21:30Z).
- **The only line it printed:** `S2 REFUSED acceptance: 472e8d3150476b9d3db6cd62ec61e7e85c4c8977 is not the head of a branch or tag of the remote https://github.com/kuds/mesozoic-labs: the acceptance run needs the pushed launch commit`
- The refusal comes before the scratch is deleted or made, so nothing was written:
  - `/home/user/mesozoic-wt/tb-accept` does not exist;
  - there is no `ACCEPTANCE.json`.

**What I could not report, because the run did not happen**
- No checks, including `resumed_from_drive` and `killed_during_a_copy`, so no kill step, no whole checkpoint on the stand-in Drive, and no resume steps.
- No line 1, line 2 or content hash, and no line 3.
- No run-directory lines.
- No values for `<<LIVE_S>>` or `<<RESUME_STEPS>>`. In `sheet_check` they come from `attempts.refused_live.lines.live` (the age) and `attempts.session1.lines.resume` (the steps).
- No record path or sha256.
- No tiny episodes, so nothing for `dev_episodes.json`.

**Rules kept**
- Bit-identity printed `45719d030359ec7d5d3914af3814d837fdc05d77652ba45c64792b511027528d c0027992de21067525cc4489b74acb157feb89f3a4c7618580addc2b341be1c1`, both at the start and after HEAD moved to 6abd75da.
- The main checkout's `git status --porcelain` was empty throughout.
- I changed no code and wrote no tracked file. I made no commit, push or tag.
- My only changes in `/home/user/mesozoic-labs/.git` are from two fetches:
  - FETCH_HEAD, now ecf95a81;
  - the remote-tracking refs `origin/claude/mesozoic-trex-stage2-refine-4jffre` (now 6abd75da) and `origin/claude/mesozoic-trex-stage2-bundle-s2-b1` (ecf95a81).
- One scratch copy of C1's manifest is at `/tmp/claude-0/-home-user-mesozoic-labs/2dca1c00-13f3-5853-acf6-8c3e3e04feb6/scratchpad/accept2/MANIFEST.C1.json`.
- No pytest was run.
