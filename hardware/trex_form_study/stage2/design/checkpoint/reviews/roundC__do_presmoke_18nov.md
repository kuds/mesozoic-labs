## (A) Git-source probe of cell 4 at C1: line 2 OK

**What I ran, and why.** At C1, cell 4 is `s2_step("inputs")`. It runs `python -m designlab.stage2.nb inputs --params <file>` with cwd set to the study directory. That step (`_step_inputs`) first opens cell 3's session state (RUN_ID, LIVE.json, QUEUE.json). It then calls `inputs(p["BUNDLE_ID"], p["DRIVE_DIR"], p["BUNDLE_SOURCE"], set_name=p["SET"], bodies=_bodies(p), local=p["LOCAL_DIR"], expect=was)`.

The signature is `inputs(bundle_id, drive, source='drive', *, set_name=None, bodies=(), local=None, expect=None)`. For the smoke the values are:
- QUEUE empty, so bodies = [SMOKE_BODY] = ["75-M-H-10"]
- SET is V6 by default
- expect is None, because it is a new run

I called `nb.inputs` directly with those values. I did not use `nb local-run`, because it runs the Stage 1 view with BUNDLE_ID "" (tiny training, a kill), so it never takes cell 4's bundle or `git:` path. I did not use `nb inputs --params` either, because it needs cell 3's session on the drive first. The inner call is the same one.

**Setup.**
- `df -h /` showed 12G free.
- No stale `tb-git-probe` existed.
- Cell 2's commands, run as written: `git clone --no-checkout https://github.com/kuds/mesozoic-labs.git /home/user/mesozoic-wt/tb-git-probe`, then `git fetch origin cc437b6a…`, `git rev-parse FETCH_HEAD^{commit}` (gave cc437b6a44d575db63e89c63e2f9dc42c4424947), and `git checkout --detach` to that commit.
- This took 31.0 s. It is a full clone, not shallow; `.git` is 448M.
- `nb.__file__` was the clone's own `designlab/stage2/nb.py`.

**Command.** cwd was `/home/user/mesozoic-wt/tb-git-probe/hardware/trex_form_study`:
```
PYTHONPATH=. MUJOCO_GL=disable PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONHASHSEED=0 bash tools/lowprio.sh /home/user/mesozoic-labs/hardware/trex_form_study/.venv/bin/python <scratchpad>/presmoke-scripts/probe_inputs.py <scratchpad>/presmoke/drive <scratchpad>/presmoke/local
```
The script calls `nb.inputs("s2-b1", drive, "git:claude/mesozoic-trex-stage2-bundle-s2-b1", set_name="V6", bodies=["75-M-H-10"], local=local, expect=None)`. It is at `/tmp/claude-0/-home-user-mesozoic-labs/2dca1c00-13f3-5853-acf6-8c3e3e04feb6/scratchpad/presmoke-scripts/probe_inputs.py`, and its output is kept beside it in `probe_out.txt`.

**Result.**
- Exit 0. Wall time 2.26 s, of which `inputs()` took 2.1 s.
- Line 2 as printed: `S2 LINE 2/4 inputs: bundle s2-b1, manifest 9227d8a0bf0b, 19 of 19 files match their sha256: OK`. This is the same text as item (e) in section 3.6 of the steps sheet at 1b5686f6.
- The returned dict:
  - ok: true
  - source: "git:claude/mesozoic-trex-stage2-bundle-s2-b1"
  - content_sha256: 9227d8a0bf0bef46769404fe0a5a0443cd2512570bf7b307d2d1633b6527f9dc
  - files_ok: 19, files: 19
  - problems: []
  - sets: {"V6": {"75-M-H-10": true}}
  - unpacked: 3 parts, 19 files, written 19, replaced 0, left 0

**Where things landed.**
- **Parts:** `<local>/inputs/s2-b1/parts/` holds `V6-75-M-H-10-1.tar` (16281600 bytes, 11d7bd3e…), `V6-75-M-H-10-2.tar` (16558080 bytes, 6739c8b7…) and `models-75-M-H-10.tar` (972800 bytes, 2e9a55fc…). All three sha256 equal the manifest's.
- **Unpacked files:** at their study paths inside the clone:
  - 2 model files in `round2/refine/feet_75/models/`
  - 17 array files in `stage2/sets/V6/75-M-H-10/{refs,transitions}/`
  - All of them are git-ignored, so `git status --porcelain` stayed empty.
- **Stand-in drive:** still empty afterwards. The `git:` path reads nothing from Drive.

**The parts came from the data branch.**
- After the call, the clone's FETCH_HEAD was `ecf95a81a6ebb43e0331b5c1db74bdea7e74aaed branch 'claude/mesozoic-trex-stage2-bundle-s2-b1'`. Before the call it was C1, from cell 2's fetch.
- `git hash-object` of each local part equals its blob on ecf95a81 (66a5dff9…, e6dc21dc…, 38acab08…).
- The data branch's MANIFEST.json is the same blob as C1's (99b0fcd2…).

**What could differ on Colab** (not tested there):
- **Paths:** the study directory is `/content/mesozoic-labs/hardware/trex_form_study`, and LOCAL is `/content/s2runs`, so the parts go to `/content/s2runs/inputs/s2-b1/parts`.
- **Git version:** 2.43.0 here; Colab's version I did not see. The commands used (`-C`, `rev-parse --show-prefix`, `cat-file blob <sha>:<path>`) are long-standing.
- **Full clone:** cell 2's plain clone fetches every branch, including the data branch (about 33 MB of parts). So cell 4's `git fetch origin <ref>` only updated FETCH_HEAD and the blobs were already local, which is why it took 2 s. A shallow or single-branch clone would download them at cell 4 instead.
- **Network:** here it went through this container's proxy; Colab goes direct to github.com without credentials. The fetch uses the same remote as cell 2's clone, so if cell 2 succeeds the remote is reachable.
- **The wrapper:** on Colab, `_step_inputs` also checks LIVE.json ownership, writes the content hash into QUEUE.json, and on failure closes the session. I did not exercise that wrapper.
- **FETCH_HEAD:** cell 4 leaves it at ecf95a81. Cell 2, when re-run, fetches REPO_REF again before reading it, so nothing depends on this.
- **Re-running cell 4:** a second run in the same runtime (`_parts_from_git` always rewrites the parts) was not tried.

**Clean-up.** I removed the clone and the scratch drive and local directory (`<scratchpad>/presmoke`). Bit-identity printed the expected pair before and after. The main checkout is clean at 1b5686f6, and I wrote nothing there.

## (B) dev_episodes.json: one entry, committed on `wt/c2`

- I ran `git -C /home/user/mesozoic-wt/c2 reset --hard 1b5686f6`, then committed **860755fd** on `wt/c2`. Nothing was pushed.
- The change is one entry added to `/home/user/mesozoic-wt/c2/hardware/trex_form_study/stage2/tb/dev_episodes.json`, in the file's own layout: indent 1, no trailing newline, the same key order as the other entries.
- I wrote the entry directly rather than through `gates2 dev-episode`, because that command would stamp done_at with the current time.
- Fields:
  - step: `V12-acceptance-b7527a66`
  - set / body: V6 / 75-M-H-10
  - commit: b7527a66528750746edf8f533c7f3b8a066aa1e3
  - done_at: 2026-11-17T22:13:46Z; listed_at: 2026-11-18T00:22:14Z
  - command: the `nb acceptance --ref b7527a66… --record …/ACCEPTANCE.json` command from the do report
  - rows_or_mode: "R5 and R6 training env steps (tiny jobs; the run's check episodes are in the reason)"
  - count: 15360

**How I counted.** I followed C2's reason texts for the second run: steps each R5 session ran until its kill, plus the steps after its resume.
- **R5 in `acceptance`:**
  - Killed on `model_1024.zip.part`, with no checkpoint whole on the stand-in Drive.
  - It had run at least 1536 steps. Two things in the record show this: the record's `progress 1536`, and session 0's line 4. In train2 (unchanged between b7527a66 and C1), line 4 is printed at the end of the third rollout, and a tiny rollout is 8 envs × 64 = 512 steps.
  - It had run at most 2048, because the copy of checkpoint 1024 had only begun.
  - Session 1 restarted it from 0 steps (record line and `restarted_from_zero`), then ran 4096.
- **R5 in `acceptance-remedies`:**
  - Killed once checkpoint 1024 was whole, so after 1024 to 2048 steps.
  - Its last session resumed from 1024 and ran 3072.
  - The refused sessions trained nothing. That is what the second run's logs show for the same code; I infer the same here.
- **R6:** 4096.
- **Steps:** the count is the upper bound, 2048 + 4096 + 4096 + 2048 + 3072 = 15360. The record shows at least 13824.
- **Check episodes:** listed in the reason, counted the way C2's checks entry counts them: 64.
  - line 3's G8a: 2 × 2
  - the trainer's G8a short form: 3 × 2
  - the R0L baseline: 3 × 6
  - the monitor at 2048 and 4096 steps: 6 × 6
  - Not counted: checks at 2048 steps cut by the kills.

**What I could not see.** The first run's logs under `/home/user/mesozoic-wt/tb-accept` were replaced by the second run's; only the 159-byte wrapper `tb-accept-bundle/acceptance.log` remains.
- The exact kill points: hence the bounds.
- Whether R5's restart from 0 made G8a's short form or the R0L baseline again: at most 2 + 6 more episodes. The code reruns G8a only if run.json lacks a pass, and the baseline only if `monitor/r0l.json` was not copied back.
- The remedies job's own checks at 2048 and 4096. I counted them from the code path and its done state; the record does not list them.

**For the lead:**
- C2 used two entries for the second run (jobs and checks). As asked, I added one entry: its count is training steps, and the 64 check episodes are only in the reason text. If you want it split the same way, it is a one-entry follow-up.
- The first run's record (ok false, 35 of 36) is not in the repository. The entry points to `/home/user/mesozoic-wt/tb-accept-bundle/stage2/tb/bundle/s2-b1/ACCEPTANCE.v12-b7527a66.json`.

**Not done:** no tests run (none read this file's content); no push, tag or upload; no code changed; nothing run on Colab.
