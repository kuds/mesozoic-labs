# build:CN (roundC, wave 4): the agent's final report, 16 Nov 2026

Working paper. Nothing here is a rule. The package is not merged: its fix round was paused for the usage limit.

WORK PACKAGE CN (the notebook): BUILD REPORT

Worktree /home/user/mesozoic-wt/wpc-cn, branch wt/wpc-cn, head 73a5112. Simulation tooling only; nothing here is a result. Nothing pushed, no history rewritten, no network call, no upload, nothing written in the main checkout. The acceptance run of design 3.10 was NOT done (the lead's step).

1. WHAT I BUILT (five new files, no other file changed)

- designlab/stage2/nb.py (3,248 lines; standard library only at import). Steps, each also a CLI subcommand: setup (pins, line 1), session (K6/K7 refusals, .part sweep, QUEUE.json, SESSION_<k>.json, LIVE.json, STATUS.json), inputs (bundle unpack + `verify --json`, bundle id and 64-hex content hash compared with the manifest's, read_inputs per set and body in a child, line 2; sources drive, rebuild, git:<ref>), preflight (archive preflight, train2 preflight, envs on both floors, G8a per body, line 3), launch (queue, fence, K2 copies, clips, job rules, line 4), smoke (Y1-Y8, SMOKE.json after Y7 and Y8), finish, release_runtime, collect, hours, local-run, acceptance. One function formats each line; LINE_RE matches them.
- stage2/stage2_training.ipynb: 9 cells, no accelerator, plain Python, each cell one step in a subprocess with its whole environment; cell 2 holds the scaffold's Git block (with `designlab` in the restart refusal) and the 40-hex refusal; cell 8 is finish then release_runtime in try/finally.
- stage2/tb/requirements-colab.txt: the nine pins.
- stage2/tb/colab_steps.txt: draft, design 3.12's eight items, numbered, 115 columns (one URL line is longer), what to expect and what to do after each step. Every value only the acceptance run or the smoke can give is a `<<FIELD>>`, listed again at the end.
- tests/test_stage2_nb.py: 33 fast tests on a stub trainer, stub clip and stub pack, 1 slow. Each part above has a test by name: cells parse; kernel imports; numpy-blocked import; pins equal the venv; the lines and failing forms; session refusals; .part sweep; queue to DONE; copy cadence and delete-one-cycle-later; second launcher refused and superseded launcher halts; job rules (retry once then HALT, exit 0/3 next job, exit 4/5 halt); kill and resume from the stand-in Drive; kill while a .part exists; interrupt; inputs on a hand-made bundle through the real bundle CLI; data branch on a scratch remote; rebuild fallback and Y8 on stubs; preflight line 3; collect; hours; release_runtime; smoke on stubs; steps sheet quotations; a local job leaves git status unchanged.

2. ACCEPTANCE (env line of the standing rules, through `bash tools/lowprio.sh`)

- Hash line after both merges: `45719d030359ec7d5d3914af3814d837fdc05d77652ba45c64792b511027528d c0027992de21067525cc4489b74acb157feb89f3a4c7618580addc2b341be1c1` (registered).
- `$PY -m pytest -p no:cacheprovider tests/test_stage2_nb.py` on 2dedc98: exit 0, `33 passed, 1 skipped in 28.16s` (two earlier runs: 29.54s, 29.23s, same counts).
- `STAGE2_SLOW=1 $PY -m pytest -p no:cacheprovider tests/test_stage2_nb.py -k local_run` on 2dedc98: exit 0, `1 passed, 33 deselected in 199.03s (0:03:19)` (on 71912ca: `1 passed, 30 deselected in 202.94s (0:03:22)`).
- `$PY -m designlab.stage2.nb local-run --drive <scratch>/cn/lr2/drive --params ... --record ...` on 2dedc98: exit 0, `S2 LOCAL RUN local-cn: OK in 199 s`, 17 of 17 checks true. Session 0 killed (whole session, SIGKILL) after 59.1 s when "checkpoint 1024 of R5-75-M-H-10-s42 is on the stand-in Drive"; local run directory deleted; RESUME at once refused in 0.5 s with `Run directory: ...: still running (LIVE.json written 1 s ago); nothing to do`; RESUME with TAKEOVER reached DONE in 139.0 s; a further RESUME refused in 0.5 s (finished). Lines as printed:
  `S2 LINE 1/4 commit: 2dedc98df91ce5bec397e51e8bdf21fb41435546 (REPO_REF is a commit: yes; checkout clean: yes)`
  `S2 LINE 2/4 inputs: bundle S1-view, manifest 80a0aafa2177, 28 of 28 files match their sha256: OK`
  `S2 LINE 3/4 G8a 75-M-H-10 set S1: zero action = R0 at 0 ms and 10 ms, 2 of 2 episodes identical (20596 physics steps): PASS`
  `Run directory: <scratch>/drive/runs/local-cn (resuming session 1: R5-75-M-H-10-s42 from 1024 steps; 0 jobs done)`
  `S2 LINE 4/4 first rate R5-75-M-H-10-s42: 454 env steps/s (8 envs, 2 workers, 4 vCPUs); 0.004096M steps in about 0.0 h`
  `S2 LINE 4/4 first rate R6-75-M-H-10-s42: 562 env steps/s (8 envs, 2 workers, 4 vCPUs); 0.004096M steps in about 0.0 h`
  Rates are tiny jobs on a loaded machine (load 2.4-3.3): provisional.
- My first local-run by hand, before the first commit, printed `FAILED in 202 s: new_run_line, resume_line`: my check counted the launch's reprint of those lines twice. Fixed; the run itself had gone through.
- Episodes on a real set: none. Every closed-loop episode ran on the S1 view of 75-M-H-10: four drills (two by pytest, two by hand), each with tiny R5 and R6 jobs, their G8a stock form, R0L baseline and two checks. One hash-only check, no episode: `nb child read-inputs --set V0 --body 75-M-H-10` on 6 files of the main checkout's frozen set, `ok: true`.
- Outputs: tests under tmp_path, my runs under <scratchpad>/cn/ (outside the repository). `git check-ignore -v stage2/tb/work/...` gives `stage2/tb/.gitignore:1:work/`; `stage2/tb/hours.json`, `runs/x/COLLECTED.json` and `checkpoint_<n>.json` are not ignored.

3. DIFF AND STATUS

`git diff claude/mesozoic-trex-stage2-refine-4jffre --stat`:
 hardware/trex_form_study/designlab/stage2/nb.py | 3248
 hardware/trex_form_study/stage2/stage2_training.ipynb | 245
 hardware/trex_form_study/stage2/tb/colab_steps.txt | 294
 hardware/trex_form_study/stage2/tb/requirements-colab.txt | 19
 hardware/trex_form_study/tests/test_stage2_nb.py | 1585
 5 files changed, 5391 insertions(+)
`git status --porcelain`: empty. The working branch's tip fdb31d5 is an ancestor of HEAD. The last merge (73a5112) brought in seven documents and no code; the test runs above are on 2dedc98.

4. FOR THE LEAD

Not done, or never run for real:
- `acceptance` is written and has never run. Only its two refusals are tested (a ref that is not 40 hex; a scratch inside the checkout). Its kill and resumes are local-run's code. It does not do step 6's ALLOW_VERSION_CHANGE / ALLOW_SOURCE_CHANGE with a faked session record, and says so in its record.
- Never executed: pip and apt in setup, the cell's git clone and fetch, the Drive mount, the real disconnect (fake modules in tests).
- smoke only on stubs. Y8's `sets build` command is a guess at what a look set needs (V6 is not in `sets.SETS` yet). Y4's job is not tiny and needs lr_probe.json and filter.json in the checkout.
- preflight's bar form (`g8a.short`) and envs on a real set: not run. A real clip: not run, because the S1 stand-in writes no traces until lookbar is merged.
- The test file has lines over 120 columns (ruff passes: E501 is ignored).

Readings I chose; each could be read otherwise:
a. A superseded launcher does not write the run's HALT.json on Drive (design 3.9 says it does). That file ends a run, and the run then belongs to the session that took over. It writes SUPERSEDED_<k>.json and a local HALT.json.
b. The resume right after a kill needs TAKEOVER, because LIVE.json is fresh. local-run and `acceptance` first show the refusal, then resume with TAKEOVER. Alternative: wait 600 s.
c. QUEUE.json is written by `session`, so a RUN_ID exists from its first session. After a failed line 2 or 3 the next run needs RESUME. Such a session closes LIVE.json, so no ten-minute wait.
d. On a resume the checkout is still cell 2's; REPO_REF "from QUEUE.json" is what is recorded. Another commit is refused without ALLOW_SOURCE_CHANGE.
e. A dirty checkout prints "checkout clean: no" and is recorded; it stops nothing.
f. Checkpoint 0 is never copied. Only the newest check gets a clip, and the queue waits up to 300 s for it at a job's end.
g. Exit codes 4 and 5 halt without a retry. SIGTERM or SIGINT ends the job, writes no HALT.json and leaves the run resumable. A failed Drive write is counted and retried, not a halt.
h. HALT.json is final for the notebook. After the trainer's exit 5 the lead must remove it by hand and pass RETRY_CHECK through an S2_PARAMS file; cell 1 has no such parameter.
i. Line 3's step count is the sum over its episodes. Line 4 reads rate.json `first_three` (two updates in) and the whole budget. Y4 reads the first three lines of updates.jsonl of a four-rollout `--no-gates` job (three updates, as Y4 says).
j. "rebuild" is allowed only when a smoke's Y8 left `inputs/<bundle_id>/rebuilt/REBUILD_OK.json` for the same content hash. That marker is my invention.
k. A smoke is not resumed; finish writes its DONE.json. collect never replaces a file that differs and skips LIVE.json.

Beyond the frozen API: a `setup` step; `finish(..., local=None)`; keyword extras on `inputs` and `preflight`; parameters beyond cell 1 for local runs (SET, TINY, DRIVE_DIR, LOCAL_DIR, MONITOR_EVERY, MONITOR_ALSO_K, PERIOD_S, LR, CONFIG, PREFLIGHT_BODIES, RETRY_CHECK, COPY_PAUSE_S, BUDGET_BY, BUDGET_SET_ON); the test hook S2_NB_SEAMS (stub commands; recorded in the session and printed).

Open points: the queue has no entry for a fine-tune (K8). The notebook's metadata does not ask for High-RAM. nb.py is in train2.SOURCE_FILES, so any change to it between sessions makes a resume refuse "other sources".

For the next packages: the lead's local launches are `nb session --params F` then `nb launch --params F` with DRIVE_DIR equal to LOCAL_DIR under stage2/tb/work, REPO_REF equal to HEAD, and BUDGET_STEPS unless TINY. `nb collect --src <run dir> --dst stage2/tb/runs/<run_id> --kind run`, then `nb hours`.

5. COMMITS (oldest first)
ec17bba the notebook's steps, the queue launcher, collect and hours (nb)
472b438 the Colab notebook: nine cells, each one step in a subprocess
3a7f7b4 the steps sheet, as a draft
7db3169 the notebook's tests on a stub trainer; local_run behind STAGE2_SLOW
71912ca Merge the working branch (rules draft with the fifth list, trainer and look-reading follow-ups)
e381389 a kill while a copy is in progress; the rebuild fallback's and an interrupt's tests
2dedc98 the launcher says when a run's end could not be written; the sheet's note on letting the machine go
73a5112 Merge the working branch (sixth list of rulings, documents only)
Every message, the two merges included, ends with the two attribution lines and names no model elsewhere.
