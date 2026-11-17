# check:CG (17 Nov): the agent's final report

Working paper. Nothing here is a rule.

## Review of work package CG (gates and validation records), branch wt/wpc-cg

**Verdict: merge after the listed fixes.** Bring in tip b13bcd4c. The tree merges cleanly into claude/mesozoic-trex-stage2-refine-4jffre and only the package's three files differ.

- Majors 1 and 2 are small fixes in the package's own files. Both must be done before the chain makes any record.
- Major 2 is a reading the builder already listed for the lead. It needs your ruling.
- No record can be made until the look sets are frozen, so nothing is lost by fixing first.

### Findings, most serious first

**1. MAJOR: the chain starts S2-4 without the committed baseline that rule M3 requires.** File: `stage2/tb/run_trackb.sh`, lines 194-196.
- Rule M3: R0L's baseline is "committed as `stage2/tb/runs/s2-4/R0L_BASELINE.json` before S2-4 starts". Its check: "the commit that adds R0L_BASELINE.json is an ancestor of S2-4's `sessions[0].commit`".
- What the chain does: `V8-baseline-commit` only copies the file and runs `cmp`. `V8-s2-4` starts right after. Nothing checks git.
- Effect: an unattended chain would start S2-4 off a baseline that is not committed. S2-4 is "run once": a second go is a change after a result.
- Fix: add a step before `V8-s2-4` that stops with a message naming the lead unless all three hold:
  - `git ls-files --error-unmatch` finds the file;
  - `git diff --quiet HEAD -- <file>` passes;
  - `git log -1 --format=%H -- <file>` is not empty.
- Also rename `V8-baseline-commit`: it copies the file and commits nothing.

**2. MAJOR (rule case decided differently; the builder listed it as a reading): G9-ref passes while part (a) fails for a reference, and nothing makes that reference unavailable.** File: `designlab/stage2/gates2.py`, lines 621-623 and 912.
- When a refcheck item fails and it is not walk/nominal or walk/v0.4, part (a) gives `ok` False and `blocks` False. G9-ref can still pass with exit 0. The reference is only listed under `not_available`.
- Rule Z2 says that reference is "treated as not available (its tasks absent, T2; its conditions 'not run', H3)". No code does that. The chain would go on to run G8a, R0L, the probes, K3 and S2-4 on it.
- Proposed fix: G9-ref does not pass (exit 1, `followed` naming those references) until the lead makes them unavailable in the set and G9-ref runs again. The alternative (pass and list them) needs changes to the env and bar code, which are not CG's files.
- Related, minor: `BLOCKING_ITEMS` (line 129) has no `start`, though the rule names "start, nominal or v0.4". A failing item is only ever attributed to its own walk or move, so the start, which every item plays, is never named. State this reading in the rule text, or name the start.
- No test feeds a failing refcheck. With `blocked = []` or with (a)'s `ok` forced True, the 7 G9-ref tests still pass.

**3. MEDIUM: kill-resume reuses finished runs without checking they were made with the settings asked.** File: gates2.py, lines 1453-1475 and 1530.
- Reproduced on a copy of the builder's stub runs, calling again with `lr=1e-3` and `config={"filter_hz": 20}`:
  `made {} pass_ True lr_given 0.001 config_given {'filter_hz': 20.0} runs lr 3e-05 runs filter [10.0]`
- K3's `n_envs` 8 and `n_steps` 1024 are never passed to the trainer and never checked. They hold today only because they are train2's defaults (`N_ENVS = 8`, `N_STEPS = 1024`).
- Fix: compare each `run.json`'s settings with the ones asked (set, body, seed, n_envs, n_steps, checkpoint_every, budget, lr, filter_hz). Move a run aside on a mismatch, and list the checks in `settings_ok`.

**4. MINOR: r6-start reuses a finished tiny run from another set.** File: gates2.py, lines 1683-1687.
- The job directory is `R6-<body>-s42`, with no set in the path.
- A done `run.json` is reused without checking its set. A V6 r6-start record could therefore cite an S1 training run.
- Fix: check `run.json`'s set and tiny budget, or put the set into the work path.

**5. MINOR: G9-ref never checks that the set is frozen, and does not check its files against the freeze.**
- The pass test at line 912 never looks at `v.freeze` (line 543). G9-ref can pass on an unfrozen set with `set_freeze: null`.
- The walk playbacks (`setio.build_track`) and the look-hash test use what is on disk and the refs entries' own sha256s. There is no `setio.read_inputs(..., require_frozen=True)` call.
- So the record does not hold the hashes of the files it played, which the brief's frozen API asks for ("inputs' hashes").
- It fails closed later: `require_g9_ref` refuses a record that names no freeze.
- Fix: for any set other than S1, refuse when the set is not frozen. Run `read_inputs` over every file played and record `inputs.files`.

**6. MINOR: test gaps.** Mutations I applied in scratch (gates2's source changed in-process by a pytest plugin) that the tests did not catch:

| Mutation | What it breaks |
|---|---|
| `blocked = []` | a failing nominal or v0.4 item no longer blocks |
| part (a) `ok = True` | a failing refcheck is ignored |
| pass test without b | part (b) no longer counts |
| pass test without c and f | parts (c) and (f) no longer count |
| G9-run (d) ignores `missing` | an expected R0L record that is absent no longer fails (d) |
| `_same` uses `np.allclose` defaults | K3's comparison gets a tolerance |
| `as_ruled=True` | a late kill counts as ruled |
| G8a ignores `compared_steps == reference_steps` | a short comparison counts as identical |

Each case passes on V0 only because another part already fails. Needed tests:
- a doctored refcheck (one failing turn item; one failing walk/v0.4 item);
- a G9-ref with its parts stubbed to pass except one;
- `missing` makes (d) fail;
- a one-ulp perturbation (`np.nextafter`) to pin tolerance zero;
- a kill whose first progress reading is past 81,920.

**7. MINOR: G8a passes a reduced record.** Line 461: `pass_` is true with `complete: false` and exits 0. A reduced run written to the gate path would read as passed by Z1's check (`identical == of`). Fix: pass only a complete record, or refuse a reduced run with `--record` under `stage2/tb/gates/`.

**8. MINOR: G9-run part (e) does not check what the reread record covers.** Line 1057 accepts any complete, trace-equal `bar-reread-1` record of 60-N-T-10. It does not check that its rows are 10 ms rows or that its conditions are turns. Fix: check rows end in `@10ms` and conditions begin with `turn/`.

**9. LOW:**
- (b) cites the comparand's files by the refs entry's sha256, not the file's own hash (line 697).
- `--refcheck PATH` skips the freeze-file check (line 575).
- The chain puts G9-ref on V6noF5 inside V1 (line 126), so every V6 step waits on V6noF5. Rules X1 and Q1 only need it before V10. For the lead.
- If `dev done` fails after a step succeeded, the entry stays "listed" for good, because the marker then skips the whole devstep (line 110).
- The `stub_kr` fixture calls `os.chdir(ROOT)` and never changes back (test line 616).

### For the lead
I agree with the builder on both points below, and checked each:
- **V3 cannot run as built.** `bar._check_rule` (bar.py 2195-2205): on a set that is not registered, only R0 rows run, so R0L is refused on V6.
- **G9-ref (f)'s 1e-6 rad join rule.** On S1 the jumps are real, not an artefact of how they are measured. `Track.joint` just before and at each join differs by the same amounts the gate records: walk to stop 3.77e-5 rad, nominal to nominal_to_brisk 1.84e-5, brisk to brisk_to_nominal 1.11e-4. Each walk segment ends exactly on a frame (fraction 0). The joins from a move into a walk are within 2e-9. So the move files' first frames differ from the walk's frame.
- The builder's other readings (section 3), chain choices (section 4) and the trainer builder's four open points (section 5) are listed correctly. None of them can be built in CG's files.

### Checked and found right
- **Rule X1 refusal:** checks schema, own sha256, set and body, pass, and the latest freeze. Mutations of each check are caught. g8a, rates, kill-resume and r6-start each call it first.
- **G8a:** the stock form on S1 and `g8a.bar_form` on a set, with Z1's ten keys, seeds and delays. A forced 1e-6 difference gives exit 1 and names condition, seed, delay and step.
- **Part (b):** the four ladder limits, including 0.02 rad and 10 % for toe-off; the L1 hash; the comparand form; and the steps, lift-offs, reference toe-off and its side of 0.02 rad for every walk and turn item.
- **Part (c):** the fifth control (tail to a tenth about its mean) fails the tail term alone. Breaking it is caught.
- **Part (f):** the screen record's hashes match the entries' on V0.
- **G9-run (d):** companions' `bar_record.path` is relative to the bar's out directory, as lookbar writes it, so V3's collect followed by g9-run lines up. Toe-clearance readings exist only on walk and turn items, matching the committed 10 ms records. A walk that falls has no window in either the bar or the look reading.
- **K3:** SIGKILL of the whole process group (pool workers and the monitor child share it). Resume from 65,536. Every checkpoint after the resume is compared. VecNormalize is read through s2load; no pickle load in gates2. Resumable, and refuses while a trainer is alive. Train2 itself refuses a run that is not tiny without a rate or a filter (ruling 5).
- **Records:** atomic writes, own sha256, provenance with the command's own outputs left out.
- **Ignore status:** `git check-ignore`: every live path is ignored; every record path is not.
- **Chain commands:** match each module's argparse (train2 run, probe-sigma, filter, probe-lr; lookbar run with `@gated`; nb collect kinds; monitor2 s2-10).
- **Chain counts:** the X1 counts add up: 240, 80, 1,179,648, 278,528, 98,304.

### Exact outputs
- `pytest -p no:cacheprovider tests/test_stage2_gates2.py`: **29 passed, 1 skipped in 75.01s (0:01:15)**.
- `STAGE2_SLOW=1 ... -k r6_start`: **1 passed, 29 deselected in 26.74s**.
- `sh -n` exit 0 and `dash -n` exit 0 (`/bin/sh` is dash).
- `TB_DRY=1 sh stage2/tb/run_trackb.sh`: exit 0, 45 steps, last line "dry run (TB_DRY): nothing was run or written". No work directory or list was created.
- Record-key hashes: `45719d030359ec7d5d3914af3814d837fdc05d77652ba45c64792b511027528d c0027992de21067525cc4489b74acb157feb89f3a4c7618580addc2b341be1c1` (the registered pair).
- `git merge-tree --write-tree` against fccebe42 (the working branch's tip, which is also the merge base): exit 0, tree d6c83af3. Only gates2.py, test_stage2_gates2.py and run_trackb.sh differ.
- Mutations: 21 run, 13 caught, 8 survived (finding 6).
- Commits f84f1ced, 3a0983a0, 8069eda9 (merge) and b13bcd4c all end with the two attribution lines. No other line names a model. None is missing the lines.
- Nothing ran that should not have. There is no `stage2/tb/gates/` or `stage2/tb/validation/` in the worktree or the main checkout, and `git status --porcelain` is empty in both. The builder's real-env training was two tiny 4,096-step kill-resume runs on S1, in the scratchpad, with no record.

### What to bring in
Tip **b13bcd4c**, as a merge, once findings 1-2 are fixed or ruled. Findings 3-5 and the tests in 6 are best done in the same small fix round.

### Not checked
- Anything on a real look set: none is frozen.
- G9-run on real lookbar output.
- K3's real settings; rates at 3 and 4 workers.
- The `-X` and V6noF5 forms of G9-ref on real data.
- r2lfit, which is not on the branch.
- A full non-dry chain run.
- Train2's and the monitor's internals beyond the calls gates2 makes.
- Colab.

Scratch files are in /tmp/claude-0/-home-user-mesozoic-labs/2dca1c00-13f3-5853-acf6-8c3e3e04feb6/scratchpad/wave5/check_cg/: the mutation results are in muts/RESULTS.txt; join_probe.py and stale_kr.py are the two checks above.
