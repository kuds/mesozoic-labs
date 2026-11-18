# Resuming the T. rex study

Written on 18 Nov 2026, when the study was paused. Section 1 is for the owner. Sections 2 to 4 are for whoever
restarts the work. Paths are relative to `hardware/trex_form_study/`.

## 1. For the owner: why it stopped, what it reached, what is kept

### Why it is paused

You paused T. rex development on 18 Nov 2026 (decision P47 in [DECISION_LOG.md](DECISION_LOG.md)). A two-legged
robot has to keep its balance at every moment, and making one walk takes many rounds of building, measuring and
adjusting real hardware, which needs equipment and time the project does not have now. So the project moves to a
Triceratops: on four legs it can walk slowly with three feet on the ground, and it stays balanced without effort.

The same day you added that the T. rex stays on the roadmap as the flagship, a milestone for end 2027 or early 2028,
with the Triceratops as the nearer demo in the third quarter of 2027. It is paused, not dropped. That is decision P51;
its row is in the decision log of the working branch from commit `c5274905` on, one commit after the pause record
this file builds on.

### What the study reached (all in a computer model: nothing was built or measured)

- **The 75 cm robot walks in the model only when its motors react quickly.** With the balance controller carried
  over from the earlier stages, it stayed up in 20 of 20 runs at 0.24 to 0.40 m/s whenever the delay between a
  command and the motor's response was 6 milliseconds or less. At 8 ms it fell in every run at 0.40 m/s. At 10 ms,
  the delay the study assumes until a real one is measured, at most 3 of 20 runs stayed up. The two-step stop worked
  in every run at 0, 4 and 6 ms. Pushes and the quick 90° turn did not pass. The tail motor was over its planning
  load at 0.35 m/s and faster. The full account is [STAGE2_75_REPORT.md](STAGE2_75_REPORT.md).
- **The walking styles are fixed.** On 17 Nov three 75 cm reference walks were frozen, so that training would always
  use the same ones:
  - the lifelike walk you chose (V6: hip bob, toe-off, body roll, a tail that swings against the body, small
    changes from step to step), with the plain head;
  - the same walk with the expressive head (a turning neck and a jaw);
  - the same walk without the roll onto the middle claw (V6noF5).

  Three earlier walks (V0, F25F9 and F25F9cl) were frozen on 15 Nov.
- **Training on Colab was ready to start.**
  - One fixed version of the code was named for it, the launch version `cc437b6a`.
  - It passed all 36 checks of a full rehearsal (36 of 36).
  - The day-one steps sheet was sent to you on 18 Nov: [stage2/tb/colab_steps.txt](stage2/tb/colab_steps.txt).
  - You chose to skip the day-one check (P50). The sheet stays usable for a restart (section 2, step 7).
- **The first local training trial gave no gain.** The trial, called S2-4, trained for 2 million steps. At the end,
  14 of 15 test runs stayed up, for 25 seconds on average. None met every limit, while the controller it was meant to
  improve on met them in 6 of 15. You chose to go on to Colab anyway (P44). That is the step the pause stopped.

### What is kept, and where

It is all in the GitHub repository `kuds/mesozoic-labs`. A *branch* there is a named line of saved versions.

| What | Where | Note |
|---|---|---|
| All the study's code, records, reports and decisions, and this file | the working branch `claude/mesozoic-trex-stage2-refine-4jffre` | It stays (P48). |
| The same work in the repository's main line | a pull request into `main`, which the lead opens and you review and merge (P48) | It keeps the study easy to find. |
| The input files for training the 75 cm robot on Colab (about 34 MB) | the data branch `claude/mesozoic-trex-stage2-bundle-s2-b1` | Never merged. The day-one sheet reads it by name, so it must not be changed (P50). |
| The other input files a fresh copy of the repository lacks (380 files, about 251 MB) | the archive branch `claude/mesozoic-trex-stage2-archive-a1` | Never merged (P49). It saves a rebuild of about 2.5 hours. |
| Four unfinished side lines of work (77 saved versions; the 60 cm walking styles among them) | one file in the working branch, `stage2/design/checkpoint/worktrees-2026-11-18.bundle` | Described in [worktrees-2026-11-18.md](stage2/design/checkpoint/worktrees-2026-11-18.md). |

**Not kept:** the step-by-step traces of the simulation runs (about 2 GB, only for diagnosis, and rebuilt exactly
from the records; P49) and the Python setup (rebuilt in about 2 minutes).

**One gap.** 52 files of the newest frozen walks (V6 with both heads, and V6noF5) were only on the cloud machine when
the archive was made (its first version, `be474190`), so that version does not hold them. They can be rebuilt from the
records (not yet tried), or the lead adds them to the archive before the cloud machine is deleted.

## 2. How to restart, step by step

### Step 0: check that the branches are there

From any machine with git:

```bash
git ls-remote https://github.com/kuds/mesozoic-labs refs/heads/claude/mesozoic-trex-stage2-refine-4jffre \
    refs/heads/claude/mesozoic-trex-stage2-bundle-s2-b1 refs/heads/claude/mesozoic-trex-stage2-archive-a1
```

- Expect three lines.
- The data branch must read `ecf95a81a6ebb43e0331b5c1db74bdea7e74aaed` (P50).
- The archive's first version was `be47419033b8705e8943072bbea1ced2ed98b32a`. A later version may add the 52 missing
  arrays; its README says what it holds.
- If the archive branch is missing, it was not pushed before the cloud machine was deleted. Then step 3 falls back on
  `restore_inputs.py restore`.

### Step 1: a fresh session on the working branch

- Open a new Claude Code session on `kuds/mesozoic-labs` and check out `claude/mesozoic-trex-stage2-refine-4jffre`.
  Use its tip, or the commit you restart from.
- Use the cloud environment "mesozoic-labs" if it still exists (made 14 Nov). Its setup script installs libOSMesa and
  builds the venv.
- Read, in order:
  1. this file;
  2. the blocks of [STATUS.md](stage2/design/checkpoint/STATUS.md) from 17 Nov, ~19:20 UTC, on;
  3. [LESSONS_LEARNED.md](LESSONS_LEARNED.md) sections 4 (the cloud container) and 5 (traps);
  4. the STANDING RULES block of the 14 Nov prompt in [NEXT_SESSION_PROMPT.md](NEXT_SESSION_PROMPT.md);
  5. [DECISION_LOG.md](DECISION_LOG.md) rows P40 to P51.
- Ask the owner for the usage meter's reading before launching agent work. STATUS.md holds the second session's
  rates, and `LESSONS_LEARNED.md` §4.7 explains why this matters.

### Step 2: the environment

[tools/README.md](tools/README.md), "A fresh container, in order", steps 1 to 3:

```bash
apt-get install -y libosmesa6
cd hardware/trex_form_study
TORCH=cpu bash tools/setup_venv.sh
```

- This builds Python 3.13.18 with the versions of `tools/requirements-study.txt`, about 1.4 GB.
- Run every Python command below from `hardware/trex_form_study` with `PYTHONPATH=. MUJOCO_GL=disable`
  (`LESSONS_LEARNED.md` §4.5).
- Run long jobs through `bash tools/lowprio.sh`, with the keep-alive of the README's step 3.

### Step 3: the inputs, from the archive branch

The archive's README gives the commands. It is `hardware/trex_form_study/stage2/archive/a1/README.md` on the archive
branch, not on this one. From the clone's root:

```bash
ROOT=$(pwd)
git fetch origin claude/mesozoic-trex-stage2-archive-a1
A=$(mktemp -d) X=$(mktemp -d)
git archive FETCH_HEAD hardware/trex_form_study/stage2/archive/a1 | tar x -C "$A"
(cd "$A/hardware/trex_form_study/stage2/archive/a1" && sha256sum parts/*.tar)   # compare with MANIFEST.json "parts"
for p in "$A"/hardware/trex_form_study/stage2/archive/a1/parts/inputs-*.tar; do tar xf "$p" -C "$X"; done
cd "$ROOT/hardware/trex_form_study"
PYTHONPATH=. .venv/bin/python tools/restore_inputs.py copy-from "$X"
PYTHONPATH=. .venv/bin/python tools/restore_inputs.py check
rm -rf "$A" "$X"
```

**What `check` should print from the archive's first version alone** (as it did in a fresh clone on 18 Nov):

- every group matching and agreeing with the records;
- `sets` with 67 files matching and 69 missing;
- exit 1.

**Completing `sets`:**

- 17 of the 69 are the `refs` and `transitions` of V6 on 75-M-H-10. They are in bundle s2-b1's three parts on the
  data branch, at `hardware/trex_form_study/stage2/tb/bundle/s2-b1/parts/*.tar`, with members at study paths.
  Extract the parts into an empty directory and run `copy-from` on it with `--group sets`. **Not tried.**
- The rest: `restore_inputs.py restore --group sets` rebuilds the frozen sets' arrays in place, and puts a file in
  place only when its sha256 is the frozen one. **Not tried for these sets.** The set on `75-M-H-10-X` needs group
  `x`, which the archive gives.

**With no archive at all,** `restore_inputs.py restore` rebuilds everything from the records. That took about 2.5 h on
2 processes on 14 Nov, for the five groups of that day (`tools/README.md`).

`check` must exit 0 before any run that reads the inputs.

### Step 4 (only if needed): the unfinished side branches

The 60 cm look sets and three superseded packages are in the git bundle. From the clone's root:

```bash
B=hardware/trex_form_study/stage2/design/checkpoint/worktrees-2026-11-18.bundle
git bundle verify "$B"
git fetch "$B" 'refs/heads/wt/wp-l:refs/heads/wt/wp-l'
```

After a `--single-branch` clone, first fetch the working branch. The details and the 18 Nov restore test are in
[worktrees-2026-11-18.md](stage2/design/checkpoint/worktrees-2026-11-18.md).

### Step 5: the bit-identity check

Every committed record's key hashes the code it was made with. The code must still be that code:

```bash
PYTHONPATH=. MUJOCO_GL=disable .venv/bin/python -c "from designlab.stage2 import bar; from designlab.stage1 import panels; print(bar.bar_sources()['sha256'], panels.panel_sources()['sha256'])"
```

It must print exactly this pair, the bar's sources and then the panels':

```
45719d030359ec7d5d3914af3814d837fdc05d77652ba45c64792b511027528d c0027992de21067525cc4489b74acb157feb89f3a4c7618580addc2b341be1c1
```

Then the regression record:

```bash
PYTHONPATH=. MUJOCO_GL=disable OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
    .venv/bin/python -m designlab.stage2.regress check
```

It must print this line and exit 0:

```
stage2/regress/regress.json: current (full tier, 256 of 256 episode runs matched, 248 distinct committed episodes, the four source hashes are today's)
```

Both commands printed exactly this on 18 Nov 2026, in a checkout of the pause's code (`b1d5e98e` plus documents) with
no inputs restored. If either differs, a file that a record key hashes has changed. Find the change before running
anything that writes records.

### Step 6: the tests

Run them after `check` exits 0, because many tests read the arrays or skip without them:

```bash
mkdir -p logs
MUJOCO_GL=disable PYTHONPATH=. .venv/bin/python -m pytest -rs -p no:cacheprovider tests > logs/pytest.log 2>&1
echo PYTEST_EXIT=$? >> logs/pytest.log
```

- **Expected count.** The last whole-suite count in the study's files is 679 passed, 107 skipped (14 Nov, at
  `e0ad284`). That was before most of Stage 2's packages and their tests (`tests/` now holds 58 files). No later
  whole-suite count is recorded, so record yours.
- **One package's tests.** Add `-o addopts=""` so that pytest prints its summary line. For example,
  `tests/test_stage2_nb.py` gave 74 passed, 1 skipped when ruling 45 was merged (17 Nov).
- **Slow tests.** `STAGE2_SLOW=1` and `STAGE1_SLOW=1` turn on the slow tests (`tools/README.md`).

### Step 7: the next step, the owner's day-one Colab smoke

**The sheet.** The sheet as sent is [stage2/tb/colab_steps.txt](stage2/tb/colab_steps.txt), at `1b5686f6` and later.
Its day one is sections 1 to 3, the smoke only.

**When it is valid (P50).** It stays valid while two things are not moved:

- the launch commit C1, `cc437b6a44d575db63e89c63e2f9dc42c4424947`;
- the data branch `claude/mesozoic-trex-stage2-bundle-s2-b1`.

The notebook fetches C1 by its 40 characters, so C1 must stay reachable on GitHub. It fetches the data branch by name,
so that branch must not be changed, renamed or deleted (STATUS.md, 18 Nov 00:40 UTC block). Check both before you
send the sheet:

```bash
git ls-remote origin refs/heads/claude/mesozoic-trex-stage2-bundle-s2-b1    # ecf95a81a6ebb43e0331b5c1db74bdea7e74aaed
git fetch origin claude/mesozoic-trex-stage2-refine-4jffre
git merge-base --is-ancestor cc437b6a44d575db63e89c63e2f9dc42c4424947 FETCH_HEAD && echo "C1 is on the working branch"
```

**After the smoke** (STATUS.md, 18 Nov 00:40 UTC block):

1. Read `SMOKE.json` from the owner's Drive.
2. Fill the smoke fields.
3. Run `nb sheet-check` without `--before-smoke`.
4. Set `BUDGET_STEPS` and the table of launches.
5. Send the sheet again.

Then come the 40M-step runs, each stopped by its own checks when it shows no gain (P44). Colab use is capped at 220
runtime-hours (P29).

**If either has moved,** a new C1 and a new acceptance run are needed, by rulings 45 to 47
([ROUND_C_RULINGS_17NOV.md](stage2/design/ROUND_C_RULINGS_17NOV.md)):

- C1 must be the head of a pushed ref when the acceptance run starts.
- Nothing is pushed to the working branch from naming C1 until the run has started.
- The main checkout stays clean while the run lasts.
- `nb acceptance --ref <C1>` must pass 36 of 36. The command of the run that passed on C1 `cc437b6a` is in
  [roundC__do_accept3_17nov.md](stage2/design/checkpoint/reviews/roundC__do_accept3_17nov.md).
- Then commit C2. It holds the sheet filled from
  [colab_steps.template.txt](stage2/tb/colab_steps.template.txt), `ACCEPTANCE.json` and `dev_episodes.json`, and
  passes `nb sheet-check --before-smoke`.
- Send the new sheet.

## 3. What was open at the pause

Each item was checked against STATUS.md (its 17 Nov 21:35 UTC block lists what was parked), the plans and the
decision log.

| Item | State at the pause | Where |
|---|---|---|
| **The 60 cm look sets** (P43) | Parked until the 75 cm training runs. V6 and V6noF5 on 60-N-T-10, and V6 on 60-N-T-10-X, are rebuilt and not frozen. All three are stopped at `nominal_to_v0.4`: its first swing asks the knee for 1.0958, 1.0964 and 1.0609 of its speed limit. The check says `merge_after_fixes`: first a test-only fix in `tests/test_stage2_moves_look.py`, then a cherry-pick, because a direct merge conflicts. 60-N-T-10 stays in the plan (P33). | branch `wt/wp-l` (tip `8bc47346`) in `worktrees-2026-11-18.bundle`; [the check](stage2/design/checkpoint/reviews/roundC__check_RB60_17nov.md); [DECISION_LOG.md](DECISION_LOG.md) P43 |
| **Servo S4** | Not started; waits for the training path. S1 to S3 are merged; S3 is gate 1's runner and the served rows SV0 and SV2. S4 measures before use: `r5env.throughput` with the servo layer, and the served stand-in rows at `N` and at each setting. | [SERVO_MODEL_DESIGN.md](stage2/design/servo_model/SERVO_MODEL_DESIGN.md), section 6, row S4 |
| **The buy gate** | Approved by the owner: the stages, gate 1's conditions 1, 2, 3 and 5, and the floors (P40, P41). Still the lead's proposals, never put to the owner: condition 4 (the tail servo), the numbers of gates 2 and 3, and the six measurements the servo note proposes to add to the bench sheet. No purchase is decided. | [BUY_GATE_DRAFT.md](BUY_GATE_DRAFT.md) |
| **Add-on items B and C** (P38) | B is the carpet range, with no verdict. C is the refit of the stand-in controller with delay (two fits). Each has a design and an independent review on file. Neither has registered rules. They run only if there is room, after their rules are rewritten, reviewed and committed. | [STAGE2_75_ADDON_PLAN.md](STAGE2_75_ADDON_PLAN.md); `stage2/design/addon/` (`design_AB.md`, `review_AB.md`, `design_C.md`, `review_C.md`) |
| **The carpet side-check** (P36) | Replaced by P37's fifth answer, "run a range, no verdict", which is item B above. The literature figures found so far are secondary only: no source document was read. | [carpet_sources_15nov.md](stage2/design/carpet_sources_15nov.md); P36, P37 |
| **Leftover: G1 on an idle machine** | G1 (the step-rate gate) is provisional for the -X bodies and for the Jetson body 75-M-H-10-J. To finish it, run `PY=.venv/bin/python bash stage2/x/run_x.sh g1` and `PY=.venv/bin/python bash stage2/j/run_j.sh g1`, each on an otherwise idle machine. | [HEAD_X.md](stage2/x/HEAD_X.md) section 3; [JETSON_BODY.md](stage2/j/JETSON_BODY.md) section 9, item 1 |
| **Leftover: J-vs-Pi panels** | Not run. The paired-seed panels compare the Jetson body with the Pi body. To pre-register them as an added tier. | [JETSON_BODY.md](stage2/j/JETSON_BODY.md) section 9, item 4; [lead_notes.md](stage2/design/lead_notes.md) (15 Nov 02:05 UTC) |
| **Leftover: group j** | Done in code: `tools/restore_inputs.py` has group `j` since `f5a0bf5e`, 30 files, and all 30 are in the archive. Its rebuild (`restore --group j`) has not been timed in a fresh worktree. | [tools/README.md](tools/README.md) |
| **Leftover: A19** | A draft row (the Jetson's power-mode budgets: 18.0 W and 28.0 W) that is not in the decision log and was not put to the owner. | [JETSON_BODY.md](stage2/j/JETSON_BODY.md) section 10 |
| **The -X head's preconditions** (V0) | Two are left (ruling 44): G1 of both -X bodies on an idle machine (above), and the jaw cover decision, because the cover overhangs its hinge and trimming it changes every -X record. The set V6 on 75-M-H-10-X is frozen. The -X head is not in bundle s2-b1; a later bundle with its own id adds it (ruling 44). Its learning-rate probe is done (1e-5, in `stage2/tb/validation/lr_probe.json`), and it is run again if the jaw cover changes the -X records. | [ROUND_C_DESIGN.md](stage2/design/ROUND_C_DESIGN.md), run plan row V0; [HEAD_X.md](stage2/x/HEAD_X.md) sections 4 and 7; [ROUND_C_RULINGS_17NOV.md](stage2/design/ROUND_C_RULINGS_17NOV.md) ruling 44 |
| **R2L fits** (V10) | Parked. The tool is merged; no fit is made (`stage2/tb/r2l/index.json` lists none). | [ROUND_C_DESIGN.md](stage2/design/ROUND_C_DESIGN.md), run plan row V10 |
| **Questions awaiting the owner** | (1) The five questions for the maker of the leg drives. The owner sends them, and gate 1 is judged after the answers or two weeks after sending (P41); no sending is recorded. (2) Buy-gate condition 4 and the numbers of gates 2 and 3 (above). (3) The draft decision-log rows F28, F29 and E, and a row owed for the add-on's result (proposed F30). (4) A19 (above). (5) The compute choice (P14 against the Jetson), which waits for the delay bench test (P30, P32). (6) P25, still open. | [MAKER_QUESTIONS.md](stage2/design/servo_model/MAKER_QUESTIONS.md); [STAGE2_75_REPORT.md](STAGE2_75_REPORT.md) section 11; [STATUS.md](stage2/design/checkpoint/STATUS.md) (16 Nov 21:13 UTC, point 5); [DECISION_LOG.md](DECISION_LOG.md) |
| **The archive's gap** | 69 frozen set arrays are not in the archive's first version. 17 of them are in bundle s2-b1. 52 (V6 with both heads, and V6noF5, on 75 cm) were only in worktrees of the cloud machine. | step 3 above; the archive's README on its branch |

## 4. What carries over to other builds

These are not specific to a biped, and a Triceratops study can take them as they are or adapt them.

| What | Paths | What it gives |
|---|---|---|
| **The Colab notebook, with Drive checkpoints and resume** | [stage2/stage2_training.ipynb](stage2/stage2_training.ipynb); [designlab/stage2/nb.py](designlab/stage2/nb.py) (launch, `acceptance`, `sheet-check`, the runtime-hour tally); [designlab/stage2/train2.py](designlab/stage2/train2.py) (`--resume`); [designlab/stage2/vecpool.py](designlab/stage2/vecpool.py); [colab_steps.template.txt](stage2/tb/colab_steps.template.txt); [requirements-colab.txt](stage2/tb/requirements-colab.txt) | Training on Colab from a fixed commit. Checkpoints are copied to Drive, and a killed runtime resumes from the last whole one. The acceptance run tested this on a stand-in Drive ([ACCEPTANCE.json](stage2/tb/bundle/s2-b1/ACCEPTANCE.json)): killed during the copy of checkpoint 2048, it resumed from 1024. Never yet run on Colab itself. |
| **The bundle tools** | [designlab/stage2/bundle.py](designlab/stage2/bundle.py) (inputs packed into deterministic tar parts with a committed manifest); the data-branch route (P46) and its reason, [drive_route.json](stage2/tb/validation/drive_route.json) (the Drive connector carried about 3.1 MB per file); [tools/restore_inputs.py](tools/restore_inputs.py) (git-ignored inputs restored by sha256, `copy-from`, `check`); git bundles for unpushed branches ([worktrees-2026-11-18.md](stage2/design/checkpoint/worktrees-2026-11-18.md)) | Getting large ignored files and unpushed work across machines, checked byte for byte. |
| **The servo-delay model and the bench sheet** | [designlab/stage1/speedfit.py](designlab/stage1/speedfit.py) (`servo_delay`: each command reaches the servos a whole number of physics steps late); [designlab/stage2/servo.py](designlab/stage2/servo.py) with [SERVO_MODEL_DESIGN.md](stage2/design/servo_model/SERVO_MODEL_DESIGN.md) (friction, gear play, voltage sag and delay, drawn at random; every value with its source and kind); [BENCH_SHEET.md](stage2/jetson75/BENCH_SHEET.md) | A harsher simulation for any legged body, and the bench measurements that would replace its guesses. The maker questions ([MAKER_QUESTIONS.md](stage2/design/servo_model/MAKER_QUESTIONS.md)) apply to any build with the same leg drives. |
| **The latency budget** | [LATENCY_AND_BUS.md](stage2/jetson75/LATENCY_AND_BUS.md), `stage2/jetson75/latency_bus.py` and `latency_bus.json` | The command-to-torque delay per compute layout (a microcontroller with CAN, or a Jetson), estimated term by term. |
| **The price pages** | [BUDGET_75.md](stage2/jetson75/BUDGET_75.md), `stage2/jetson75/budget_75.py` and `budget_75.json`; the published pages are listed in [lead_notes.md](stage2/design/lead_notes.md) ("Owner pages published") and [LESSONS_LEARNED.md](LESSONS_LEARNED.md) section 8; the page tools are [tools/report_page.py](tools/report_page.py), [tools/shot_page.js](tools/shot_page.js) and [designlab/page_kit.py](designlab/page_kit.py) | Catalog prices with sources, as a record (P31), and how a Markdown report becomes the owner's page. |
| **The pre-registration and decision-log process** | [DECISION_LOG.md](DECISION_LOG.md) (each owner answer with the options put and the one chosen); [STAGE2_75_PLAN.md](STAGE2_75_PLAN.md) and [STAGE2_TRACKB_PLAN.md](STAGE2_TRACKB_PLAN.md) (rules committed before the first run, and rules "changed after a result" labelled so); the lead's rulings, for example [ROUND_C_RULINGS_17NOV.md](stage2/design/ROUND_C_RULINGS_17NOV.md); the independent checks in `stage2/design/checkpoint/reviews/`; [LESSONS_LEARNED.md](LESSONS_LEARNED.md) sections 3 (method) and 4 (the cloud container) | How a result is kept honest: rules fixed before runs, two seeds for any verdict, an independent check of every build and report, and nothing adopted without the owner. |
| **The container tools** | [tools/README.md](tools/README.md): `keepalive.sh`, `memguard.sh`, `lowprio.sh`, `setup_venv.sh` | Long jobs in a cloud container that may restart. |
