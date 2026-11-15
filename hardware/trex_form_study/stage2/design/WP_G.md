# Work package G: the closed-loop bit-identity regression and the one Stage 1 edit

You are building the guard of a robotics simulation study (a robotic T. rex biped; everything is simulation).
You need no other context than this brief and the design it points into:
`/tmp/claude-0/-home-user-mesozoic-labs/2dca1c00-13f3-5853-acf6-8c3e3e04feb6/scratchpad/design/ROUND_A_DESIGN.md`
("the design"; read its sections 1 (D18, D19), 2.3's `regress.py` block, 2.4's first row, 4.2).

Background you must know. Stage 1's records are cited, never rerun. Every committed Stage 1 and Stage 1b episode
record holds the sha256 of its 50 Hz trace. Re-running those episodes with unchanged code reproduces the traces
byte for byte (shown on 14 Nov for 160 turn episodes of `60-N-T-10` and four episodes of `75-M-H-10`). That is the
only closed-loop proof available that an edit to a Stage 1 module changes nothing by default. Your package turns
it into a tool, a committed record and a test, and then uses it to land one small edit.

## Goal

1. `designlab/stage2/regress.py`: re-run committed episodes through the stock `eval` command into a temporary
   directory and compare each trace's sha256 **and** each episode record with the committed ones.
2. `stage2/regress/regress.json`: the committed record of the full tier, made on the code as you leave it.
3. The edit to `designlab/stage1/control.py` of the design's section 2.4, proven default-identical by 1 and 2.

## Files you own (and no others)

- `designlab/stage2/regress.py`, `tests/test_stage2_regress.py`, `stage2/regress/regress.json`
- the edit to `designlab/stage1/control.py` (only the lines named below)

Do not edit any other file, and no other line of `control.py`. `designlab/stage2/s2io.py` (work package R, merged
before you start) gives you `require_one_thread`, `write_json`, `sha256_file`, `STUDY`.

## Setup

Worktree: the absolute path the lead gives you; work only there.
```
cd <your worktree>/hardware/trex_form_study
PY=/home/user/mesozoic-labs/hardware/trex_form_study/.venv/bin/python
export PYTHONPATH=. MUJOCO_GL=disable OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
$PY tools/restore_inputs.py copy-from /home/user/mesozoic-labs/hardware/trex_form_study --group picks-models picks-arrays
$PY tools/restore_inputs.py check --group picks-models picks-arrays       # exit 0 before anything else
```
The Stage 1b turn rows read their gains from `stage1/runs/2026-11-10-speed` (committed JSON; see
`tools/README.md`, "Eval traces", for the exact command and what it needs). One heavy process at a time
(`--workers 1`); run the full tier through `bash tools/lowprio.sh`. Compute cap: 2 core-hours (the full tier
twice: before and after the edit).

## `regress.py`

```python
CASES: tuple     # dicts: {tier, run (the committed run dir the records are in), inputs, body, row_module | None, rows, conditions, seeds}
def run(tier="fast", record=None, procs=1, log=print) -> dict
def check(record=STUDY / "stage2/regress/regress.json") -> list[str]     # differences between the record and today's code; [] is good
def main(argv=None) -> int     # CLI: python -m designlab.stage2.regress {run|check} [--tier fast|full] [--record PATH] [--procs 1]
```
Cases, exactly the design's section 4.2 table:
- fast: (a) `75-M-H-10`, records under `stage1/runs/2026-11-08`, `--rows R0,R2/42 --conditions
  start_stop/twice,walk/brisk --episodes 1`; (b) `60-N-T-10`, records under `stage1/runs/2026-11-10-speed`,
  `--row-module designlab.stage1.speedfit --rows R2s/42@10ms,R2@10ms --conditions
  turn/walking_0.35_left,turn/in_place_0.50_left --episodes 1 --inputs stage1/runs/2026-11-08`.
- full: fast, plus `--episodes 1` of every hard-floor condition (`stand/hard`, the six walks, the six turns,
  `start_stop/twice`, the eight pushes) of `75-M-H-10` and `60-N-T-10` under `R0,R2/42` from
  `stage1/runs/2026-11-08` (R0 runs hard floor only; skip conditions a row does not run), plus all 20 seeds of
  `--behaviors turn` under `R2s/42@10ms,R2@10ms` on `60-N-T-10` (the 160 Stage 1b episodes).
Each case is one subprocess: `python -m designlab.stage1 eval --body B [--row-module M] --rows ... --conditions
... --episodes N --workers 1 --inputs I --out <temporary dir>`, environment: the three thread variables at 1,
`MUJOCO_GL=disable`, `STAGE1B_SPEED_RUN` removed. Never pass a committed run directory as `--out` (`eval`
overwrites records whose key differs). Then, per episode: `trace_equal` = sha256 of the regenerated
`eval/traces/<row>/<behavior>/<condition>/<seed>.npz` equals `episodes[i].trace.sha256` of the committed record at
`<run>/<body>/eval/<row>/<behavior>/<condition>.json`; `record_equal` = the regenerated episode dict equals the
committed one on every key but `wall_s` (compare after a JSON round trip; floats must be exactly equal).
`run` refuses to start (exit 2, message) if the arrays or the round-two model files are missing (otherwise
`plant.model_files` would write `stage1/models/<body>/` into the checkout), asserts `git status --porcelain` is
empty before and after, and with `--record` writes the record of the design's 4.2 through `s2io.write_json`:
cases with both flags, `matched`, `of`, the four source hashes read at run time
(`panels.panel_sources()["sha256"]`, `plant.sources_sha256()`, the tuner's and the speed fit's:
`tune.tuner_sources()`, `speedfit.speed_sources()`; read their return shapes in the code), versions, thread
count, commit, dirty flag, time. Exit 1 on any difference. `check` compares the record's four hashes with
today's and requires `tier == "full"` and `matched == of`.

Resumable: keep each case's result in `<temporary root>/cases/<case id>.json` under a scratch root given by
`--scratch` (default a `logs/regress/` directory, ignored) so a killed full run continues; the record is written
only when every case is there.

## The `control.py` edit (do it only after the full tier passes on the unedited code)

In `designlab/stage1/control.py`:
- `Track.add_periodic` (`:967-978`): `K = int(ref.arrays["window_strides"]) if "window_strides" in ref.arrays else
  1`; pass `period=ref.period / K, frames_per_stride=ref.n // K` to `frame_info`; set the new field
  `stride_s=ref.period / K` on the `Segment`. Leave `period=ref.period`, `n=ref.n`, `step` as they are (the window
  keeps driving the wrap).
- `Segment` (`:750-766`): add `stride_s: float = 0.0` as the last field.
- `_episode` (`:2300`): `sg.period` -> `(sg.stride_s or sg.period)` in the `per_ref` expression.
Nothing else. With no `window_strides` key (no file of Stage 1 or of round A has one) `K = 1`, `x / 1 == x` and
`n // 1 == n` exactly, so every existing episode must be unchanged. This edit moves four source hashes; that is
intended, and it is why it lands now, before any Stage 2 record exists.

## Tests to write (`tests/test_stage2_regress.py`)

- `test_regress_record_is_current` (fast, needs no arrays): `regress.check()` returns `[]`.
- `test_regress_fast` (behind `STAGE2_SLOW=1`; skips without arrays): `regress.run("fast")` matches 8 of 8.
- `test_window_strides_key` (fast, synthetic): a `refs.Reference` built from a Stage 1 walk's arrays tiled to two
  strides with `window_strides = 2` gives a `Segment` whose `info["period"]` equals the one-stride reference's,
  whose `period` is the window and whose `joint(t)` equals the one-stride track's at the same times; without the
  key the segment equals today's (skip without arrays).
- `test_cases_only_name_committed_records`: every case's committed record path exists (JSON is committed).

## Acceptance (report each command's own exit code)

```
bash tools/lowprio.sh $PY -m designlab.stage2.regress run --tier full --scratch logs/regress-before     # on the unedited tree: all match
# ... make the control.py edit ...
bash tools/lowprio.sh $PY -m designlab.stage2.regress run --tier full --scratch logs/regress-after --record stage2/regress/regress.json
$PY -m designlab.stage2.regress check                                               # exit 0
$PY -m pytest -p no:cacheprovider tests/test_stage2_regress.py tests/test_stage1.py tests/test_stage1_panels.py tests/test_stage1_speedfit.py
git status --porcelain
```
Done when the full tier matches every trace and every episode record before and after the edit, the record is
committed, and the Stage 1 test files above pass (some tests pin source-module lists, not hashes; if any test
pins a hash that your edit moves, stop and report it instead of editing that test). Say in your report how many
episodes ran and the wall time.

Note for the lead (put it in your report): the record must be made again at the freeze commit, after the Jetson
body's merge, with the same `run --tier full --record` command; until then `test_regress_record_is_current`
fails, by design.

## Commit

Two commits on your own branch in your worktree: (1) `regress.py`, its tests and a record of the unedited tree if
you wish; (2) the `control.py` edit with the new record. Messages end with the attribution lines your session's
system reminder gives, exactly; name no model anywhere else. Do not merge, do not push unless the lead says so.
