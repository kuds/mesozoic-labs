# T. rex stance evidence (2026-10-12 to 2026-10-17)

Evidence for [TREX_STANCE_SYMMETRY_AND_FOOT_2026_10.md](../TREX_STANCE_SYMMETRY_AND_FOOT_2026_10.md),
frozen with it on 2026-10-17. These are copies of the files that decided or
document each verdict, taken from the session's scratch tree before it is lost.
The scripts are frozen copies as they ran: research tools, imported by nothing,
outside the library, its tests and CI (CI's ruff, mypy and pytest cover
`environments/` only). Checkpoints, rollouts, traces and per-episode judge files
are not here; where they are, and their sha256, is in the note's §14 and in
`r10_screen/fetch_record.json`.

## What changed in the copies

[SOURCES.txt](SOURCES.txt) lists every file with the scratch file it came from,
that file's sha256, and what the copy changed. Three kinds of change, nothing
else:

- **prefix:** the absolute path of the session's scratch directory is replaced
  by `scratchpad` (and the session directory above it by `session`). Paths in
  these files such as `scratchpad/foot/...` or `foot/...` name that tree; only
  the files listed here survive. Scripts with a `SCRATCH` constant
  (`judge_r10.py`, `score_r10.py`) now resolve it relative to the working
  directory.
- **whitespace:** trailing whitespace removed and a final newline added, as the
  repository's pre-commit hooks require; JSON content is unchanged.
- **trailer:** `symmetry/frozen_params.json` held the scratch clone's two
  commit-trailer lines (`repo.commit_trailer`); they are replaced by a
  placeholder. Its sha256 is therefore not the `58b2c8fb…` that READY-v1
  recorded before the symmetry screen trained (the original's sha256 is in
  SOURCES.txt). `judge_r10.py` and `score_r10.py` assert that original value,
  and `verify_final.py` and `recount_final.py` the value READY-v1 passes them,
  so against this copy they stop with a sha256 mismatch; every threshold they
  read is unchanged (`decide_final.py` and `test_decide_final.py` run on it as
  it is).

[SHA256SUMS](SHA256SUMS) pins every file here but this README
(`sha256sum -c SHA256SUMS` in this directory).

## Contents

**`run_a/`: run A and the 2026-10-12 foot and CoP investigation (§3).**

- `report.md`: the corrected report (mechanism, reward, per-species anatomy,
  probes, recommendation, open questions, 11 corrections). It cites further
  scratch files under `foot/`; the ones that back the note are copied beside
  it.
- `stance_gate_report_20261011_165957.txt`: run A's `stance_quality/v2` report
  from Drive (1/40 clean).
- `ablate_table.txt` (held-pose ablation of run A's mean action),
  `thresholds.txt` (ramped single-ankle thresholds per model),
  `an_moment.txt` (the verifier's ankle-moment balance).
- `reward_pairs.txt` (per-term paired returns), `observability.txt` (CoP fits
  from observation channels).
- `noise_saturation.txt`, `noise_s48.txt`, `noise_s56.txt`: the statue and the
  trained policies under the policies' own noise.
- `probe0_analysis.txt`, `probe1_results.txt`, `probe2_results.txt`: the three
  1M probes on `main` `8565d299`. Read probe 2 with the report's Corrections 8
  (its episode-level p-values are not evidence).
- `anatomy_summary.txt`, `statue_per_species.txt`: statue panels per species.

**`noise_rank/`: the offline noise-ranking check (§4.1).** `summary.txt` is the
corrected verdict (it supersedes a first version, not copied);
`pass_corrected.txt` the corrected pass table; `verify_summary.txt` the
independent re-derivation that withdrew the "does not reward falling early"
claim.

**`cap_screens/`: the screens before symmetry and the roll diagnosis (§4).**

- `screen_1m_analysis.txt`, `screen_3m_noise_analysis.txt`,
  `screen_3m_cop_authority_analysis.txt`, `screen_6m_authority_analysis.txt`,
  `screen_hip_roll_analysis.txt`: each screen's analysis with its
  predeclared criteria. In `screen_3m_noise_analysis.txt` the matched-sigma
  noise comparison ("about 8x ... 14x") and the proposed 12M entry rule built
  on it are withdrawn; `control_s{42,52,60}_uniform_sigma016.json` are the
  control under the same uniform σ 0.16 (9 + 11 + 17 = 37 of 60 falls).
- `roll_diag_analysis.txt`: the lateral pad-roll diagnosis.
  `roll_verify_reward_terms.json` is the independent check's per-term returns;
  it corrects the analysis's "the reward does not price the lean" and its
  option (b) range (net spread ±32).

**`symmetry/`: the symmetry screen, arms L and C, verdict O9 (§5).**

- `frozen_params.json`: every frozen parameter, threshold, seed and checkpoint
  (redacted as above).
- `interim.json`, `verdict.json`: the 3M interim and the 6M verdict;
  `J1_recount.log`, `J2_recount.log`: the independent recount (120 and 537
  quantities, no disagreement); `rerun_ledger.json`: the two infrastructure
  re-runs.
- `verdict_scripts/`: `verify_final.py`, `recount_final.py`,
  `decide_final.py`, `arbiter_final.py` and `test_decide_final.py`,
  byte-identical to the versions whose sha256 `frozen_params.json` records.
- `mirror_ratio.json` (why the KL-form mirror loss, not augmentation),
  `lean_calibration.json` (the lateral lean term's weight and width),
  `ema_credit.json` (PPO's credit for an EMA reward term),
  `groundwork_verify_summary.json` (independent check of the mirror map,
  equivariance, π_sym and lean calibration, with its discrepancies list).

**`r10_screen/`: the r10 foot and arm L screen (§6).**

- `criteria.txt`: K1-K4, written before training, with the dated tie-break
  clarification. `colab_steps.txt`: how the three Colab runs were set up.
- `score.txt`, `score.json`: the scored verdict beside r9 arm L, the best
  stance by the fixed rule and the report-only block. Two reading notes: the
  best-stance listing prints joint-space falls only, though the rule sums
  joint-space and own-sigma falls (the pick is the same); and `score.json`'s
  r9 `foot_cop_fh` fields count unmeasured episodes as over the rail and are
  unused (r9's K2 uses its pad counts).
- `judge_r10.py` (judges one checkpoint, binding plant, task, seed and
  commit), `score_r10.py` (computes K1-K4), `test_score_r10.py` (33 boundary
  cases), `validate_r9.json` (the judge reproducing an r9 judge file before
  use).
- `verify_recompute.py`, `verify_recompute.json`: the independent
  recomputation of every cell from the per-episode rows.
- `report_only_summary.json`: the notebook's end-of-stage certificates and the
  6M-12M in-training statistics. `fetch_record.json`: Drive file ids, sizes,
  md5 and sha256 of the 36 files fetched.

**`diagnosis/`: why seeds 52 and 60 stood on their toes (§7).**
`findings_v3.txt` is the current version. `findings_v1_superseded.txt` is kept
because its figures reached the public #623 comment: its noise hypothesis (H1)
is withdrawn, its edge "1.06" and its slopes -192 / -196 are superseded (v3,
section 4 and its header).

**`pilot/`: the walking pilot's pre-registration (§9; not trained).**

- `criteria.txt`: the fixed criteria, byte-identical (sha256 `1cdd495c…`).
- `pilot_score.py` (applies the rule), `pilot_judge.py` (runs the probe and
  binds the run, including the parent's sha256), `pilot_parent.py` (the
  screen's parent rule), `gait_probe_pilot.py` (the adapter that runs the
  frozen [gait probe](../gait_2026_09/gait_probe.py) on today's API),
  `make_gait_cases.py` and `fixtures/gait_cases.json` (the probe-computed hop
  and walk cases the rule is tested against), `test_pilot_score.py`,
  `test_pilot_parent.py`.
- `duty_cut_summary.json`: the cost table behind the 0.80 duty guard.
  `pilot_parent_on_screen_score.log.txt`: `pilot_parent.py` on the screen's
  score, naming r10 arm L seed 42 6M, the rule's choice that the maintainer
  departed from.
- `pilot_video.py` and the Colab steps stay with the session; the branch
  `claude/pilot-walk-r9` (`eaeef50f`) holds the parent checkpoint and its
  `SOURCE.txt`.

**`telemetry/`.** `posture_drift_replay.json`: the `PostureDrift` default
replayed on the r10 runs' `gate_progress.npz`, on the 20:20 snapshot and on the
full series (two in a row against three).

## How to re-check each verdict

Commands as run on 2026-10-17 from a copy of this directory (Python 3.13;
`PYTHONDONTWRITEBYTECODE=1`). None needs the repository except where stated.

| Check | Command | Result |
|---|---|---|
| The symmetry screen's outcome from its stored per-arm records | in `symmetry/verdict_scripts/`: `python3 -c "import json, decide_final as D; P = json.load(open('../frozen_params.json')); V = json.load(open('../verdict.json')); print(D.outcome(V['arms']['L'], V['arms']['C'], P))"` | `O9`, no PR arm, no 12M arms |
| The symmetry decision table | `python3 test_decide_final.py ../frozen_params.json` | `"ok": true`, every combination mapped |
| The r10 verdict from the per-episode rows | `python3 r10_screen/verify_recompute.py <judge dir> out.json` on the nine judge JSONs (not copied; regenerate them with `judge_r10.py` from the Drive checkpoints in `fetch_record.json` on a clone of `claude/screen-trex-r10-arm-l` at `026eb833`) | identical to `verify_recompute.json`: K1-K4 all FAIL |
| The r10 score | `python3 r10_screen/score_r10.py <judge dir> --json score.json`, with the r9 judge files and the original `frozen_params.json` under `scratchpad/foot/...` | identical to `score.json` apart from the judge directory's path |
| The r10 scorer at every bar | `python3 r10_screen/test_score_r10.py` (reads the original `frozen_params.json` at `scratchpad/foot/symmetry/design/`) | 33/33 |
| The pilot rule | in `pilot/`: `python3 test_pilot_score.py` | 203/203 (probe recomputation skipped) |
| The pilot rule, with every fixture row recomputed by the frozen probe | `python3 test_pilot_score.py <tree>` with `<tree>` a checkout of `claude/pilot-walk-r9` at `eaeef50f` and numpy, MuJoCo and the repository's dependencies installed | 254/254 |
| The parent rule and the judge's bindings | `python3 test_pilot_parent.py` | 40/40 |
| The files themselves | `sha256sum -c SHA256SUMS` | all 70 files OK |

`test_pilot_score.py` prints sample "Walking pilot score" blocks while it runs;
they score synthetic panels, not the pilot.

## Lint

Under the repository's ruff configuration (ruff 0.16.8), `ruff check` passes on
14 of the 16 scripts here. It reports 11 findings in
`r10_screen/verify_recompute.py` (a verifier's quick script: a multi-module
import, an unused `os`, statements joined by `;` and `:`) and 2 in
`symmetry/verdict_scripts/recount_final.py` (an unused `math`, a variable named
`I`). `ruff format --check` would reformat all 16. They are left as they ran,
so that each stays the file that decided its verdict (and the verdict scripts
keep the sha256 the screen recorded).
