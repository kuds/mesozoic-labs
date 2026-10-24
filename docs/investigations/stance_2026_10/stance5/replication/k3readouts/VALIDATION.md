# k3readout5r.py: validation against stance5's skeptic-checked K3 values (report only)

Written 2026-10-23 02:16 UTC (review-fix round; replaces the 01:25 UTC version). REPORT ONLY: these readouts enter no
verdict, decision line or tie-break (`R/readouts_addendum_k3.txt` governs; `R` = `SP/foot/stance5/replication`, `SP` =
the scratchpad). The addendum says "the tool and its validation ... are recorded below when built", but the addendum is
pinned by the tool (sha256 7a79e9f6..., any other addendum is refused) and was not edited, so the record is this file
and `README.md`.

## Result

**Every checked row matches: 223 checked rows, 0 mismatches.** 20 further rows are information only: values k3diag
never rolled (nohp and hpC at s42 6.3M; nohp, hpD and hpC at s52 final), the addendum-window stagger next to k3diag's
other windows, the per-episode max stagger and the judge/own sigma ratios.

| What | File | sha256 |
|---|---|---|
| tool | `R/k3readouts/k3readout5r.py` | d0aed5a17bb0ea82b5496b12421e3010e25a6afcb77ea7abecc4cd99e44517ac |
| runner | `R/k3readouts/k3readout_one5r.sh` | 5e3c66cddaabcfabd231ed503b5b5d4fbed5654cf13adc7e1a8b28d97188f727 |
| comparison script | `R/k3readouts/validate_k3r.py` (numpy only, reads files; run through iso.py) | 282f6a3d0099f64eacc145c2b02bb97346989b1f98a25ee9689acdaf2a7373c3 |

The previous record (tool b69ff098..., runner 7ac36ccc...) is superseded. All five stance5 reference slots were rolled
again with the current tool and runner, through the runner, on 2026-10-23 01:52-02:08 UTC (two at a time):
`out/stance5_LTSB_s42/k3r_6291456.2.json`, `k3r_7340032.1.json`, `k3r_final.1.json`,
`out/stance5_LTSB_s52/k3r_final.1.json` and `out/stance5_LTSB_s60/k3r_final.1.json` (each time the tool printed
"... exists; wrote ... instead"). validate_k3r.py checks the newest output written by the current tool (by
`identity.tool_sha256`, not by file name) and compares every older output of the slot with it readout by readout: the
six b69ff098 outputs (s42 6.3M twice, 7.3M and final, s52 and s60 finals) are identical in every readout they hold.

A first re-validation of s42 7.3M and final in this round (01:48 UTC, logs `..._7935.log` and `..._7944.log`) was cut by
a container restart after two conditions; it wrote no output.

## Key values (current tool)

| Checkpoint | full (judge) | matches_judge / reward | hip_pitch | nohp | hpD | hpC | kick 0.3 | kick 0.2 | kick falls before step 400 | response /rad | own hp sigma | stagger sd mm: 200..L-1 / v2 window (falls in window) | seconds |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| s42 6,291,456 | 0 (0) | yes / yes | 0 (0) | 0 (none) | 0 (0) | 0 (none) | 0 (0) | 0 (0) | 0 / 0 | -1.037 (-1.037) | 0.0204 (0.0204; 0.020-0.022) | 10.67 / 10.67 (10.7) (0) | 344.8 |
| s42 7,340,032 | 9 (9) | yes / yes | 8 (8) | 0 (0) | 5 (5) | 0 (0) | 20 (20) | 0 (0) | 0 / 0 | -0.514 (-0.514) | 0.0187 (0.0187) | 61.68 / 47.98 (48.0) (9) | 308.0 |
| s42 final | 13 (13) | yes / yes | 15 (15) | 2 (2) | 15 (15) | 1 (1) | 20 (20) | 3 (3) | 0 / 0 | -0.342 (-0.342) | 0.0148 (0.0148) | 61.36 / 44.08 (44.1) (13) | 275.6 |
| s52 final | 0 (0) | yes / yes | 1 (1) | 0 (none) | 0 (none) | 0 (none) | 20 (20) | 0 (0) | 0 / 0 | -0.587 (-0.587) | 0.0139 (0.0139) | 15.09 / 15.09 (15.1) (0) | 313.7 |
| s60 final | 3 (3) | yes / yes | 2 (2) | 0 (0) | 0 (0) | 0 (0) | 20 (20) | 0 (0) | 0 / 0 | -0.448 (-0.448) | 0.0143 (0.0143) | 20.87 / 20.75 (20.8) (2) | 300.3 |

Reference in brackets: the judge file (full); verify_1 p*/q*.jsonl (hip_pitch, nohp, hpD, hpC at 7.3M, final and
s60 final); k3diag ablation_matrix.csv and rollouts/npz (hip_pitch at 6.3M, s52 and s60); verify_2 v2_probe.json and
probe npz (hpD at 6.3M, the response, own sigma, the v2-window stagger); pulse.csv and rollouts/npz (kicks, and the
kick falls before step 400 from their per-episode lengths); report.md sec. 2 and 4 (own sigma ranges). "none" = k3diag
never rolled it. Fall counts were also compared per episode (lengths and termination reasons) wherever the reference
kept them, and the response and v2-window stagger were recomputed from verify_2's probe arrays at full precision
(bit-identical medians). "full" reproduces every judge file per episode in seed, length and termination reason, and in
reward to 0.0 (the tolerance is 1e-6). "falls in window" (episodes with 200 < L < 1000) was checked against the judge
files' own per-episode lengths: at s60 final one of the three falls ends at step 143, before the window, so 2.

Further checks: own hip-pitch sigma at 6.3M is 0.0205 (s52) and 0.0215 (s60) by the tool's `--check-only` load, so
all three band seeds sit in report.md's 0.020-0.022; the judge/own hip-pitch sigma ratio over the five slots runs
6.9-10.2, report.md's range. Per-episode max of abs(stagger) over 200..L-1 (largest episode): 48.3, 286.3, 205.4, 64.7 and
98.1 mm (at 7.3M and final the largest come from falls).

## Review findings and what was done

Two reviews (fidelity, isolation) found nothing blocking and 13 minor points. Each was checked against the code before
acting; the tests are under `fix_r2/` (driver scripts `tests.sh`, `selftests_again.sh`, `e2e.sh`; the earlier
`fix_r1/` holds the same tests from before the restart) and are rows of the table below (run "fix").

| # | Finding | Real? | Done | Evidence |
|---|---|---|---|---|
| 1 | summary line and headline `sd_mm` include the topple steps of falls | yes (by the addendum's window) | the window is kept (the addendum's readout, like for like with the references); the summary line now prints `stagger_sd=`, `stagger_sd_v2_window=` and `stagger_falls_in_window=`; each stagger block has `falls_in_window` and `sd_mm_note`; README explains which figure compares with k3diag | 5 summary-line rows, 5 falls_in_window rows (vs judge lengths) |
| 2 | `valid` ignored the per-episode reward difference | yes (v2_probe required < 1e-6) | `reward_matches_judge` (max abs diff < 1e-6) recorded and required by `valid` | reward rows (all 0.0); tamper test: reward +1e-3 on episode 0, smoke 1: matches_judge true, reward_matches_judge false, valid false, exit 4 |
| 3 | runner refused a missing judge file itself, writing no refused JSON | yes | the runner no longer pre-refuses: it warns and lets the tool refuse, which writes `out/<RUN>/refused_<steps>.json` (exit 3); the runner logs a `refused` line and says so on stderr | runner copy (only `K3R_TEST_ROOT` changed): exit 3, refused JSON written, stderr message, runner.log line |
| 4 | a fall before the kick would count as a kick fall | yes (robustness only; matches k3pulse) | k3pulse's count kept; `falls_before_kick` (L <= 400) and its seeds recorded beside it and printed in the summary | 10 rows vs k3roll npz lengths: 0 everywhere |
| 5 | LTSBr sigma re-executes judge main's sidecar statements | yes, by necessity | no code change; README states that the exact equality with the judge file's sigma_vector is the binding guarantee, and that 70 / 80 share one vector | LTSBr e2e rows (sidecar 89778b3f..., equality true) |
| 6 | runner hangs when `--arm` is last | yes | `--arm` needs a value (exit 2 with usage) | six argument tests under `timeout 5`: exit 2, no hang |
| 7 | runner silent on a crash | yes | runner prints `k3readout FAILED rc=N, see <log>` on stderr (and messages for exit 3 and 4) | runner copy with a missing TOOL: exit 1, message |
| 8 | the judge file's recorded `judge_sha256` was not checked | yes | refused unless it is `sha256:` + the pinned judge module's sha256 | judge-file copy with `sha256:06d39f7d...`: exit 3, refused JSON; every real judge file passes (5 rows) |
| 9 | a slot outside the addendum gives a plain valid output | yes | `addendum_slot` (`readout` / `reference` / `no`) at top level and in the summary line; a NOTE for `no` | s70 4194304 copy: NOTE, `addendum_slot=no`; the five reference slots say `reference` |
| 10 | `K3R_TEST_ROOT` not confined | yes | refused (exit 2, nothing read or written) unless it resolves inside `R/k3readouts`, excluding the directory itself, `out/` and `logs/` | 8 tests (/tmp, k3readouts, out, out/<RUN>, logs, R, `..` path, symlink to out): exit 2, nothing written |
| 11 | exit 2 meant both bad arguments and an invalid output | yes | invalid output now exits 4; 2 is bad arguments only; README and the tool's docstring say to select outputs by `valid`, not by name | tamper test exit 4; e2e test-root outputs exit 4 |
| 12 | the addendum's pin vs "recorded below when built" | yes | kept as is: the record lives in README.md and this file; the addendum is unchanged (7a79e9f6...); README says re-pin deliberately if the addendum is ever amended | pins before / after |
| 13 | per-call log name could collide and be truncated | yes (in principle) | the log name carries the runner's PID and is opened under `noclobber` | log names `..._<utc>_<pid>.log` in runner.log |

The two runner tests (3, 7) used byte copies of the runner with one line changed; their per-call logs and
`start` / `end` / `refused` lines are in `logs/` and `logs/runner.log` (the refused line names the
`fix_r2/r_missing` test root; the crash line is LTSB s42 final rc=1 at 01:58:10 UTC).

## Stagger windows (why two sd figures)

The addendum's readout is the sd of `info["foot_stagger_m"]` over steps 200..L-1 pooled per condition. For a falling
episode that window includes the topple itself, so at s42 7.3M and final it reads 61.7 and 61.4 mm and partly measures
the falls (`falls_in_window` 9 and 13). k3diag's figures use other windows: verify_2 (v2_probe_ana.py) stops falls at
L-151 (48.0 and 44.1 mm, reproduced to full precision), and k3gain (policy_gain.csv) gives 48.6 and 47.7 mm; report.md
check 3 says "48-50 mm (the exact figure depends on the window)". With no falls in the window (s42 6.3M, s52 final) the
two windows coincide. Every output and summary line carries both (`sd_mm`, `sd_mm_v2_window`) and `falls_in_window`.

## The LTSBr path (replication mode) end to end

No replication 6.3M or final judge file exists yet, so `--arm LTSBr` was run on byte copies of two replication judge
files (s70 at 4,194,304 and s80 at 3,145,728) under a `K3R_TEST_ROOT` (outputs there are invalid by design, exit 4):
first with tool b69ff098 (`selftest/t8_ltsbr_e2e`), again with the current tool (`fix_r2/e2e`, 2026-10-23 02:09 UTC).
Both times: the sigma came through judge_stance5r.py's sidecar path (sha256 89778b3f... checked, source_sha256 the
pinned frozen_params) and equals the judge file's sigma_vector exactly; "full" reproduced the judge file per episode
(matches_judge true, reward difference 0.0, reward_matches_judge true); the two tools' readouts are identical. Both
slots are outside the addendum, so the current outputs say `addendum_slot=no`. Seconds: 336.1 (s70) and 322.2 (s80).

Refusal self-tests t1-t7 (fixtures unchanged under `selftest/`) were re-run with the current tool and each exited 3
with its original reason and a refused JSON: judge file missing; model sha256 differs; sigma differs (LTSB and LTSBr);
judge file names another sidecar sha256; judge file judged at another clone HEAD; judge file of another mode. A dirty
clone was not simulated (the clones are read-only); that check is the judge's own `git_state` compared with the pinned
HEAD, before and after the rolls (`clone_clean_after` true in every output).

## Timings (seconds per checkpoint, all seven conditions of 20 episodes)

Current tool: 344.8 (s42 6.3M), 308.0 (s42 7.3M), 275.6 (s42 final), 313.7 (s52 final), 300.3 (s60 final); LTSBr
end-to-end 336.1 and 322.2. Previous tool: 350.9 and 333.1 (s42 6.3M), 305.5, 268.9, 315.4, 269.2; 293.7 and 331.1.
Each condition takes 22-55 s (kick +/-0.3 runs short when every episode falls). Two runs at a time, alongside the two
judge / panel queue workers, on the 4-CPU box.

## Pins and clones

Before and after this round: judge_stance5r.py 16d39f7d..., score_stance5r.py 55c37ba7..., criteria.txt a38536de...,
sigma_s70_s80.json 89778b3f..., judge_stance5.py d9286724..., readouts_addendum_k3.txt 7a79e9f6..., queue5r.sh
0547d452...: unchanged. `SP/fix/scratch_copband` clean at cf80ecb1 and `SP/fix/screen_s4_LTS` clean at c1e146a8
before and after.

## Full table (`validate_k3r.py`, run through iso.py, 2026-10-23 02:15 UTC)

| run | steps | readout | expected | got | match | source |
|---|---|---|---|---|---|---|
| s42 | 6291456 | checked output k3r_6291456.2.json written by the current tool | d0aed5a1 | d0aed5a1 | yes | identity.tool_sha256 |
| s42 | 6291456 | k3r_6291456.json (tool b69ff098) readouts == k3r_6291456.2.json | identical | identical | yes | determinism |
| s42 | 6291456 | k3r_6291456.1.json (tool b69ff098) readouts == k3r_6291456.2.json | identical | identical | yes | determinism |
| s42 | 6291456 | full falls | 0 | 0 | yes | judge file |
| s42 | 6291456 | full matches_judge (seed, length, termination per episode) | true | true | yes | judge file episodes_detail |
| s42 | 6291456 | full reward max abs diff vs judge | 0 | 0 | yes | judge file episodes_detail |
| s42 | 6291456 | full reward_matches_judge (< 1e-6, part of valid) | true | true | yes | tool |
| s42 | 6291456 | valid | true | true | yes | tool |
| s42 | 6291456 | addendum_slot (stance5 reference slot) | reference | reference | yes | addendum WHERE AND WHEN |
| s42 | 6291456 | judge file judge_sha256 == pinned judge_stance5.py | sha256:d9286724 | sha256:d9286724 | yes | judge file |
| s42 | 6291456 | stagger falls_in_window (full; 200 < L < 1000) | 0 | 0 | yes | judge file episodes_detail lengths |
| s42 | 6291456 | summary line carries both stagger figures and falls_in_window | stagger_sd=10.67mm stagger_sd_v2_window=10.67mm stagger_falls_in_window=0 | stagger_sd=10.67mm stagger_sd_v2_window=10.67mm stagger_falls_in_window=0 | yes | tool summary |
| s42 | 6291456 | hip_pitch per-episode lengths + terms | identical | identical | yes | rollouts/npz (k3roll hip_pitch) |
| s42 | 6291456 | hip_pitch falls | 0 (ablation_matrix.csv) | 0 | yes | ablation_matrix.csv |
| s42 | 6291456 | nohp falls | no reference (not rolled by k3diag) | 0 | info | - |
| s42 | 6291456 | hpD per-episode lengths | identical | identical | yes | verify_2/probe npz (v2_probe hpD) |
| s42 | 6291456 | hpD falls | 0 (v2_probe.json) | 0 | yes | v2_probe.json |
| s42 | 6291456 | hpC falls | no reference (not rolled by k3diag) | 0 | info | - |
| s42 | 6291456 | kick +/-0.3 falls | 0 (pulse.csv) | 0 | yes | pulse.csv |
| s42 | 6291456 | pulse_anti per-episode lengths + terms | identical | identical | yes | rollouts/npz (k3roll pulse_anti) |
| s42 | 6291456 | kick +/-0.3 falls_before_kick (L <= 400) | 0 | 0 | yes | rollouts/npz (k3roll pulse_anti) lengths |
| s42 | 6291456 | kick +/-0.2 falls | 0 (pulse.csv) | 0 | yes | pulse.csv |
| s42 | 6291456 | pulse_anti20 per-episode lengths + terms | identical | identical | yes | rollouts/npz (k3roll pulse_anti20) |
| s42 | 6291456 | kick +/-0.2 falls_before_kick (L <= 400) | 0 | 0 | yes | rollouts/npz (k3roll pulse_anti20) lengths |
| s42 | 6291456 | direct response Jq median (3 dp, per rad) | -1.037 | -1.037 | yes | v2_probe.json |
| s42 | 6291456 | stagger sd, v2 window (mm, 0.1) | 10.7 | 10.67 | yes | v2_probe.json |
| s42 | 6291456 | own hip-pitch sigma (4 dp) | 0.0204 | 0.0204 | yes | v2_probe.json |
| s42 | 6291456 | direct response Jq median, full precision | -1.0368301272392273 | -1.0368301272392273 | yes | verify_2/probe npz, v2_probe_ana reduction |
| s42 | 6291456 | stagger sd v2 window, full precision (mm) | 10.671913 | 10.67 | yes | verify_2/probe npz |
| s42 | 6291456 | hip-pitch r-l angle sd, v2 window (mrad) | 19.711781 | 19.71 | yes | verify_2/probe npz |
| s42 | 6291456 | full per-episode lengths vs v2 probe | identical | identical | yes | verify_2/probe npz |
| s42 | 6291456 | own hip-pitch sigma in the band's 6.3M range | 0.020-0.022 | 0.0204 | yes | report.md sec. 2 line 266 |
| s42 | 6291456 | judge sigma / own sigma, hip pitch r, l (report only) | - | 7.06, 6.76 | info | tool |
| s42 | 6291456 | stagger sd, addendum window 200..L-1 (mm; report only) | k3gain window 9.4 | 10.67 | info | policy_gain.csv (other window) |
| s42 | 6291456 | stagger: per-episode max of abs(stagger) over 200..L-1, largest episode (mm; report only) | - | 48.3 | info | tool |
| s42 | 7340032 | checked output k3r_7340032.1.json written by the current tool | d0aed5a1 | d0aed5a1 | yes | identity.tool_sha256 |
| s42 | 7340032 | k3r_7340032.json (tool b69ff098) readouts == k3r_7340032.1.json | identical | identical | yes | determinism |
| s42 | 7340032 | full falls | 9 | 9 | yes | judge file |
| s42 | 7340032 | full matches_judge (seed, length, termination per episode) | true | true | yes | judge file episodes_detail |
| s42 | 7340032 | full reward max abs diff vs judge | 0 | 0 | yes | judge file episodes_detail |
| s42 | 7340032 | full reward_matches_judge (< 1e-6, part of valid) | true | true | yes | tool |
| s42 | 7340032 | full per-episode lengths + terms | identical | identical | yes | verify_1/p2.jsonl |
| s42 | 7340032 | valid | true | true | yes | tool |
| s42 | 7340032 | addendum_slot (stance5 reference slot) | reference | reference | yes | addendum WHERE AND WHEN |
| s42 | 7340032 | judge file judge_sha256 == pinned judge_stance5.py | sha256:d9286724 | sha256:d9286724 | yes | judge file |
| s42 | 7340032 | stagger falls_in_window (full; 200 < L < 1000) | 9 | 9 | yes | judge file episodes_detail lengths |
| s42 | 7340032 | summary line carries both stagger figures and falls_in_window | stagger_sd=61.68mm stagger_sd_v2_window=47.98mm stagger_falls_in_window=9 | stagger_sd=61.68mm stagger_sd_v2_window=47.98mm stagger_falls_in_window=9 | yes | tool summary |
| s42 | 7340032 | hip_pitch per-episode lengths + terms | identical | identical | yes | verify_1/p2.jsonl |
| s42 | 7340032 | hip_pitch per-episode lengths + terms | identical | identical | yes | rollouts/npz (k3roll hip_pitch) |
| s42 | 7340032 | hip_pitch falls | 8 (verify_1/p2.jsonl) / 8 (ablation_matrix.csv) | 8 | yes | verify_1/p2.jsonl; ablation_matrix.csv |
| s42 | 7340032 | nohp per-episode lengths + terms | identical | identical | yes | verify_1/q1.jsonl |
| s42 | 7340032 | nohp falls | 0 (verify_1/q1.jsonl) | 0 | yes | verify_1/q1.jsonl |
| s42 | 7340032 | hpD per-episode lengths + terms | identical | identical | yes | verify_1/p2.jsonl |
| s42 | 7340032 | hpD per-episode lengths | identical | identical | yes | verify_2/probe npz (v2_probe hpD) |
| s42 | 7340032 | hpD falls | 5 (verify_1/p2.jsonl) / 5 (v2_probe.json) | 5 | yes | verify_1/p2.jsonl; v2_probe.json |
| s42 | 7340032 | hpC per-episode lengths + terms | identical | identical | yes | verify_1/p2.jsonl |
| s42 | 7340032 | hpC falls | 0 (verify_1/p2.jsonl) | 0 | yes | verify_1/p2.jsonl |
| s42 | 7340032 | kick +/-0.3 falls | 20 (pulse.csv) | 20 | yes | pulse.csv |
| s42 | 7340032 | pulse_anti per-episode lengths + terms | identical | identical | yes | rollouts/npz (k3roll pulse_anti) |
| s42 | 7340032 | kick +/-0.3 falls_before_kick (L <= 400) | 0 | 0 | yes | rollouts/npz (k3roll pulse_anti) lengths |
| s42 | 7340032 | kick +/-0.2 falls | 0 (pulse.csv) | 0 | yes | pulse.csv |
| s42 | 7340032 | pulse_anti20 per-episode lengths + terms | identical | identical | yes | rollouts/npz (k3roll pulse_anti20) |
| s42 | 7340032 | kick +/-0.2 falls_before_kick (L <= 400) | 0 | 0 | yes | rollouts/npz (k3roll pulse_anti20) lengths |
| s42 | 7340032 | direct response Jq median (3 dp, per rad) | -0.514 | -0.514 | yes | v2_probe.json |
| s42 | 7340032 | stagger sd, v2 window (mm, 0.1) | 48.0 | 47.98 | yes | v2_probe.json |
| s42 | 7340032 | own hip-pitch sigma (4 dp) | 0.0187 | 0.0187 | yes | v2_probe.json |
| s42 | 7340032 | direct response Jq median, full precision | -0.5136185586452484 | -0.5136185586452484 | yes | verify_2/probe npz, v2_probe_ana reduction |
| s42 | 7340032 | stagger sd v2 window, full precision (mm) | 47.975615 | 47.98 | yes | verify_2/probe npz |
| s42 | 7340032 | hip-pitch r-l angle sd, v2 window (mrad) | 55.411621 | 55.41 | yes | verify_2/probe npz |
| s42 | 7340032 | full per-episode lengths vs v2 probe | identical | identical | yes | verify_2/probe npz |
| s42 | 7340032 | judge sigma / own sigma, hip pitch r, l (report only) | - | 7.73, 7.39 | info | tool |
| s42 | 7340032 | stagger sd, addendum window 200..L-1 (mm; report only) | k3gain window 48.6 | 61.68 | info | policy_gain.csv (other window) |
| s42 | 7340032 | stagger: per-episode max of abs(stagger) over 200..L-1, largest episode (mm; report only) | - | 286.3 | info | tool |
| s42 | final | checked output k3r_final.1.json written by the current tool | d0aed5a1 | d0aed5a1 | yes | identity.tool_sha256 |
| s42 | final | k3r_final.json (tool b69ff098) readouts == k3r_final.1.json | identical | identical | yes | determinism |
| s42 | final | full falls | 13 | 13 | yes | judge file |
| s42 | final | full matches_judge (seed, length, termination per episode) | true | true | yes | judge file episodes_detail |
| s42 | final | full reward max abs diff vs judge | 0 | 0 | yes | judge file episodes_detail |
| s42 | final | full reward_matches_judge (< 1e-6, part of valid) | true | true | yes | tool |
| s42 | final | full per-episode lengths + terms | identical | identical | yes | verify_1/p2.jsonl |
| s42 | final | valid | true | true | yes | tool |
| s42 | final | addendum_slot (stance5 reference slot) | reference | reference | yes | addendum WHERE AND WHEN |
| s42 | final | judge file judge_sha256 == pinned judge_stance5.py | sha256:d9286724 | sha256:d9286724 | yes | judge file |
| s42 | final | stagger falls_in_window (full; 200 < L < 1000) | 13 | 13 | yes | judge file episodes_detail lengths |
| s42 | final | summary line carries both stagger figures and falls_in_window | stagger_sd=61.36mm stagger_sd_v2_window=44.08mm stagger_falls_in_window=13 | stagger_sd=61.36mm stagger_sd_v2_window=44.08mm stagger_falls_in_window=13 | yes | tool summary |
| s42 | final | hip_pitch per-episode lengths + terms | identical | identical | yes | verify_1/p2.jsonl |
| s42 | final | hip_pitch per-episode lengths + terms | identical | identical | yes | rollouts/npz (k3roll hip_pitch) |
| s42 | final | hip_pitch falls | 15 (verify_1/p2.jsonl) / 15 (ablation_matrix.csv) | 15 | yes | verify_1/p2.jsonl; ablation_matrix.csv |
| s42 | final | nohp per-episode lengths + terms | identical | identical | yes | verify_1/q2.jsonl |
| s42 | final | nohp falls | 2 (verify_1/q2.jsonl) | 2 | yes | verify_1/q2.jsonl |
| s42 | final | hpD per-episode lengths + terms | identical | identical | yes | verify_1/p2.jsonl |
| s42 | final | hpD per-episode lengths | identical | identical | yes | verify_2/probe npz (v2_probe hpD) |
| s42 | final | hpD falls | 15 (verify_1/p2.jsonl) / 15 (v2_probe.json) | 15 | yes | verify_1/p2.jsonl; v2_probe.json |
| s42 | final | hpC per-episode lengths + terms | identical | identical | yes | verify_1/p2.jsonl |
| s42 | final | hpC falls | 1 (verify_1/p2.jsonl) | 1 | yes | verify_1/p2.jsonl |
| s42 | final | kick +/-0.3 falls | 20 (pulse.csv) | 20 | yes | pulse.csv |
| s42 | final | pulse_anti per-episode lengths + terms | identical | identical | yes | rollouts/npz (k3roll pulse_anti) |
| s42 | final | kick +/-0.3 falls_before_kick (L <= 400) | 0 | 0 | yes | rollouts/npz (k3roll pulse_anti) lengths |
| s42 | final | kick +/-0.2 falls | 3 (pulse.csv) | 3 | yes | pulse.csv |
| s42 | final | pulse_anti20 per-episode lengths + terms | identical | identical | yes | rollouts/npz (k3roll pulse_anti20) |
| s42 | final | kick +/-0.2 falls_before_kick (L <= 400) | 0 | 0 | yes | rollouts/npz (k3roll pulse_anti20) lengths |
| s42 | final | direct response Jq median (3 dp, per rad) | -0.342 | -0.342 | yes | v2_probe.json |
| s42 | final | stagger sd, v2 window (mm, 0.1) | 44.1 | 44.08 | yes | v2_probe.json |
| s42 | final | own hip-pitch sigma (4 dp) | 0.0148 | 0.0148 | yes | v2_probe.json |
| s42 | final | direct response Jq median, full precision | -0.342115581035614 | -0.342115581035614 | yes | verify_2/probe npz, v2_probe_ana reduction |
| s42 | final | stagger sd v2 window, full precision (mm) | 44.082154 | 44.08 | yes | verify_2/probe npz |
| s42 | final | hip-pitch r-l angle sd, v2 window (mrad) | 52.722610 | 52.72 | yes | verify_2/probe npz |
| s42 | final | full per-episode lengths vs v2 probe | identical | identical | yes | verify_2/probe npz |
| s42 | final | own hip-pitch sigma | 0.0148 | 0.0148 | yes | report.md sec. 4 line 313 |
| s42 | final | judge sigma / own sigma, hip pitch r, l (report only) | - | 9.75, 9.32 | info | tool |
| s42 | final | stagger sd, addendum window 200..L-1 (mm; report only) | k3gain window 47.7 | 61.36 | info | policy_gain.csv (other window) |
| s42 | final | stagger: per-episode max of abs(stagger) over 200..L-1, largest episode (mm; report only) | - | 205.4 | info | tool |
| s52 | final | checked output k3r_final.1.json written by the current tool | d0aed5a1 | d0aed5a1 | yes | identity.tool_sha256 |
| s52 | final | k3r_final.json (tool b69ff098) readouts == k3r_final.1.json | identical | identical | yes | determinism |
| s52 | final | full falls | 0 | 0 | yes | judge file |
| s52 | final | full matches_judge (seed, length, termination per episode) | true | true | yes | judge file episodes_detail |
| s52 | final | full reward max abs diff vs judge | 0 | 0 | yes | judge file episodes_detail |
| s52 | final | full reward_matches_judge (< 1e-6, part of valid) | true | true | yes | tool |
| s52 | final | valid | true | true | yes | tool |
| s52 | final | addendum_slot (stance5 reference slot) | reference | reference | yes | addendum WHERE AND WHEN |
| s52 | final | judge file judge_sha256 == pinned judge_stance5.py | sha256:d9286724 | sha256:d9286724 | yes | judge file |
| s52 | final | stagger falls_in_window (full; 200 < L < 1000) | 0 | 0 | yes | judge file episodes_detail lengths |
| s52 | final | summary line carries both stagger figures and falls_in_window | stagger_sd=15.09mm stagger_sd_v2_window=15.09mm stagger_falls_in_window=0 | stagger_sd=15.09mm stagger_sd_v2_window=15.09mm stagger_falls_in_window=0 | yes | tool summary |
| s52 | final | hip_pitch per-episode lengths + terms | identical | identical | yes | rollouts/npz (k3roll hip_pitch) |
| s52 | final | hip_pitch falls | 1 (ablation_matrix.csv) | 1 | yes | ablation_matrix.csv |
| s52 | final | nohp falls | no reference (not rolled by k3diag) | 0 | info | - |
| s52 | final | hpD falls | no reference (not rolled by k3diag) | 0 | info | - |
| s52 | final | hpC falls | no reference (not rolled by k3diag) | 0 | info | - |
| s52 | final | kick +/-0.3 falls | 20 (pulse.csv) | 20 | yes | pulse.csv |
| s52 | final | pulse_anti per-episode lengths + terms | identical | identical | yes | rollouts/npz (k3roll pulse_anti) |
| s52 | final | kick +/-0.3 falls_before_kick (L <= 400) | 0 | 0 | yes | rollouts/npz (k3roll pulse_anti) lengths |
| s52 | final | kick +/-0.2 falls | 0 (pulse.csv) | 0 | yes | pulse.csv |
| s52 | final | pulse_anti20 per-episode lengths + terms | identical | identical | yes | rollouts/npz (k3roll pulse_anti20) |
| s52 | final | kick +/-0.2 falls_before_kick (L <= 400) | 0 | 0 | yes | rollouts/npz (k3roll pulse_anti20) lengths |
| s52 | final | direct response Jq median (3 dp, per rad) | -0.587 | -0.587 | yes | v2_probe.json |
| s52 | final | stagger sd, v2 window (mm, 0.1) | 15.1 | 15.09 | yes | v2_probe.json |
| s52 | final | own hip-pitch sigma (4 dp) | 0.0139 | 0.0139 | yes | v2_probe.json |
| s52 | final | direct response Jq median, full precision | -0.587270975112915 | -0.587270975112915 | yes | verify_2/probe npz, v2_probe_ana reduction |
| s52 | final | stagger sd v2 window, full precision (mm) | 15.089148 | 15.09 | yes | verify_2/probe npz |
| s52 | final | hip-pitch r-l angle sd, v2 window (mrad) | 25.493507 | 25.49 | yes | verify_2/probe npz |
| s52 | final | full per-episode lengths vs v2 probe | identical | identical | yes | verify_2/probe npz |
| s52 | final | own hip-pitch sigma | 0.0139 | 0.0139 | yes | report.md sec. 4 line 313 |
| s52 | final | judge sigma / own sigma, hip pitch r, l (report only) | - | 10.16, 10.17 | info | tool |
| s52 | final | stagger sd, addendum window 200..L-1 (mm; report only) | k3gain window 14.5 | 15.09 | info | policy_gain.csv (other window) |
| s52 | final | stagger: per-episode max of abs(stagger) over 200..L-1, largest episode (mm; report only) | - | 64.7 | info | tool |
| s60 | final | checked output k3r_final.1.json written by the current tool | d0aed5a1 | d0aed5a1 | yes | identity.tool_sha256 |
| s60 | final | k3r_final.json (tool b69ff098) readouts == k3r_final.1.json | identical | identical | yes | determinism |
| s60 | final | full falls | 3 | 3 | yes | judge file |
| s60 | final | full matches_judge (seed, length, termination per episode) | true | true | yes | judge file episodes_detail |
| s60 | final | full reward max abs diff vs judge | 0 | 0 | yes | judge file episodes_detail |
| s60 | final | full reward_matches_judge (< 1e-6, part of valid) | true | true | yes | tool |
| s60 | final | full per-episode lengths + terms | identical | identical | yes | verify_1/p3.jsonl |
| s60 | final | valid | true | true | yes | tool |
| s60 | final | addendum_slot (stance5 reference slot) | reference | reference | yes | addendum WHERE AND WHEN |
| s60 | final | judge file judge_sha256 == pinned judge_stance5.py | sha256:d9286724 | sha256:d9286724 | yes | judge file |
| s60 | final | stagger falls_in_window (full; 200 < L < 1000) | 2 | 2 | yes | judge file episodes_detail lengths |
| s60 | final | summary line carries both stagger figures and falls_in_window | stagger_sd=20.87mm stagger_sd_v2_window=20.75mm stagger_falls_in_window=2 | stagger_sd=20.87mm stagger_sd_v2_window=20.75mm stagger_falls_in_window=2 | yes | tool summary |
| s60 | final | hip_pitch per-episode lengths + terms | identical | identical | yes | rollouts/npz (k3roll hip_pitch) |
| s60 | final | hip_pitch falls | 2 (ablation_matrix.csv) | 2 | yes | ablation_matrix.csv |
| s60 | final | nohp per-episode lengths + terms | identical | identical | yes | verify_1/q4.jsonl |
| s60 | final | nohp falls | 0 (verify_1/q4.jsonl) | 0 | yes | verify_1/q4.jsonl |
| s60 | final | hpD per-episode lengths + terms | identical | identical | yes | verify_1/p3.jsonl |
| s60 | final | hpD falls | 0 (verify_1/p3.jsonl) | 0 | yes | verify_1/p3.jsonl |
| s60 | final | hpC per-episode lengths + terms | identical | identical | yes | verify_1/p3.jsonl |
| s60 | final | hpC falls | 0 (verify_1/p3.jsonl) | 0 | yes | verify_1/p3.jsonl |
| s60 | final | kick +/-0.3 falls | 20 (pulse.csv) | 20 | yes | pulse.csv |
| s60 | final | pulse_anti per-episode lengths + terms | identical | identical | yes | rollouts/npz (k3roll pulse_anti) |
| s60 | final | kick +/-0.3 falls_before_kick (L <= 400) | 0 | 0 | yes | rollouts/npz (k3roll pulse_anti) lengths |
| s60 | final | kick +/-0.2 falls | 0 (pulse.csv) | 0 | yes | pulse.csv |
| s60 | final | pulse_anti20 per-episode lengths + terms | identical | identical | yes | rollouts/npz (k3roll pulse_anti20) |
| s60 | final | kick +/-0.2 falls_before_kick (L <= 400) | 0 | 0 | yes | rollouts/npz (k3roll pulse_anti20) lengths |
| s60 | final | direct response Jq median (3 dp, per rad) | -0.448 | -0.448 | yes | v2_probe.json |
| s60 | final | stagger sd, v2 window (mm, 0.1) | 20.8 | 20.75 | yes | v2_probe.json |
| s60 | final | own hip-pitch sigma (4 dp) | 0.0143 | 0.0143 | yes | v2_probe.json |
| s60 | final | direct response Jq median, full precision | -0.44814586639404297 | -0.44814586639404297 | yes | verify_2/probe npz, v2_probe_ana reduction |
| s60 | final | stagger sd v2 window, full precision (mm) | 20.752525 | 20.75 | yes | verify_2/probe npz |
| s60 | final | hip-pitch r-l angle sd, v2 window (mrad) | 26.998332 | 27.00 | yes | verify_2/probe npz |
| s60 | final | full per-episode lengths vs v2 probe | identical | identical | yes | verify_2/probe npz |
| s60 | final | judge sigma / own sigma, hip pitch r, l (report only) | - | 9.94, 10.51 | info | tool |
| s60 | final | stagger sd, addendum window 200..L-1 (mm; report only) | k3gain window 19.2 | 20.87 | info | policy_gain.csv (other window) |
| s60 | final | stagger: per-episode max of abs(stagger) over 200..L-1, largest episode (mm; report only) | - | 98.1 | info | tool |
| band | 5 slots | judge / own hip-pitch sigma, range over the validated slots (1 dp) | 6.9-10.2 | 6.9-10.2 (s42 6291456 to s60 final) | yes | report.md sec. 2 line 267 |
| s52 | 6291456 | own hip-pitch sigma in the band's 6.3M range (check-only) | 0.020-0.022 | 0.0205 | yes | report.md sec. 2 line 266; stance5_LTSB_s52_k3r_6291456_20261023T011813Z_check.log |
| s60 | 6291456 | own hip-pitch sigma in the band's 6.3M range (check-only) | 0.020-0.022 | 0.0215 | yes | report.md sec. 2 line 266; stance5_LTSB_s60_k3r_6291456_20261023T011822Z_check.log |
| s70 | 4194304 | LTSBr e2e (selftest/t8_ltsbr_e2e, tool b69ff098): judge file copy byte-identical to R/judged | true | true | yes | selftest/t8_ltsbr_e2e |
| s70 | 4194304 | LTSBr e2e (selftest/t8_ltsbr_e2e, tool b69ff098): sigma from the sidecar (sha256 89778b3f...) == judge file sigma_vector | true | true | yes | selftest/t8_ltsbr_e2e |
| s70 | 4194304 | LTSBr e2e (selftest/t8_ltsbr_e2e, tool b69ff098): full matches_judge | true | true | yes | selftest/t8_ltsbr_e2e |
| s70 | 4194304 | LTSBr e2e (selftest/t8_ltsbr_e2e, tool b69ff098): full reward max abs diff vs judge | 0 | 0 | yes | selftest/t8_ltsbr_e2e |
| s70 | 4194304 | LTSBr e2e (selftest/t8_ltsbr_e2e, tool b69ff098): full falls / judge falls | 0 | 0 | yes | selftest/t8_ltsbr_e2e |
| s70 | 4194304 | LTSBr e2e (selftest/t8_ltsbr_e2e, tool b69ff098): valid (false by design under K3R_TEST_ROOT) | false | false | yes | selftest/t8_ltsbr_e2e |
| s80 | 3145728 | LTSBr e2e (selftest/t8_ltsbr_e2e, tool b69ff098): judge file copy byte-identical to R/judged | true | true | yes | selftest/t8_ltsbr_e2e |
| s80 | 3145728 | LTSBr e2e (selftest/t8_ltsbr_e2e, tool b69ff098): sigma from the sidecar (sha256 89778b3f...) == judge file sigma_vector | true | true | yes | selftest/t8_ltsbr_e2e |
| s80 | 3145728 | LTSBr e2e (selftest/t8_ltsbr_e2e, tool b69ff098): full matches_judge | true | true | yes | selftest/t8_ltsbr_e2e |
| s80 | 3145728 | LTSBr e2e (selftest/t8_ltsbr_e2e, tool b69ff098): full reward max abs diff vs judge | 0 | 0 | yes | selftest/t8_ltsbr_e2e |
| s80 | 3145728 | LTSBr e2e (selftest/t8_ltsbr_e2e, tool b69ff098): full falls / judge falls | 0 | 0 | yes | selftest/t8_ltsbr_e2e |
| s80 | 3145728 | LTSBr e2e (selftest/t8_ltsbr_e2e, tool b69ff098): valid (false by design under K3R_TEST_ROOT) | false | false | yes | selftest/t8_ltsbr_e2e |
| s70 | 4194304 | LTSBr e2e (fix_r2/e2e, tool d0aed5a1): judge file copy byte-identical to R/judged | true | true | yes | fix_r2/e2e |
| s70 | 4194304 | LTSBr e2e (fix_r2/e2e, tool d0aed5a1): sigma from the sidecar (sha256 89778b3f...) == judge file sigma_vector | true | true | yes | fix_r2/e2e |
| s70 | 4194304 | LTSBr e2e (fix_r2/e2e, tool d0aed5a1): full matches_judge | true | true | yes | fix_r2/e2e |
| s70 | 4194304 | LTSBr e2e (fix_r2/e2e, tool d0aed5a1): full reward max abs diff vs judge | 0 | 0 | yes | fix_r2/e2e |
| s70 | 4194304 | LTSBr e2e (fix_r2/e2e, tool d0aed5a1): full falls / judge falls | 0 | 0 | yes | fix_r2/e2e |
| s70 | 4194304 | LTSBr e2e (fix_r2/e2e, tool d0aed5a1): valid (false by design under K3R_TEST_ROOT) | false | false | yes | fix_r2/e2e |
| s70 | 4194304 | LTSBr e2e (fix_r2/e2e, tool d0aed5a1): written by the current tool | d0aed5a1 | d0aed5a1 | yes | fix_r2/e2e |
| s70 | 4194304 | LTSBr e2e (fix_r2/e2e, tool d0aed5a1): reward_matches_judge | true | true | yes | fix_r2/e2e |
| s70 | 4194304 | LTSBr e2e (fix_r2/e2e, tool d0aed5a1): addendum_slot (not an addendum slot) | no | no | yes | fix_r2/e2e |
| s70 | 4194304 | LTSBr e2e (fix_r2/e2e, tool d0aed5a1): readouts == selftest/t8_ltsbr_e2e/out/stance5_LTSB_s70/k3r_4194304.json | identical | identical | yes | determinism |
| s80 | 3145728 | LTSBr e2e (fix_r2/e2e, tool d0aed5a1): judge file copy byte-identical to R/judged | true | true | yes | fix_r2/e2e |
| s80 | 3145728 | LTSBr e2e (fix_r2/e2e, tool d0aed5a1): sigma from the sidecar (sha256 89778b3f...) == judge file sigma_vector | true | true | yes | fix_r2/e2e |
| s80 | 3145728 | LTSBr e2e (fix_r2/e2e, tool d0aed5a1): full matches_judge | true | true | yes | fix_r2/e2e |
| s80 | 3145728 | LTSBr e2e (fix_r2/e2e, tool d0aed5a1): full reward max abs diff vs judge | 0 | 0 | yes | fix_r2/e2e |
| s80 | 3145728 | LTSBr e2e (fix_r2/e2e, tool d0aed5a1): full falls / judge falls | 0 | 0 | yes | fix_r2/e2e |
| s80 | 3145728 | LTSBr e2e (fix_r2/e2e, tool d0aed5a1): valid (false by design under K3R_TEST_ROOT) | false | false | yes | fix_r2/e2e |
| s80 | 3145728 | LTSBr e2e (fix_r2/e2e, tool d0aed5a1): written by the current tool | d0aed5a1 | d0aed5a1 | yes | fix_r2/e2e |
| s80 | 3145728 | LTSBr e2e (fix_r2/e2e, tool d0aed5a1): reward_matches_judge | true | true | yes | fix_r2/e2e |
| s80 | 3145728 | LTSBr e2e (fix_r2/e2e, tool d0aed5a1): addendum_slot (not an addendum slot) | no | no | yes | fix_r2/e2e |
| s80 | 3145728 | LTSBr e2e (fix_r2/e2e, tool d0aed5a1): readouts == selftest/t8_ltsbr_e2e/out/stance5_LTSB_s80/k3r_3145728.json | identical | identical | yes | determinism |
| fix | - | K3R_TEST_ROOT confinement conf_R: exit 2, refused message | 2, message | 2, message | yes | fix_r2/tests.sh |
| fix | - | K3R_TEST_ROOT confinement conf_dotdot: exit 2, refused message | 2, message | 2, message | yes | fix_r2/tests.sh |
| fix | - | K3R_TEST_ROOT confinement conf_here: exit 2, refused message | 2, message | 2, message | yes | fix_r2/tests.sh |
| fix | - | K3R_TEST_ROOT confinement conf_logs: exit 2, refused message | 2, message | 2, message | yes | fix_r2/tests.sh |
| fix | - | K3R_TEST_ROOT confinement conf_out: exit 2, refused message | 2, message | 2, message | yes | fix_r2/tests.sh |
| fix | - | K3R_TEST_ROOT confinement conf_outsub: exit 2, refused message | 2, message | 2, message | yes | fix_r2/tests.sh |
| fix | - | K3R_TEST_ROOT confinement conf_symlink: exit 2, refused message | 2, message | 2, message | yes | fix_r2/tests.sh |
| fix | - | K3R_TEST_ROOT confinement conf_tmp: exit 2, refused message | 2, message | 2, message | yes | fix_r2/tests.sh |
| fix | - | K3R_TEST_ROOT confinement: nothing written under out/ (or /tmp/out, R/out) | nothing | nothing | yes | fix_r2/conf_written.txt |
| fix | - | slot outside the addendum (s70 4194304 copy): NOTE and addendum_slot=no | rc 0, NOTE, no | rc 0 | yes | fix_r2/tests.sh note |
| fix | - | judge file recording another judge_sha256: refused, refused JSON | rc 3, reason | rc 3, 1 reason(s) | yes | fix_r2/c_jsha |
| fix | - | runner, judge file missing: the tool refuses and writes the refused JSON; stderr says so | rc 3, refused JSON, stderr | rc 3, JSON | yes | fix_r2/runner_testroot.sh |
| fix | - | runner, tool crash: stderr 'k3readout FAILED rc=1, see <log>' | rc 1, message | rc 1 | yes | fix_r2/runner_crash.sh |
| fix | - | reward tamper (+1e-3, smoke 1): matches true, reward_matches false, valid false, exit 4 | rc 4, True, False, False, 0.001 | rc 4, matches True, reward_matches False, valid False, diff 0.001 | yes | fix_r2/c_reward |
| fix | - | self-test t1_missing re-run with the current tool: exit 3, refused JSON, reason 'judge_6291456.json missing' | 3, reason | 3, reason | yes | fix_r2/selftests_again.sh |
| fix | - | self-test t2_model re-run with the current tool: exit 3, refused JSON, reason 'model sha256' | 3, reason | 3, reason | yes | fix_r2/selftests_again.sh |
| fix | - | self-test t3_sigma re-run with the current tool: exit 3, refused JSON, reason 'sigma differs from' | 3, reason | 3, reason | yes | fix_r2/selftests_again.sh |
| fix | - | self-test t4_sidecar re-run with the current tool: exit 3, refused JSON, reason 'judge file sigma file' | 3, reason | 3, reason | yes | fix_r2/selftests_again.sh |
| fix | - | self-test t5_head re-run with the current tool: exit 3, refused JSON, reason 'judge file was judged at 0000000' | 3, reason | 3, reason | yes | fix_r2/selftests_again.sh |
| fix | - | self-test t6_foreign re-run with the current tool: exit 3, refused JSON, reason 'mode 'quick'' | 3, reason | 3, reason | yes | fix_r2/selftests_again.sh |
| fix | - | self-test t7_sigma_r re-run with the current tool: exit 3, refused JSON, reason 'sigma differs from' | 3, reason | 3, reason | yes | fix_r2/selftests_again.sh |
| fix | - | runner arguments [stance5_LTSB_s70 final --arm] (timeout 5 s): exit 2, no hang | 2 | 2 | yes | fix_r2/args_tests.txt |
| fix | - | runner arguments [stance5_LTSB_s70 final --arm LTSB] (timeout 5 s): exit 2, no hang | 2 | 2 | yes | fix_r2/args_tests.txt |
| fix | - | runner arguments [stance5_LTSB_s70 123] (timeout 5 s): exit 2, no hang | 2 | 2 | yes | fix_r2/args_tests.txt |
| fix | - | runner arguments [stance5_LTSB_s42 final --arm LTSBr] (timeout 5 s): exit 2, no hang | 2 | 2 | yes | fix_r2/args_tests.txt |
| fix | - | runner arguments [stance5_LTSB_s70] (timeout 5 s): exit 2, no hang | 2 | 2 | yes | fix_r2/args_tests.txt |
| fix | - | runner arguments [stance5_LTSB_s70 final --bogus] (timeout 5 s): exit 2, no hang | 2 | 2 | yes | fix_r2/args_tests.txt |

all checked rows match: True checked rows: 223 mismatches: 0 info rows: 20
