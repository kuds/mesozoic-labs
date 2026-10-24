# Stance5 replication: report-only K3 mechanism readouts (k3readouts)

REPORT ONLY. These readouts enter no verdict, no decision line and no tie-break, and they are not a veto (see
`R/readouts_addendum_k3.txt`, which governs; `R` = `SP/foot/stance5/replication`, `SP` = the scratchpad
`scratchpad`). A readout that cannot be
computed changes nothing. `criteria.txt`, the judge, the scorer, the sidecar and the queue tools are not edited, and
nothing here touches them; the judge modules are imported read-only.

The addendum says "the tool and its validation ... are recorded below when built". The addendum is pinned by the tool
(`ADDENDUM_SHA256` = 7a79e9f6..., checked first: any other addendum is refused), so the tool's record is kept here and in
`VALIDATION.md`, not appended to the addendum. Appending to the addendum would make every run refuse until the pin is
re-set on purpose.

## Files

| File | What it is |
|---|---|
| `k3readout5r.py` | the tool (one checkpoint per call) |
| `k3readout_one5r.sh` | the runner: `k3readout_one5r.sh RUN STEPS [--arm LTSB\|LTSBr] [--check-only]` |
| `out/<RUN>/k3r_<steps>.json` | one output per run and slot (never overwritten: a repeat writes `k3r_<steps>.<n>.json`) |
| `out/<RUN>/refused_<steps>.json` | a refusal record (the reasons, and the identity read so far; never overwritten either) |
| `logs/<RUN>_k3r_<steps>_<utc>_<pid>[_check].log`, `logs/runner.log` | one log per call (opened with noclobber); start / end / refused lines |
| `validate_k3r.py`, `VALIDATION.md` | the validation against stance5's skeptic-checked values |
| `selftest/` | refusal self-tests (t1-t7), the LTSBr end-to-end test on a byte copy of a replication judge file (t8, under `K3R_TEST_ROOT`, so marked invalid by design) and the development smoke output (never read by anything) |
| `review_fidelity/`, `review_isolation/` | the two reviews' own test material (read by nothing) |
| `fix_r1/`, `fix_r2/` | the review-fix round's tests (runner copies, tampered judge-file copies; see `VALIDATION.md`) |

## Use

    R=SP/foot/stance5/replication
    $R/k3readouts/k3readout_one5r.sh stance5_LTSB_s70 6291456          # replication (default --arm LTSBr)
    $R/k3readouts/k3readout_one5r.sh stance5_LTSB_s80 final
    $R/k3readouts/k3readout_one5r.sh stance5_LTSB_s42 final --arm LTSB  # stance5 reference / validation

The addendum's slots are 6,291,456 and final (12,001,280), each once its judge file exists. A call for a slot whose
`R/judged/<RUN>/judge_<steps>.json` is missing is refused by the tool (exit 3, `out/<RUN>/refused_<steps>.json` written,
nothing rolled; the runner says so up front and adds a `refused` line to `logs/runner.log`), so it can be called after
each judge. The runner accepts any multiple of 524288 that has a judge file; a slot the addendum does not name is
rolled, but its output says `"addendum_slot": "no"` (also in the summary line, and the tool prints a NOTE): a
measurement, not one of the addendum's readouts.

The runner runs the tool exactly as the k3diag runners ran theirs: `ISO_WT=SP/fix/scratch_copband nice -n 19 python3
SP/foot/stance3/tools/iso.py k3readout5r.py ...` with OMP/MKL/OPENBLAS threads 1, `PYTHONPYCACHEPREFIX=SP/caches/pyc`
and `MPLCONFIGDIR=SP/caches/mpl`, from this directory, with `K3R_TEST_ROOT` unset. At most two k3readout runs at a time
(two `flock` slots under `logs/`); a third call exits 9 and runs nothing. A checkpoint takes about 4.5-6 minutes on one
CPU with two runs and the judge queue sharing the box (7 conditions of 20 episodes, about 20-55 s each; measured timings
are in `VALIDATION.md`).

Exit codes (tool and runner):

| code | meaning |
|---|---|
| 0 | a valid output written (`--check-only`: every check passed, nothing written) |
| 2 | bad arguments, or a `K3R_TEST_ROOT` outside `R/k3readouts`: nothing run, nothing written |
| 3 | refused: a refused JSON is written with the reasons (`--check-only`: nothing written) |
| 4 | an output written but NOT valid ("full" did not reproduce the judge file, the clone changed during the rolls, or a smoke / test-root run) |
| 9 | runner only: two k3readout runs already active |
| 1 or other | a crash; the runner prints `k3readout FAILED rc=N, see <log>` on stderr |

The runner prints the tool's `K3R ...` summary line (and any NOTE or "exists; wrote" line); on exit 3, 4 or a crash it
also says so on stderr. Readers select outputs by their `valid` and `addendum_slot` fields, never by file name: an
invalid first output keeps the `k3r_<steps>.json` name and a later valid repeat becomes `k3r_<steps>.1.json`.

## Refusals

The tool refuses (exit 3, `out/<RUN>/refused_<steps>.json`) when:
- the judge module is not the pinned file (`judge_stance5r.py` 16d39f7d... for LTSBr, `judge_stance5.py` d9286724... for
  LTSB), or the addendum is not the pinned file (7a79e9f6...);
- the judge file is missing, unreadable, foreign (schema, arm LTSB, mode full, seed, steps, smoke false, noise_run, or a
  recorded `judge_sha256` other than the pinned judge module's) or incomplete (no `judge_seconds`, no 20-episode
  `noise_matched_control_joint`), or was judged at another clone HEAD;
- the checkpoint zip or VecNormalize pickle is missing, or its sha256 differs from the judge file's;
- the sigma source fails its checks or the sigma differs from the judge file's (see Sigma below);
- the clone `SP/fix/scratch_copband` is not clean at cf80ecb14d1630ce3b4dcdea28a05835d143470b (the judge's own
  `isolate` and `git_state`, the latter read-only with `--no-optional-locks`), or `environments` would not import from it.

If "full" does not reproduce the judge file exactly, the output is still written, marked `"valid": false`, and the tool
exits 4. Two flags say why: `"matches_judge": false` when a per-episode seed, length or termination reason differs, and
`"reward_matches_judge": false` when an episode's reward differs from the judge file's by 1e-6 or more (verify_2's
`v2_probe.py` line 182 required the same). `valid` needs both. Such an output must not be read as a readout.

## What is measured (definitions and the lines they copy)

Common loop (`SP/foot/stance5/k3diag/verify_1/vroll.py` lines 108-147, itself the judge's `noise_matched` in joint
space): one fresh env per condition with the judge's `SubstepContactRecorder(env, Morphology)` attached; spawns 3042-3061
(`J.NOISE_SEED`, `J.NOISE_EPISODES`); `rng = numpy.random.default_rng(episode seed)`; every step the observation is
normalised with the checkpoint's VecNormalize stats, `mu = action_net(mlp_extractor.forward_actor(extract_features(o)))`,
`cmd = clip(mu)`, `z = rng.standard_normal(15)`; a fall is an episode shorter than the 1000-step horizon. Checkpoints are
loaded only by the judge module's own `load_checkpoint`.

1. Falls of 20 under noise, `act = clip(cmd + e)` with `n = sigma * z` (vroll.py `transform`, lines 35-64):
   `full` e = n (must reproduce the judge file; `matches_judge`, `reward_matches_judge`); `hip_pitch` n on actuators 3
   and 7 only; `nohp` n on every actuator except 3 and 7; `hpD` d = (n[3] - n[7]) / 2 applied as (d, -d) on 3, 7; `hpC`
   c = (n[3] + n[7]) / 2 applied as (c, c). hpD + hpC = the hip-pitch noise exactly; both are at the judge's real
   amplitude (not k3roll's boosted hp_anti / hp_sym).
2. The kick probe (`k3diag/rollouts/scripts/k3roll.py` lines 57-62 and 214-225, conditions `pulse_anti` and
   `pulse_anti20`): deterministic, no noise (the condition's sigma mask is all zeros, so `act = clip(cmd + 0 * z)`, the
   clipped mean); on control steps 400-419 (the step counter before `env.step`, from 0; 20 steps x 0.01 s = 0.2 s,
   starting 4.0 s after reset) the kick is added to that command and clipped again: `act[3] = clip(act[3] + a_r)`,
   `act[7] = clip(act[7] + a_l)`, (a_r, a_l) = (+0.3, -0.3) or (+0.2, -0.2) (right +, left -). Falls of 20 over the
   same spawns, counted as `length < 1000` (`k3pulse.py` line 43). k3pulse does not check that an episode lived to the
   kick, and neither does this count; `falls_before_kick` (episodes with length <= 400, which never received the kick)
   is recorded beside it and printed in the summary line. It has been 0 in every output so far.
3. Direct response (`k3diag/verify_2/v2_probe.py` lines 28, 90-94, 101-105, 117, 120-139; reduction
   `v2_probe_ana.py`): during `full`, at every visited state, the raw observation and six nudged copies go through the
   same normalisation and actor as one float32 batch. The hip-pitch angle channels are obs index `jnt_qposadr - 7` of
   the r / l hip-pitch joints (checked against qpos at every reset). The nudges are q_r + h, q_l - h and q_r - h,
   q_l + h with h = 0.01 rad; the velocity channels `(nq - 7) + dofadr - 6` get +/- 0.05 rad/s; both angles + h / - h
   give the common mode. Jq = [(mu_r - mu_l)(+h) - (mu_r - mu_l)(-h)] / (4 h), per rad of q_r - q_l. The probe never
   touches the env or the rng. Aggregate: window steps 200 .. L-1 for survivors and 200 .. L-151 for falls, episodes
   with 60 or fewer window steps skipped, all steps pooled; the readout is the median (`Jq_median`). Also recorded: the
   mean, p10 / p90, the per-episode-median range, Jv (per rad/s) and Jq_common.
4. Own sigma: `exp(model.policy.log_std)` of the loaded checkpoint, per actuator. Also `sigma_judge / sigma_own` per
   actuator, and the hip-pitch mean (v2_probe's `own_sigma_hp`).
5. Stagger: the env's `info["foot_stagger_m"]` (the heading-frame fore-aft offset of the right foot from the left),
   read after each `env.step` (v2_probe.py line 160). Per noise condition: the sd over window steps 200 .. L-1 pooled over
   the episodes (the addendum's readout, `sd_mm`), the per-episode max |stagger| over the same window, and, for
   comparison with k3diag, the sd over v2_probe_ana's window (`sd_mm_v2_window`) and the sd of q_r - q_l over that
   window (`dq_sd_mrad_v2_window`).
   **`sd_mm` includes the topple steps of falling episodes.** Whenever a condition has falls inside its window
   (`falls_in_window` > 0, i.e. episodes with 200 < L < 1000) it partly measures the falls rather than the stagger (and
   the per-episode max then usually comes from a fall); `sd_mm_note` says so in the JSON. `sd_mm_v2_window` stops each
   fall 150 steps before its end and is the figure comparable with k3diag's 48-50 mm. The summary line prints both
   (`stagger_sd=`, `stagger_sd_v2_window=`) and `stagger_falls_in_window=`. Example: s42 7,340,032 (9 falls) reads
   61.68 mm in the addendum window and 47.98 mm in the v2 window (k3diag 48.0). The window is applied the same way to the
   stance5 references, so replication-against-reference comparisons of `sd_mm` are like for like. k3diag's report also
   quotes a third window (200 .. fall onset, k3analyze.py).

n = 20 per condition: differences of 2-3 falls are within binomial noise (addendum).

## Sigma

- `--arm LTSBr` (runs s70, s80): judge_stance5r.py's own code path for the sidecar seeds, `main()` lines 1221-1236,
  re-executed statement for statement on the module's constants: frozen_params sha256 == `FROZEN_PARAMS_SHA256`; the
  sidecar `CONTROL_SIGMA_SIDECAR` read, its sha256 == `CONTROL_SIGMA_SIDECAR_SHA256` (89778b3f...), its `source_sha256`
  == `FROZEN_PARAMS_SHA256`, then `control_sigma[str(seed)]`; the vector as `np.asarray(..., float64)` (noise_matched
  line 870). Refused unless it equals the judge file's `noise_matched_control_joint.sigma_vector` exactly (and its
  `sigma_source`), and unless the judge file's `control_sigma_file` / `control_sigma_file_sha256` name the same file
  and hash. The judge has no callable that returns the sigma (and `main` cannot be run partially), so those statements
  are copied; the binding guarantee is the exact equality with the vector the pinned judge wrote into the judge file,
  so the copy cannot drift silently. The sidecar's vectors for seeds 70 and 80 are identical, so a 70 / 80 sigma swap
  could not be seen (and would change nothing); a run swap is caught by the seed, run-list and model-sha256 checks.
- `--arm LTSB` (runs s42, s52, s60): frozen_params `criteria.H4.control_sigma[seed].vector` as vroll.py lines 93-95
  read it (sha256-checked), equal to the judge file's sigma_vector exactly.

## Output (`k3r_<steps>.json`, schema stance5r-k3readout/v1)

- `valid`, `matches_judge`, `reward_matches_judge`, `clone_clean_after`, `addendum_slot` (`readout`, `reference` or
  `no`), `summary` (the stdout line);
- `identity`: tool and runner path and sha256, addendum sha256, judge module and sha256, judge file and sha256 (and the
  judge sha256 the judge file records, which must be the pinned one), clone HEAD / status / branch before and after, the
  modules' origins, model and vecnorm path and sha256 (and the judge file's), sigma file, sha256, source string and
  vector, seeds, horizon, control_dt, the slot's place in the addendum, the definitions and source lines;
- `conditions` (full, hip_pitch, nohp, hpD, hpC): falls, fall_seeds, lengths, terms, seconds; full also matches_judge,
  reward_matches_judge, the judge's falls and fall seeds, and the largest reward difference against the judge file;
- `kick` (pulse_anti = +/-0.3, pulse_anti20 = +/-0.2): falls, fall_seeds, lengths, terms, falls_before_kick and their
  seeds;
- `response`, `own_sigma`, `stagger` (per noise condition, with `falls_in_window` and `sd_mm_note`), `timing_seconds`.
No per-step arrays are written; an output is about 20 KB.

The summary line reads, for example:
`K3R stance5_LTSB_s42 7340032 arm=LTSB addendum_slot=reference valid=True full=9/20 (judge 9, matches_judge=True,
reward_matches_judge=True) hip_pitch=8 nohp=0 hpD=5 hpC=0 kick0.3=20 kick0.2=0 kick_falls_before_kick=0/0
response=-0.514/rad own_hp=0.0187 stagger_sd=61.68mm stagger_sd_v2_window=47.98mm stagger_falls_in_window=9 (...s)`.

## Validation

`VALIDATION.md` holds the comparison with stance5's skeptic-checked values (judge files, verify_1, verify_2, k3diag
tables and rollouts), regenerated by `validate_k3r.py` (run through iso.py), and the review-fix round's tests.
Result (2026-10-23 02:15 UTC, tool d0aed5a1..., runner 5e3c66cd...): s42 at 6,291,456, 7,340,032 and final, s52 final
and s60 final, all re-rolled with this tool; 223 checked rows, 0 mismatches; "full" reproduces every judge file per
episode (seed, length, termination, and reward to 0.0); the earlier tool's six outputs are identical in every readout;
the LTSBr path reproduces two replication judge files (byte copies, under `K3R_TEST_ROOT`) with both tools; the 13
review findings, what was done about each and the tests are listed there.
