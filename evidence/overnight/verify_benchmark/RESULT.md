# VERIFY_BENCHMARK: independent check of the BENCHMARK stage ("is 1.0 m/s a valid walking benchmark?")

Folder: `/tmp/claude-0/-home-user-mesozoic-labs/aff94cce-91ba-5515-b81b-a9f018e73dc9/scratchpad/work/overnight/verify_benchmark/`. Paths below are relative to it unless absolute.
Worked 02:34-02:57 UTC on 2026-11-04 (verify deadline 10:00 UTC).

Ground rules kept:
- Nothing trained or fitted.
- Repository, export (main 4d5f4e7) and physics untouched. The export directory is read-only (`dr-xr-xr-x`), and `find -newermt "2026-11-03 23:00"` on it returns 0 files.
- Hypothetical caps and weights were recomputed offline from my own recorded per-step arrays. No env attribute was set.
- One simulation process at a time, all with nice -n 10, OMP_NUM_THREADS=1, TMPDIR=tmp/ and the optB venv python -I -B. No extra packages were needed.
- All my code is new (`scripts/measure_v.py`, `vroll.py`, `vana.py`, `vhist.py`, `vhist2.py`, `vprobe.py`). None of the BENCHMARK scripts were imported. The walkers are the EXISTENCE / SPEED controller classes and parameter files, which are the objects under test.

## Verdict: CONFIRMED WITH CORRECTIONS

Every number I recomputed reproduces, most of them exactly. The corrections are to labels, one physics revision, a mass range, and two over-generalisations. None changes the four headlines.

## What I re-ran and recomputed

1. **Froude table, independently** [measured: out/measure_v_stdout.txt, out/measure_v.json].
   - Method: I built each species' locomotion env. L is the mean hip_pitch `xanchor` z at the 'home' keyframe, on my own MjData. Mass is the pelvis subtree mass. Gate and cap come from each `locomotion.toml` via tomllib and from the built env. g = 9.81 is the model's |g|.

   | | mass kg | L home (mine = probe) | statue hips t ≥ 2 s | CoM | gate: Fr / v̂ | cap: Fr / v̂ | v at Fr 0.25 / 0.5 |
   |---|---|---|---|---|---|---|---|
   | T. rex | 85.720 | 0.87660 | 0.87706 (+0.46 mm) | 0.86709 | 1.0: 0.1163 / 0.3410 | 2.5: 0.7268 / 0.8525 | 1.4662 / 2.0736 |
   | compsognathus | 1.000 | 0.24439 | 0.24447 | 0.22831 | 0.08: 0.0027 / 0.0517 | 0.25: 0.0261 / 0.1615 | 0.774 / 1.095 |
   | velociraptor | 13.500 | 0.50554 | 0.50553 | 0.46308 | 2.0: 0.8066 / 0.8981 | 2.5: 1.2602 / 1.1226 | 1.114 / 1.575 |

   - Statue: 10 of 10 full episodes per species on seeds 3042-3051.
   - Converted to the T. rex body: the raptor gate is 2.634 m/s and the compsognathus gate 0.152 m/s.
   - Config comment, locomotion.toml:139-148: "Fr 0.14-0.16" implies L 0.64-0.73 m. At the comment's own CoM heights (0.72-0.88 m), 1.0 m/s is Fr 0.116-0.142, so the comment does not agree with itself either. The raptor gate is Fr 0.807 at hip height and 0.880 at CoM height, not ~1.07.
   - The stage's statements on all of this are confirmed.
2. **Literature conversions and arithmetic** [inputs: literature, recalled, unverified; arithmetic measured].
   - v_model = v_full·√(0.8766/L_full), with L_full = 3.0-3.5 m. Preferred walk 1.28 → 0.641-0.692; 5.3 → 2.652-2.865; 5-11 → 2.50-5.95; 8 → 4.00-4.32.
   - The inverse puts this model's 1.0 m/s at 1.850-1.998 m/s full size.
   - Ideal linear start: a 0.5 / 3 s ramp needs 1.026 / 1.176 m/s steady.
   - All of these are confirmed.
3. **Re-rolls with my own loop** [measured: out/v_<tag>.json/.npz, out/v_<tag>_stdout.txt]. The loop is plain `env.reset(seed)`, `ctl.reset()`, `env.step`.
   - Behaviours: the statue and all 10 walkers (FINAL3, f2v0.25, V01, V05, V23, V24, V18, V35, V27, V29), seeds 3042-3051, 110 episodes. Also V35 on seeds 0 and 1.
   - Per-term sum: summing the 29 `reward_*` info terms in the env's own left-associated order, plus the fall penalty on a non-success terminal step, equals the env reward exactly (max error 0.0) on every step of every episode. Summing all `reward_*` keys gives the same, because `reward_floor_impact` and `reward_airborne_substeps` are present but weighted 0.
   - My own `qvel[0:2]·_initial_prey_dir_2d` equals `info["forward_vel"]` exactly.
   - `reward_forward` equals `2.0·clip(fv/2.5, −1, 1)` exactly.
   - R(2.0, 2.5) recomputed matches the summed env reward to within 4.5e-13.
   - Against BENCHMARK: per-step reward and forward_vel arrays are **bitwise equal on all 110 episodes**, and returns differ by 0.0.
   - Against REWARD (which set the weight in its env): statue, FINAL3 and f2v0.25 match at w 0.2 and 2.0 to within 2.3e-12 for returns and 5.4e-13 for γ = 0.995 discounted returns.
4. **Code read for the gate and the reward** [measured: code reading].
   - Gate code: `eval_diagnostics.py:1320-1347` sums `info["forward_vel"]` over every step of each eval episode and divides by that episode's step count. `curriculum/manager.py:259,417-419` compares the mean over episodes with `min_avg_forward_vel`. The post-stage judge path (`evaluation.py:99-124`, `reporting/gates.py:1599-1609`) uses the same per-episode mean.
   - `forward_vel` is the root freejoint velocity along `_initial_prey_dir_2d`, set at reset (`tyrannosaurus_rex/base_env.py:916,1381`).
   - The locomotion `gate_kind` `reward_and_length/v1` (`gate_schema.py:69-79`) has no gait key.
   - `forward_vel_weight` and `forward_vel_max` enter only through `reward_forward` and `reward_backward` (`shared/base_env.py:646-656`). `backward_vel_penalty_weight` is 0 on the built env, so the offline recompute is exact.
   - The ramp is absolute: 0.2 → 2.0, linear over 2M steps from step 0 (`advancement.py:658-683`, `train_base.py:1355-1366`). That makes 0.47 at 300k and 1.1 at 1M correct.

## Headline-by-headline

1. **1.0 m/s is a sensible walking speed. Fr 0.116; statue hips within 0.5 mm; V18 steady 0.98 on 10/10 with no flight.** CONFIRMED [measured].
   - L 0.8766 m; settled hips +0.46 mm; Fr 0.1163, v̂ 0.341; Fr 0.25 → 1.466 m/s, Fr 0.5 → 2.074 m/s.
   - V18 on 3042-3051: 10/10 full, steady (5-10 s) 0.979 [0.912, 1.001].
   - My floor-truth count finds 0 airborne control steps for V18. A foot counts as down when the floor normal force is > 0.1 N on at least 3 of 5 substeps. 0.2% of substeps have both feet unloaded.
   - The frozen probe's `analyze_episode`, run by me (`out/vprobe_stdout.txt`), gives class walk 10/10, alternation ≥ 0.919, flight 0, double support 0.172-0.182.
   - Label slip: the Fr 0.25 and Fr 0.5 reference points in this headline are recalled literature. The headline labels the line [measured; reading inferred]. The key results and the stage RESULT table label them correctly.
2. **The gate is an episode mean from a standing start, so a walk needs about 1.30-1.33 steady (1.03-1.18 with an ideal start). V35 averages 0.88, V27 0.992 with 5/10 seeds ≥ 1.0.** CONFIRMED [measured; code read; arithmetic].
   - Episode / steady ratio: V18 0.767, V35 0.754, V27 0.750 (1/ratio = 1.30-1.33); the slower walks 0.81-0.91.
   - V35 episode mean 0.882 [0.788, 0.910], 0/10 seeds pass.
   - V27 0.992 [0.932, 1.023], 5/10 seeds ≥ 1.0 (3043, 3044, 3045, 3047, 3049; 3048 is 0.9988).
   - The 1.30-1.33 figure holds for these controllers' 3.5-4 s start-up ramps (V18 3.5 s, V35/V27 4.0 s, all from 0.2 m/s after a 0.5 s settle; read from the param files). It is a controller property, as the stage's limits say.
3. **The gate checks no gait. The September hops passed at Fr 0.13-0.29.** CONFIRMED [code read; doc claim].
   - No gait key exists in `reward_and_length/v1`.
   - Doc lines 86 and 89 of `docs/GAIT_QUALITY_PLAN_2026_09.md` give 1.05 and 1.57 m/s two-footed hops, 35% and 43% airborne, alternation 0.010 for the first. Fr is 0.128 and 0.287 on the current L.
   - Correction: those runs were on **physics r7**, not "r8/r9" as the stage's limits say. `docs/investigations/STANCE_HACK_AUDIT_2026_10.md:484` lists `20260914_123816` as r7. `docs/NEXT_STEPS.md:264` lists `20260925_033501` as "physics r7". r8 landed 2026-10-06 per `GAIT_QUALITY_PLAN_2026_09.md:15,673`. Their Fr therefore uses r10's L, assuming a hip height that may differ slightly on r7 [inferred].
4. **The reward pays speed linearly to 2.5 m/s. Steady slopes 771-881 per 10 s per m/s at w 2.0; 80 per episode per m/s at w 0.2; the crouched V23 and V24 lose at w 0.2.** CONFIRMED [measured].
   - Steady-window slopes at w 2.0, cap 2.5 (V24→V18 / V18→V35 / V35→V27): +881 / +771 / +845.
   - Cap 1.2: +1678 / +1006 / +706. Cap 1.0: +1615 / +648 / +323.
   - Paired Δ versus the statue at w 0.2: V23 −41.0 (0/10), V24 −16.5 (0/10).

## Key results checked (all reproduced, exactly or to the printed digit) [measured]

- Δ versus statue at w 0.2 / 2.0, cap 2.5: FINAL3 +4.9 / +141.3, f2v0.25 +4.1 / +207.5, V05 +18.2 / +329.4, V18 +13.4 / +555.7, V35 +21.0 / +658.0, V27 +35.5 / +751.5.
- Δ with cap 1.2: V18 +77.5 / +1197.0, V27 +104.2 / +1438.7. With cap 1.0: V18 +95.7 / +1378.2, V27 +114.9 / +1546.2.
- At the ramp start, every non-falling walk except V23 wins 10/10 under either cap, by +21.3 to +114.9.
- Ramp line (w 0.2 / 0.47 / 1.1 / 2.0): FINAL3 +4.9 / +25.4 / +73.1 / +141.3; V27 +35.5 / +142.9 / +393.5 / +751.5.
- Discounted from reset at w 0.2: V23 −6.23, V24 −4.71, V18 −3.20, V35 −2.56, V27 −1.27.
- Non-forward deficits: FINAL3 −10.2, V05 −16.4, V23 −73.3 (height −64.2), V24 −61.0, V18 −46.8, V35 −49.7, V27 −44.1.
- Within-stride spread: V18 p5/p50/p95 0.724 / 0.957 / 1.302, 43.3% of steady steps > 1.0. V27: 91.2% > 1.0 and 60.8% > 1.2.
- V27 forward pay at w 2.0: 794 (cap 2.5), 1479 (cap 1.2), 1586 (cap 1.0). Hop bounds: 840 / 1256 at cap 2.5.
- V29: 8/10 full (falls at 712 and 855 steps), episode mean 0.990.
- V35 on seeds 0 and 1 reproduces SPEED's recorded final root position bitwise.
  - My first comparison seemed to fail, by about 2 mm. I had compared my freejoint `qpos[0:2]` with the probe's `root_pos`, which is the pelvis body `xpos`.
  - `vhist2.py` ran the probe's `rollout_episode` and my plain loop on fresh envs. Both give the identical final qpos (9.017018590096768, −0.2622447583948429), and the probe's `root_pos` equals SPEED's value bitwise.
  - So the trajectories are identical, and the gap came only from how I measured position. `vhist.py` also showed that `RobotDescription(env)` leaves the env's MjData unchanged.

## Corrections to the stage's text

1. **Physics revision of the September hops: r7, not r8/r9** (limits section; RESULT.md line 241). The sources are the ones cited under headline 3.
2. **Mass scaling.** "~110-175 kg [from a recalled adult mass of 6-9 t]": 110-175 kg is the 7 t case for L_full 3.5-3.0 m. With 6-9 t the range is 94-225 kg [arithmetic; inputs recalled, unverified].
3. **"At equal speed a hop and a walk earn the same under every cap setting."** This is exact only while the per-step speed stays below the cap.
   - With a 1.0 or 1.2 cap, `clip` is concave. At equal mean speed, a gait with larger per-step swings earns less, and the stage's own walks already spend 43-91% of steady steps above 1.0. So the hop bounds are upper bounds, as the table says, not equalities.
   - The conclusion that a cap does not reward alternation stands [inferred].
4. **Upright break-even range.** With V01 included, the upright walks break even at 0.13-0.25 m/s (V01 0.251) at w 0.2 and cap 2.5, and at 0.05-0.12 m/s with caps 1.2/1.0. The stage says 0.13-0.23 and 0.05-0.11. Its own output lists V01 [measured].
5. **Key-results grouping.** "upright V05 +18.2, V18 +13.4, V35 +21.0, V27 +35.5" reads as if V18, V35 and V27 were upright. They use the same crouch as V23 and V24 (`z_com` 0.93 against 0.985 for V05/FINAL3; param files). They win at w 0.2 because they are faster. The stage's body text treats them as crouched.
6. **Literature labels.**
   - Headlines 1 and 4 use Fr 0.25 (human preferred walk) and Fr 0.5 (walk-run transition) without the [literature, recalled, unverified] label. The key results do label them.
   - All other literature figures are correctly labelled recalled and unverified.
   - From my own recall [literature, recalled, unverified]:
     - The figures attributed to van Bijlert, van Soest & Schulp 2021 (1.28 m/s), Sellers et al. 2017 (~5.3 m/s), Hutchinson & Garcia 2002 and Hutchinson 2004 (5-11 m/s) and Sellers & Manning 2007 (~8 m/s) match what I remember.
     - Alexander & Jayes 1983 is a quadruped dynamic-similarity paper, so it is a weak source for the bipedal walk-run Fr ≈ 0.5. Alexander 1989 is the better one of the two cited.
   - No network was used, so none of this could be checked.

## My own gait metric versus the probe [measured]

- My floor-truth flight and double support agree with the probe: V27 flight max 0.0022, double support 0.144-0.158; V18 double support 0.172-0.182.
- My crude alternation index is lower (0.71-0.86 minimum over the walks). It counts every up→down transition, including brief re-touchdowns by the same foot. The probe merges near-simultaneous events. I therefore rely on the frozen probe (`vprobe.py`) for alternation, which gives V18 ≥ 0.919 and V27 ≥ 0.951 as the stage says.

## Limits of this check

- Same hand-built controllers and the same seeds 3042-3051. The re-rolls test reproducibility and the arithmetic, not whether learned policies would behave the same.
- The hop speeds stay doc claims; nothing was simulated for them.
- Literature figures could not be verified without network access.

## Slips (mine)

1. Two read-only waits on `out/vbatch_log.txt` overlapped: a background until-loop and a Monitor until-loop. Both are noted in CMDLOG.
2. One shell call (`sleep 60; cat out/vbatch_log.txt`) was blocked by the harness and never ran. It is not in CMDLOG.
3. Scripts were written with the Write tool and one Edit; each has a CMDLOG NOTE. The first shell call and the logger's creation are NOTE lines written after the fact.
4. My first V35 seed-0/1 comparison mixed two position definitions (see above). Two extra read-only diagnostic runs were needed to resolve it (`vhist.py`, `vhist2.py`).
5. The harness wrote its own background-task output files outside my folder, under `/tmp/claude-0/-home-user-mesozoic-labs/aff94cce-91ba-5515-b81b-a9f018e73dc9/tasks/`.
6. CMDLOG contains the device-directory string `/dev` twice, only as grep patterns used to audit logs. It never names the null or zero device.

## Files

- `CMDLOG.txt` and `RESULT.md`.
- `scripts/`: `L.sh`, `measure_v.py`, `vroll.py`, `vbatch.sh`, `vana.py`, `vhist.py`, `vhist2.py`, `vprobe.py`.
- `out/`:
  - `measure_v.json` and `measure_v_stdout.txt`;
  - `v_<tag>.json`, `v_<tag>.npz` and `v_<tag>_stdout.txt` for the 11 behaviours and `V35_s01`;
  - `test_V18*`;
  - `vana.json` and `vana_stdout.txt`;
  - `vhist_stdout.txt`, `vhist2_stdout.txt`, `vprobe_stdout.txt`;
  - `vbatch_log.txt`.
