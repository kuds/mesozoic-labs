# VERIFY_LOGSTD: independent check of the LOGSTD stage (stance parent under set policy log_std)

Folder: `/tmp/claude-0/-home-user-mesozoic-labs/aff94cce-91ba-5515-b81b-a9f018e73dc9/scratchpad/work/overnight/verify_logstd/` (called V below). The stage folder is `.../overnight/logstd/` (called S). Worked 05:12-06:04 UTC (verify write-up due 10:30).

**Verdict: CONFIRMED WITH CORRECTIONS.** Every count in the headline that I re-ran reproduces within noise:
- with my own script and two independent noise streams;
- on the stage's reset seeds and on 80 fresh ones;
- with a bit-exact reproduction of the stage's torch stream.

Three statements need correcting: the "no-fall" label on -2.0, one sentence in the limits about pooled rates, and how firm the -1.875 / -1.8125 edge is.

All of this was CPU evaluation only. Nothing was trained. The repository, the export, the physics and the checkpoint were not changed: the checkpoint sha256 is the same before and after (`out/ckpt_sha256_start.txt`, `out/ckpt_sha256_end.txt`) and matches the stage's. Simulations ran one process at a time, under nice -n 10, with OMP_NUM_THREADS=1 and TMPDIR=V/tmp. Every command is in `CMDLOG.txt`.

## What I ran (my own script, `scripts/vlog.py`)

- **Env and loading.**
  - Env: `build_env(ModelId("tyrannosaurus_rex","anatomical"),"locomotion")` from the export main_4d5f4e7.
  - Parent: `stage1_final.zip` with `stage1_final_vecnorm.pkl`, loaded by the export's `load_sb3_checkpoint`, with frozen statistics (asserted `training False`, `norm_reward False`).
- **log_std on an in-memory copy.**
  - The policy is `copy.deepcopy`'d and log_std is set on the copy only. The loaded model's own log_std is asserted equal to the checkpoint values after every episode.
  - The policy has exactly one parameter named `log_std`: a 15-vector, no gSDE, `DiagGaussianDistribution`.
  - At step 0 of every episode, the copy's action distribution has stddev = exp(set value) on every actuator (relative error ≤ 5.1e-8). The original keeps the checkpoint's std (≤ 3.1e-8). The copy's mean equals the original's mean exactly.
  - [measured: out/va_stdout.txt section D]
- **Noise streams.** None of these is used by the stage, except rT, which is the stage's own draw:
  - `vP`: numpy Philox(SeedSequence([seed, 0x5A17, 41])), a = clip(mean + std·N(0,1), -1, 1).
  - `vT`: SB3's own sampler on the copy (`get_distribution(obs).get_actions(deterministic=False)`), with torch.manual_seed(7,000,003 + seed).
  - `rT`: the stage's and REWARD stage's torch draw (manual_seed(seed), `predict(deterministic=False)`). It is used only to check their records.
- **Seeds.** Reset seeds 3042-3081 (panel S, the stage's). Fresh panels F = 9100-9139 and G = 9140-9179.
- **Readouts.** Episodes run at w 0.2; the return at w 2.0 = ret + 9·(forward-term sum). On 9 episodes I re-ran at w 2.0 directly: same length and final state, |difference| ≤ 1.8e-12. [measured]
- **Contact rule (mine, not the probe's).** I read the env's per-substep touch-sensor block (`_substep_foot_force_block`, read-only copy).
  - A foot is down when the force exceeds 0.1 N on at least half of the substeps, with a 2-step debounce.
  - A swing is a lift-off followed by a touchdown. Clearance is the maximum site z during the swing minus the median site z while down.
  - This is a different data source from the stage's floor-contact probe, so lift counts are comparable only roughly.

## Stand counts (stood = 1000 steps without termination) [measured: out/va_stdout.txt, out/va3_stdout.txt]

| log_std (std) | stage S torch/np/np2 | mine S vP / vT | mine F vP / vT | mine G vP / vT | all independent panels pooled |
|---|---|---|---|---|---|
| own (0.007-0.111) | 40/40/40 | 40 / – | – | 40 / – | 200/200 |
| -2.5 (0.082) | 40/40/40 | 40 / – | – | – | 160/160 |
| -2.25 (0.105) | 40/40/40 | 40 / – | – | – | 160/160 |
| **-2.0 (0.135)** | **40/40/40** | **40 / 40** | **40 / 40** | **39 / 40** | **359/360** |
| -1.9375 (0.144) | 40/39/39 | 40 / 39 | – | – | 197/200 |
| **-1.875 (0.153)** | **40/38/38** | **39 / 38** | **40 / 38** | **38 / –** | **309/320 (0.966)** |
| -1.8125 (0.163) | 38/37/35 | 37 / 36 | 38 / – | 34 / – | 255/280 (0.911) |
| -1.75 (0.174) | 32/31/28 | 33 / 29 | 37 / 33 | 30 / – | 253/320 (0.791) |
| **-1.5 (0.223)** | **4/2/3** | **4 / 2** | 1 / – | – | 16/240 (0.067) |
| deterministic / statue | 40 / 40 | 40 / 40 | – | det 40 | – |

- **Stage counts recounted from its own JSONs.**
  - All 37 stage runs hold seeds 3042-3081 (40 unique seeds each).
  - The stored log_std matches each run's label.
  - The stood counts and causes match its headline exactly.
  - The floors f-2.0 and f-1.75 store uniform vectors.
  - [measured: out/va_stdout.txt section A]
- **rT against the stage's torch JSON.** For -1.5, -2.0 and -1.875, all 40 seeds have the same length and the same end; |Δreturn| ≤ 2.3e-12. Run on my deep copy, the stage's draw gives its numbers. [measured: section C]
- **rT against the REWARD stage's records.**
  - At -1.5 (trex_parent_ls15 + _p2, w 0.2), all 40 seeds have the same length and end, |Δreturn| = 0, and 36/40 end early in both.
  - At -2.0 (trex_parent_ls20), 20/20 seeds match.
  - So "the torch stream reproduces REWARD's 36/40 seed for seed" holds. [measured: out/va2_stdout.txt section 1]
- **Floors.**
  - The largest learned std is 0.1107 (tail_1_yaw, log_std -2.2006), below exp(-2.0) = 0.1353. So max(learned, exp(-2.0)) and max(learned, exp(-1.75)) are uniform.
  - My f-2.0 vP run is identical to u-2.0 vP in 40/40 episodes (length, return and final state).
  - Learned std values: knee 0.0070, hip pitch 0.0127, ankle 0.0125, hip roll 0.0520. [measured]

## Other headline numbers re-measured [measured: out/va_stdout.txt section B]

- **Return per 10 s at w 0.2 / w 2.0.**

  | setting | mine | stage |
  |---|---|---|
  | statue | 1094.7 / 1092.6 | same |
  | deterministic | 1095.9 / 1093.7 | same |
  | own std | 1095.6 / 1093.4 | same |
  | -2.0 | 1086.2-1087.9 / 1070.6-1071.6 (five runs); 1076.4 / 1059.6 (G vP, with the one fall) | 1086.9-1087.5 / 1070.4-1071.2 |
  | -1.875 | 1046-1084 / 1020-1059 | inside the stage's ranges |
  | -1.5 | 314-376 / 303-349 | 348-422 / 321-395 |

  - The -1.5 values overlap the stage's ranges; my lowest (314 / 303) is on fresh seeds.
  - -2.0 against the statue on S: -7.6 at w 0.2 and -21.6 at w 2.0. About 15.6 of the latter is the forward term (forward sum -1.79 against -0.23 at w 0.2, times 10).
- **Displacement over 10 s along the reset prey direction (full-length episodes).**
  - -2.0: -0.215 to -0.231 m, sd 0.036-0.050 (stage -0.226, sd 0.037).
  - -1.875: -0.333 to -0.346 m, sd 0.051-0.064 (stage -0.338, sd 0.050).
  - -1.75: about -0.47 to -0.49 m.
  - Every full-length episode from -2.25 to -1.75 ends behind its start. The most forward ends are -0.043 m (-2.25), -0.066 m (-2.0, G) and -0.203 m (-1.875).
- **Swings (my touch-sensor rule).**

  | setting | episodes with a swing | swings per episode | stage |
  |---|---|---|---|
  | own std | 0/80 | – | 0/120 |
  | -2.5 | 1/40 | – | 1/120 |
  | -2.25 | 15/40 | 0.53 | 65/120, 0.70 |
  | -2.0 | 239/240 | 7.3-7.8 | 120/120, 7.33 |
  | -1.875 | – | 15.7-16.9 | 16.38 |
  | -1.75 | – | 26.6-29.8 | 27.88 |

  - Median swing 0.02 s and median clearance 1.95-2.4 mm, from -2.0 to -1.875.
  - Episodes with a swing of at least 2 cm: 0 of 240 at -2.0 and 0 of 200 at -1.875.
  - L/R alternation is 0.49-0.60 from -2.0 to -1.5 (random is 0.5; a walk is 1.0).
  - The micro-lift picture is confirmed by an independent contact source.
- **Applied std.** The [-1, 1] clip leaves 0.986-1.012 of the set std on every actuator of every stochastic run (stage 0.986-1.014). The filter factor sqrt(α/(2-α)) = 0.489 at α 0.38587 checks as arithmetic. I did not re-measure the filtered std itself. [measured / arithmetic]
- **Arithmetic.**
  - Pooled 1.00 / 0.98 / 0.97 / 0.92 / 0.76 / 0.08 = 120, 118, 116, 110, 91 and 9 of 120.
  - The Wilson lower bound for 116/120 is 0.917, and P(≥36/40 | 0.917) = 0.768 (the stage's "77%").
  - Early ends fall at 1.24-9.89 s, means 4.2-6.5 s. [measured: out/va2_stdout.txt, out/va3_stdout.txt]

## Corrections

1. **"-2.0 is the no-fall start (0/120)" is true only on the seeds 3042-3081.**
   - On those seeds it stood 200/200 over five independent streams.
   - On fresh seeds it fell once in 160: seed 9159, vP, `fallen` at 7.32 s.
   - Both the deterministic parent and its own std stood 40/40 on that panel, so the fall is caused by the noise.
   - Pooled: 359/360. -2.0 still clears 36/40 in every panel by a wide margin, but "no-fall" should read "about 0.3% falls (1/360)". [measured]
2. **The limits sentence "with pooled rates the line falls between -1.875 (0.967) and -1.8125 (0.917)" is wrong against the 36/40 = 0.90 bar.**
   - 0.917 is above 0.90, so on pooled point rates -1.8125 passes. The line falls between -1.8125 and -1.75.
   - The sentence is true only if "passes" means the Wilson lower bound is at least 0.90 (0.917 at -1.875 against 0.853 at -1.8125).
   - With my panels, pooled -1.8125 is 255/280 = 0.911, at the bar. [arithmetic, measured]
3. **The -1.875 / -1.8125 edge is softer than "-1.8125 fails".**
   - -1.8125 stood 37/40 and 36/40 in my two streams on the stage's seeds, and 38/40 on panel F. It failed on panel G (34/40) and in the stage's np2 (35/40), so it passes 5 of 7 panels. Its true rate is about 0.91, which gives P(≥36/40) ≈ 0.7.
   - -1.875 passed all 8 panels (stage 3, mine 5; minimum 38/40). So the stage's rule (pass in every stream) still picks -1.875, and that is confirmed.
   - On the brief's grid, -1.75 failed 7 of 8 panels. One fresh panel reached 37/40, and the pooled rate is 0.79.
   - So "-2.0 is the largest grid value that passes" holds as a rate statement. A single 40-episode panel can pass at -1.75.
   - [measured; inferred for the rates]

No other headline claim failed. The other judgements were not tested here beyond the stand-rate measurements above: the cliff between std 0.135 and 0.22, tapping in place at -2.0 with no step, and -1.875 as the edge. Those measurements are consistent with them. [inferred]

## Checks on method and rules

- **Where the stage set log_std.** It set log_std with `policy.log_std.copy_` on the policy object returned by `load_sb3_checkpoint` (scan2.py lines 87-105). That object is the per-process in-memory load. It is not a separate deepcopy, and nothing was saved. The checkpoint sha256 is unchanged in both their records and mine. My deepcopy route gives the same trajectories (rT matches). I see no rule breach, only a narrower reading of "copy". [measured: code read, sha256]
- **What the stage log_std drives.** It is the only `log_std` parameter of the DiagGaussian actor. Their stored `log_std_used` matches every label. [measured]
- **Device references.** I found no reference to the device directory in the stage's CMDLOG, scripts or RESULT.md. I checked with a list of every redirect target and a search for paths under the system device directory. The stage's own slip list matches its CMDLOG NOTE/SLIP entries. [measured]
- **Size.** The stage folder is 21 MB; mine is 2.6 MB.

## Limits of this verification

- My contact rule (touch sensors, majority of substeps) differs from the stage's floor-contact probe, so I compare lift and swing counts only roughly. I did not run the frozen gait probe or its gait-class labels.
- The filtered (post-filter) std and the per-seed fall-overlap statistic were checked only as arithmetic or by eye (section 3 of out/va2_stdout.txt), not re-derived.
- Like the stage, this is a fixed policy with frozen observation statistics. It says nothing about what PPO does after the warm start.

## Rule slips (all in CMDLOG.txt)

- **Not logged as they ran.** The first `date; ls` (05:12:06Z) and one `tail -2` of CMDLOG ran before or outside the log wrapper. Both are logged after the fact as NOTE lines.
- **Wrong note time.** One NOTE gave a wrong time for writing vlog.py. A CORRECTION line follows it.
- **System python3 used as a text editor.** It inserted run lines into chain2.sh, with no -I or -B and not under nice. No simulation and no JSON read. Logged as SLIP.
- **Uninformative exit codes in chain1.** The `[exit N]` lines of chain1.sh record the status of a `date` substitution, not of the python run, so they are always 0. This was found by a test and fixed in chain2 to chain4. Every chain1 run was checked through its STOOD line and JSON instead.
- **Not under nice.** The harness-side waits and watches did not run under nice: the Monitor `tail -f` watch and the background `until` wait loops. Two foreground waits were moved to the background by the harness's 600 s limit, and one such waiter overlapped a later one. All of them were read-only polling. The harness wrote its own task-output files outside V. I read one of them once (logged).
- **Torch cache.** Torch created `tmp/torchinductor_root` inside V's TMPDIR.
- **A pattern for the device directory.** To check the stage, I used one generic search pattern for paths under the system device directory. It was a generic directory pattern, not the name of the device the rules forbid.
- **Not done.** No training, no repository, export, physics or checkpoint change, no network, no git writes, no writes outside V, and no overlapping simulations.

## Files

- **Scripts** (`scripts/`):
  - `vlog.py`: episodes.
  - `chain1.sh` to `chain4.sh`: sequential queues.
  - `log.sh`: logging wrapper.
  - `va.py`, `va2.py` and `va3.py`: JSON readers and arithmetic.
- **Outputs** (`out/`):
  - `<setting>_<stream>[_F|_G].json` and `*_stdout.txt`: per-run records.
  - `va_stdout.txt`: main tables and checks.
  - `va2_stdout.txt`: REWARD reproduction, arithmetic and per-seed falls.
  - `va3_stdout.txt`: panels pooled across stage and mine, and Fisher tests of panel S against F+G. There is no significant seed-panel effect: p = 0.16 at -1.75 and 0.29 at -2.0.
  - `ckpt_sha256_*.txt`: checkpoint hashes.
