# Offline battery: r9 arm L symmetry-screen checkpoints

**This is an offline evaluation of uncertified scratch-recipe checkpoints. It is a report only, certifies nothing, and is not gate evidence.**

Citation prefixes (all paths are absolute):
- `B/` = `scratchpad/foot/battery/`
- `H/` = `scratchpad/foot/next_signals/howclose/`
- `N/` = `scratchpad/foot/next_signals/`
- `V/` = `B/verify_skeptic/` (the verifier's re-derivations)

## 1. What was run

**Checkpoints.** The r9 arm L symmetry screen (MirrorKLPPO with scratch `experiment_*` keys), seeds 42, 52 and 60, at 6,291,456 steps and at robust_best (s60 524,288; s42 655,360; s52 917,504).

**Not gate evidence.** They were trained under task_sha256 a2af3c5c…, but these certificates roll bde44225…, so the judge would refuse them [H/evidence_table.txt l.93; B/fresh_seeds/offline_certificates.txt].

**Setup.** Read-only main 2e734e29 (physics r9), with arm L's reward-only options off, and no training.

| Part | What | Outputs |
|---|---|---|
| Fresh seeds | 8 six-heading certificates from the earlier `run_cert.py`: s60 and s52 at 6M and robust_best, on seeds 5042-5081 and 9042-9081 | B/fresh_seeds/ |
| Shared fall | Seed 7047 replayed at six headings: s60 6M, s52 6M and the statue | B/shared_fall/ |
| Push | main's impulse probe as shipped, plus a 16-seed copy with a fine grid and forward/backward rows; s60/s52/s42 6M and the statue | B/push/ |
| Own sigma | Heading-0 v2 panels (seeds 3042-3081) with actions sampled from each policy's own Gaussian, beside deterministic re-runs | B/own_sigma/ |

**Checks.** A verifier re-derived the claims [V/]. Its corrections are applied, and refuted claims are dropped (see the last section).

**Resources.** Outputs total 77 MB [B/report/disk_usage.txt]. Each part ran at most 2 heavy processes, but push and own-sigma overlapped, so for a while 4 ran at once [B/own_sigma/logs/queue.log].

## 2. Fresh-seed certificates

| Checkpoint | 3042 | 5042 (new) | 7042 (audit) | 9042 (new) | Pooled clean seeds | 95% CI | P(one 40-seed panel passes) at rate [CI low] |
|---|---|---|---|---|---|---|---|
| s60 6M | PASS 40/40 | PASS 40/40 | PASS 39/40 | PASS 39/40 | 158/160 = 0.9875 | 0.9556-0.9985 | 0.998 [0.899] |
| s52 6M | PASS 38/40 | PASS 37/40 | PASS 38/40 | PASS 39/40 | 152/160 = 0.9500 | 0.9039-0.9782 | 0.862 [0.456] |
| s60 robust_best | PASS 40/40 | PASS 40/40 | not run | PASS 40/40 | 120/120 = 1.000 | 0.9697-1.000 | 1.000 [0.968] |
| s52 robust_best | PASS 39/40 | PASS 40/40 | not run | PASS 40/40 | 119/120 = 0.9917 | 0.9544-0.9998 | 1.000 [0.892] |
| s42 6M | FAIL 29/40 | not run | not run | not run | — | — | — |

Sources:
- B/fresh_seeds/fresh_seed_summary.txt and .json (`pooled.*`), plus H/offline_certificates.txt for 3042 and 7042. Intervals are exact Clopper-Pearson.
- Last column: B/report/panel_pass_prob.json. It treats seeds as independent and ignores the fall rail, so it is an illustration only.

**The pass replicates.** s60 6M, s52 6M and both robust_best checkpoints pass both fresh panels. s42 6M was not re-run.

**No fresh falls.** Every policy episode reached the horizon, and each seed's label was the same at all six headings [V/vfresh.out].

**Pad CoP is the bar that binds.** Every unclean fresh seed failed pad CoP fore-aft (above 0.80).

**s52 6M on 5042 is at the minimum passing count.** Its LCB of 0.8174 is above the 0.80 bar, but 36/40 would give 0.7856 [V/vfresh.out].

**Seed 9062.** The statue falls on it. Both 6M checkpoints are unclean there, and both robust_best checkpoints are clean [B/fresh_seeds/fresh_seed_summary.txt].

The pooled rates are not a gate statistic.

## 3. The shared fall on seed 7047

**Figure: B/shared_fall/shared_fall_7047.png** (heading 0; all six headings repeat exactly).

| Event (control steps) | s60 6M | s52 6M | statue |
|---|---|---|---|
| Hips vs home at step 41 (joint angle, deg, R/L) | +2.4 / +0.8 | +0.3 / +0.1 | −3.8 / −3.7 (sags) |
| Pad CoP first on the toe edge, both feet | 5 | 6 | 5, but only 4% of the time |
| Share of steps 11 to the end with the CoP on the toe edge | 93% | 95% | 4% (heel edge 7%) |
| XCoM past the pad toe line for good | 14 | 18 | never (min margin +0.034 m) |
| XCoM leaves the pad + digit-tip support for good | 89 | 124 | never (min +0.019 m) |
| Topple growth rate vs omega0 (1/s) | 2.54 vs 3.46 | 2.36 vs 3.44 | — |
| End | head_contact, step 173 | nosedive, step 212 | stands, 1000 steps |
| Largest leg force fraction; friction use (shear / mu·N) | 0.20; ≤0.39 | 0.18; ≤0.42 | 0.08; 0.09 |

Sources: B/shared_fall/shared_fall_summary.json; V/vfall.out.

**Mechanism.**
1. **The reset leaves the animal tipped forward.** After step 1 the pads are about 5-6 deg toe-down, and the policies' XCoM is already at or 1 mm past the pad toe line (0.0697 and 0.0689 m, against 0.0687 m). Of the 40 audit seeds, this is the most toe-down and most forward start.
2. **The statue survives by folding.** Its hips sag and its heels come down.
3. **The policies hold the hips at or above home instead.** The pads stay at least 3 deg toe-down, the CoP pins on the toe edge, and the passive digits are the only support left.
4. **The topple.** The XCoM creeps out, and the body topples like an inverted pendulum.

It is the only fall either policy has on seeds 7042-7081. No actuator saturates, and nothing slips.

## 4. Push-and-release

**The probe as shipped cannot tell the roles apart.** main's grid (0.5/1/2 m/s) starts above every role's envelope. Every sagittal row falls, lateral rows survive at most 2/8, and main's verdict reads "Neither side recovers from anything" [B/push/push_summary.json `repo_probe_rows`; B/push/repo_probe/*/stance_gate_probe_impulse.txt].

**The 16-seed copy reproduces main exactly** on 104/104 shared episodes per role [push_summary.json `crosscheck_telemetry_vs_repo_probe`].

| Role | Forward | Backward | Left | Right | Parked pad CoP, R/L (+ toe) | b_pad fwd / back (m) | omega0·b fwd / back (m/s) |
|---|---|---|---|---|---|---|---|
| s60 6M | 0.05 / 0.20 | 0.30 / 0.35 | 0.35 / 0.40 | 0.30 / 0.35 | +0.31 / +0.36 | 0.046 / 0.104 | 0.16 / 0.35 |
| s52 6M | 0.35 / 0.40 | 0.10 / 0.15 | 0.20 / 0.35 | 0.25 / 0.35 | −0.39 / −0.36 | 0.098 / 0.048 | 0.33 / 0.17 |
| s42 6M | 0.15 / 0.15 | 0.30 / 0.35 | 0.30 / 0.40 | 0.35 / 0.40 | +0.31 / +0.41 | 0.052 / 0.096 | 0.18 / 0.33 |
| statue | 0.15 / 0.25 | 0.10 / 0.30 | 0.35 / 0.45 | 0.35 / 0.40 | +0.16 / +0.19 | 0.067 / 0.075 | 0.23 / 0.26 |

Direction columns are envelopes in m/s: all 16 seeds survive / at least 8 survive. b_pad is the distance from the CoP to the pad edge. Source: push_summary.json `roles.<role>` (`envelope_*`, `margin_arithmetic`, `control.pad_cop_fore_aft_signed_median_per_foot_steps_200_499`).

- **The weak direction follows the parked CoP.** The toe-loaded roles are weak forward; the heel-loaded s52 is weak backward.
- **The arithmetic agrees in size, not seed by seed.** Observed over predicted is 0.78-1.18 at the median, but strict per-seed agreement is only 2-13 of 16 [V/vpush_c.out].
- **At the envelope the CoP reaches the pad edge.** Every sagittal faller's XCoM crossed it (616/616) [V/vpush_c.out].
- **Lateral envelopes are about 0.6 of omega0·b for every role, the statue included.** The cause was not tested. Nobody steps [B/push/push_report.txt §6c-d].
- **No policy beats the statue in every direction.** At 8/16 the statue's weakest direction is the best of all the roles. At 16/16, s42 beats it and s52 ties.
- **The certificate is blind to this.** s60 passes the pad-CoP bar easily (median 0.36, worst 0.78) [H/cert_margins.txt], yet it has the smallest forward margin. The bar bounds |CoP|, not which side it sits on.

## 5. Own-sigma panels

| Checkpoint | Mean leg std | Stochastic falls | Clean of 40, stochastic / deterministic | Pad CoP median, stochastic / deterministic | Pad CoP worst, stochastic / deterministic | Main stochastic failures |
|---|---|---|---|---|---|---|
| s60 6M | 0.045 | 0 | 39 / 40 | 0.549 / 0.363 | 0.803 / 0.780 | pad CoP, 1 seed (3080) |
| s52 6M | 0.036 | 0 | 38 / 38 | 0.504 / 0.399 | 0.929 / 0.997 | pad CoP, same 2 seeds |
| s42 6M | 0.043 | 0 | 31 / 29 | 0.679 / 0.688 | 1.000 / 0.997 | pad CoP 8, corner lift 4 |
| s60 robust_best | 0.152 | 0 | 0 / 40 | 0.873 / 0.455 | 0.909 / 0.624 | settle force 40 (median 2.36 BW), pad CoP 40, window drift 8, sole contacts 1 |
| s52 robust_best | 0.114 | 0 | 0 / 39 | 0.819 / 0.492 | 0.919 / 0.801 | settle force 40 (1.94 BW), pad CoP 32 |
| s42 robust_best | 0.136 | 0 | 0 / 40 | 0.841 / 0.327 | 0.873 / 0.613 | settle force 40 (2.17 BW), pad CoP 40 |

Sources: B/own_sigma/own_sigma_report.txt, own_sigma_summary.json and panels/*/panel.json; V/vsigma.out. The settle-force bar is 1.5 BW.

**The stochastic policy never falls.** The only drift is s60 robust_best: 8 window-displacement failures, which trip the hop_or_fall rail.

**At 6M the verdicts hold.** But s60's pad CoP median rises by 0.18, which costs it seed 3080.

**robust_best is clean only when run deterministically.** Its own leg noise is about 3 times the 6M noise. The deterministic re-runs match the certificate cells row for row.

## 6. How close: robust or on the rail?

**Robust where the certificate looks.** The passes replicate on fresh seeds with no falls. s60 6M loses only 2 of 160 seeds, and on one of them the statue falls too.

**On the rail elsewhere:**
- s52 6M is at the minimum count on 5042. At its pooled rate, about one fresh panel in seven would fail.
- Both 6M audits on 7042 sit on the hop_or_fall rail: one more fall would fail them [B/fresh_seeds/fresh_seed_summary.txt].
- The policies park the CoP off-centre and have no more push margin than the statue.
- On seed 7047, that posture loses a seed the statue survives.
- robust_best depends on deterministic actions.

**Verdict.** s60 6M is a real but thin pass; s52 6M is on the rail.

## 7. What it suggests next

The maintainer's rule of 2026-10-16 puts any reward change out of scope. That includes a CoP-centring term, re-weighting, the lateral-lean term, and committing arm L's options [N/recommendation.md l.95; N/questions/inventory.txt l.4]. Nothing below changes reward, task or gate, and each code item needs a PR and maintainer approval.

1. **Report signed pad CoP** beside |pad CoP|.
2. **Extend main's impulse probe**, report-only: add forward and backward rows and a 0.05-0.5 m/s grid.
3. **Add a report-only own-sigma panel beside robust_best.** How selection uses it is the maintainer's call.
4. **Run more offline panels for s52 6M if its pass matters.** s42 6M has no fresh panels.
5. **Treat seed 7047 as a reset-noise tail.** Reset noise is the maintainer's task decision.

## Verifier corrections applied

| Claim | Correction |
|---|---|
| s52 5042 "LCB exactly at the bar" | 0.8174 is above the bar; the run is at the minimum passing count |
| 7047 spawn "XCoM 3 mm short of the toe line, speed 0.044" | Dropped. These were the statue's values after step 1, mixed with the policies' speed. The policies are at or past the toe line. |
| "Both falls end inside the settle window" | Dropped; only s60's does (s52 ends at step 212) |
| Hip posture compared servo targets with the statue's joint angles | Now compared on joint angles |
| Push matches "seed by seed" | Agreement in size only. The s42 example is moved to its envelope, and the statue comparison is qualified to 8/16. |
| robust_best leg noise "3-4x" | About 3 times; one s60 robust_best sole-contacts failure added |
| Reward/statue ratios | Not used; main's statue reference averages only full-horizon episodes |
