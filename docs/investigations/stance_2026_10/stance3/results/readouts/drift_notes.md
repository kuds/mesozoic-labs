# Whole-body fore-aft drift and the candidate-term landscape (stance3, report only)

Written 2026-10-19 ~02:30-03:30 UTC. Report and scratch only: nothing on Drive or GitHub was touched, no git command ran in
/home/user/mesozoic-labs, and both clones (`fix/screen_s3_L` at 7e6b557d, `fix/screen_s3_LT` at 1af2d4d7) were only
imported through `tools/iso.py`; `git status --porcelain` is empty for both before and after (checked at the end).
Every rollout ran as ONE process at a time (OMP/MKL/OPENBLAS threads 1, torch threads 1, PYTHONDONTWRITEBYTECODE=1,
PYTHONPYCACHEPREFIX and MPLCONFIGDIR set). "Measured" means computed here from files or rollouts; "inferred" is marked.

Paths below are relative to `SP/foot/stance3/night/diag_drift/` (SP = the session scratchpad).

## Files

- `scripts/ledger.py` -> `tables/ledger_<run>.csv`, `tables/ledger_summary.txt` (item 1, eval ledgers, no simulation).
- `scripts/warn.py` -> `tables/warn_summary.txt` (item 3).
- `scripts/roll.py`: an unchanged copy of `diag/r4194304/posture/roll.py` (the judge's env construction: the clone's
  stance stage `[env]` kwargs, `load_sb3_checkpoint` + frozen VecNormalize, deterministic clipped predict, seeds
  3042-3081 in order, heading 0, a balance-mode SubstepContactRecorder). `scripts/run_rolls.sh` ran it sequentially for
  5.24M, 6.29M and 8.39M of all three runs -> `out/<run>_<steps>.{npz,json,log}`, `out/rolls.log`. The 4.2M rollouts
  and the statue are reused from `diag/r4194304/posture/out/`.
- `scripts/terms.py` -> `tables/terms_episodes.csv` (one row per policy episode, 480 rows), `tables/terms_statue.csv`,
  `tables/terms_checkpoints.json`, `tables/terms_summary.txt` (item 2).
- `scripts/slopes.py` -> `tables/slopes.txt` (analytic restoring slopes); `scripts/measured.py` -> `tables/measured.txt`
  (binned costs from the rolled panels); `scripts/split.py` -> `tables/split.txt` (left-right readings vs the K2 flags,
  settle-phase costs of the falls, neck pitch per checkpoint).


## 1. The fore-aft trajectory in the eval ledgers (measured)

Source: each run's `eval_episodes.csv` (refreshed ~01:30 UTC): Run 1 71 evaluations to 9,306,112; Run 2 67 to
8,781,824; Run 3 69 to 9,043,968; 10 episodes each. Means are over full-horizon episodes. The COM is
`com_fore_aft_mean_m` in the foot-site frame (statue -16.1 mm). Leg angles are `leg_q_l_r_mean_*_deg`, offsets from the
home keyframe; the ankle is also given against the settled ankle (1.3713665 rad = home + 1.6735 deg). The CoP split is
the per-episode |r - l| of `<label>_foot_cop_fore_aft`. Per-evaluation tables: `tables/ledger_<run>.csv`.

### Phase means

| run | span | COM mm | ankle vs settled deg | mean abs ankle L-R deg | CoP r / l | CoP split | XcoM heel margin mean / min-of-episodes mm | clean/10 | pad over-bar/10 | K2b-outside/10 |
|---|---|---|---|---|---|---|---|---|---|---|
| Run 1 L s60 | 1.0-2.3M | -6.9 | -1.10 | 1.85 | +0.20 / +0.18 | 0.50 | 81.4 / 75.9 | 9.0 | 1.0 | 2.4 |
| | 3.1-4.3M | -38.8 | +2.42 | 1.49 | -0.26 / -0.25 | 0.12 | 46.7 / 30.4 | 9.9 | 0.1 | 0.1 |
| | 4.4-6.2M | -51.1 | +4.18 | 2.34 | -0.43 / -0.45 | 0.13 | 34.1 / 24.1 | 9.6 | 0.1 | 0.1 |
| | 6.4-7.2M | -0.1 | -2.45 | 1.35 | +0.29 / +0.28 | 0.36 | 75.6 / 49.7 | 9.0 | 1.0 | 2.8 |
| | 7.5-8.4M | -49.1 | +3.24 | 2.50 | -0.40 / -0.43 | 0.26 | 38.4 / 31.7 | 9.7 | 0.3 | 0.0 |
| | 8.5-9.3M | -41.2 | +2.51 | 1.89 | -0.33 / -0.29 | 0.22 | 45.4 / 37.0 | 9.6 | 0.3 | 0.3 |
| Run 2 LT s60 | 1.0-2.3M | -12.6 | -0.46 | 0.27 | +0.13 / +0.09 | 0.43 | 76.6 / 74.1 | 8.7 | 0.4 | 1.4 |
| | 3.1-4.3M | -17.8 | +0.38 | 0.55 | +0.05 / +0.01 | 0.95 | 70.9 / 63.8 | 0.6 | 2.4 | 4.3 |
| | 4.4-6.2M | -21.5 | +0.54 | 0.32 | +0.01 / -0.06 | 0.77 | 66.9 / 60.7 | 1.0 | 1.4 | 2.4 |
| | 6.4-7.2M | -26.7 | +0.55 | 0.31 | -0.13 / -0.07 | 0.51 | 61.7 / 54.8 | 9.5 | 0.5 | 0.5 |
| | 7.5-8.4M | -28.8 | +0.22 | 0.34 | -0.18 / -0.08 | 0.50 | 59.7 / 53.9 | 9.4 | 0.3 | 0.1 |
| | 8.5-8.8M | -29.0 | +0.36 | 0.14 | -0.14 / -0.12 | 0.35 | 59.9 / 56.4 | 10.0 | 0.0 | 0.0 |
| Run 3 LT s52 | 1.0-2.3M | -20.8 | +0.23 | 0.32 | +0.00 / -0.03 | 0.19 | 67.5 / 60.9 | 9.9 | 0.0 | 0.0 |
| | 3.1-4.3M | -33.3 | +1.00 | 0.34 | -0.19 / -0.19 | 0.29 | 54.4 / 46.2 | 9.9 | 0.0 | 0.0 |
| | 4.4-6.2M | -27.8 | +0.70 | 0.29 | -0.09 / -0.13 | 0.39 | 60.7 / 56.8 | 9.2 | 0.3 | 0.2 |
| | 6.4-7.2M | -30.7 | +1.40 | 0.59 | -0.16 / -0.15 | 0.61 | 58.5 / 53.9 | 8.8 | 1.2 | 1.0 |
| | 7.5-8.4M | -39.6 | +2.14 | 1.26 | -0.26 / -0.31 | 0.70 | 48.3 / 40.3 | 6.7 | 3.3 | 1.9 |
| | 8.5-9.0M | -36.7 | +1.94 | 1.21 | -0.21 / -0.27 | 0.70 | 51.6 / 40.0 | 7.0 | 3.0 | 0.8 |

(Run 2's clean count of 0-1 from 2.23M to 6.03M is the tail_3 saturation habit, not posture.)

### Swing amplitude and timing

Turning points of the 3-evaluation rolling-mean COM (reversals of at least 15 mm), and the raw extremes:

- **Run 1 (arm L s60): three swings of 50-65 mm.**
  - Plateau -5.5..-8.5 mm (mean -6.9) from 0.8M to 2.23M.
  - Back 1: from 2.36M to the raw minimum -58.3 mm at 5.77M (ankle +4.97 deg past settled, heel margin 27.0 mm),
    about -51 mm over 3.4M steps. The 8-evaluation OLS slope peaked at -36 mm/M (3.15M), with a 4.46-4.59M dip to
    -56.5.
  - Forward: -58.3 (5.77M) -> +6.4 mm at 6.69M (ankle -3.26 deg, i.e. closed past settled), +64.7 mm in 0.92M steps;
    +26.9 mm in the single evaluation 6.29 -> 6.42M; slope peak +80 mm/M at 6.68M.
  - Back 2: +6.4 (6.69M) -> -54.7 (8.00M), -61 mm in 1.3M steps; -19 mm in the single evaluation 7.34 -> 7.47M; slope
    peak -64 mm/M at 7.86M.
  - Partial return: -36.8 mm at 9.31M (+18 mm).
  - Range after 3.15M 64.7 mm, sd 17.4 mm; |COM - statue| beyond 30 mm in 20 of 48 evaluations; the ankle spans
    8.2 deg (-3.26..+4.97 around settled); largest 8-evaluation (1.05M) change +59.7 mm (5.77 -> 6.82M).
  - The swing is ankle-carried both ways. Across evaluations from 1M, COM = -8.2 mm per deg of ankle (r -0.987). The
    hip and knee co-move: hip -0.80 deg and knee -0.44 deg per ankle deg. The 4.2M foot-rooted sensitivities (ankle
    -10.6, knee -5.3, hip -0.8 mm/deg) predict the per-evaluation COM with r 0.999 and slope 1.08.
- **Run 2 (L+T s60): no swing; a slow creep back.**
  - -9.8 mm (1.84M) -> -30.5 (8.00M); the range after 1M is 20.6 mm, and 14.3 mm after 3.15M (sd 4.6).
  - The 8-evaluation slope never exceeds 7 mm/M in size after 1M. The ankle spans 1.62 deg (-0.74..+0.88 around
    settled).
  - |COM - statue| <= 14.4 mm in every evaluation. Largest 8-evaluation change -8.0 mm.
- **Run 3 (L+T s52): a heel-ward drift, a partial return, then a slow slide.**
  - -18.6 (1.57M) -> -35.5 at 3.67M (ankle +1.2) -> back to -24.3 at 5.64M -> -44.1 at 8.13M (ankle +2.2..+2.3,
    heel margin 42.6 mm). The 6.03-8.13M slide is -18.6 mm over 2.1M steps (slope -5..-9 mm/M).
  - Range after 1M 25.5 mm, after 3.15M 19.8 mm (sd 4.9); ankle span 2.39 deg (-0.05..+2.33 around settled);
    |COM - statue| <= 28.0 mm; largest 8-evaluation change -10.4 mm.

### Does L+T bound the whole-body drift better than arm L? (measured; seed caveat inferred)

Yes for the whole-body COM, by about 3-4 times in range and 6-7 times in the worst 1M change:

| after 3.15M (K5 window) | Run 1 L s60 | Run 2 LT s60 | Run 3 LT s52 |
|---|---|---|---|
| COM range / sd (mm) | 64.7 / 17.4 | 14.3 / 4.6 | 19.8 / 4.9 |
| max abs(COM - statue) (mm) | 42.2 | 14.4 | 28.0 |
| largest 8-eval change after 1M (mm) | +59.7 | -8.0 | -10.4 |
| ankle range around settled (deg) | 8.23 | 0.74 | 2.02 |
| peak abs(8-eval slope) after 1M (mm/M) | 80 | 7 | 11 |
| mean abs(ankle L-R) in its clean span (deg) | 1.76 (0.92-2.23M) | 0.27 (6.16-8.78M) | 0.30 (4.85-6.03M) |

- Runs 1 and 2 share seed 60 (initial weights and reset stream), so their contrast is the cleanest one: the same seed
  swung 65 mm under arm L and crept 20 mm under L+T.
- Run 3 is a different seed. Seed 52 drifted heel-ward under scratch arm L too: -44/-37/-38 mm against the statue at
  4/5/6M (criteria.txt, scratch harness). Run 3's judged offsets were -15.7/-9.3/-11.6 mm at 4.2/5.24/6.29M, but -21.2
  at 8.39M. Smaller, but a cross-harness comparison (inferred).
- L+T bounds the common-mode drift but does not anchor it. Both L+T runs moved monotonically back late:
  - Run 2 by -13 mm from 3.15M to 8.0M;
  - Run 3 by -19 mm from 6.03M to 8.13M.
- What failed in L+T is differential, not common-mode:
  - Run 2's 4.2M K2 failure was the spawn stagger (run log).
  - Run 3's 8.39M K2a failure is a split stance: one foot on its heel, the other mid or forward. It coincides in the
    ledger with the CoP split rising from 0.39 to 0.70 and abs(ankle L-R) from 0.30 to 1.26 deg, while the COM moved
    only about 12 mm.
- Arm L tolerates a larger per-episode L-R ankle asymmetry: about 1.8 deg against 0.3 under L+T, in the clean spans of
  each. Inferred: the narrow 0.05 rad tolerance prices each ankle separately.
- K2 is asymmetric, and Run 1 shows it.
  - Its backward phases (COM -51 mm, both CoPs about -0.44) are K2-clean on the ledger: K2b 0.1/10, pad 0.1/10,
    clean 9.6-9.9.
  - Its forward phase (COM about 0) has K2b-outside 2.8/10 and pad over-bar 1.0/10.
  - From the statue's whole-foot CoP (about +0.04..+0.07 span half-lengths), the signed bar [-0.929, +0.571] leaves
    about 0.99 half-lengths (69 mm) heel-ward and 0.51 (36 mm) toe-ward.
  - So a backward drift stays K2-clean on the panel while the XcoM heel margin erodes: from 81 to 27-34 mm, with a
    minimum over episodes of 20-24 mm.

## 2. Candidate terms on the judged checkpoints (measured from 480 rolled episodes + the statue)

### Reproduction (measured)

All 12 rollouts reproduce their judge files exactly: per-seed length and termination 0 mismatches in 480 episodes,
max |difference| 0 in both per-foot `<label>_foot_cop_fore_aft` and in `com_mtp3_head_m`. The single statue rollout
(L clone) matches the statue rows of all 12 judge files exactly (39/40 full horizon, 3081 falls). Rollout log:
`out/rolls.log`, 02:35-02:52 UTC, one process at a time, ~110 s each. The 4.2M rollouts are the ones in
`diag/r4194304/posture/out` (checked again here).

### Term definitions (window = steps 200-999 of each episode; per-step cost, x800 = per window)

- **anchor**: 0.4 x mean over the two ankles of (1 - exp(-(e/0.10 rad)^2)), e = q_ankle - 1.3713665 rad (the
  recommendation's candidate, paid after step 200).
- **lhp_L / lhp_LT**: the leg_home_pose deficit w (1 - Q) exactly as the clones compute it
  (`environments/tyrannosaurus_rex/base_env.py:171-180, 1046-1053` and `environments/shared/reward_functions.py:401-441`,
  byte-identical in both clones). Q is the mean over the 8 leg joints (r/l hip pitch, hip roll, knee, ankle) of
  0.75 exp(-(e/tol)^2) + 0.25 exp(-(e/(6 tol))^2), with e taken from `_LEG_SETTLED_QPOS`. Arm L: w 0.5, tol 0.10;
  L+T: w 1.0, tol 0.05. It is paid every step; only the window is summed here.
- **band0**: 0.25 (1 - exp(-((x - x0)/25 mm)^2)), x = COM ahead of the mean MTP III along the heading (the judge's
  `com_mtp3_head`), x0 = 63.93 mm (statue panel median).
- **band15**: band0 with a +-15 mm dead band. **band50**: band0 with a 50 mm shoulder.
- **band_d**: the pre-run design band from `foot/tonight/design_incentive/costs.py`: [0, 84] mm free, 25 mm shoulder,
  w 0.25.
- **The statue's own costs**: anchor 0.0006 (0.5/window), lhp_L 0.0024 (1.9), lhp_LT 0.0177 (14.1), band0 0.0038,
  band15 0.0001, band_d 0.

### Per checkpoint (full-horizon episodes; `tables/terms_summary.txt`, `tables/terms_episodes.csv`)

| ckpt | clean | K2a/K2b | noise j/o | ankle vs settled | abs(ankle L-R) | dx mm | anchor (/win) | lhp_L (/win) | lhp_LT (/win) | band0 (/win) | band15 | band50 | band_d |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| R1 4.19M | 38 | 2/2 | 3/2 | +1.80 | 1.77 | -23.0 | 0.051 (40) | 0.044 (35) | 0.230 (184) | 0.140 (112) | 0.031 | 0.049 | 0 |
| R1 5.24M | 38 | 2/2 | 2/1 | +4.67 | 2.86 | -36.1 | 0.191 (153) | 0.088 (71) | 0.383 (306) | 0.211 (169) | 0.123 | 0.101 | 0 |
| R1 6.29M | 39 | 1/1 | 7/1 | +1.67 | 1.60 | -19.5 | 0.043 (34) | 0.025 (20) | 0.151 (120) | 0.113 (90) | 0.013 | 0.036 | 0 |
| R1 8.39M | 38 | 0/0 | 7/0 | +3.12 | 2.44 | -28.3 | 0.113 (90) | 0.044 (35) | 0.228 (182) | 0.178 (142) | 0.065 | 0.069 | 0 |
| R2 4.19M | 0 | 10/14 | 0/0 | +0.25 | 0.48 | -1.5 | 0.003 (2) | 0.004 (3) | 0.027 (21) | 0.006 (5) | 0 | 0.002 | 0 |
| R2 5.24M | 0 | 6/10 | 1/0 | +0.55 | 0.52 | -4.9 | 0.006 (4) | 0.004 (3) | 0.028 (23) | 0.012 (10) | 0 | 0.003 | 0 |
| R2 6.29M | 36 | 2/6 | 0/0 | +0.67 | 0.36 | -8.5 | 0.006 (5) | 0.004 (3) | 0.027 (22) | 0.028 (23) | 0 | 0.008 | 0 |
| R2 8.39M | 39 | 1/0 | 2/0 | +0.18 | 0.26 | -12.5 | 0.001 (1) | 0.008 (6) | 0.060 (48) | 0.056 (45) | 0.000 | 0.016 | 0 |
| R3 4.19M | 38 | 2/2 | 1/1 | +0.49 | 0.20 | -15.9 | 0.004 (3) | 0.003 (2) | 0.022 (18) | 0.084 (67) | 0.002 | 0.025 | 0 |
| R3 5.24M | 40 | 0/0 | 1/0 | +0.63 | 0.40 | -9.2 | 0.006 (5) | 0.003 (2) | 0.023 (19) | 0.032 (26) | 0 | 0.008 | 0 |
| R3 6.29M | 40 | 0/0 | 0/0 | +1.08 | 0.34 | -11.7 | 0.015 (12) | 0.005 (4) | 0.034 (27) | 0.049 (40) | 0 | 0.013 | 0 |
| R3 8.39M | 34 | 6/3 | 0/0 | +2.00 | 0.99 | -21.3 | 0.050 (40) | 0.016 (13) | 0.106 (84) | 0.128 (102) | 0.020 | 0.042 | 0 |

### Do the terms separate failing episodes and checkpoints from clean ones?

**Checkpoints: only within a run, and only where the failure comes with common-mode drift.**
- **Within Run 3** every term rises monotonically into the failing 8.39M: anchor 0.004 -> 0.006 -> 0.015 -> 0.050;
  lhp_LT 0.022 -> 0.023 -> 0.034 -> 0.106.
- **Across runs they invert.** Run 1's checkpoints carry the highest costs (anchor 0.04-0.19 per step) yet pass K2 at
  6.29M and 8.39M. Run 2's K2-failing 4.2M and 5.24M carry the lowest (anchor 0.003-0.006).
- Spearman over the 12 checkpoints against K2a+K2b is -0.16 to -0.35 for every term.
- Against joint+own noise falls it is +0.41 to +0.67, but that is the arm L vs L+T split (inferred). Inside Run 1, the
  costliest checkpoint (5.24M, anchor 0.191) has 2+1 noise falls and the cheapest (6.29M, 0.043) has 7+1.
- The design band band_d is 0 at every checkpoint. It never prices backward drift up to 64 mm.

**K2-flagged episodes against ok ones (full horizon; AUC = P(flagged costs more)).**

| group | flagged / ok | anchor | lhp_L | lhp_LT | band0 | band15 | abs(ankle L-R) |
|---|---|---|---|---|---|---|---|
| all 12 checkpoints | 37 / 436 | 0.33 | 0.38 | 0.38 | 0.19 | 0.33 | 0.71 |
| Run 3 8.39M | 6 / 34 | **0.50** | 0.86 | 0.84 | 0.74 | 0.75 | **1.00** |
| Run 2 4.2 + 5.24M | 24 / 56 | 0.51 | 0.65 | 0.65 | 0.38 | 0.52 | **0.95** |
| Run 2 6.29M | 6 / 34 | | | | | | **0.98** |

- **The failures in this screen are left-right splits, not common-mode drift.**
  - In Run 3 at 8.39M, each of the 6 K2a episodes has one ankle 2.2-3.4 deg past settled and the other 0.1-0.9.
    abs(L-R) is 2.13-2.83 deg, against at most 1.96 in the 34 ok episodes (median 0.68).
  - The heel-loaded foot is the one whose ankle opened more: 4 episodes left heel at -0.86..-0.96, 2 right heel at
    -0.97.
  - Their COM is only a few mm further back than the ok episodes': -25.3 against -20.6 mm.
  - The anchor (a mean over the ankles) is 0.050 against 0.050.
- **Under L+T the signed CoP split tracks the ankle difference.** Run 2: -1.50 per deg, r -0.89; Run 3: -0.67 per deg,
  r -0.87. Under arm L there is no such link (r +0.31), with a 1.6-2.9 deg resting L-R spread.
- Inferred: Run 2's stagger shows up in joint space as this ankle difference (0.47-1.39 deg in flagged episodes,
  against a median of 0.31 in ok ones). That would explain why "both ankles within 0.005 rad of the settled target"
  held on the panel mean while single episodes split.
- **The full leg_home_pose separates better than the anchor or the band** (lhp_LT AUC 0.84 at Run 3 8.39M). It prices
  each joint separately, including hip pitch and roll.

**Falls.**
- The window costs of the 7 panel falls exceed every ok episode's (AUC 0.83-1.00). That is trivial: the window
  contains the collapse.
- In the settle phase (steps 100-199), the anchor already marks Run 3's two lateral 4.2M falls. Seeds 3050 and 3063
  score 0.252 and 0.207, above all 38 full-horizon episodes (median 0.006): the right ankle opens early.
- It does not mark Run 1's five falls: they sit at the 11th-95th percentile of their checkpoint. Two are backward
  tail_contact at 4.2M (window COM -59 and -85 mm); one is tail_contact at 5.24M; the other two (5.24M seed 3063 and
  6.29M seed 3063) end with the window COM 146 and 231 mm ahead.

### Restoring slope at 3-6 deg of ankle (`tables/slopes.txt` arithmetic; `tables/measured.txt` from the episodes)

Per-step d(cost)/d(deg) for a heel-ward (ankle-opening) common-mode drift. Per window, multiply by 800.

| term | 3 deg | 4 deg | 5 deg | 6 deg | mean 3-6 | measured, episodes at 2.5-6 deg (n 81) |
|---|---|---|---|---|---|---|
| anchor (w 0.4, 0.10 rad) | 0.056 | 0.060 | 0.057 | 0.049 | 0.057 (45/win) | 0.051 |
| lhp_L, ankle share | 0.013 | 0.014 | 0.014 | 0.012 | 0.014 (11/win) | |
| lhp_LT, ankle share | 0.047 | 0.028 | 0.013 | 0.006 | 0.022 (18/win) | |
| lhp_L, Run 1 drift path (hip -0.80, knee -0.44 deg per ankle deg) | 0.026 | 0.029 | 0.030 | 0.028 | 0.029 (23/win) | 0.025 |
| lhp_LT, same path | 0.113 | 0.087 | 0.060 | 0.041 | 0.075 (60/win) | 0.086 |
| band0 (25 mm, at -10.6 mm/deg) | 0.054 | 0.020 | 0.005 | 0.001 | 0.016 | 0.023 |
| band15 | 0.091 | 0.070 | 0.032 | 0.009 | 0.051 | 0.045 |
| band50 | 0.045 | 0.044 | 0.037 | 0.027 | 0.039 | 0.025 |
| band_d [0, 84] | 0 | 0 | 0 | 0 | 0 | 0 |

Per mm of COM behind the statue (episodes at 20-60 mm behind, n 149): anchor 0.0075, lhp_L 0.0035, lhp_LT 0.0125,
band0 0.0050, band15 0.0072, band50 0.0041, band_d 0 per step per mm. Across episodes the COM moves -7.4 mm per deg of
ankle (r -0.91).

- **At 3-6 deg the anchor gives about 4 times arm L's ankle-share slope and about 2 times arm L's whole-leg slope.**
- **Along the drift path the hip and knee co-move, and there L+T's whole-leg term is already as steep as the anchor or
  steeper** (measured 0.086 against 0.051 per deg). It fades past about 5 deg: 0.041 at 6 and 0.029 at 7, where the
  anchor keeps 0.049 and 0.038.
- **The anchor's advantage over L+T therefore lies past about 5.5 deg.** No L+T checkpoint got near that: after 1M, Run 2 and
  Run 3 stayed within -0.74..+2.33 deg in every evaluation.
- **The measured 2.5-6 deg slopes come almost entirely from Run 1's episodes** (arm L). Run 3's largest per-episode
  ankle is +3.4 deg.
- **The COM band needs a wide shoulder to matter at these amplitudes.** With a 25 mm shoulder and no dead band it is
  nearly flat past 4 deg (40 mm). The pre-run [0, 84] mm design is flat for any backward drift short of 64 mm (about
  6 deg). It prices only the old forward lean (0.049 at -3 deg, 0.137 at -4 deg).

### Neck guard (measured neck pitch offset from the settled 0.21432 rad, + = head lower; COM worth inferred)

- Run 2's neck rose steadily: -2.2, -3.4, -6.2 and -14.4 deg at 4.2, 5.24, 6.29 and 8.39M, while dx went -1.5 to
  -12.5 mm.
- Run 3's neck went from raised to lowered: -22.6, -10.0, +4.4 and +5.3 deg.
- Run 1's stayed at +0.2..+4.7 deg.
- Using the 4.2M estimate (22.6 deg is about -6 mm, so about 0.27 mm/deg), these neck moves are worth at most about
  7-8 mm of COM (inferred).
- So a COM band can be paid partly through the neck, but only for a small share of a 30-60 mm drift. The guard
  matters more against nosedive and head contact than against COM gaming (inferred).

## 3. Early warning from the eval ledger (measured; lead in evaluations of 131,072 steps)

`tables/warn_summary.txt`. Two views were run.
- **A: baseline-relative.** Each signal is compared with the mean ± 3 sd of a span the judge or ledger calls clean,
  in the bad direction. The count signals use fixed rules instead: clean <= 8, pad over-bar >= 2, K2b-outside >= 2.
  "Confirmed" means two consecutive evaluations beyond the line, which is the earliest a live monitor can act.
- **B: fixed thresholds (PostureTrend-style).** No per-run baseline. Each rule also gets its false-alarm count inside
  clean spans: Run 1 0.92-2.23M, Run 2 6.16-8.78M, Run 3 4.85-6.55M.

### Run 3's slip

Baseline 4.85-6.03M, 10 evaluations, each 9-10/10 clean; judged 40/40 at 5.24M.

The events:
- **E1, 6.82M:** the first evaluation with clean 7 and pad 3.
- **E2, 7.86M:** the slip proper, at clean 6 and pad 4, then 5/5 and 6/4 by 8.13M.
- **J, 8.39M:** the judged K2a 6/40.

| signal (view A) | first beyond / confirmed | lead to E1 / E2 / J (first) | lead to E2 / J (confirmed) |
|---|---|---|---|
| ankle vs settled (> +0.93 deg; baseline +0.73 ± 0.07) | 6.16M / 6.29M | 5 / 13 / 17 | 12 / 16 |
| COM (< -32.5 mm; baseline -26.2 ± 2.1) | 7.08M / 7.21M | - / 6 / 10 | 5 / 9 |
| abs(ankle L-R) (> 0.57 deg; baseline 0.30 ± 0.09) | 6.82M / 6.95M | 0 / 8 / 12 | 7 / 11 |
| CoP split (> 0.67; baseline 0.39 ± 0.09) | 6.82M / 7.21M | 0 / 8 / 12 | 5 / 9 |
| rear-most foot CoP (< -0.45) | 6.82M / 7.21M | 0 / 8 / 12 | 5 / 9 |
| clean <= 8, pad over-bar >= 2 | 6.82M / 7.21M | 0 / 8 / 12 | 5 / 9 |
| XcoM heel margin mean (< 55 mm; baseline 62.6 ± 2.5) | 7.21M / 7.60M | - / 5 / 9 | 2 / 6 |
| XcoM heel margin min (< 45 mm) | 7.73M / 7.86M | - / 1 / 5 | 0 / 4 |

- **The ankle moved first.** It rose from +0.73 to +1.02 deg past settled at 6.16M and +1.17 at 6.29M, then climbed
  steadily to +2.33 at 7.86M.
- **That gives 12-13 evaluations (1.6-1.7M steps) of warning before the slip, and 16-17 before the judged failure.**
  The 6.29M checkpoint, judged 40/40, sits at the start of this climb.
- **Caveat (measured):** the ankle signal alone is not specific. Run 3's ankle already sat at +0.92..+1.23 deg from
  3.0M to 4.06M, and the ledger was 9-10/10 clean then. The 3-sd line is only 0.2 deg above a tight baseline.
- **The direct K2 symptoms all appear at E1, 6.82M:** the split signals (abs(ankle L-R), CoP split, rear-foot CoP) and
  the counts. That is 8 evaluations before the slip and 12 before J.

### Run 1's swings

| swing (event) | what moved first (view A) | lead |
|---|---|---|
| Back 1 (B1: COM first more than 30 mm behind the statue, 3.67M) | COM and XcoM heel margin mean, confirmed 2.49M; ankle and heel margin min, confirmed 2.62M; rear-most foot CoP, confirmed 3.02M | 9 / 8 / 5 evaluations |
| Forward (F: +15 mm in one evaluation, 6.29M) | nothing crossed its 3-sd line before 6.29M | 0 |
| Back 2 (B2: -19 mm in one evaluation, 7.47M) | XcoM heel margin mean (7.21M, confirmed 7.34M); COM and ankle (7.34M) | 1-2 |

- **No count signal warned of a backward phase.** Clean count, pad over-bar and K2b-outside stayed at 9-10, 0 and 0
  throughout both backward phases. Their only alarms in Run 1 came in the forward phase: K2b-outside 3-6/10 at
  6.55-6.95M.
- **The reversals are abrupt.** The forward reversal took the COM from -50 to -8 mm in two evaluations, and nothing
  in the ledger preceded it. Only the slow onset of a drift is detectable early.

### Fixed rules (view B)

| rule | Run 1 first alarm / false alarms in clean span / lead to B1, F, B2 | Run 2 first / false | Run 3 first / false / lead to E1, E2, J |
|---|---|---|---|
| R1 abs(COM - statue) > 20 mm, 2 in a row | 3.28M / 0 of 11 / 3, 16, 6* | never / 0 of 21 | 7.73M / 0 of 14 / -, 1, 5 |
| R2 abs(8-eval COM slope) > 15 mm/M | 2.75M / 0 of 11 / 7, 12*, 8 | never / 0 | never / 0 |
| R3 abs(ankle - settled) > 1.5 deg, 2 in a row | 3.15M / 0 of 11 / 4, 16, 25* | never / 0 | 7.08M / 0 / -, 6, 10 |
| R4 mean abs(ankle L-R) > 0.75 deg, 2 in a row | 1.05M / **10 of 11** / - | never / 0 | 7.21M / 0 / -, 5, 9 |
| R5 CoP split > 0.70, 2 in a row | never / 0 | 3.02M (in its saturation phase) / 0 | 7.21M / 0 / -, 5*, 3 |
| R6 XcoM heel margin mean < 50 mm, 2 in a row | 3.54M / 0 / 1, 16, 25* | never / 0 | 8.00M / 0 / -, -, 3 |
| R7 clean <= 8 in 2 of the last 3 | 1.44M / 2 of 11 | 2.36M / 3 of 21 | 4.72M / 1 of 14 / 16*, 6, 10 |
| R8 pad over-bar >= 4 in the last 3 | 1.44M / 3 of 11 | 2.88M / 0 | 6.82M / 0 / 0, 8, 12 |

\* The alarm had lapsed for more than 2 evaluations before the event.

- **No single rule catches both failure shapes.**
  - The slope rule R2 is the only fast one for Run 1's swings (7-8 evaluations ahead of B1 and B2). It never fires on
    Run 3's slow slide, which never exceeded 9 mm/M.
  - The level rules R1, R3 and R6 catch both runs. On Run 3 they come late: 1-6 evaluations before the slip, 3-10
    before J.
  - R4 (ankle L-R) is clean for the L+T runs but fires constantly on arm L, whose resting asymmetry is 1.8 deg.
- **A combined PostureTrend reading would fire about 8-13 evaluations before Run 3's slip, with no false alarm in any
  clean span here (inferred).** Its parts:
  - the level of abs(ankle - settled) or of abs(COM - statue), as a trend against the run's own last 8 evaluations
    rather than an absolute bar;
  - the 8-evaluation COM slope;
  - the CoP split and abs(ankle L-R), with an arm-specific baseline.
- **Checkpoint selection (inferred):**
  - Prefer a checkpoint whose preceding 8 evaluations show a flat COM and ankle against their own mean (within about
    2 sd) and a CoP split below 0.5.
  - The S3 "late-stable" rule (8 evaluations of 10/10 full horizon with 0 over-bar) picks Run 3's 4.19M, not 5.24M or
    6.29M, because of single 9/10 evaluations at 5.37M, 5.77M and 6.03M. It is over-strict on the count signals and
    blind to the ankle and COM trend.

## 4. Selection hint for walking pilot 2's stance parent (report only; measured from the 12 judge files)

The rule: most clean panel episodes, with ties going to fewer joint-space plus own-sigma noise falls. All 12 judged J
checkpoints, ranked:

| rank | checkpoint | clean | LCB | K2a / K2b | roll | noise joint + own | v2 |
|---|---|---|---|---|---|---|---|
| 1 | **Run 3 stance3_armLT_s52 @ 6,291,456** | **40/40** | 0.928 | 0 / 0 | 0 | **0 + 0** | PASS |
| 2 | Run 3 @ 5,242,880 | 40/40 | 0.928 | 0 / 0 | 0 | 1 + 0 | PASS |
| 3 | Run 2 @ 8,388,608 | 39/40 | 0.887 | 1 / 0 | 0 | 2 + 0 | PASS |
| 4 | Run 1 @ 6,291,456 | 39/40 | 0.887 | 1 / 1 | 0 | 7 + 1 | PASS |
| 5 | Run 3 @ 4,194,304 | 38/40 | 0.851 | 2 / 2 | 0 | 1 + 1 | FAIL |
| 6 | Run 1 @ 5,242,880 | 38/40 | 0.851 | 2 / 2 | 0 | 2 + 1 | FAIL |
| 7 | Run 1 @ 4,194,304 | 38/40 | 0.851 | 2 / 2 | 0 | 3 + 2 | FAIL |
| 8 | Run 1 @ 8,388,608 | 38/40 | 0.851 | 0 / 0 | 2 | 7 + 0 | PASS |
| 9 | Run 2 @ 6,291,456 | 36/40 | 0.786 | 2 / 6 | 0 | 0 + 0 | FAIL |
| 10 | Run 3 @ 8,388,608 | 34/40 | 0.725 | 6 / 3 | 3 | 0 + 0 | FAIL |
| 11-12 | Run 2 @ 4.2M and 5.24M | 0/40 | 0 | 10/14, 6/10 | 0 | 0, 1 | FAIL |

- **The strongest J checkpoint so far is Run 3 (stance3_armLT_s52) at 6,291,456.** It is clean on 40/40, has no noise
  falls, K2a 0, K2b 0 and roll 0, and offset x_c -11.6 mm from the statue (score_8388608.txt).
- **It is a non-final checkpoint, from a run that has since failed K2 (at 8.39M).** The pre-registered rule names the
  best FINAL checkpoint; this hint is report-only, over J checkpoints judged so far.
- **Caveat (measured, item 3):** 6.29M is where Run 3's ankle began to open (+1.02 deg past settled at 6.16M, +1.17 at
  6.29M, against a +0.73 baseline). 5.24M, the runner-up (1 joint-noise fall), sits inside the flat baseline, and its
  candidate-term costs are below 6.29M's (item 2). If the pilot is sensitive to a parent that is already trending, 5.24M
  is the conservative pick; the stated rule picks 6.29M.
- **Neck (measured, item 2):** Run 3's neck habit changed. It was raised 22.6 deg against the settled neck at 4.2M,
  -10.0 deg at 5.24M, and +4.4 deg (head slightly lower) at 6.29M.

## Implications for the next round (inferred unless stated)

1. **The common-mode fore-aft drift is already bounded by L+T, which is config-only.**
   - Measured: on seed 60, the COM range dropped from 64.7 to 14.3 mm after 3.15M, and the worst 1M change from
     +59.7 to -8.0 mm.
   - The ankle anchor (w 0.4, 0.10 rad) would add a slope of about 0.05-0.06 per step per degree at 3-6 deg. L+T's
     whole-leg term already supplies a comparable slope along the measured drift path: 0.086 against 0.051 per deg.
   - The anchor's real gain is past about 5.5 deg, which no L+T checkpoint approached.
   - So "add the anchor" (decision line 3b) is unlikely to fix what failed K2 in this screen.
2. **What failed K2 under L+T is a left-right foot split:**
   - Run 2: spawn stagger, kept.
   - Run 3 at 8.39M: one foot on its heel, its ankle opened 2.2-3.4 deg; the other foot mid or forward.
   - The joint-space image of the split is the L-R ankle difference. It separates the K2-flagged episodes at
     AUC 0.95-1.00 (measured), where the anchor and COM band score 0.38-0.74.
   - A next fore-aft term should price that difference, either (a) or (b):
     - (a) a split anchor on q_ankle_r - q_ankle_l around 0, or on the fore-aft stagger of the two feet;
     - (b) a per-foot CoP term on each foot's whole-foot CoP (a signed K2b-shaped band per foot).
   - Either needs code and moves the task digest. Mirror symmetry (mirror_kl) does not remove per-episode splits.
3. **The COM band as designed before the runs ([0, 84] mm) has zero slope for any backward drift short of 64 mm.**
   - It would not have priced one step of Run 1's or Run 3's heel-ward drift (largest per-episode drift 51.6 mm).
   - A centred band needs a wide shoulder (50 mm) or a dead band of about 15 mm plus a 25 mm shoulder to keep a slope
     at 30-60 mm.
   - It is blind to the splits.
   - The neck moves seen here (up to 28 deg) are worth at most about 8 mm, so a neck guard is cheap insurance, not
     the main issue.
4. **Arm L's large, abrupt reversals were not preceded by any ledger signal.**
   - The forward swing took +27 mm in one evaluation.
   - Only the slow onset of a drift is detectable early: 8-9 evaluations ahead for Run 1's first backward drift, and
     12-13 for Run 3's ankle before its slip.
   - PostureTrend should combine three readings, as trends against each run's own last 8 evaluations:
     - a level reading: the ankle against settled, the COM against the statue;
     - a slope reading: the 8-evaluation COM slope, beyond 15 mm/M;
     - a split reading: the CoP split and abs(ankle L-R).
   - It should not alarm on counts alone. The clean count and pad over-bar are direction-blind and lag; they stayed at
     10/10 through Run 1's 50 mm backward phases.
5. **K2's signed bar gives about 2 times more heel-ward room than toe-ward room from the statue's CoP.**
   - Backward drift therefore passes K2 while the XcoM heel margin halves (Run 1: 81 -> 27 mm).
   - If K3 joint-noise robustness matters, the heel margin is a better posture readout than K2b for backward
     drift.
   - Within Run 1, though, the checkpoint COM did not order the noise-fall counts (5.24M: most backward, 2/20;
     6.29M: mid-swing, 7/20). So the K3 failure is not explained by COM position alone.
   - Measured from the judge files' noise blocks: under joint-space noise the surviving full-horizon episodes stand
     forward of the deterministic panel on the same 20 seeds. The median x_c rises 19-31 mm in Run 1 (e.g. 6.29M:
     43.2 -> 62.1 mm), 13-18 mm in Run 2 and 9-14 mm in Run 3. Joint noise is not a neutral perturbation of the
     fore-aft posture (survivor selection is part of it; inferred).
6. **Checkpoint choice:**
   - Run 3 at 6,291,456 is the strongest J checkpoint by the stated rule.
   - 5,242,880 is the conservative alternative: identical clean count, 1 joint-noise fall, before the ankle trend.
   - Neither is a final checkpoint, and Run 3 has since failed K2.

## Checks at the end

- Clone status: see the last lines of this file (git -C <clone> status --porcelain, read-only).
screen_s3_L: head 7e6b557d porcelain-lines 0
screen_s3_LT: head 1af2d4d7 porcelain-lines 0
