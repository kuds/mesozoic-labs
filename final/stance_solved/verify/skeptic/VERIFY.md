# VERIFY: the "quality skeptic" note (`ST/final/stance_solved/skeptic/NOTE.md`)

Read only. No simulation, training, Drive, network or git write. My probes are `fall_seeds.py` / `.out` and `thresholds.out`, all from existing files. The full reasoning is in `NOTE.md` here.

**Verdict: the numbers are sound. Three claims are wrong, a few wordings are overstated, and one important blocker is missing.**

**Confirmed against their sources**
- Every figure I checked matches its source.
  - Head heights and the shares of episodes below 0.90 m; s60's 9 falls in 910; the K3 fall counts.
  - The s80 tail_2 series; the action-noise values; the gate margin table; the actuator maxima.
  - Returns; foot slide; unclean-episode counts.
- **Citations:** SCORE l.321/324/325/329/332/337/341/347/357/361/362; RUNLOG l.585/594/601/607/620; GQP l.516/700; BRP l.1319; BR `train_base.py` l.872; BR `stance.toml` l.315-318; BR `reward_functions.py` l.648-668; KNOWN_ISSUES l.1324-1347.

**Wrong**
- **"Maintainer has not ruled on head height."** The maintainer ruled on 2026-10-25 (D-D46) that head posture is report-only, with no gate change [RUNLOG l.607 (2); KNOWN_ISSUES l.1336-1338].
- **"#491 not found in MAIN's docs."** Four MAIN docs cite it: REFERENCE_MICRODUCK l.57, TREX_STAGE1_BOUNCE l.5/182/194, SIM_TO_REAL_PLAN l.186.

**Overclaims**
- **"Nothing would hold the head up in walking."** The walking config penalises nose-diving and ends episodes on head contact [locomotion.toml l.10, l.21-22].
- **"Three things block naming a parent."** The only formal block is the absent M4 record [SCORE l.361].
  - s60's late fall rate is 3/280, within the gate's tolerance: at most 1 hop-or-fall in 40 and 0.95 full horizon [M3 thresholds].
  - The body itself rates s80's tail_2 and the head MINOR.
- **"More robust to command noise than the statue."** The two tests are unmatched.
  - Knee and ankle joint sigmas are 0.066-0.082, below the statue test's 0.10 [k3_panel.json].
  - The own-noise test runs at std about 0.06, close to deterministic, and the OFF runs also pass it 0/240.
- **"Closing everything needs CPU only."** The tail_2 endpoint needs GPU, as the note's own body says.

**Missing**
- **No warm start as built.** The code refuses a warm start from a PHASE checkpoint into any stage that declares no phase input, which includes walking. Main cannot load these checkpoints [BR `train_base.py` l.667-676]. The fix is code only (Option B).
- **Screen recipe.** The stance was trained on the screen branch's recipe. Main has no r10 stance seed [MAIN `stages.toml` l.33].
- **Noise.** PHASE action noise (0.057-0.060) is below OFF's (0.065-0.070) and below pilot 1's parent (0.095), which read STILL.
- **Joint-noise falls.** s60's 7/240 is comparable to OFF s60's and OFF s42's 6/240.
- **s80 tail_1** is also rising, from +0.006 to +0.0415.
- **Head lowering is paid for.** The neck-posture term costs about 3-5 per episode, so the head is lowered by choice, not by drift alone.
- **Main's own warning.** KNOWN_ISSUES l.1264 (HIGH) says stance gates admit chatter and marching stances, which supports the "floor" point.
