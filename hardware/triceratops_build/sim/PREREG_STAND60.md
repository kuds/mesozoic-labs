# Pre-registration: four-foot standing check, 60 cm build (option C)

Written 18 Nov 2026, **before** `check_stand60.py` was run on the 60 cm models. Committed with the models, the
generator and the script in one commit. Any rule changed after a result will be labelled "changed after a result" here
and in the report. The rules follow `PREREG_STAND.md` (the 75 cm check), changed only where the build differs.

**Disclosure.** Before writing these rules I had seen the generator's printout (`build_model60.py`): total mass 11.310 kg
(listed) and 12.441 kg (closed); centre of mass x +29.1 / +26.3 mm; a lever estimate of the front share,
(x_com + 260 mm) / 520 mm, of 55.6 % and 55.1 %. I had also checked that the models compile, that no parts touch at the
neutral pose (0 contacts) and that the soles start 7.2 mm above the floor, and I had looked at two rendered stills.
I had not simulated the 60 cm models.

## What is run

- Models: `generated/trike60_C_{listed,closed}.xml`, built by `build_model60.py` at this commit, mujoco 3.10.0.
- Seeds, perturbation, command, duration, settled window: as in `PREREG_STAND.md` (seed 0 reported; seeds 1 and 2
  judged, every hinge ±0.02 rad uniform and the base 5 mm higher; all servos held at zero for 10 s at a 1 ms step;
  means over the last 1 s).
- Budget: 6 runs of 10 s simulated time. Stop rule: if the script errors, fix the script and rerun all runs.

## Rules

| Rule | What | Pass if |
|---|---|---|
| R1 | Build | Model compiles; total mass equals the 60 cm parts list's totals (listed 11.310 kg, closed 12.441 kg: the generator's mass ledger, which the 60 cm parts list `PARTS_60CM.md` will carry) within ±0.01 kg |
| R2 | Stays standing | As in `PREREG_STAND.md`: root height ≥ 0.90 × start, tilt ≤ 5°, no NaN, no MuJoCo warning; root speed RMS < 5 mm/s in the window |
| R3 | Four loaded feet | Every foot ≥ 10 % of the weight; the four sum to the weight within ±2 % |
| R4a | Front share | Front feet carry 50-60 % of the weight |
| R4b | (dropped) | The 75 cm rule compared with the parts list's independent as-built estimate; the 60 cm build has none yet. The lever estimate above is reported, not judged |
| R4c | Left-right balance | Left feet carry 50 ± 2 % |
| R5a | Continuous load (fail rule) | Every leg joint's settled torque ≤ its drive's thermal-equilibrium torque at the joint: X4 roll and pitch 7.5 N·m; hind knee (X4, 1:1) 7.5 × 0.90 = 6.75 N·m; front elbow (X6, 1:1) 12 × 0.90 = 10.8 N·m |
| R5b | Against the scaled 75 cm figures (flag, not a fail) | Closed variant: front flexion ≤ 1.15 × 4.03 N·m and roll ≤ 1.15 × 2.14 N·m, the parts list's T20 four-foot standing figures (8.1 and 4.3 N·m at 20 kg) × (12.441 / 20) × 0.8 (torque ∝ weight × leg length). Listed: the same × 11.310 / 12.441 |
| R6 | No saturation | No actuator at ≥ 99 % of its force limit in the window |
| R7 | Head and tail servos | Neck pitch torque ≤ 0.63 N·m (XC430-W240, one third of its 1.9 N·m stall at 12 V, an estimate; web search 18 Nov 2026) and tail pitch torque ≤ 0.50 N·m (XL430-W250, one third of 1.5 N·m) |

**Verdict per variant:** PASS if R1, R2, R3, R4a, R4c, R5a, R6 and R7 hold for both judged seeds. R5b is a flag.

## What this check does not cover

As in `PREREG_STAND.md`: walking, peak torques, speed limits, heat over time, delay and noise. The 60 cm part masses
and positions are planning estimates; nothing here is measured.

## Amendment 1 (18 Nov 2026, after the first run and the independent check): revision 2 of the models

**Changed after a result.** The first run (above; `results/stand60_2026-11-18/`) judged revision 1 of the models
(commit 6739087d). The independent check then found errors in the model and one rule that could not fail. Revision 2
of `build_model60.py` changes the models as listed in its docstring and in `REPORT_60CM.md`. It is judged again here,
in `results/stand60_2026-11-18b/`, by the same script with the changes below. The first run's verdicts stay on record;
they describe revision 1.

**Disclosure.** Before writing this amendment I had seen revision 2's generator printout: 11.540 kg listed and
12.694 kg closed; centre of mass x +26.5 / +27.7 mm; lever estimate of the front share 55.1 % / 55.3 %. I had also
seen its packaging check (`check_layout60.py`: no intersecting parts in the bay; flexion drives at least 17.5 mm from
the other drives and 26.3 mm from the body proxy over the joint grid). I had not simulated revision 2.

**Rules changed:**
- **R1 becomes a regression check** (it compared the model with its own ledger, so it could not fail unless the
  build is not repeatable): total mass equals the ledger, 11.540 / 12.694 kg, within ±0.01 kg.
- **R1b, new flag (not a fail):** the closed total lies within the budget's option C estimate, 11.8-12.2 kg
  (`budget/size_budget_out.txt`). Revision 2's 12.694 kg will flag; the report explains it.
- **R5b's reference figures** scale with revision 2's closed mass: front flexion 8.1 × (12.694 / 20) × 0.8 = 4.11 N·m
  and roll 4.3 × (12.694 / 20) × 0.8 = 2.18 N·m (× 1.15 as before); listed × 11.540 / 12.694.
- Everything else is unchanged.

**Corrections to the text above (no rule affected):** the disclosure's "soles start 7.2 mm above the floor" is the
sole centres' height; the sole bottoms touch the floor at the start. The judged seeds settle to the same pose (to about
1e-8), as at 75 cm, so they differ only in the first seconds; the two seeds test the transient, and the settled loads
are in effect one result.

## Amendment 2 (18 Nov 2026, after the recheck of revision 2): revision 3 of the models

**Changed after a result.** The recheck found the ankle springs scaled from the wrong reference (T20's 20 kg, not PR
#669's own 15 kg model) and the Pilz relay block intersecting two parts (the packaging check had skipped it). Revision
3 corrects these (`build_model60.py` docstring). Revision 2's standing results (`results/stand60_2026-11-18b/`) stay on
record. Revision 3 is judged by the same rules as amendment 1 (the totals are unchanged at 11.540 / 12.694 kg; the
reserve's growth factor changes from × 1.2146 to × 1.2088 because the servo cases now grow too), in
`results/stand60_2026-11-18c/`.

**Disclosure.** Before writing this, I had seen revision 3's generator printout (totals as above; centre of mass x
+26.5 / +27.6 mm) and its packaging check (no intersections, Pilz block included). I had not simulated revision 3.
