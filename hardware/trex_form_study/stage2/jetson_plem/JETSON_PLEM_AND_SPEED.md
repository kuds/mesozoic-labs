# Stage 2 study: Jetson Orin Nano Super + Partial PLEM and walking speed ceilings (`60-N-T-10` and `75-M-H-10`)

**Simulation and verification only; nothing is adopted into the Round 2 baseline (`round2/`), per P14 and P24, and none of the hash-locked Stage 1 modules (`designlab/stage1/`) are modified (`STAGE1B_PLAN.md` decision 5).**

Generated from `stage2/jetson_plem/jetson_plem_study.py` (`stage2/jetson_plem/jetson_plem_study.json`). Both baseline builds (`60-N-T-10` in `round2/refine/narrow_carpet_60/candidates/60-N-T-10.json`; `75-M-H-10` in `round2/refine/feet_75/candidates/75-M-H-10.json`) are rebuilt from their Round 2 records (`designlab/stage1/refs.py`) and verified against their committed MJCF `xml_sha256`, Stage 1 `SHA256_PREFIX`, single-support static holds, double-support static torques, and nominal walk RMS requirements (`max_rel_diff <= 1e-9`). Automated regression checks live in `tests/test_stage2_jetson_plem.py`; itemized prototype costs for the recommended `75-M-H-10` + Jetson configuration live in `stage2/jetson_plem/BUILD_BUDGET_75_JETSON.md`.

---

## 1. Jetson Orin Nano Super + Partial PLEM packaging, shell, and power verification

Replacing the Round 2 Raspberry Pi 5 + AI HAT+ (`rpi5_8gb_ai_hat_plus_13tops`: `85 × 56 × 30 mm`, `0.100 kg`, `9.5 W` w/ USB SSD) with the catalog NVIDIA Jetson Orin Nano Super Developer Kit (`nvidia_orin_nano_super_devkit`: `103 × 90.5 × 34.77 mm`, `0.174 kg`) inside `designlab/pipeline.py` (`build_candidate`):

### 1.1 Internal chest packaging and exterior shell envelope

| Metric (`jetson_plem_study.json`) | `60-N-T-10` (Drop-in 110 mm PDB) | `60-N-T-10` (94 mm PDB Fix) | `75-M-H-10` (Drop-in 110 mm PDB) |
|---|---:|---:|---:|
| **Same-body internal hardware overlaps** | `1` (`power_distribution` vs `left_hip_roll_motor`) | **`0`** | **`0`** |
| **Minimum PDB clearance (`min_pdb_gap_mm`)** | `-6.95 mm` (interference) | **`+10.00 mm`** | **`+20.70 mm`** |
| **Crown height (`crown_mm`)** | `600.72 mm` (`+0.00 mm`) | `600.72 mm` (`+0.00 mm`) | `752.71 mm` (`+0.00 mm`) |
| **Exterior torso shell width (`torso_width_mm`)** | `126.84 mm` (`+0.00 mm`) | `126.84 mm` (`+0.00 mm`) | `201.16 mm` (`+0.00 mm`) |
| **Overall robot width (`max_width_mm`, across thighs)** | `274.37 mm` (`+0.00 mm`) | `274.37 mm` (`+0.00 mm`) | `313.44 mm` (`+0.00 mm`) |
| **Raw DevKit + tray mass delta over Pi 5** | `+80.1 g` | `+74.2 g` (`94 mm` PDB) | `+78.7 g` |

- **`75-M-H-10` (75 cm Hybrid, `H = 0.5019 m`):** Drops straight into the existing chest compute bay with **`0` same-body hardware overlaps** (`min_pdb_gap_mm = +20.70 mm` between `hw_power_distribution` and `hw_compute_tray`) and **`0.00 mm` exterior torso or robot width change**.
- **`60-N-T-10` (60 cm Tapered, `H = 0.4047 m`):** Because `designlab/build.py` places the tail-base `power_distribution` box `10 mm` behind the longer `120 mm` Jetson compute tray, keeping the default `110 mm` 24 V PDB shifts it rearward into `hw_left_hip_roll_motor` by **`-6.95 mm`** (while `torso_width_mm = 126.84 mm` and `max_width_mm = 274.37 mm` remain unchanged). Shortening the 24 V PDB bay from `110 mm` to **`94 mm`** (`PDB_SHORT_LEN_M = 0.094`) clears every trunk component with **`0` overlaps** (`min_pdb_gap_mm = +10.00 mm`).

### 1.2 `6S -> 12 V` DC-DC buck regulation across `21.0–25.2 V`

The Orin Nano Super carrier (`P3768`) accepts `9–20 V` DC, so it cannot tie directly to a `6S` LiPo (`21.0 V` cutoff to `25.2 V` full charge; `< 27 V` clamped QDD regen bus). Two catalog Pololu step-down regulators (`designlab/data/parts.json`) cover both JetPack 6.2 power modes with wide input-voltage and thermal headroom:
- **`15 W` mode (+`2.5 W` NVMe/fan = `17.5 W` avg):** `pololu_d36v50f12` (`12 V / 4.5 A / 54 W`, `50 V` max input, `7 g`, `$39.95`) runs at **`33.3 %`** peak rated load across the full `21.0–25.2 V` swing.
- **`25 W` Super mode (+`3.5 W` NVMe/fan = `28.5 W` avg):** `pololu_d42v110f12` (`12 V / 9 A / 108 W`, `60 V` max input, `15 g`, `$69.95`) runs at **`26.4 %`** peak rated load across both `15 W` and `25 W` modes.

### 1.3 Partial PLEM real-time architecture, net mass, and battery runtime

Full WIM-Corporation PLEM targets multi-bus humanoid hardware (`x86`/Orin + EtherCAT/multi-CAN), whereas a 14-DoF T. rex (`12` CAN QDD leg drives + `2` serial expression servos) benefits directly from **Partial PLEM**:
1. Patching JetPack 6.2 Linux with **`PREEMPT_RT`** and isolating **`CPU 5`** (`isolcpus=5 nohz_full=5 rcu_nocbs=5`) for a lock-free `500–1000 Hz` C++/Rust state estimator, safety barrier, and `R2`/`R5` controller.
2. Driving the 12 MyActuator CAN QDDs directly from `CPU 5` over a galvanically isolated USB-CAN FD interface (`pcan_usb_fd`, `0.5 W`), eliminating the unwritten `4.0 W` Teensy 4.1 serial-bridge hop (`COMPUTE.md` bench test 8; `DECISION_LOG.md` F7) in favor of a lightweight `1.5 W` pelvis IMU + hardware E-stop watchdog (`2.0 W` total vs `4.0 W`).
3. Communicating between `CPU 0–4` + Orin GPU (TensorRT FP16 perception, OSNet re-ID, depth occupancy) and `CPU 5` over lock-free POSIX shared-memory ring buffers on a single `CLOCK_MONOTONIC` clock.

| Avionics Stack (`jetson_plem_study.json`) | Regulator | Net Mass Delta vs R2 | `60-N-T-10` Walk Power | `60-N-T-10` Runtime (`3300 / 5000 mAh`) | `75-M-H-10` Walk Power | `75-M-H-10` Runtime (`3300 / 5000 mAh`) |
|---|---|---:|---:|---:|---:|---:|
| **Round 2 Baseline (`Pi 5 + AI HAT+` + Teensy)** | *(internal 5 V)* | `0.0 g` | `69.32 W` | `47.56 / 72.06 min` | `89.77 W` | `36.73 / 55.64 min` |
| **Jetson `15 W` + Teensy 4.1 Bridge** | `pololu_d36v50f12` | `+81.0 g` | `77.34 W` | `42.63 / 64.58 min` | `97.79 W` | `33.71 / 51.08 min` |
| **Jetson `15 W` + Partial PLEM (`CPU 5` + `PCAN-USB FD`)** | `pololu_d36v50f12` | **`+17.0 g`** | **`75.07 W`** | **`43.92 / 66.54 min`** | **`95.51 W`** | **`34.52 / 52.30 min`** |
| **Jetson `25 W` Super + Partial PLEM** | `pololu_d42v110f12` | `+25.0 g` | `86.90 W` | `37.94 / 57.48 min` | `107.34 W` | `30.71 / 46.53 min` |

Every configuration clears `BEHAVIOR_SPEC.md` §4's `15–20 min` continuous walking requirement on `lipo_6s_3300` (`73.26 Wh`, `75 %` usable = `54.95 Wh`) by **`1.5–2.2x`**.

---

## 2. Walking speed envelope and safe maintainable top speed (`60-N-T-10` vs `75-M-H-10`)

Safe maintainable speed is bounded by the **lowest** of five independent physical and control ceilings:
1. **Actuator back-EMF corner speed** across the `6S` LiPo discharge (`25.2 V` full charge, `24.0 V` screen w/ `1.15x` planning margin, `21.0 V` depleted cutoff).
2. **Continuous `A1` thermal duty** (`0.57 × rated`, including MJCF joint damping/friction on the `STS3215` tail and `STS3250` / `XM540-W270` neck).
3. **Single-support floor traction `mu_req`** (`F21`: `0.90` clean wood/tile, `0.54` smooth vinyl).
4. **Closed-loop dynamic stability under realistic actuation delay** (`0–10 ms` via `designlab/stage1/speedfit.py`, `STAGE1B_SPEED.md` §4).
5. **Autonomous obstacle stopping inside the Chest `VL53L5CX` (`15 Hz`) ToF floor window** (`0.843 m` usable on 60 cm, `1.010 m` on 75 cm after `0.10 m` obstacle standoff; skull `OAK-D` camera cannot see the floor closer than `2.40 m` ahead of the toes).

### 2.1 Speed ceilings summary (`Standard V0` vs `F25 + F9` speed-tuned + counter-phased tail)

In the baseline `V0` gait (`designlab/gait.py`), `tail_amp = +0.12 rad` swings in phase with leg angular momentum instead of opposing it (`stage2/research/research_look.md` §3.1 `F9`), and shallow crouch (`0.03 H`) forces high mid-swing knee angular velocity. Applying `F25` hard-floor gait tuning (`crouch = 0.07 H` on 60 cm / `0.05 H` on 75 cm, `lift = 0.081 H`, `duty = 0.55` at `>= 0.36 m/s`, `stride_rel` anchored at `GAIT_TUNE.md`'s `0.713 / 0.863`) plus **`F9` counter-phased tail (`tail_amp = -0.125 rad`)** cuts `mu_req` by `25–36 %`, eliminates `75-M-H-10`'s `STS3215` tail `A1` thermal overload (`1.030 -> 0.413`), and raises drive speed ceilings while clearing all mid-swing pad-height and joint-range gates:

| Speed Milestone / Ceiling | `60-N-T-10` (60 cm Tapered) | `75-M-H-10` (75 cm Hybrid) | Governing Limit / Physical Mechanism |
|---|---|---|---|
| **Round 2 nominal walk (`Fr = 0.012`)** | **`0.218 m/s`** (`0.78 km/h`) | **`0.243 m/s`** (`0.87 km/h`) | `3/3` closed-loop at `0–6 ms`; clears `mu = 0.54` smooth vinyl (`mu_req = 0.31 / 0.23` w/ `F9`). |
| **Brisk walk (`Fr = 0.025`)** | **`0.315 m/s`** (`1.13 km/h`) | **`0.351 m/s`** (`1.26 km/h`) | Standard `V0` drops to `1/3` on 60 cm (`mu_req = 0.75`); `F25 + F9` reaches `3/3` (`mu_req = 0.55 / 0.39`). |
| **Stage 2 agility bar (`STAGE2_PLAN.md` §1)** | **`0.400 m/s`** (`tuned+F9` ratio `0.771`, `mu_req = 0.683`) | **`0.400 m/s`** (`tuned+F9` ratio `0.723`, `mu_req = 0.413`) | Standard `V0` on 60 cm needs `mu_req = 0.964 > 0.90` (`0/3` closed-loop); `F25 + F9` reaches `3/3` at `0–4 ms`. |
| **Animal pace (`Fr = 0.05`)** | **`0.446 m/s`** (`tuned+F9` ratio `0.804`, `mu_req = 0.809`) | **`0.496 m/s`** (`tuned+F9` ratio `0.751`, `mu_req = 0.572`) | `F9` counter-phased tail cuts 75 cm `STS3215` tail `A1` load from `1.030` (fail) to `0.413` (pass); `3/3` closed-loop even at `21.0 V` and `mu = 0.54`. |
| **Standard `V0` `24 V` planning ceiling (`1.00`)** | **`0.481 m/s`** (`knee` speed; `mu_req = 1.23`) | **`0.531 m/s`** (`knee` speed; `mu_req = 0.87`) | `myactuator_rmd_x4_p36_36_e_l` knee speed (`8.69 rad/s` @ `24 V` w/ `1.15x` margin). |
| **Floor grip ceiling (`mu = 0.90` wood/tile, `F25 + F9`)** | **`~0.48 m/s`** (`mu_req = 0.91` at `0.48 m/s`) | **`~0.65 m/s`** (`mu_req = 0.870` at `0.640 m/s`) | 60 cm binds first on stance-pad yaw spin (`F21`); 75 cm's wider/longer `feet_75` pads hold grip to `0.65 m/s`. |
| **Chest ToF safe obstacle-stop ceiling (worst-case stride)** | **`0.485 m/s`** (Pi 5, `200 ms`) -> **`0.550 m/s`** (Jetson+PLEM, `60 ms`) | **`0.540 m/s`** (Pi 5, `200 ms`) -> **`0.610 m/s`** (Jetson+PLEM, `60 ms`) | `VL53L5CX` `15 Hz` 2-frame debounce + outer floor edge (`0.943 / 1.110 m`) minus `0.10 m` standoff (`designlab/stage1/transitions.py`). |
| **`F25 + F9` `21.0 V` depleted-pack drive ceiling** | **`0.559 m/s`** (`hip_pitch` / `knee`) | **`0.651 m/s`** (`knee`, `0.980` ratio at `0.640 m/s`) | Zero torque-speed clipping from `25.2 V` full charge down to `21.0 V` (`3.50 V/cell`) cutoff. |
| **Unmargined `25.2 V` full-charge burst** | **`~0.645 m/s`** (traction-limited indoors) | **`~0.760 m/s`** (`mu_req = 1.17`, high-grip track only) | Absolute back-EMF corner on a fresh `25.2 V` pack. |

- **Stage 0 physical hardware ceilings vs. closed-loop operating speeds:**
  - **`60-N-T-10` hardware ceiling:** **`0.45–0.48 m/s`** (`1.62–1.73 km/h`) indoors (floor-grip limited at `mu = 0.90`; drive- and thermal-safe to `0.56 m/s`), while closed-loop `R2s` walking remains controller-limited around `0.31–0.40 m/s` pending Stage 2's learned `R5` policy (`STAGE1B_SPEED.md` §2).
  - **`75-M-H-10` hardware ceiling vs. `R2` closed-loop "Shown" speed:** Hardware screens allow **`0.54–0.61 m/s` autonomously** (Chest ToF safe-stop ceiling upgraded from `0.540` to **`0.610 m/s`** by Jetson + Partial PLEM) and **`0.64–0.65 m/s` open-loop drive/traction ceiling** (`21.0 V` back-EMF and `mu = 0.90` grip). Under Stage 1/2's strict 20-episode (`seeds 3042–3061`), 2-controller (`R2/42` and `R2/43`) `tune.measure` `10 s`-window rule at `<= 2 ms` bus delay (`§2.3`), `75-M-H-10` achieves **`0.513–0.516 m/s` (`20/20 · 20/20`)** in straight-line walking (both `mu = 0.90` and `mu = 0.54`) and 2-step stops, while **`0.40–0.45 m/s`** remains the practical all-around agile speed once `~22 N` walking nudges and `90°` turns are combined (`STAGE2_PLAN.md` §1, §5).
- **Expressive head (`-X`, `+0.150 kg` forward, `STAGE2_PLAN.md` S2-2) speed note:** Leg drive speed ceilings change by `< 1 %` on both sizes, whereas passive `STS3250` neck-pitch continuous load on `60-N-T-10-X` rises to `0.90–0.96 × A1` across `0.22–0.56 m/s` versus a comfortable `0.72–0.74 × A1` on `75-M-H-10-X`'s `XM540-W270`.

### 2.2 Closed-loop actuation-delay cliff (`0–10 ms`) and Chest ToF stopping horizon

- **Actuation delay vs. chest weight:** Probing `speedfit.servo_delay` (`0, 2, 4, 6, 10 ms`) on `F25 + F9` closed-loop plants shows **`100 %` (`3/3`) survival at `0–2 ms`** across `0.22–0.52 m/s` on `60-N-T-10` (`R2s`) and `0.24–0.58 m/s` on `75-M-H-10` (`R2`), whereas `6–10 ms` serial-bridge latencies degrade survival sharply at `>= 0.40–0.50 m/s` (`jetson_plem_study.json` `closed_loop_latency_probe_040ms`: `3/3` at `0–2 ms` vs `0/3` and `1/3` at `10 ms`). Injecting `+17 g` (`Jetson + Partial PLEM @ 2 ms`) up to `+397 g` (full perception + compute chest delta) directly into `plant.trunk_mass` confirms **bus latency—not Jetson's chest mass—governs high-speed stability**.
- **Chest ToF safe-stop horizon:** Because the skull `OAK-D` camera cannot see the floor closer than `2.396 / 2.395 m` ahead of the toes (`SCREENING.md` §5), low obstacles (`>= 0.10 m`, `BEHAVIOR_SPEC.md` §3) inside `2.40 m` rely on the chest `VL53L5CX` `15 Hz` ToF window (`0.8428 m` usable on `60-N-T-10`, `1.0100 m` on `75-M-H-10`). Cutting host perception + IPC delay from `200 ms` (`Pi 5 + AI HAT+`) to `60 ms` (`Jetson TensorRT FP16 + Partial PLEM`) saves `140 ms × v` of unbraked travel, raising worst-case stride-boundary safe speed from **`0.485 -> 0.550 m/s`** (`60-N-T-10`) and **`0.540 -> 0.610 m/s`** (`75-M-H-10`), and mean-phase safe speed from **`0.710 -> 0.840 m/s`** and **`0.815 -> 0.900 m/s`**.

### 2.3 `75-M-H-10` 20-episode (`seeds 3042–3061`) closed-loop decomposition: gait/judging vs. Jetson leg control

Evaluating both official Stage 1 controllers (`R2/42` and `R2/43`) over the full 20-seed panel (`tune.measure` `10 s` steady window; `F25_F9_cl` adds closed-loop duty/lift margin `duty = 0.59, lift_frac = 0.095` to `F25 + F9`) separates **simulation/gait effects** from **compute-platform effects**:

| `75-M-H-10` Condition (`20` seeds each, `R2/42 · R2/43`) | Delay | Floor / Plant | Stay Up (`42 · 43`) | Meet Every Limit (`42 · 43`) | Median Speed (`m/s`) | Binding Limit / Takeaway |
|---|---:|---|---|---|---|---|
| **Stage 1 `V0` brisk (`0.351 m/s`)** | `0 ms` | `mu = 0.90` (Pi 5) | `20/20 · 20/20` | `0/20 · 0/20` | `0.360 · 0.361` | Fails **only** in-phase `STS3215` tail `A1` (`1.05x`). |
| **Stage 1 `V0` animal pace (`0.496 m/s`)** | `0 ms` | `mu = 0.90` (Pi 5) | `20/20 · 20/20` | `0/20 · 0/20` | `0.506 · 0.490` | Fails tail `A1` (`1.18–1.31x`), `duty` (`0.546 / 0.518`), and `R2/43` ankle speed (`1.17x`). |
| **`F25 + F9` (`duty = 0.55`, `0.496 m/s`)** | `0 ms` | `mu = 0.90` (Pi 5) | `20/20 · 20/20` | `20/20 · 0/20` | `0.511 · 0.510` | Cures tail `A1` (`0.58x`) & ankle speed (`0.74x`); `R2/43` misses only `duty = 0.542 < 0.55`. |
| **`F25_F9_cl` (`duty = 0.59`) @ `0.400 / 0.450 / 0.496 m/s`** | `0 ms` | `mu = 0.90` & `0.54` (Pi 5) | **`20/20 · 20/20`** | **`20/20 · 20/20`** | **`0.415 / 0.466 / 0.514`** | **Shown on both grips** (`18–20/20` also keep Stage 2 `<= 0.25 m` drift) with **zero hardware change**. |
| **`F25_F9_cl` @ `0.550 m/s`** | `0 ms` | `mu = 0.90` (Pi 5) | **`20/20 · 20/20`** | **`20/20 · 20/20`** | **`0.570 · 0.568`** | `hip_pitch` reaches `0.966x` of `24 V` speed limit (true `0 ms` closed-loop speed ceiling). |
| **`F25_F9_cl` + Jetson `+397 g` trunk mass (`0.496 m/s`)** | `0 / 2 ms` | `mu = 0.90` (Jetson) | **`20/20 · 20/20`** | **`20/20 · 20/20`** | **`0.515 · 0.516`** | `< 0.001 m/s` difference vs. Pi 5 at equal delay: **Jetson does not alter raw motor speed/torque**. |
| **`F25_F9_cl` @ `0.400` & `0.496 m/s` (Partial PLEM `2 ms`)** | **`2 ms`** | `mu = 0.90` | **`20/20 · 20/20`** | **`20/20 · 20/20`** | **`0.416 / 0.515`** | Partial PLEM (`CPU 5` + `pcan_usb_fd`) holds the `2 ms` pass zone (`toe = 22.8–23.6 mm`). |
| **`F25_F9_cl` @ `0.496 m/s` (`4 ms` vs `6–10 ms` bridge lag)** | `4 / 6–10 ms` | `mu = 0.90` | `20·20` / **`0·0`** | `5·1` / **`0·0`** | `0.512` / *falls* | `4 ms` loses toe clearance (`18.9–19.3 mm`); **`6–10 ms` (`S2-7`) collapses `0/20` at both `0.40` & `0.50 m/s`**. |
| **Stage 2 `stop2` (`<= 1.5` strides, `<= 1.5 m/s²`) @ `0.40` & `0.50 m/s`** | `2 ms` | `mu = 0.90` | **`20/20 · 20/20`** | **`20/20 · 20/20`** | `0.67 / 0.73` strides | `0.79 / 0.92 m/s²` peak decel; 2-step stopping passes cleanly up to `0.50 m/s`. |
| **Stage 2 walking nudge (`~22 N`, forward / left) @ `0.40` vs `0.50 m/s`** | `2 ms` | `mu = 0.90` | `20·20` (`15·15` L) -> `20·17` (`13·11` L) | `18·15` -> `19·13` (`9·7` L) | `0.416 / 0.515` | Mid-stride shoves (and turns) degrade above `0.40 m/s`, anchoring Stage 2's `0.40 m/s` all-around bar. |

---

## 3. Behavior-by-behavior impact of Jetson Orin Nano Super + Partial PLEM

| Behavior & Repository Source | Impact | Mechanism & Quantitative Evidence |
|---|---|---|
| **Fast walking (`0.40 m/s` bar, `Fr = 0.05` animal pace, `0.52–0.61 m/s` top end)**<br>`BEHAVIOR_SPEC.md` §1, `STAGE1_PROBE.md` `G2, G3`, `STAGE2_PLAN.md` S2-7 | **Strong Help** (w/ `F25 + F9`) | Partial PLEM (`PREEMPT_RT` + isolated `CPU 5` + direct `pcan_usb_fd`) holds host-to-CAN actuation delay to `~1.5–3 ms`, keeping both robots in the `3/3` closed-loop survival zone instead of falling off the `6–10 ms` serial-bridge cliff. |
| **Walking bump recovery & standing shoves**<br>`STAGE1_PROBE.md` `G1, G11`, `STAGE2_PLAN.md` §5 | **Moderate Help** | Preserves feedback phase margin for step-placement (`capture`) and ankle-roll reflexes and executes Stage 2's `50–250 Hz` `R5` policy in `< 0.4 ms` on `CPU 5` (software gain/policy refit still required for `G1`). |
| **Stage 2 learned policy (`R5`) + character layer blending**<br>`BEHAVIOR_SPEC.md` §2, `STAGE2_PLAN.md` S2-3 | **Strong Help** | Runs `R5` locomotion policy, BD-X-style standing/walking/show policy blending, and Olaf-style real-time servo winding thermal observers (`stage2/research/research_look.md` §2) concurrently without GPU contention. |
| **Selected-person following (`1.5–2.5 m`, `>= 1.0 m` min, `>= 90 %` in-view)**<br>`BEHAVIOR_SPEC.md` §3 | **Strong Help** | `67 TOPS` Orin GPU runs full-res (`640 px`) TensorRT FP16 person + pose keypoints + stereo depth alignment at `30+ Hz` (`~8–12 ms`), resolving truncated close-range legs/torsos (`COMPUTE.md` bench test 5) far better than `416 px` INT8 crops. |
| **Multi-person crossing & occlusion re-ID (`<= 0.5 s` ambiguity stop, `>= 80 %` reacquire)**<br>`BEHAVIOR_SPEC.md` §3, `COMPUTE.md` bench test 4 & 6 | **Transformative Help** | Batched GPU OSNet / CLIP-ReID embeddings + ByteTrack + depth segmentation run in `< 6 ms` for 5–10 people concurrently instead of serializing crops behind the detector on a single PCIe Gen2x1 NPU. |
| **Obstacle avoidance (`>= 0.10 m`) & Chest ToF emergency stop**<br>`BEHAVIOR_SPEC.md` §3, `STAGE1_PROBE.md` `G12` | **Strong Help** | Cuts perception-to-command latency from `~175–200 ms` to `~45–60 ms`, raising worst-case Chest ToF safe-stop speed by **`+0.065 / +0.070 m/s`** (`0.485 -> 0.550` / `0.540 -> 0.610 m/s`) while supporting GPU 3D occupancy mapping (`nvblox`) across the `2.40 m` floor blind gap. |
| **Head attitude & `< 20 ms` camera-pose alignment**<br>`BEHAVIOR_SPEC.md` §3, `COMPUTE.md` bench test 8 | **Strong Help** | Eliminating the Teensy serial bridge timestamps neck encoders and pelvis IMU on the same `CLOCK_MONOTONIC` clock as USB `OAK-D` frames (`< 1–2 ms` skew vs `< 20 ms` requirement). |
| **Expressive character moves: `Look at a person`, `Head-first turn (F11)`, `Gaze-stabilised walk (F10)`, `Roar`, `Boom`, `Snap`, `Startle`**<br>`stage2/research/research_look.md` §5.1 | **Strong Help** | Enables simultaneous 3D face/eye saliency + audio DOA (`Look at a person`), `45–60 ms` path-to-head anticipation (`F11`), `500 Hz` IMU-to-neck gaze stabilization (`F10`), `< 5 ms` RT jaw-audio sync (`Roar` / `Boom` / `Snap`), and `25–40 ms` audio/visual `Startle` reflex. |
| **Floor friction `mu_req`, thigh-twist turn range, sway & carpet (`G4, G5, G6, G7, G9, G10`)**<br>`STAGE1_PROBE.md` §8 | **Neutral** | Governed by cast-PU tread `mu` and `F9` tail phase (`G4`), `±0.30 rad` hip-yaw stops (`G5`), carpet compliance (`G6, G9`), reference ZMP inset (`G7`), and foot pad area (`G10`). |
| **Continuous walking runtime (`15–20 min` target)**<br>`BEHAVIOR_SPEC.md` §4 | **Minor Trade-Off** | Costs `3.64 min` on `60-N-T-10` (`47.56 -> 43.92 min`) and `2.21 min` on `75-M-H-10` (`36.73 -> 34.52 min`) in `15 W` + Partial PLEM mode, remaining **`1.7–2.2x` above** the `20 min` specification. |

---

## 4. Platform and body-size engineering recommendation

1. **Recommended compute stack — NVIDIA Jetson Orin Nano Super (8 GB) + Partial PLEM (`15 W` default / `25 W` Super optional via `pololu_d42v110f12` + `PCAN-USB FD`):**
   - Retires `COMPUTE.md`'s highest software/schedule risks (Hailo-8L multi-crop re-ID serialization, single PCIe lane contention with NVMe, and the unwritten Teensy 4.1 clock-sync/CAN bridge firmware) while adding only **`+17 g`** net mass and **`+$50` (Low) to `+$373` (Nominal)** over the Round 2 Pi 5 + AI HAT+ + Teensy baseline (`< 2 %` of robot cost).
2. **Recommended build size — `75-M-H-10` (75 cm Hybrid, `14.85 kg` base):**
   - Accepts the Orin Nano Super DevKit with **`0` internal hardware overlaps** (`+20.70 mm` PDB clearance, no custom `94 mm` PDB redesign needed), shows **`0.514 m/s` (`20/20 · 20/20`)** closed-loop straight-line walking at `<= 2 ms` on both `mu = 0.90` and `mu = 0.54` (`0.40–0.45 m/s` all-around w/ nudges & turns; **`0.61–0.65 m/s`** Stage 0 hardware ceiling vs `0.45–0.48 m/s` on `60-N-T-10`), carries the `XM540-W270` neck servo needed for Stage 2's `-X` expressive head (`0.71 × A1` vs `60-N-T-10-X`'s tight `0.85–0.96 × A1`), and stays inside `DECISION_LOG.md` `P4`'s `$10k–$20k` prototype budget target (`$17,216` nominal self-assembled / `$20,030` turnkey outsourced; see `stage2/jetson_plem/BUILD_BUDGET_75_JETSON.md`).
