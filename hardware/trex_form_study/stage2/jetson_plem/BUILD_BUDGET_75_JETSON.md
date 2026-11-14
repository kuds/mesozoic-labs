# Stage 2 build budget & BOM: `75-M-H-10` (75 cm Hybrid) + Jetson Orin Nano Super + Partial PLEM

**Planning and verification estimate only; nothing is adopted into the Round 2 baseline (`round2/`), per `DECISION_LOG.md` P14 and P24.**

This document itemizes a complete single-prototype build budget for **`75-M-H-10`** (`round2/refine/feet_75/candidates/75-M-H-10.json`, `14.85 kg` base) equipped with the **NVIDIA Jetson Orin Nano Super Developer Kit (8 GB)** and **Partial PLEM** real-time bus architecture evaluated in `stage2/jetson_plem/JETSON_PLEM_AND_SPEED.md` (`stage2/jetson_plem/jetson_plem_study.py` / `stage2/jetson_plem/jetson_plem_study.json`), unified with all adopted `P21` materials, external 24 V power-path, and fall-arrest rig items from `round2/refine/MATERIALS.md`.

## Executive summary (`1` physical robot + bench/safety rig + spares + contingency)

- **In-house final assembly** *(frame plates CNC/laser-cut and skins MJF-printed outside; mechanical fit-up and harness wiring performed in-house)*:
  - **Low:** **`$12,586`** | **Nominal (Target):** **`$17,216`** | **High:** **`$24,077`** *(comfortably inside `DECISION_LOG.md` `P4`'s `$10k–$20k` prototype target)*
- **Turnkey outsourced assembly** *(includes bounded subcontract shop mechanical/electrical integration per `DECISION_LOG.md` `P4` and `designlab/cost.py`)*:
  - **Low:** **`$14,282`** | **Nominal (Target):** **`$20,030`** | **High:** **`$28,215`**
- **Pure Jetson + Partial PLEM delta vs. Round 2 Raspberry Pi 5 baseline (`$18,521` nominal in `75-M-H-10.json`):**
  - Only **`+$50` (Low)** to **`+$373` (Nominal)** (`< 2 %` of total build cost). The remaining `~$1,136` nominal difference vs. `75-M-H-10.json` comes from rolling `round2/refine/MATERIALS.md`'s adopted `P21` upgrades (MJF PA11 tough skins, `LTC4417` external 24 V power path + dummy puck, and overhead fall-arrest tether boom) plus character audio into one unified table.

All component prices and scaling factors come from `round2/refine/feet_75/candidates/75-M-H-10.json`, `designlab/cost.py`, `designlab/data/actuators.json`, `designlab/data/parts.json`, `round2/refine/MATERIALS.md`, and `stage2/x/X_COSTS.md`.

---

## 1. Master build budget summary (`75-M-H-10` + Jetson Orin Nano Super + Partial PLEM)

| # | Subsystem / Budget Line | Key Hardware & Scope | Low | Nominal (Target) | High |
|---:|---|---|---:|---:|---:|
| 1 | **Leg actuators (`12` CAN QDDs)** | `2x` MyActuator `RMD-X6-P20-60-E` (`$428` ea, hip pitch) + `10x` `RMD-X4-P36-36-E-L` (`$269` ea, hip roll, thigh twist, knee, ankles A/B) | `$3,546.00` | `$3,546.00` | `$3,723.30` |
| 2 | **Expression actuators (base head & tail)** | `1x` ROBOTIS `DYNAMIXEL XM540-W270-T` (`$482.89`, neck pitch) + `1x` Feetech `STS3215` 12 V (`$22.00`, tail yaw) | `$504.89` | `$504.89` | `$530.13` |
| 3 | **Compute & NVMe storage** | NVIDIA **Jetson Orin Nano Super DevKit 8 GB** (`nvidia_orin_nano_super_devkit`, `$249` MSRP Low/Nom; `$399` reseller High) + 512 GB–1 TB M.2 2280 NVMe SSD (`$55–$95`) + vibration tray, Wi-Fi antennas & locking DC lead | `$335.00` | `$395.00` | `$599.00` |
| 4 | **Partial PLEM real-time bus & IMU** | Galvanically isolated USB-CAN FD (`CANable 2.0 Isolated` `$65` Low; industrial **`PEAK PCAN-USB FD` `$368`** Nom/High, `pcan_usb_fd`) + pelvis IMU (`BNO085` / `ICM-42688`) + `U2D2` / TTL servo adapter + `1.5 W` hardware E-stop watchdog | `$175.00` | `$498.00` | `$560.00` |
| 5 | **Vision, ToF & foot sensors** | `1x` Luxonis `OAK-D Lite` (`$269` + locking USB3 cable) + `2x` Pololu `VL53L5CX` / `VL53L8CX` (`$40–$50` + mux) + stance-pad contact/load sensing (`$113–$340`) | `$437.34` | `$568.09` | `$759.21` |
| 6 | **Character audio & mic I/O** | I2S Class-D amp + chest bass transducer (`Roar` / `Boom` / `Snap`) + MEMS mic array (`Startle` & growth voice trigger) | `$35.00` | `$65.00` | `$120.00` |
| 7 | **6S power bus, 12 V buck & `P21` DC input** | Custom 24 V PDB + Pololu `pololu_d42v110f12` (`12 V / 9 A / 108 W`, `$69.95`, supports `15 W` & `25 W Super`; or `pololu_d36v50f12` `$39.95` Low) + `LTC4417` external 24 V priority power path, `< 27 V` regen clamp & Cat-1 E-stop (`MATERIALS.md` §6) | `$515.00` | `$740.00` | `$1,120.00` |
| 8 | **Primary battery & tail-base cassette** | `1x` Gens ace `G-Tech 3300mAh 6S 60C` (`lipo_6s_3300`, `$107.19`, `73.26 Wh`) + tail-base quick-swap cassette + `0.678 kg` tethered dummy puck (`P21`) | `$167.19` | `$217.19` | `$277.19` |
| 9 | **Wiring harness & service loops** | High-flex silicone CAN FD & XT60/XT30 leg/trunk harness, neck camera/servo service loop, strain reliefs (`cost.py` `x1.29` size factor) | `$193.02` | `$308.83` | `$450.38` |
| 10 | **6061-T6 frame, bearings, linkages & feet** | Outsourced laser/CNC 6061-T6 twin-plate skeleton (`2.54–3.0 mm`), crossed-roller/angular-contact joint bearings, ankle pushrods, wider/longer `feet_75` soles & cast `50–60 Shore A` PU studded treads (`4.07 kg`) | `$2,947.58` | `$4,503.25` | `$6,550.19` |
| 11 | **MJF PA11 skins, inserts & bumpers (`P21`)** | Tough MJF **PA11** graded-wall (`1.0–1.4 mm`) covers (`1,852 cm^3`), heat-set brass inserts, vapour/dye finish, EVA strike bumpers & joint gaiters | `$1,132.87` | `$1,950.63` | `$3,408.60` |
| 12 | **Bench supply, charger & overhead boom (`P21`)** | Mean Well `RSP-750-24` (`24 V / 31 A`) tether supply, dual 6S smart charger, 5 m `SJOOW 16/2` cable, and overhead fall-arrest gantry/jib boom with `< 0.5 kg` tool balancer | `$850.00` | `$1,450.00` | `$2,500.00` |
| 13 | **Spares kit (`cost.py` standard)** | `2x` spare `RMD-X6-P20-60-E` (`$856`), `1x` spare `6S 3300mAh` pack (`$107.19`), plus spare `X4-36-L` / `STS3215`, pushrod ends, PU tread pads & fasteners | `$1,113.19` | `$1,313.19` | `$1,613.19` |
| — | **Subtotal (self-assembled in-house)** | *All fabricated parts, actuators, electronics, rig & spares delivered* | **`$11,952.08`** | **`$16,060.07`** | **`$22,211.19`** |
| — | **Contingency (`5.3 % / 7.2 % / 8.4 %`)** | *Freight, actuator import duties/tariffs, taxes & minor bracket rev* | `$633.46` | `$1,156.33` | `$1,865.74` |
| | **TOTAL — Self-assembled build** | | **`$12,585.54`** | **`$17,216.40`** | **`$24,076.93`** |
| 14 | **Outsourced subcontract integration (`P4`)** | Bounded contract shop mechanical fit-up, harness routing & bench checkout (+ proportional contingency) | `+$1,695.98` | `+$2,813.68` | `+$4,138.45` |
| | **TOTAL — Turnkey outsourced build** | | **`$14,281.52`** | **`$20,030.08`** | **`$28,215.38`** |

---

## 2. Detailed electronics & actuator BOM

### 2.1 Actuators (`14` base drives; `16` with optional `-X` head)

| Joint Group | Qty | Catalog / Vendor Part | Unit Price | Extended | Notes |
|---|---:|---|---:|---:|---|
| **Hip pitch** | `2` | MyActuator `RMD-X6-P20-60-E` (`myactuator_rmd_x6_p20_60_e`) | `$428.00` | `$856.00` | Confirm V4 `X6-60` vs V4.1 `X6-50-L` availability when ordering (`designlab/data/actuators.json`). |
| **Hip roll (tandem)** | `2` | MyActuator `RMD-X4-P36-36-E-L` (`myactuator_rmd_x4_p36_36_e_l`) | `$269.00` | `$538.00` | Centreline tandem layout (`y_r = 0`). |
| **Hip yaw (thigh twist)** | `2` | MyActuator `RMD-X4-P36-36-E-L` | `$269.00` | `$538.00` | `P18` targeted upgrade over `AK45-36`. |
| **Knee** | `2` | MyActuator `RMD-X4-P36-36-E-L` | `$269.00` | `$538.00` | `8.69 rad/s` planning speed (`0.65 m/s` walk safe w/ `F25 + F9`). |
| **Ankle pitch / roll (`A & B`)** | `4` | MyActuator `RMD-X4-P36-36-E-L` | `$269.00` | `$1,076.00` | Mounted high on calf; parallel pushrods to MTP joint. |
| **Neck pitch** | `1` | ROBOTIS `DYNAMIXEL XM540-W270-T` (`robotis_xm540_w270`) | `$482.89` | `$482.89` | `0.66 × A1` base, `0.71 × A1` w/ `-X` expressive head. |
| **Tail yaw** | `1` | Feetech `STS3215` 12 V (`feetech_sts3215_12v`) | `$22.00` | `$22.00` | `0.41 × A1` at `0.50 m/s` with `F9` counter-phased tail. |
| **Total installed actuators** | **`14`** | | | **`$4,050.89`** | *(Plus `2x X6-60` `$856` in Line 13 spares)* |

### 2.2 Jetson Orin Nano Super + Partial PLEM avionics stack

| Role | Recommended Part | Nominal Price | Rationale |
|---|---|---:|---|
| **Onboard compute** | NVIDIA **Jetson Orin Nano Super Developer Kit (8 GB)** (`nvidia_orin_nano_super_devkit`) | `$249.00` *(MSRP)* | `67 TOPS` CUDA/TensorRT GPU + 6-core Cortex-A78AE (`CPU 5` isolated for RT control). Drops into `75-M-H-10` chest with `0` overlaps (`+20.70 mm` PDB clearance). |
| **NVMe storage** | 512 GB–1 TB M.2 2280 TLC NVMe SSD | `$65.00` | Required for JetPack 6.2 Super mode + high-rate telemetry/bag logging under walking vibration. |
| **6S -> 12 V buck** | Pololu **`D42V110F12`** (`pololu_d42v110f12`, `#5677`, `12 V / 9 A / 108 W`, `60 V` max in, `15 g`) | `$69.95` | `$30` over `pololu_d36v50f12` (`$39.95`) and supports both `15 W` and `25 W Super` modes across `21.0–25.2 V` at `26.4 %` peak load. |
| **Galvanically isolated CAN FD** | **PEAK-System `PCAN-USB FD`** (`pcan_usb_fd`, `IPEH-004022`, `500 V` isolation, `68 g`) | `$368.00` | Mainline Linux `SocketCAN` driver; galvanic isolation prevents 12-motor QDD regenerative ground transients from resetting Jetson USB ports (`$65` isolated `CANable 2.0 FD` in Low tier). |
| **Servo bus + pelvis watchdog** | ROBOTIS `U2D2` + TTL bus adapter + `BNO085` / `ICM-42688` SPI IMU + RP2040/STM32 hardware E-stop supervisor | `$130.00` | Drives `XM540` neck + `STS3215` tail and enforces hardware CAN torque-off if the `500 Hz` Jetson `CPU 5` heartbeat drops. |
| **Head stereo + chest ToF** | Luxonis `OAK-D Lite` (`$269`) + `2x` Pololu `VL53L8CX` `#3419` (`2 × $24.95`) | `$318.90` | `+$10` upgrade from `VL53L5CX` to `VL53L8CX` extends `5000 lux` indoor ambient floor range from `1.7 m` to `2.8 m` (`designlab/data/parts.json`). |

---

## 3. Reconciliation against the Round 2 `75-M-H-10` record (`$18,521` nominal)

1. **Round 2 `75-M-H-10.json` turnkey baseline** (`Pi 5 + AI HAT+`, PA12 skins, before `P21` external power path / tether boom): **`$18,521`**
2. **Swap `Pi 5 + AI HAT+` (`$410` line) -> Jetson Orin Nano Super + NVMe (`$395` at `$249` MSRP):** **`-$15`** *(or `+$109` at `$399` reseller High)*
3. **Add `pololu_d42v110f12` `12 V / 9 A` buck (`+$70`) + industrial `PCAN-USB FD` Partial PLEM bus (`+$318`):** **`+$388`** *(net Jetson + Partial PLEM hardware delta = **`+$373`**)*
4. **Include adopted `P21` items (`round2/refine/MATERIALS.md` §2, §4, §6) + character audio:**
   - MJF PA11 tough skins + EVA strike bumpers: **`+$60`**
   - `LTC4417` external 24 V priority power path + `< 27 V` regen clamp + Cat-1 E-stop + `0.678 kg` dummy puck: **`+$300`**
   - Overhead fall-arrest / tether boom added to bench lot: **`+$600`**
   - Character audio amp/transducer/mic + proportional `7.2 %` contingency: **`+$166`**
5. **Unified turnkey outsourced total:** **`$20,030`** *(or **`$17,216`** self-assembled in-house)*

---

## 4. Optional Stage 2 upgrades menu (not required on Day 1)

| Optional Upgrade | Nominal Delta | Mass Delta | When to Add |
|---|---:|---:|---|
| **Stage 2 expressive head (`75-M-H-10-X`: neck yaw + hinged jaw, `stage2/x/X_COSTS.md`)** | **`+$195`** (`2x STS3215` `$44` + thin-section yaw bearing `$75` + brackets/cables `$76`) | `+0.150 kg` *(+ bearing `~0.04 kg`)* | Once Stage 2 `L1` look review confirms hardware neck-yaw (`Look at a person`, `Head-first turn`) and jaw (`Roar`); `XM540` neck stays at `0.71 × A1`. |
| **Wide-FOV + active IR stereo camera (`Luxonis OAK-D Pro W`)** | **`+$260`** (`$529` vs `$269`) | `+0.030 kg` | For dim museum halls or low-texture corridors (`127°` stereo HFOV + IR dot projector). |
| **Extended-endurance pack (`Gens ace 6S 5000mAh`, `lipo_6s_5000`, `111 Wh`)** | **`+$30` / pack** (`$136.99` vs `$107.19`) | `+0.249 kg` | Extends `75-M-H-10` Jetson `15 W` walking runtime from `34.52 min` to `52.30 min` (`46.53 min` in `25 W Super`); verify cassette length (`165 mm` vs `138 mm`). |
| **7075-T6 high-yield hip & thigh brackets (`MATERIALS.md` §3)** | **`+$120`** | `+0.042 kg` | `1.8x` yield strength over 6061-T6 on high-moment hip/knee plates for repeated fall-testing insurance. |
