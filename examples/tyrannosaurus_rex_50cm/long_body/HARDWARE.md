# Longer medium candidate: hardware dimensions and mass placement

The [hardware manifest](../long_body_hardware.json) separates published part
dimensions from installation space reserved in the [longer candidate](README.md).
The current continuous torso shell also has a [surface check](torso_validation.json)
against the complete installed reserves, including a 3 mm allowance. The original
ellipsoid containers remain hidden planning bounds, separate from the visible
shell and its four convex collision sections around the hip opening.
It replaces this candidate's earlier generic electronics proxies with sourced
part envelopes, explicit quantities and estimated assembly allocations. The
[silhouette-C reference](../README.md) remains unchanged.

Sources were reviewed on 2026-10-26. Dimensions below are full dimensions in
millimeters, in the supplier's order; `catalog_axes` records their permutation
into the owning body's local x/y/z axes. `null` means unknown. A zero tolerance
in the manifest means no additional supplier tolerance is recorded, not that
the part has perfect dimensions. An envelope fit is a packaging screen, not
qualification of a finished assembly.

## Catalog evidence

| Part and quantity | Published dimensions and nominal mass | Evidence and limits |
| --- | --- | --- |
| Jetson Orin Nano Super 8GB P3766 kit, 1 | 103 × 90.5 × 34.77 mm; mass unknown | [NVIDIA December 2024 datasheet](https://dam-cdn.nvd.orangelogic.com/AssetLink/2ug686w80406gxf6q8uv7r3l7265m2n8.pdf), linked from the [official product page](https://www.nvidia.com/en-us/autonomous-machines/embedded-systems/jetson-orin/nano-super-developer-kit/). Includes feet, carrier, module and thermal solution. The older distributor value of 100 × 79 × 21 mm is superseded. NVMe, connected cables and installation airflow remain unqualified. |
| Gens ace G-Tech 3300mAh 6S 60C EC5 battery, 1 | L×W×H: 138±5 × 42±2 × 43±2 mm; 506±20 g | [Exact product](https://genstattu.com/gens-ace-g-tech-3300mah-6s-60c-22-2v-lipo-battery-pack-with-ec5-plug/). The modeled body uses the tolerance maximum, 143 × 44 × 45 mm. Listed discharge leads are 120 mm long; connector, lead routing and restraint need additional space. |
| Teensy 4.1, 1 | PCB: 60.96 × 17.78±0.6 × 1.57 mm; mass unknown | [PJRC drawing](https://www.pjrc.com/teensy/dimensions_teensy41.png). The modeled 60.96 × 18.38 × 15 mm envelope includes maximum recorded PCB width and a **planning** height for components/headers. Full populated height and USB cable routing are unverified. |
| CAN transceiver boards, 2 | Exact part, dimensions and mass unknown | Each 30 × 20 × 15 mm box is a planning reservation. No selected product supports that envelope yet. |
| Pololu D36V50F12, 3; D36V50F5, 2 | Each 25.4 × 25.4 × 9.525 mm; 7 g | [12 V specifications](https://www.pololu.com/product/4095/specs), [5 V specifications](https://www.pololu.com/product/4091/specs). Both dimensions and mass exclude optional headers. Mounting, wiring and thermal clearance remain unqualified. |
| ROBOTIS U2D2, 2 | 48 × 18 × 14.9 mm; 9 g | [Official manual](https://emanual.robotis.com/docs/en/parts/interface/u2d2/). Connected cable envelopes are additional. |
| OAK-D Lite fixed-focus camera, 1 | Width×height×depth: 91 × 28 × 17.5 mm; 61 g | [Luxonis product](https://checkout.luxonis.com/products/oak-d-lite-1). These are enclosure dimensions; bracket, USB connector and cable bend remain unqualified. |
| XM430-W210-T, 2 | W×H×D: 28.5 × 46.5 × 34 mm; 82 g | [Official specifications](https://emanual.robotis.com/docs/en/dxl/x/xm430-w210/#specifications), [drawing](https://www.robotis.com/service/download.php?no=157). The 34 mm body depth excludes protruding horn, idler and fastener geometry. |
| XL330-M288-T, 4 | W×H×D: 20 × 34 × 26 mm; 18 g | [Official specifications](https://emanual.robotis.com/docs/en/dxl/x/xl330-m288/#specifications), [drawing](https://www.robotis.com/service/download.php?no=1986). The 26 mm depth includes a 23 mm case and 3 mm horn; the illustrated optional idler configuration requires 29 mm. Mounting configuration is unresolved. |
| MYACTUATOR RMD-X4-P12.5-10, 12 | Related global revision: Ø55±0.1 × 55.5±0.5 mm axial; 330 g | [Global specification graphic](https://static.wixstatic.com/media/cab28a_2b2f7baa22b54695b5fca3156d882392~mv2.jpg) identifies P12.5-10-C/-E, while the [page heading](https://www.myactuator.com/x4-10details) says P12-10-E. The [selected US listing](https://myactuator.us/products/rmd-x-series/rmd-x4-p12-5-10) could not be verified. This is **related-revision evidence**, not confirmed selected-SKU CAD. |

The candidate retains a conservative **60 mm RMD planning diameter** and
increases axial length from 55 to **56 mm**, covering the related drawing's
maximum axial tolerance. The ankle-roll case centers move **1 mm rearward** for
the **3 mm case-clearance target** while the joint axes remain unchanged. This
placement still needs a real bracket, output support and connector design;
the planning case does not establish those clearances. Nominal 24 V motor
specifications also do not establish compatibility with a fully charged 6S
battery.

## Installation reserves and mass accounting

The following are estimated installed assemblies. Their centers are torso-local
placements in the manifest, not supplier centers of gravity. The battery's
fore-aft position can also be overridden by `redesign.Layout.battery_x_m`.

| Assembly | Reserved x×y×z space, mm | Default center x/y/z, mm | Total mass allocation | Known catalog contribution and residual |
| --- | --- | --- | --- | --- |
| Battery | 160 × 90 × 65 | -170 / 0 / 72 | 550 g | Nominal battery 506 g leaves **44 g** for restraint. At the published 526 g maximum, only **24 g** remains; installed mass is not qualified. |
| Compute | 120 × 110 × 65 | -15 / 0 / 170 | 300 g | Total budget for kit, storage, cooling and mounting. Kit mass is unknown, so a supported residual cannot yet be calculated. |
| Control/interfaces | 110 × 85 × 25 | 112 / 0 / 145 | 180 g | Two U2D2 units consume 18 g, leaving **162 g estimated** for Teensy, CAN, other interfaces, IMU/audio and mounts. A separate 20 g foot-sensor allocation preserves the original 200 g group. |
| Torso power | 100 × 85 × 35 | 112 / 0 / 100 | 300 g | Five regulators consume 35 g, leaving **265 g estimated** for distribution, wiring and mounts. A separate 50 g leg-wiring allocation preserves the original 350 g group. |

The expanded packing check exposed two conflicts hidden by the earlier generic
proxies. The battery assembly center is lowered from 90 to **72 mm** in the
torso frame: its installation reserve ends at **104.5 mm**, below the tail
motor body's **112 mm** lower face, leaving **7.5 mm** with the corrected
motor orientation. The jaw motor's center moves **37.5 mm along positive local
Y** and **7.5 mm below the joint**, clear of the nod motor body while preserving
the jaw joint pivot and axis.

The ROBOTIS drawings put the shaft along catalog depth, index 2, with an offset
within the width/height face. Relative to the case center, the offset along
catalog height is **+12 mm for XM430** (46.5/2 − 11.25) and **+7.5 mm for
XL330** (34/2 − 9.5). The head yaw/nod, jaw and tail case orientations are
corrected, and the case and estimated mass centers, including those of the
shoulder motors, are placed so these shaft lines meet their modeled joints.
The screen checks both shaft direction and perpendicular line offset for all
18 motors. Axial horn placement, brackets, idlers, bearings and mechanical
couplings still need an installation design; matching a shaft line does not
qualify those parts or establish the vendor center of gravity.

Catalog masses **consume these existing assembly budgets**; adding visible
part envelopes does not add their masses again. Motors and the camera retain
their existing individual ledger entries. Residuals are available allocations,
not measured material. No complete assembly has been weighed.

The compute tray's 120 × 110 × 65 mm space is an installation allowance around
the complete 103 × 90.5 × 34.77 mm kit. It is neither a module-only dimension nor
a vendor-approved cooling volume. The control layout explicitly reserves space
for both CAN boards and both U2D2 interfaces; the power layout contains all five
regulators. The internal part boxes are separate from the outer installation
reserves so a small board cannot silently stand in for its whole assembly.

The modeled assembly centers and box inertias remain estimates, including the
distribution of unweighed residual mass. The Jetson's offset inside its tray
does not establish the tray assembly's actual center of mass. ROBOTIS publishes
[XM430](https://www.robotis.com/service/download.php?no=717) and
[XL330](https://www.robotis.com/service/download.php?no=2136) inertia references,
but their CAD origins and axes must be mapped before importing those CG values.
No vendor CG has been substituted into this candidate.

## Regression expectations and remaining work

The [hardware screen](../hardware_checks.py) checks **187 clearance pairs at
790 poses**: six assembly pairs, 76 pairs between four assemblies and 19 mounted
parts, and 105 mounted-part pairs. The 66 RMD-to-RMD pairs remain covered by the
mechanical screen. Sampling includes home, the 729-pose expression grid and
independent five-point sweeps of each leg joint. The
[hardware report](hardware_validation.json) records passing known limits and
all 18 shaft-direction and 18 shaft-line checks. Overall hardware status
remains **incomplete** because the
unverified parts, assembly masses and installation details above are unresolved.

Changes to the manifest, generator or layout should preserve these checks:

- Compare every modeled catalog envelope with the recorded dimensions,
  positive tolerances and axis mapping; reject undersized battery, complete
  Jetson and actuator packages. Keep unknown or related-revision evidence
  visible rather than treating it as exact supplier qualification.
- Check that each mapped motor shaft direction matches its joint axis and
  that the shaft line passes through the joint anchor after the supplier's
  case-center offset is applied. A fitting box alone cannot establish this.
- Reconcile quantities with the BOM: one Jetson, battery and Teensy, two CAN
  boards, two U2D2 units, three 12 V and two 5 V regulators, one camera,
  twelve RMD, two XM430 and four XL330 motors. Verify mass allocations and
  nonnegative residuals without counting catalog mass twice.
- Check each installed reserve inside its declared torso container with the
  configured 3 mm reduced-axis allowance; check contents within the reserve,
  internal overlaps and selected motor/assembly clearances at sampled poses.
  These convex outer shell boundaries are planning containers, not manufactured
  wall-offset surfaces.
- Rebuild after mass or center changes and rerun the mechanical, standing and
  crouch-to-stand reviews. Compare the fixed-scale exterior, service views and
  recorded motion to catch unexpected placement or silhouette changes.
- Exercise negative cases that deliberately shrink a reserve, overlap parts,
  remove a required component, understate a mass budget or alter a catalog
  axis mapping. Such mutations should produce an actionable failure.

The regression expectations above do not establish results for every proposed
negative case. Packing coverage does not include combined leg/expression
motions, continuous paths, deformation, service extraction or cable routing.
Select the CAN/NVMe parts, measure populated board and
assembly masses, resolve motor revisions and mounting configurations, and
design restraints, headers, fasteners, connectors and cooling before treating
these planning reserves as a fabrication layout. Updated simulation evidence
belongs in the [candidate review](README.md).
