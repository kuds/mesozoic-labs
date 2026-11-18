Verdict: sound_after_fixes

The analysis is sound once the fixes below are made. Every PLEM citation was checked against its file and line, and they hold apart from the items listed. Recommendation C follows from the facts, and none of the fixes reverses it. Notes are in /tmp/claude-0/-home-user-mesozoic-labs/2dca1c00-13f3-5853-acf6-8c3e3e04feb6/scratchpad/plem/check/REVIEW_NOTES.md. The corrected document is in /tmp/claude-0/-home-user-mesozoic-labs/2dca1c00-13f3-5853-acf6-8c3e3e04feb6/scratchpad/plem/check/PLEM_FIT_corrected.md. The original PLEM_FIT.md was not overwritten.

Factual errors:
1. **Orin Nano Super price.** "$249 list / $399 street" is wrong. NVIDIA's list price has been $399 since July 2026 (project compute_2026-11-07.json line 195; CNX Software, Notebookcheck and Hackster, Jul 2026). The AGX Orin's $3,499 (was $1,999) is confirmed by the same sources.
2. **Shell test count.** There are 8 shell test scripts, not 9: two in tools/test and six in setup/rt/test.
3. **JetPack sessions.** "Three [sessions] on JetPack 7.2.1" is wrong. hardware-evidence.md:6 says all sessions ran on JetPack 7.2.1. The last three used the devkit's built-in port.
4. **The clone.** "This clone has a single commit" describes a depth-1 shallow clone (.git/shallow exists). a9a5d12 is a documentation commit made after the v0.11.0 release, and no tags were fetched.
5. **2 ms drives.** "A drive that only accepts a 2 ms cycle cannot run under PLEM" is too strong. The bus cycle must be 200–1000 µs, but schema §5 (cia402-drive-model.md:854-856) allows the SYNC0 cycle to be a multiple of it. Only 1 ms has run on hardware (ethercat_slave_spec.hpp:114-116).
6. **Option B total.** The line items in section 4.1 add to +$3,309–3,459, so the total is +$3.3–3.5k, not "+$3.3–3.6k".
7. **Drive prices.** $269 and $428 (the $3,864 total) are MyActuator's own US-store prices. Third-party US retail is about twice that: RobotShop lists the X4-36-E at $603–620 and the X6-60-E at $742–760 (web, 18 Nov 2026), about $7.8–8.0k for the twelve. This applies to A and B alike, so the comparison stands, but the absolute budget is at risk. Flagged.
8. **Orin Nano Ethernet.** An unverified recollection was replaced with forum evidence. The port is a Realtek RTL8111/8168 running NVIDIA's r8168 driver. The wizard's native-driver list names r8169, so as shipped it would pick the generic path.
9. **maxon HEJ 70.** Corrected to 50 N·m peak and 1.05 kg (HEJ 70-48-50). maxon's family page lists 30, 62 and 140 N·m at 0.57, 1.05 and 2.00 kg. The original "63 N·m, rev 1.2" was not reproduced, and CiA 402 support is not confirmed.
10. **Synapticon.** STO is standard, and Safe Motion over FSoE is an option. Torque and mass are now given per Circulo size (Circulo 12: 40/110 N·m, 1.5–1.87 kg).
11. **RobStride.** The table said "CAN only". It speaks the MIT protocol and, per its firmware notes, CANopen, with no EtherCAT found. RS03 (0.88 kg) and RS04 (120 N·m, 1.42 kg) were added.

Overstatements, now relabelled or softened:
12. **Delay sensitivity.** "Loop delay is not what limits [the quadruped]" is now marked as a hypothesis. PR #669 itself sweeps 0–10 ms of command delay (PDF p.4) and asks for p99 tick-to-torque delay (p.5). The gait is 1.5 strides/s with a 167 ms swing, and static stability is planned, not shown.
13. **Pro licensing.** "Pro is licensed per device" is now marked as inference.
14. **Pilot cost.** "The pilot costs no critical-path time" now holds only if the pilot is time-boxed and staffed apart from the Teensy firmware work.
15. **The T. rex.** The recommendation now says the flagship inherits the same blockers: no brake support, AGX Orin only, 8 axes per bus. A disable stop is a fall for a biped, so the flagship case is open, not presumed.

Citations and completeness:
16. **IgH version.** 1.6.13 is at README.md:156-157 and install.md:15, not README.md:17-20.
17. **Velocity limits.** They are required on every axis, not only on position and velocity axes.
18. **Stale-command watch.** On a bus with no torque axis it runs only while the bus is released (getting-started.md:272-273).
19. **Full test suite.** It passed in two hardware sessions (hardware-evidence.md:256-258 and 274-275), not one.
20. **Telemetry contents.** The list now includes the module serial number, temperatures, GPU load and drive fault codes, and the IP address that WIM's server logs can record (telemetry.md:17-18, 27).

Added evidence:
21. **New findings added.**
    - The add-drive skill is told to stop on a brakeless drive (SKILL.md:71-74).
    - A controller manager running 50 ms late latches a protective stop (getting-started.md:629).
    - A code comment puts the hardware E-stop "at the drive/STO level" (control_mode_flags.hpp:27-28).
    - PLEM's only position unit is motor-encoder counts × gear ratio (schema 261-262). This adds a question for MyActuator about what 0x6064 counts.
    - Jetson supply risk: devkits were reported out of stock after the July 2026 price rise.
    - PLEM has a public drive-support issue template.

Could not verify:
- MyActuator's EtherCAT PDF: cdn.robotshop.com is outside the session allowlist, so only search extracts were used.
- PLEM's GitHub stars, issues and contributors (GitHub not queried).
- The ₩/$ exchange rate.
- Whether wim-product-page.pages.dev is WIM's official page. Its ₩7.9M "Controller Only" figure was confirmed in a search extract.
