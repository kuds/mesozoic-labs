# The centre-of-mass revisit: its report, the checks on trained policies and the creep scan (2026-10-29)

Evidence for decisions D-D53 and D-D54
([BEHAVIOR_RECIPES_PLAN.md](../../../BEHAVIOR_RECIPES_PLAN.md) §6.2) and for
[TREX_TAIL_SATURATION_HABIT_2026_10.md](../../TREX_TAIL_SATURATION_HABIT_2026_10.md),
where `REPORT.txt` is COMREV, `policies/` POL and `verify_policies/` VPOL.
After the entropy continuation screen's reading
([`../ecs_results/`](../ecs_results/README.md)) the maintainer took up the
centre-of-mass follow-up parked on 2026-10-28 (D-D53): four lenses on a
scratch clone of `main` at `378ae2f8`, with an adversarial verifier for each
measured lens, simulation only, no training; then, at the maintainer's
answers of 17:28 UTC (D-D54), a scan of two more models for the Compsognathus
robot's creep. Exploratory and report only: nothing here is pre-registered,
and nothing changes a gate, reward, plant or criterion. The tail clamp test,
which `REPORT.txt` §4 reports, is in [`../tail_probes/`](../tail_probes/README.md).

[SOURCES.txt](../SOURCES.txt) lists each file's source, its sha256 and the
changes the copy made; [SHA256SUMS](../SHA256SUMS) pins them (every file of
the folder but this README). The sources are in the session's scratch
(`scratchpad2/com_revisit/`): `REPORT.txt`, and each folder here the folder
of the same name under `notes/`.

- **`REPORT.txt`**: the revisit's report to the maintainer (written at about
  17:26 UTC): §1 the absolute stance targets, §2 the Compsognathus robot's
  creep, §3 the centre-of-mass checks on the 24 entropy-screen end policies
  and their two starts, §4 the tail clamp test, §5 the ratchet and #650's
  open items, §6 the decisions the maintainer answered at 17:28 UTC, §7 the
  files. Its two workflow run ids are removed (**redact**); it is otherwise
  byte-identical.
- **`policies/`** (the policies lens) and **`verify_policies/`** (its
  verifier): the 26 policies, deterministic, seeds 3042-3061, with #650's
  reduction, against the neutral-action statue: no same-side lean or creep;
  railed and unrailed policies lean alike; the fore-aft posture gap mostly
  inherited from the starts, the railed lineage's centre of mass about 5 mm
  further back at its ends (11 of 12; `verify_policies/lineage.txt`); the r9
  checkpoint's lean a plant-transfer effect (the lens's explanation
  refuted); one end policy, `s80_A_1321`, falling on seed 3072. `NOTES.txt`
  in each; `policies/PREREG.txt` is the CoM-P1 check fixed before the end
  policies were read, and `policy_com_check.patch` the test helpers that
  carry it (an uncommitted patch, not applied anywhere).
- **`scan/`** (the creep scan) and **`verify_scan/`** (its verifier): the
  anatomical Compsognathus and the Velociraptor, 481 episodes each at the CoM
  tests' and their own stance noise, no plant change: no creep on either
  model (0 of 480 noisy spawns each); the Velociraptor's foot-travel flag at
  its training noise is a slow, decaying relaxation, not a creep. The
  verifier confirms the scan and corrects four of its statements (F2, F3, F4
  and F10, `verify_scan/NOTES.txt`; its outputs in `verify_scan/out/`). The
  scan's per-episode JSONL outputs are not copied; `scan_table.txt` and the
  other tables are.

What each folder keeps (the same rule as `../tail_probes/`): the files at the
lens folder's top level that are text, its notes, tables, scripts, logs and
non-empty error logs, its run markers, its JSON outputs of at most 130 KB and
its JSONL outputs of at most 60 KB, and from a verifier's `out/` folder its
text outputs, logs and run markers, all byte-identical to their sources
(but `REPORT.txt`), absolute session-scratch paths and checkout paths
included. Not kept: every other subdirectory (rollouts, per-episode JSONL,
tracks, the lens's code copy), the
`.npz` files, and the larger outputs (`policies/analysis_w*.json`, about
0.25 MB each, whose `.txt` renderings are kept; `verify_policies/recompute.json`,
5.8 MB; `scan/scan.jsonl` and `scan/fixcheck.jsonl`, per-episode). The targets,
creep and design lenses and their verifiers, which do not bear on the tail
habit, are not copied; `REPORT.txt` §1, §2 and §5 summarise them, and the
centre-of-mass test change of D-D54 (1) is outside this record.

Kept as their sources have them, so to leave out of a manual pre-commit run:
`policies/check_final.json`, `check_m1_held.json` and `check_r9.json` have no
final newline; `policies/analysis_w2.txt`, `analysis_w5.txt` and
`analysis_w9.txt` each have one line ending in a space, and
`policies/policy_com_check.patch` two (its blank context lines).
