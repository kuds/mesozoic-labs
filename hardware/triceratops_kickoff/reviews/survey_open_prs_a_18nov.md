## Review of the six open PRs on kuds/mesozoic-labs (read 18 Nov 2026)

Sources: GitHub PR metadata and descriptions from `pull_request_read get`, plus local `git fetch` of each head with diffs against its merge-base. main is at 0748ac12, the merge of #667. #667 changes only `hardware/trex_form_study/` (2,314 files). The working branch `claude/mesozoic-trex-stage2-refine-4jffre` is at c5274905 (P51). That is newer than the b1d5e98e named in my brief. I posted nothing, merged nothing, closed nothing and wrote no tracked file.

**#585 "Add physical gait reports and opt-in locomotion certification"**
- kuds; `main` ← `codex/gait-certification` (171450b6). Created and last updated 2 Oct 2026. Not a draft.
- 2 commits, 32 files, +4,835/−17, all in `environments/shared/*`, `docs/GAIT_CERTIFICATION.md` and CI.
- What it does: a physics-substep gait recorder (foot force, swing, phase, slip, flight) for all six species, and an opt-in `locomotion_gait/v1` gate with a binomial lower bound. Report only; no recipe adopts it.
- State: `mergeable_state: dirty` (conflicts). Its merge-base is 668 commits behind main.
- On main: the recorder, events and morphology layer reached main in another form. `environments/shared/gait/recorder.py` was added by 0d80e368 (6 Oct, the floor-truth stance gate `stance_quality/v2`, D-D23). The gait judging, `curriculum/gait_gate.py`, `scripts/gait_report.py` and `docs/GAIT_CERTIFICATION.md` are not on main. main's `GAIT_QUALITY_PLAN_2026_09.md` still lists `locomotion_gait/v1` as planned (PR-G5).
- Superseded: #596 contains both of its commits unchanged and says it replaces it. #596 also finds that #585 rejected every genuine gait (compsognathus 0/10, velociraptor 0/10).
- **Recommendation: close as superseded by #596.**

**#596 "Walk-first gait certification, #585 fixes, and the T. rex gait reward"**
- kuds; `main` ← `claude/gait-checker-trex-gait-r1` (1e494323). Created and last updated 5 Oct. Draft.
- 25 commits, 62 files, +14,039/−124.
- What it does:
  - a walk-first checker `locomotion_gait/v2` with `biped_walk`, `quadruped_walk` and `biped_alternating` profiles;
  - fixes P1–P15 and PL-1–PL-7 in #585;
  - a gait reward kit (`gait_rewards.py`);
  - T. rex `gait-r1` locomotion config. Its description says to hold it until a 1.5M-step pilot passes, or revert b0714f9.
- State: `dirty`, 634 commits behind main. It edits paths main has since renamed: `configs/trex/locomotion.toml` and `environments/trex/envs/trex_env.py` are absent on main, which uses `environments/tyrannosaurus_rex/`. Main has the shared floor-truth pieces (recorder, events, morphology), but none of `metrics.py`, `labels.py`, `preflight.py`, `report.py`, `seeds.py`, `gait_gate.py`, `binomial.py` or `gait_rewards.py`. `MEASUREMENT_VERSION` on main is `floor-truth/v2`.
- The T. rex part (`gait-r1`) is moot now that the T. rex is paused. The S2-4 probe on the study branch gave no gain.
- The checker matters for the quadruped. It has a `quadruped_walk` profile, girdle participation and duty/load rails. Its own residuals note "no genuine quadruped" in calibration, and quadruped red-team passes on 3:2 cadence and running bouts.
- **Recommendation: keep parked, then extract the checker.** Do not merge as is. When Triceratops locomotion training starts, port the checker without `gait-r1` onto main's current layout. That is the PR's own fallback path: revert b0714f9 and merge the checker, plumbing and kit. Items to extract:
  - `docs/GAIT_CERTIFICATION.md`, including the known-residuals section;
  - the `quadruped_walk` profile and its tests;
  - the evidence and benchmark tables in the PR description.

  Then close #596.

**#623 "T. rex physics r10: a load-bearing tridactyl foot and the whole-foot stance rail (D-D34)"**
- kuds; `main` ← `claude/trex-foot-r10` (0ee0917d). Created 16 Oct, last updated 17 Oct. Draft.
- 12 commits, 52 files, +2,728/−544.
- What it does: a new T. rex foot (sole box, digits, MTP springs), the `max_foot_cop_fore_aft` stance rail, the `foot_toe_line.py` script, and `MEASUREMENT_VERSION` v3.
- Its own condition: hold until a 3-seed screen shows the r10 foot is at least as good as r9.
- State: `dirty`, 346 commits behind main.
- Superseded on main. The name "r10" went to a different change: "T. rex physics r10: preload the tail pitch springs" (D-D44), merged as #643 on 25 Oct. main's `configs/plant_versions.toml:783-784` reads: "r10 is the counter's next integer, not the foot the parked draft #623 screened under that name (D-D34), which never became part of main." `BEHAVIOR_RECIPES_PLAN.md:1251` says the same.
- **Recommendation: close as superseded** by #643/D-D44 and by the paused T. rex. If a lesson is wanted for the T. rex restart (end 2027 / early 2028), first save the PR description's evidence (V1/V2/V3 push margins, the H4 noise fall counts, the CoP-bar calibration) and the never-landed `docs/investigations/TREX_FOOT_R10_2026_10.md`. They are T. rex anatomical-sim specific and give the Triceratops nothing directly.

**#635 "Add T-Rex concepts with shared validation and medium hardware reconciliation"**
- kuds; `main` ← `codex/mini-trex-forward-knee-example` (398c02c1). Created 24 Oct, last updated 26 Oct. Not a draft.
- 26 commits, 204 files, +166,099/−18.
- What it adds:
  - `examples/` mini, medium (50 cm) and metre (100 cm) T. rex concepts;
  - shared MJCF, render and validation utilities;
  - medium hardware reconciliation: Jetson kit 103 × 90.5 × 34.77 mm per the description, RMD case length 56 mm, an 8.207 kg allocation;
  - a metre BOM (RobStride, Jetson Orin Nano Super, Teensy, OAK-D Lite) estimated in the description at $11,521 hardware and $17,286 for a first prototype. That figure is the PR's own claim and not re-verified.
- State: `mergeable_state: unstable` (checks not green, as reported), 227 commits behind main. None of `examples/tyrannosaurus_rex_50cm`, `examples/tyrannosaurus_rex_100cm`, `examples/mini_tyrannosaurus_rex` or `examples/shared` is on main.
- Superseded in practice by the T. rex form study. #667 (on main) and the working branch carry the 75 cm Jetson design and budget, and the study's `BASELINE.md` lists #635/#648/#649 as an unmerged stack it deliberately does not build on.
- Stacked on it: #648, which in turn carries #649 → #651 → #662 → #663/#664.
- **Recommendation: close as superseded by the form study (#667 and the coming study PR).** Extract first, as a short doc on main:
  - `examples/tyrannosaurus_rex_50cm/long_body/HARDWARE.md`: sourced part dimensions and the "installation reserve vs part body" packing method;
  - `examples/tyrannosaurus_rex_100cm/hardware.md` and `bom.csv`: a sourced RobStride-based price point, useful as a comparison row for the 15–20 kg Triceratops actuator choice.

  Do not merge 166k lines of generated example files.

**#646 "docs: design MicroDuck-inspired gait and sim-to-real improvements"**
- kuds; `main` ← `codex/microduck-transfer-design` (8a6a91b4). Created and last updated 26 Oct. Not a draft.
- 1 commit, 2 files, +361: new `docs/MICRODUCK_TRANSFER_DESIGN_2026_10.md` plus a link in `docs/README.md`.
- State: `mergeable_state: clean`, 182 commits behind main.
- Content: species-neutral; zero mentions of "T. rex". Sections cover gait measurement, sensor-realizable observations, actuator identification and a one-actuator benchmark, calibrated randomization, an export/deployment contract, recovery and policy composition, and a delivery sequence.
- On main: a different, shorter note, `docs/hardware/REFERENCE_MICRODUCK_2026_10.md` (b7746480, 31 Oct). It is a report-only reference framed around the T. rex tail habit. The design doc itself is not on main.
- **Recommendation: merge.** It is docs only, mergeable, and directly useful for the quadruped's sim-to-real path: servo identification and the export contract match the PLEM/Jetson plans. Two things to do first, or right after:
  - refresh its references to D-D17/D-D33/D-D44 and to the "existing hardware-build recommendation", which predate the Triceratops pivot;
  - cross-link it with `REFERENCE_MICRODUCK_2026_10.md` and `SIM_TO_REAL_PLAN.md`.

  It was not CI-run locally; its description defers to Actions.

**#648 "Compare mini T. rex torso layouts and leg folding directions"**
- kuds; `codex/mini-trex-forward-knee-example` (#635) ← `codex/mini-juvenile-hardware-fit` (00589040). Created and last updated 28 Oct. Draft.
- 4 commits of its own, 49 files, +55,255, all under `examples/mini_tyrannosaurus_rex/{juvenile,leg_study,layout_study}` and T. rex tests.
- What it does: a 3-layout torso study and forward- vs rearward-folding legs for a ~0.75 kg, 10-motor mini T. rex. Its recommendation is torso A with forward-folding legs.
- State: `clean` against its base #635, so it can only land after #635. Not on main.
- Superseded by the form study's direction (a 75 cm Jetson build, not a mini). Nothing in it transfers to a 15–20 kg quadruped.
- **Recommendation: close as superseded by the form study**, together with #635. Its children #649 → #651 → #662 → #663/#664 are outside my brief but sit on this stack and would lose their base. Review them at the same time. Optional extract: one paragraph of its leg-fold finding (a symmetric 35 mm box-envelope limit) into the T. rex study's restart notes.

### Summary table

| PR | Base ← head | Created / updated | Size | Draft / merge state | On main or superseded? | Recommendation |
|---|---|---|---|---|---|---|
| #585 | main ← codex/gait-certification | 2 Oct / 2 Oct | 2 commits, 32 files, +4,835/−17 | no / dirty | recorder layer on main via D-D23 (0d80e368); the gate is not; replaced by #596 | Close, superseded by #596 |
| #596 | main ← claude/gait-checker-trex-gait-r1 | 5 Oct / 5 Oct | 25 commits, 62 files, +14,039/−124 | draft / dirty | checker not on main; paths renamed since; T. rex `gait-r1` moot | Keep parked; when quadruped training starts, extract the checker (`quadruped_walk`, `GAIT_CERTIFICATION.md`) without gait-r1, then close |
| #623 | main ← claude/trex-foot-r10 | 16 Oct / 17 Oct | 12 commits, 52 files, +2,728/−544 | draft / dirty | superseded: "r10" landed as tail springs (#643, D-D44); main records #623 as never merged | Close; optionally save its evidence for the T. rex restart |
| #635 | main ← codex/mini-trex-forward-knee-example | 24 Oct / 26 Oct | 26 commits, 204 files, +166,099/−18 | no / unstable | not on main; superseded by the form study (#667 + study PR) | Close; first extract `long_body/HARDWARE.md` and the 100 cm `hardware.md`/`bom.csv` |
| #646 | main ← codex/microduck-transfer-design | 26 Oct / 26 Oct | 1 commit, 2 files, +361 | no / clean | not on main (main has only the shorter REFERENCE_MICRODUCK note) | Merge after refreshing decision references and cross-links |
| #648 | #635 ← codex/mini-juvenile-hardware-fit | 28 Oct / 28 Oct | 4 commits, 49 files, +55,255 | draft / clean (on #635) | not on main; mini T. rex direction superseded | Close with #635; children #649–#664 also need a decision |

Not done: I ran no CI or tests and opened no check-run logs. #635's "unstable" is GitHub's reported state only. I did not read full file contents beyond headings and the greps cited.