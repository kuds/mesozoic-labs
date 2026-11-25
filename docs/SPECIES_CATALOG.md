# Species catalog

<!-- Written whole by `python -m environments.shared.species_catalog`, which CI runs with `--check`;
do not edit by hand. -->

Each model's full entry and the published run summaries. The [root README](../README.md#species)
has one row per model, and each model's page on [mesozoiclabs.com](https://mesozoiclabs.com)
renders the same entry.

## Species

<!-- BEGIN GENERATED: SPECIES -->
The active-species tables are generated from `configs/species_manifest.toml`, the layered plant contract,
the executable Gymnasium environments, compiled MJCF models, and current stage TOML files. The budgets and
gates shown are for the Stable-Baselines3 curriculum path. Do not edit the generated block by hand.

<a id="velociraptor_mongoliensis-anatomical"></a>

### Velociraptor mongoliensis (Anatomical)

Swift Bipedal Predator. **Specialty:** Fast bipedal running.

| Generated specification | Value |
|---|---|
| Observation dimension | 67 |
| Action dimension / actuators | 17 |
| Generalized coordinates / velocities | nq=30, nv=29 |
| Compiled dynamic model mass | 15.0 kg |
| Plant contract revisions | policy r15; physics r6; visual r7 ([details](PLANT_CONTRACT.md)) |
| Model | `environments/velociraptor_mongoliensis/anatomical/assets/velociraptor_mongoliensis_anatomical.xml` |

| Current stage | Recipe | Warm-start from | Objective | SB3 configured budget | SB3 early-advancement gate |
|---|---|---|---|---:|---:|
| 1 — Stance | stance (deliverable) | — | Learn to stand and balance without falling | 12M | seeds clean at all six headings (whole scene turned -90, -45, 0, 45, 90, 180°) LCB95 ≥ 0.8 over ≥ 40 seeds (settle 100 steps); at every heading, clean stance episodes LCB95 ≥ 0.8 over the same seeds; an episode is clean when it reaches the horizon with min_all_feet_support ≥ 0.98, max_touchdown_rate ≤ 0.25, max_window_displacement_m ≤ 0.1, min_foot_load_share ≥ 0.4, max_actuator_saturation_fraction ≤ 0.1, max_settle_airborne_substeps ≤ 0, max_settle_peak_floor_force_bw ≤ 2, min_foot_load_share_windowed ≥ 0.35, max_foot_contact_fraction ≤ 0.02, max_phantom_support_fraction ≤ 0.05, max_nonfoot_load_fraction ≤ 0.01, max_settle_stance_width_change_m ≤ 0.05, min_support_geom_duty ≥ 0.5, min_support_geom_coverage ≥ 0.8, max_episode_yaw_change_deg ≤ 25, max_settle_touchdowns ≤ 2, min_foot_load_share_statue_ratio ≥ 0.8; full-horizon episodes ≥ 95.0% at every heading; hop-or-fall episodes ≤ 1 at every heading; reward rail ≥ 1950 at every heading; reward rail ≥ 0.6 × the statue's at the same heading; verdict from the six-heading stance_heading_report.json on the handoff pair (post-stage; fail-closed when absent, never the yaw-zero panel) |
| 2 — Locomotion | walk (deliverable) | 1 — Stance | Walk: follow a forward command with heading hold, and stand still on a zero command | 8M | cruise episodes (a non-zero reset command) success LCB95 ≥ 0.6 and survival LCB95 ≥ 0.8 over ≥ 20 episodes; a cruise episode succeeds when it reaches the horizon, settles its initial heading within 1 s + |error| / yaw_rate_max and holds it 1 s, stays in tolerance on ≥ 60.0% of its steps after the settle, makes ≥ 0.5 × the commanded speed along the heading after 2 s and passes the gait clause; stand episodes (an exactly zero command) over ≥ 5: ≥ 60.0% hold the stop (it settles, and holds on ≥ 60.0% of the steps after 1 s) and ≤ 0 fall; gait clause (gait_limbs 2): min_alternation_index ≥ 0.75, max_simultaneous_fraction ≤ 0.15, max_hop_flight_fraction ≤ 0.15, min_lead_swaps_per_stride ≥ 1, max_foot_contact_fraction ≤ 0.02, max_phantom_support_fraction ≤ 0.05; handoff chosen by the gated statistic on a separate selection panel; verdict from command_tracking_report.json on that pair (post-stage; fail-closed when absent) |

**Backend-specific success semantics:**
- **Stable-Baselines3 — Walk command tracking (command_tracking/v1):** The walk checkpoint clears the command_tracking/v1 gate (walk profile, decision D-D61), judged post-stage on the handoff its gated statistic chose on a separate selection panel. The walk follows a forward cruise command, a speed in [0.5, 1] x the model's cruise speed at a heading within 0.26 rad of the spawn's, drawn at reset and held for the whole episode, and about 15% of its episodes are stand episodes, an exactly zero command for the whole episode (decision D-D62). On the cruise family the one-sided 95% lower bounds on survival and on success meet the configured bounds: a cruise episode succeeds when it reaches the horizon, settles its initial heading, stays in tolerance on the configured share of its steps after the settle, makes the configured fraction of the commanded speed along the heading and passes the biped gait clause (no flight cap: its cruise reaches Froude 0.82); on the stand family the configured share holds the stop and no episode falls. The families' counts and bounds are exported to summary.json from the admitted command_tracking_report.json (decision D-D61), and the catalog headlines the cruise success bound and the stand success fraction once a result summary is listed.

**Per-deliverable success semantics:**
- **stance (1 — Stance) · Stable-Baselines3 — Stance quality at six headings (stance_quality_heading/v1):** The stance checkpoint clears the stance_quality_heading/v1 gate (the six-heading certificate, decision D-D32 and the maintainer's decision of 2026-11-06), judged post-stage on the handoff pair: the 40 certification-panel seeds are rolled with the whole scene turned -90, -45, 0, 45, 90 and 180 degrees; at every heading the stance_quality/v2 decision on floor truth (horizon reached, no hop or impact in the settle window, both feet down and loaded, the MTP pad and digits III and IV of each foot loaded, no actuator held at its limit) passes against the statue rolled at that heading, and the one-sided 95% lower bound on the share of seeds clean at every heading meets the configured bound, with the reward rails as collapse floors only. The clean-seed count and its bound are recorded in gate_verdict.json but not exported to summary.json until a later phase (decision D-A9, deferred by D-B15), so the catalog names them with no value.
- **walk (2 — Locomotion) · Stable-Baselines3 — Walk command tracking (command_tracking/v1):** The walk checkpoint clears the command_tracking/v1 gate (walk profile, decision D-D61), judged post-stage on the handoff its gated statistic chose on a separate selection panel. The walk follows a forward cruise command, a speed in [0.5, 1] x the model's cruise speed at a heading within 0.26 rad of the spawn's, drawn at reset and held for the whole episode, and about 15% of its episodes are stand episodes, an exactly zero command for the whole episode (decision D-D62). On the cruise family the one-sided 95% lower bounds on survival and on success meet the configured bounds: a cruise episode succeeds when it reaches the horizon, settles its initial heading, stays in tolerance on the configured share of its steps after the settle, makes the configured fraction of the commanded speed along the heading and passes the biped gait clause (no flight cap: its cruise reaches Froude 0.82); on the stand family the configured share holds the stop and no episode falls. The families' counts and bounds are exported to summary.json from the admitted command_tracking_report.json (decision D-D61), and the catalog headlines the cruise success bound and the stand success fraction once a result summary is listed.

[Full documentation →](../environments/velociraptor_mongoliensis/README.md)

<a id="tyrannosaurus_rex-anatomical"></a>

### Tyrannosaurus rex (Anatomical)

Apex Predator. **Specialty:** Heavy-bodied bipedal balance and walking.

| Generated specification | Value |
|---|---|
| Observation dimension | 59 |
| Action dimension / actuators | 15 |
| Generalized coordinates / velocities | nq=28, nv=27 |
| Compiled dynamic model mass | 85.5 kg |
| Plant contract revisions | policy r17; physics r12; visual r6 ([details](PLANT_CONTRACT.md)) |
| Model | `environments/tyrannosaurus_rex/anatomical/assets/tyrannosaurus_rex_anatomical.xml` |

| Current stage | Recipe | Warm-start from | Objective | SB3 configured budget | SB3 early-advancement gate |
|---|---|---|---|---:|---:|
| 1 — Stance | stance (deliverable) | — | Learn to stand and balance without falling | 12M | seeds clean at all six headings (whole scene turned -90, -45, 0, 45, 90, 180°) LCB95 ≥ 0.8 over ≥ 40 seeds (settle 200 steps); at every heading, clean stance episodes LCB95 ≥ 0.8 over the same seeds; an episode is clean when it reaches the horizon with min_all_feet_support ≥ 0.98, max_touchdown_rate ≤ 0.25, max_window_displacement_m ≤ 0.1, min_foot_load_share ≥ 0.4, max_actuator_saturation_fraction ≤ 0.1, max_settle_airborne_substeps ≤ 0, max_settle_peak_floor_force_bw ≤ 1.5, min_foot_load_share_windowed ≥ 0.35, max_foot_contact_fraction ≤ 0.02, max_phantom_support_fraction ≤ 0.05, max_nonfoot_load_fraction ≤ 0.02, max_settle_stance_width_change_m ≤ 0.08, max_sole_tilt_deg ≤ 2, max_sole_corner_lift_m ≤ 0.004, min_sole_contacts ≥ 1.5, max_episode_yaw_change_deg ≤ 25, max_settle_touchdowns ≤ 2, max_sole_cop_fore_aft ≤ 0.8; full-horizon episodes ≥ 95.0% at every heading; hop-or-fall episodes ≤ 1 at every heading; reward rail ≥ 2260 at every heading; reward rail ≥ 0.6 × the statue's at the same heading; verdict from the six-heading stance_heading_report.json on the handoff pair (post-stage; fail-closed when absent, never the yaw-zero panel) |
| recovery — Recovery | recover (deliverable) | 1 — Stance | Hold the stance against scheduled external pushes and recover from each | 3M | recovery success LCB95 ≥ 0.3; paired Δ vs each required frozen null LCB95 ≥ 0.2; re-entry ≤ 100 steps + 50-step dwell; ≥ 40 episodes/evaluation; verdict from the frozen gate_resolution.json (post-stage; fail-closed when absent or stale) |
| 2 — Locomotion | walk (deliverable) | 1 — Stance | Walk: follow a forward command with heading hold, and stand still on a zero command | 8M | cruise episodes (a non-zero reset command) success LCB95 ≥ 0.6 and survival LCB95 ≥ 0.8 over ≥ 20 episodes; a cruise episode succeeds when it reaches the horizon, settles its initial heading within 1 s + |error| / yaw_rate_max and holds it 1 s, stays in tolerance on ≥ 60.0% of its steps after the settle, makes ≥ 0.5 × the commanded speed along the heading after 2 s and passes the gait clause; stand episodes (an exactly zero command) over ≥ 5: ≥ 60.0% hold the stop (it settles, and holds on ≥ 60.0% of the steps after 1 s) and ≤ 0 fall; gait clause (gait_limbs 2): min_alternation_index ≥ 0.75, max_simultaneous_fraction ≤ 0.15, max_hop_flight_fraction ≤ 0.15, min_lead_swaps_per_stride ≥ 1, max_foot_contact_fraction ≤ 0.02, max_phantom_support_fraction ≤ 0.05, max_flight_fraction ≤ 0.05; handoff chosen by the gated statistic on a separate selection panel; verdict from command_tracking_report.json on that pair (post-stage; fail-closed when absent) |

**Backend-specific success semantics:**
- **Stable-Baselines3 — Walk command tracking (command_tracking/v1):** The walk checkpoint clears the command_tracking/v1 gate (walk profile, decision D-D61), judged post-stage on the handoff its gated statistic chose on a separate selection panel. The walk follows a forward cruise command, a speed in [0.5, 1] x the model's cruise speed at a heading within 0.26 rad of the spawn's, drawn at reset and held for the whole episode, and about 15% of its episodes are stand episodes, an exactly zero command for the whole episode (decision D-D62). On the cruise family the one-sided 95% lower bounds on survival and on success meet the configured bounds: a cruise episode succeeds when it reaches the horizon, settles its initial heading, stays in tolerance on the configured share of its steps after the settle, makes the configured fraction of the commanded speed along the heading and passes the biped gait clause with its flight cap; on the stand family the configured share holds the stop and no episode falls. The families' counts and bounds are exported to summary.json from the admitted command_tracking_report.json (decision D-D61), and the catalog headlines the cruise success bound and the stand success fraction once a result summary is listed.

**Per-deliverable success semantics:**
- **stance (1 — Stance) · Stable-Baselines3 — Stance quality at six headings (stance_quality_heading/v1):** The stance checkpoint clears the stance_quality_heading/v1 gate (the six-heading certificate, decision D-D32), judged post-stage on the handoff pair: the 40 certification-panel seeds are rolled with the whole scene turned -90, -45, 0, 45, 90 and 180 degrees; at every heading the stance_quality/v2 decision on floor truth (horizon reached, no hop or impact in the settle window, both feet down and loaded, flat pads, no actuator held at its limit) passes against the statue rolled at that heading, and the one-sided 95% lower bound on the share of seeds clean at every heading meets the configured bound, with the reward rails as collapse floors only. The clean-seed count and its bound are recorded in gate_verdict.json but not exported to summary.json until a later phase (decision D-A9, deferred by D-B15), so the catalog names them with no value.
- **recover (recovery — Recovery) · Stable-Baselines3 — Recovery under pushes (recovery_quality/v1):** The recovery checkpoint clears the recovery_quality/v1 gate, judged post-stage against the run's frozen gate_resolution.json: recovery-success LCB95 and the paired delta against each required frozen null meet the frozen thresholds, with re-entry inside the configured step budget and dwell. The statistics are not exported to summary.json until a later phase (decision D-A9, deferred by D-B15).
- **walk (2 — Locomotion) · Stable-Baselines3 — Walk command tracking (command_tracking/v1):** The walk checkpoint clears the command_tracking/v1 gate (walk profile, decision D-D61), judged post-stage on the handoff its gated statistic chose on a separate selection panel. The walk follows a forward cruise command, a speed in [0.5, 1] x the model's cruise speed at a heading within 0.26 rad of the spawn's, drawn at reset and held for the whole episode, and about 15% of its episodes are stand episodes, an exactly zero command for the whole episode (decision D-D62). On the cruise family the one-sided 95% lower bounds on survival and on success meet the configured bounds: a cruise episode succeeds when it reaches the horizon, settles its initial heading, stays in tolerance on the configured share of its steps after the settle, makes the configured fraction of the commanded speed along the heading and passes the biped gait clause with its flight cap; on the stand family the configured share holds the stop and no episode falls. The families' counts and bounds are exported to summary.json from the admitted command_tracking_report.json (decision D-D61), and the catalog headlines the cruise success bound and the stand success fraction once a result summary is listed.

[Full documentation →](../environments/tyrannosaurus_rex/README.md)

<a id="brachiosaurus_altithorax-anatomical"></a>

### Brachiosaurus altithorax (Anatomical)

Gentle Giant Herbivore. **Specialty:** Long-necked quadrupedal walking.

| Generated specification | Value |
|---|---|
| Observation dimension | 81 |
| Action dimension / actuators | 30 |
| Generalized coordinates / velocities | nq=38, nv=37 |
| Compiled dynamic model mass | 175.3 kg |
| Plant contract revisions | policy r12; physics r7; visual r4 ([details](PLANT_CONTRACT.md)) |
| Model | `environments/brachiosaurus_altithorax/anatomical/assets/brachiosaurus_altithorax_anatomical.xml` |

| Current stage | Recipe | Warm-start from | Objective | SB3 configured budget | SB3 early-advancement gate |
|---|---|---|---|---:|---:|
| 1 — Stance | stance (deliverable) | — | Learn to stand on four legs without falling | 6M | reward ≥ 1050; episode length ≥ 950; ≥ 10 episodes/evaluation; 3 consecutive passes |
| 2 — Locomotion | walk (deliverable) | 1 — Stance | Walk: follow a forward command with heading hold, and stand still on a zero command | 16M | cruise episodes (a non-zero reset command) success LCB95 ≥ 0.6 and survival LCB95 ≥ 0.8 over ≥ 20 episodes; a cruise episode succeeds when it reaches the horizon, settles its initial heading within 1 s + |error| / yaw_rate_max and holds it 1 s, stays in tolerance on ≥ 60.0% of its steps after the settle, makes ≥ 0.5 × the commanded speed along the heading after 2 s and passes the gait clause; stand episodes (an exactly zero command) over ≥ 5: ≥ 60.0% hold the stop (it settles, and holds on ≥ 60.0% of the steps after 1 s) and ≤ 0 fall; gait clause (gait_limbs 4): min_pair_alternation_index ≥ 0.75, min_lead_swaps_per_stride ≥ 1, max_flight_fraction ≤ 0.05, min_limb_duty ≥ 0.2; handoff chosen by the gated statistic on a separate selection panel; verdict from command_tracking_report.json on that pair (post-stage; fail-closed when absent) |

**Backend-specific success semantics:**
- **Stable-Baselines3 — Walk command tracking (command_tracking/v1):** The walk checkpoint clears the command_tracking/v1 gate (walk profile, decision D-D61), judged post-stage on the handoff its gated statistic chose on a separate selection panel. The walk follows a forward cruise command, a speed in [0.5, 1] x the model's cruise speed at a heading within 0.26 rad of the spawn's, drawn at reset and held for the whole episode, and about 15% of its episodes are stand episodes, an exactly zero command for the whole episode (decision D-D62). On the cruise family the one-sided 95% lower bounds on survival and on success meet the configured bounds: a cruise episode succeeds when it reaches the horizon, settles its initial heading, stays in tolerance on the configured share of its steps after the settle, makes the configured fraction of the commanded speed along the heading and passes the quadruped gait clause (pair alternation, limb duty and the flight cap) and the lead-swap bar; on the stand family the configured share holds the stop and no episode falls. The families' counts and bounds are exported to summary.json from the admitted command_tracking_report.json (decision D-D61), and the catalog headlines the cruise success bound and the stand success fraction once a result summary is listed.

**Per-deliverable success semantics:**
- **stance (1 — Stance) · Stable-Baselines3 — Reward-gated stance (reward_and_length/v1):** The stance checkpoint clears the reward_and_length/v1 gate: mean evaluation reward at or above the statue-derived collapse rail and a near-full-horizon mean episode length over the required consecutive evaluations. The zero-action statue clears this gate, so the stance deliverable is labelled by its gate kind here rather than claimed as certified stance quality; stance_quality/v1 waits on shin instrumentation (a kneeling pose currently reads identically to airborne) (plan §4.8).
- **walk (2 — Locomotion) · Stable-Baselines3 — Walk command tracking (command_tracking/v1):** The walk checkpoint clears the command_tracking/v1 gate (walk profile, decision D-D61), judged post-stage on the handoff its gated statistic chose on a separate selection panel. The walk follows a forward cruise command, a speed in [0.5, 1] x the model's cruise speed at a heading within 0.26 rad of the spawn's, drawn at reset and held for the whole episode, and about 15% of its episodes are stand episodes, an exactly zero command for the whole episode (decision D-D62). On the cruise family the one-sided 95% lower bounds on survival and on success meet the configured bounds: a cruise episode succeeds when it reaches the horizon, settles its initial heading, stays in tolerance on the configured share of its steps after the settle, makes the configured fraction of the commanded speed along the heading and passes the quadruped gait clause (pair alternation, limb duty and the flight cap) and the lead-swap bar; on the stand family the configured share holds the stop and no episode falls. The families' counts and bounds are exported to summary.json from the admitted command_tracking_report.json (decision D-D61), and the catalog headlines the cruise success bound and the stand success fraction once a result summary is listed.

[Full documentation →](../environments/brachiosaurus_altithorax/README.md)

<a id="dibothrosuchus_elaphros-anatomical"></a>

### Dibothrosuchus elaphros (Anatomical)

Gracile Erect-Limbed Crocodylomorph. **Specialty:** Erect-limbed quadrupedal walking.

| Generated specification | Value |
|---|---|
| Observation dimension | 75 |
| Action dimension / actuators | 27 |
| Generalized coordinates / velocities | nq=35, nv=34 |
| Compiled dynamic model mass | 8.6 kg |
| Plant contract revisions | policy r11; physics r4; visual r3 ([details](PLANT_CONTRACT.md)) |
| Model | `environments/dibothrosuchus_elaphros/anatomical/assets/dibothrosuchus_elaphros_anatomical.xml` |

| Current stage | Recipe | Warm-start from | Objective | SB3 configured budget | SB3 early-advancement gate |
|---|---|---|---|---:|---:|
| 1 — Stance | stance (deliverable) | — | Hold the erect quadrupedal stance without collapsing into a sprawl | 6M | reward ≥ 1500; episode length ≥ 950; ≥ 10 episodes/evaluation; 3 consecutive passes |
| 2 — Locomotion | walk (deliverable) | 1 — Stance | Walk: follow a forward command with heading hold, and stand still on a zero command | 12M | cruise episodes (a non-zero reset command) success LCB95 ≥ 0.6 and survival LCB95 ≥ 0.8 over ≥ 20 episodes; a cruise episode succeeds when it reaches the horizon, settles its initial heading within 1 s + |error| / yaw_rate_max and holds it 1 s, stays in tolerance on ≥ 60.0% of its steps after the settle, makes ≥ 0.5 × the commanded speed along the heading after 2 s and passes the gait clause; stand episodes (an exactly zero command) over ≥ 5: ≥ 60.0% hold the stop (it settles, and holds on ≥ 60.0% of the steps after 1 s) and ≤ 0 fall; gait clause (gait_limbs 4): min_pair_alternation_index ≥ 0.75, min_lead_swaps_per_stride ≥ 1, max_flight_fraction ≤ 0.05, min_limb_duty ≥ 0.2; handoff chosen by the gated statistic on a separate selection panel; verdict from command_tracking_report.json on that pair (post-stage; fail-closed when absent) |

**Backend-specific success semantics:**
- **Stable-Baselines3 — Walk command tracking (command_tracking/v1):** The walk checkpoint clears the command_tracking/v1 gate (walk profile, decision D-D61), judged post-stage on the handoff its gated statistic chose on a separate selection panel. The walk follows a forward cruise command, a speed in [0.5, 1] x the model's cruise speed at a heading within 0.26 rad of the spawn's, drawn at reset and held for the whole episode, and about 15% of its episodes are stand episodes, an exactly zero command for the whole episode (decision D-D62). On the cruise family the one-sided 95% lower bounds on survival and on success meet the configured bounds: a cruise episode succeeds when it reaches the horizon, settles its initial heading, stays in tolerance on the configured share of its steps after the settle, makes the configured fraction of the commanded speed along the heading and passes the quadruped gait clause (pair alternation, limb duty and the flight cap) and the lead-swap bar; on the stand family the configured share holds the stop and no episode falls. The families' counts and bounds are exported to summary.json from the admitted command_tracking_report.json (decision D-D61), and the catalog headlines the cruise success bound and the stand success fraction once a result summary is listed.

**Per-deliverable success semantics:**
- **stance (1 — Stance) · Stable-Baselines3 — Reward-gated stance (reward_and_length/v1):** The stance checkpoint clears the reward_and_length/v1 gate: mean evaluation reward at or above the statue-derived collapse rail and a near-full-horizon mean episode length over the required consecutive evaluations. The zero-action statue clears this gate, so the stance deliverable is labelled by its gate kind here rather than claimed as certified stance quality; stance_quality/v1 waits on a stance-quality and perturbation preflight this plant has not had (plan §4.8).
- **walk (2 — Locomotion) · Stable-Baselines3 — Walk command tracking (command_tracking/v1):** The walk checkpoint clears the command_tracking/v1 gate (walk profile, decision D-D61), judged post-stage on the handoff its gated statistic chose on a separate selection panel. The walk follows a forward cruise command, a speed in [0.5, 1] x the model's cruise speed at a heading within 0.26 rad of the spawn's, drawn at reset and held for the whole episode, and about 15% of its episodes are stand episodes, an exactly zero command for the whole episode (decision D-D62). On the cruise family the one-sided 95% lower bounds on survival and on success meet the configured bounds: a cruise episode succeeds when it reaches the horizon, settles its initial heading, stays in tolerance on the configured share of its steps after the settle, makes the configured fraction of the commanded speed along the heading and passes the quadruped gait clause (pair alternation, limb duty and the flight cap) and the lead-swap bar; on the stand family the configured share holds the stop and no episode falls. The families' counts and bounds are exported to summary.json from the admitted command_tracking_report.json (decision D-D61), and the catalog headlines the cruise success bound and the stand success fraction once a result summary is listed.

[Full documentation →](../environments/dibothrosuchus_elaphros/README.md)

<a id="compsognathus_longipes-anatomical"></a>

### Compsognathus longipes (Anatomical)

Small Bipedal Theropod. **Specialty:** Small-bodied bipedal walking.

| Generated specification | Value |
|---|---|
| Observation dimension | 51 |
| Action dimension / actuators | 14 |
| Generalized coordinates / velocities | nq=24, nv=23 |
| Compiled dynamic model mass | 1.0 kg |
| Plant contract revisions | policy r7; physics r4; visual r4 ([details](PLANT_CONTRACT.md)) |
| Model | `environments/compsognathus_longipes/anatomical/assets/compsognathus_longipes_anatomical.xml` |

| Current stage | Recipe | Warm-start from | Objective | SB3 configured budget | SB3 early-advancement gate |
|---|---|---|---|---:|---:|
| 1 — Stance | stance (deliverable) | — | Hold an upright stance with foot support | 11M | seeds clean at all six headings (whole scene turned -90, -45, 0, 45, 90, 180°) LCB95 ≥ 0.8 over ≥ 40 seeds (settle 200 steps); at every heading, clean stance episodes LCB95 ≥ 0.8 over the same seeds; an episode is clean when it reaches the horizon with min_all_feet_support ≥ 0.98, max_touchdown_rate ≤ 0.25, max_window_displacement_m ≤ 0.05, min_foot_load_share ≥ 0.4, max_actuator_saturation_fraction ≤ 0.1, max_settle_airborne_substeps ≤ 15, max_settle_peak_floor_force_bw ≤ 3, min_foot_load_share_windowed ≥ 0.35, max_foot_contact_fraction ≤ 0.02, max_phantom_support_fraction ≤ 0.05, max_nonfoot_load_fraction ≤ 0.01, max_settle_stance_width_change_m ≤ 0.011, min_support_geom_duty ≥ 0.5, min_support_geom_coverage ≥ 0.8, max_sole_tilt_deg ≤ 2, max_sole_corner_lift_m ≤ 0.003, min_sole_contacts ≥ 1.5, max_sole_cop_fore_aft ≤ 0.7, max_window_airborne_substeps ≤ 40, max_window_peak_floor_force_bw ≤ 2, min_foot_load_share_statue_ratio ≥ 0.8; full-horizon episodes ≥ 95.0% at every heading; reward rail ≥ 2740 at every heading; reward rail ≥ 0.6 × the statue's at the same heading; verdict from the six-heading stance_heading_report.json on the handoff pair (post-stage; fail-closed when absent, never the yaw-zero panel) |
| recovery — Recovery | recover (deliverable) | 1 — Stance | Pilot: recover an upright stance after calibrated horizontal pushes | 3M | recovery success LCB95 ≥ 0.5; paired Δ vs each required frozen null LCB95 ≥ 0.1; re-entry ≤ 40 steps + 20-step dwell; ≥ 40 episodes/evaluation; verdict from the frozen gate_resolution.json (post-stage; fail-closed when absent or stale) |
| 2 — Locomotion | walk (deliverable) | 1 — Stance | Walk: follow a forward command with heading hold, and stand still on a zero command | 3M | cruise episodes (a non-zero reset command) success LCB95 ≥ 0.6 and survival LCB95 ≥ 0.8 over ≥ 20 episodes; a cruise episode succeeds when it reaches the horizon, settles its initial heading within 1 s + |error| / yaw_rate_max and holds it 1 s, stays in tolerance on ≥ 60.0% of its steps after the settle, makes ≥ 0.5 × the commanded speed along the heading after 2 s and passes the gait clause; stand episodes (an exactly zero command) over ≥ 5: ≥ 60.0% hold the stop (it settles, and holds on ≥ 60.0% of the steps after 1 s) and ≤ 0 fall; gait clause (gait_limbs 2): min_alternation_index ≥ 0.75, max_simultaneous_fraction ≤ 0.15, max_hop_flight_fraction ≤ 0.15, min_lead_swaps_per_stride ≥ 1, max_foot_contact_fraction ≤ 0.02, max_phantom_support_fraction ≤ 0.05, max_flight_fraction ≤ 0.05; handoff chosen by the gated statistic on a separate selection panel; verdict from command_tracking_report.json on that pair (post-stage; fail-closed when absent) |

**Backend-specific success semantics:**
- **Stable-Baselines3 — Walk command tracking (command_tracking/v1):** The walk checkpoint clears the command_tracking/v1 gate (walk profile, decision D-D61), judged post-stage on the handoff its gated statistic chose on a separate selection panel. The walk follows a forward cruise command, a speed in [0.5, 1] x the model's cruise speed at a heading within 0.26 rad of the spawn's, drawn at reset and held for the whole episode, and about 15% of its episodes are stand episodes, an exactly zero command for the whole episode (decision D-D62). On the cruise family the one-sided 95% lower bounds on survival and on success meet the configured bounds: a cruise episode succeeds when it reaches the horizon, settles its initial heading, stays in tolerance on the configured share of its steps after the settle, makes the configured fraction of the commanded speed along the heading and passes the biped gait clause with its flight cap; on the stand family the configured share holds the stop and no episode falls. The families' counts and bounds are exported to summary.json from the admitted command_tracking_report.json (decision D-D61), and the catalog headlines the cruise success bound and the stand success fraction once a result summary is listed.

**Per-deliverable success semantics:**
- **stance (1 — Stance) · Stable-Baselines3 — Stance quality at six headings (stance_quality_heading/v1):** The stance checkpoint clears the stance_quality_heading/v1 gate (the six-heading certificate, decision D-D32), judged post-stage on the handoff pair: the 40 certification-panel seeds are rolled with the whole scene turned -90, -45, 0, 45, 90 and 180 degrees; at every heading the stance_quality/v2 decision on floor truth (horizon reached, no hop or impact in the settle window, the two-foot flight time and the landing peak bounded over the scored window, both feet down and loaded, the pad and all three digits of each foot loaded, flat pads, no actuator held at its limit) passes against the statue rolled at that heading, and the one-sided 95% lower bound on the share of seeds clean at every heading meets the configured bound, with the reward rails as collapse floors only. The clean-seed count and its bound are recorded in gate_verdict.json but not exported to summary.json until a later phase (decision D-A9, deferred by D-B15), so the catalog names them with no value.
- **recover (recovery — Recovery) · Stable-Baselines3 — Recovery under pushes (recovery_quality/v1):** The recovery checkpoint clears the recovery_quality/v1 gate, judged post-stage against the run's frozen gate_resolution.json: recovery-success LCB95 and the paired delta against each required frozen null meet the frozen thresholds, with re-entry inside the configured step budget and dwell. The statistics are not exported to summary.json until a later phase (decision D-A9, deferred by D-B15).
- **walk (2 — Locomotion) · Stable-Baselines3 — Walk command tracking (command_tracking/v1):** The walk checkpoint clears the command_tracking/v1 gate (walk profile, decision D-D61), judged post-stage on the handoff its gated statistic chose on a separate selection panel. The walk follows a forward cruise command, a speed in [0.5, 1] x the model's cruise speed at a heading within 0.26 rad of the spawn's, drawn at reset and held for the whole episode, and about 15% of its episodes are stand episodes, an exactly zero command for the whole episode (decision D-D62). On the cruise family the one-sided 95% lower bounds on survival and on success meet the configured bounds: a cruise episode succeeds when it reaches the horizon, settles its initial heading, stays in tolerance on the configured share of its steps after the settle, makes the configured fraction of the commanded speed along the heading and passes the biped gait clause with its flight cap; on the stand family the configured share holds the stop and no episode falls. The families' counts and bounds are exported to summary.json from the admitted command_tracking_report.json (decision D-D61), and the catalog headlines the cruise success bound and the stand success fraction once a result summary is listed.

[Full documentation →](../environments/compsognathus_longipes/README.md)

<a id="compsognathus_longipes-robot"></a>

### Compsognathus longipes (Robot)

Twelve-Servo Biped Prototype. **Specialty:** Servo-driven bipedal walking with fixed head and tail.

| Generated specification | Value |
|---|---|
| Observation dimension | 41 |
| Action dimension / actuators | 12 |
| Generalized coordinates / velocities | nq=19, nv=18 |
| Compiled dynamic model mass | 1.6 kg |
| Plant contract revisions | policy r4; physics r2; visual r2 ([details](PLANT_CONTRACT.md)) |
| Model | `environments/compsognathus_longipes/robot/assets/compsognathus_longipes_robot.xml` |

| Current stage | Recipe | Warm-start from | Objective | SB3 configured budget | SB3 early-advancement gate |
|---|---|---|---|---:|---:|
| 1 — Stance | stance (deliverable) | — | Hold an upright stance with foot support | 11M | reward ≥ 1800; full-horizon episodes ≥ 95.0%; unsupported duty ≤ 0.02; unsupported duty 95% upper bound ≤ 0.02; ≥ 40 episodes/evaluation; 3 consecutive passes |
| recovery — Recovery | recover (deliverable) | 1 — Stance | Pilot: recover an upright stance after calibrated horizontal pushes | 3M | recovery success LCB95 ≥ 0.5; paired Δ vs each required frozen null LCB95 ≥ 0.1; re-entry ≤ 40 steps + 20-step dwell; ≥ 40 episodes/evaluation; verdict from the frozen gate_resolution.json (post-stage; fail-closed when absent or stale) |
| 2 — Locomotion | walk (deliverable) | 1 — Stance | Walk: follow a forward command with heading hold, and stand still on a zero command | 3M | cruise episodes (a non-zero reset command) success LCB95 ≥ 0.6 and survival LCB95 ≥ 0.8 over ≥ 20 episodes; a cruise episode succeeds when it reaches the horizon, settles its initial heading within 1 s + |error| / yaw_rate_max and holds it 1 s, stays in tolerance on ≥ 60.0% of its steps after the settle, makes ≥ 0.5 × the commanded speed along the heading after 2 s and passes the gait clause; stand episodes (an exactly zero command) over ≥ 5: ≥ 60.0% hold the stop (it settles, and holds on ≥ 60.0% of the steps after 1 s) and ≤ 0 fall; gait clause (gait_limbs 2): min_alternation_index ≥ 0.75, max_simultaneous_fraction ≤ 0.15, max_hop_flight_fraction ≤ 0.15, min_lead_swaps_per_stride ≥ 1, max_foot_contact_fraction ≤ 0.02, max_phantom_support_fraction ≤ 0.05, max_flight_fraction ≤ 0.05; handoff chosen by the gated statistic on a separate selection panel; verdict from command_tracking_report.json on that pair (post-stage; fail-closed when absent) |

**Backend-specific success semantics:**
- **Stable-Baselines3 — Walk command tracking (command_tracking/v1):** The walk checkpoint clears the command_tracking/v1 gate (walk profile, decision D-D61), judged post-stage on the handoff its gated statistic chose on a separate selection panel. The walk follows a forward cruise command, a speed in [0.5, 1] x the model's cruise speed at a heading within 0.26 rad of the spawn's, drawn at reset and held for the whole episode, and about 15% of its episodes are stand episodes, an exactly zero command for the whole episode (decision D-D62). On the cruise family the one-sided 95% lower bounds on survival and on success meet the configured bounds: a cruise episode succeeds when it reaches the horizon, settles its initial heading, stays in tolerance on the configured share of its steps after the settle, makes the configured fraction of the commanded speed along the heading and passes the biped gait clause with its flight cap; on the stand family the configured share holds the stop and no episode falls. The families' counts and bounds are exported to summary.json from the admitted command_tracking_report.json (decision D-D61), and the catalog headlines the cruise success bound and the stand success fraction once a result summary is listed.

**Per-deliverable success semantics:**
- **stance (1 — Stance) · Stable-Baselines3 — Stance quality (stance_quality/v1):** The stance checkpoint clears the stance_quality/v1 gate: the unsupported-duty 95% upper bound and the full-horizon episode fraction meet the configured bounds over 40-episode evaluations, with the reward rail as a collapse floor only. The duty and full-horizon statistics are not exported to summary.json until a later phase (decision D-A9, deferred by D-B15), so the catalog names them with no value.
- **recover (recovery — Recovery) · Stable-Baselines3 — Recovery under pushes (recovery_quality/v1):** The recovery checkpoint clears the recovery_quality/v1 gate, judged post-stage against the run's frozen gate_resolution.json: recovery-success LCB95 and the paired delta against each required frozen null meet the frozen thresholds, with re-entry inside the configured step budget and dwell. The statistics are not exported to summary.json until a later phase (decision D-A9, deferred by D-B15).
- **walk (2 — Locomotion) · Stable-Baselines3 — Walk command tracking (command_tracking/v1):** The walk checkpoint clears the command_tracking/v1 gate (walk profile, decision D-D61), judged post-stage on the handoff its gated statistic chose on a separate selection panel. The walk follows a forward cruise command, a speed in [0.5, 1] x the model's cruise speed at a heading within 0.26 rad of the spawn's, drawn at reset and held for the whole episode, and about 15% of its episodes are stand episodes, an exactly zero command for the whole episode (decision D-D62). On the cruise family the one-sided 95% lower bounds on survival and on success meet the configured bounds: a cruise episode succeeds when it reaches the horizon, settles its initial heading, stays in tolerance on the configured share of its steps after the settle, makes the configured fraction of the commanded speed along the heading and passes the biped gait clause with its flight cap; on the stand family the configured share holds the stop and no episode falls. The families' counts and bounds are exported to summary.json from the admitted command_tracking_report.json (decision D-D61), and the catalog headlines the cruise success bound and the stand success fraction once a result summary is listed.

[Full documentation →](../environments/compsognathus_longipes/README.md)

<a id="allosaurus_fragilis-anatomical"></a>

### Allosaurus fragilis (Anatomical)

Jurassic Tridactyl Theropod. **Specialty:** Level-ground walk along a commanded heading, over an eight-metre course.

| Generated specification | Value |
|---|---|
| Observation dimension | 41 |
| Action dimension / actuators | 10 |
| Generalized coordinates / velocities | nq=19, nv=18 |
| Compiled dynamic model mass | 1482.2 kg |
| Plant contract revisions | policy r2; physics r2; visual r2 ([details](PLANT_CONTRACT.md)) |
| Model | `environments/allosaurus_fragilis/anatomical/assets/allosaurus_fragilis_anatomical.xml` |

| Current stage | Recipe | Warm-start from | Objective | SB3 configured budget | SB3 early-advancement gate |
|---|---|---|---|---:|---:|
| 1 — Stance | stance (deliverable) | — | Hold an upright, foot-supported standing posture. | 3M | reward ≥ 1200; full-horizon episodes ≥ 95.0%; unsupported duty ≤ 0.02; unsupported duty 95% upper bound ≤ 0.02; ≥ 40 episodes/evaluation; 3 consecutive passes |
| 2 — Locomotion | walk (deliverable) | 1 — Stance | Walk: follow a forward command with heading hold, and stand still on a zero command | 6M | cruise episodes (a non-zero reset command) success LCB95 ≥ 0.6 and survival LCB95 ≥ 0.8 over ≥ 20 episodes; a cruise episode succeeds when it reaches the horizon, settles its initial heading within 1 s + |error| / yaw_rate_max and holds it 1 s, stays in tolerance on ≥ 60.0% of its steps after the settle, makes ≥ 0.5 × the commanded speed along the heading after 2 s and passes the gait clause; stand episodes (an exactly zero command) over ≥ 5: ≥ 60.0% hold the stop (it settles, and holds on ≥ 60.0% of the steps after 1 s) and ≤ 0 fall; gait clause (gait_limbs 2): min_alternation_index ≥ 0.75, max_simultaneous_fraction ≤ 0.15, max_hop_flight_fraction ≤ 0.15, min_lead_swaps_per_stride ≥ 1, max_foot_contact_fraction ≤ 0.02, max_phantom_support_fraction ≤ 0.05, max_flight_fraction ≤ 0.05; handoff chosen by the gated statistic on a separate selection panel; verdict from command_tracking_report.json on that pair (post-stage; fail-closed when absent) |

**Backend-specific success semantics:**
- **Stable-Baselines3 — Walk command tracking (command_tracking/v1):** The walk checkpoint clears the command_tracking/v1 gate (walk profile, decision D-D61), judged post-stage on the handoff its gated statistic chose on a separate selection panel. The walk follows a forward cruise command, a speed in [0.5, 1] x the model's cruise speed at a heading within 0.26 rad of the spawn's, drawn at reset and held for the whole episode, and about 15% of its episodes are stand episodes, an exactly zero command for the whole episode (decision D-D62). On the cruise family the one-sided 95% lower bounds on survival and on success meet the configured bounds: a cruise episode succeeds when it reaches the horizon, settles its initial heading, stays in tolerance on the configured share of its steps after the settle, makes the configured fraction of the commanded speed along the heading and passes the biped gait clause with its flight cap; on the stand family the configured share holds the stop and no episode falls. The families' counts and bounds are exported to summary.json from the admitted command_tracking_report.json (decision D-D61), and the catalog headlines the cruise success bound and the stand success fraction once a result summary is listed. Its course_complete event, the pelvis 7.65 m along the reset's commanded heading (the eight-metre course less 0.35 m) with a foot loaded, is recorded as a cruise episode's success key and is no gate criterion; a stand episode has no course.

**Per-deliverable success semantics:**
- **stance (1 — Stance) · Stable-Baselines3 — Supported standing (stance_quality/v1):** At least 95% of standing trials last ten seconds. After a two-second settling period, the mean fraction of time on neither foot and its upper confidence bound each stay at or below 2%. Mean reward must reach 1,200 points.
- **walk (2 — Locomotion) · Stable-Baselines3 — Walk command tracking (command_tracking/v1):** The walk checkpoint clears the command_tracking/v1 gate (walk profile, decision D-D61), judged post-stage on the handoff its gated statistic chose on a separate selection panel. The walk follows a forward cruise command, a speed in [0.5, 1] x the model's cruise speed at a heading within 0.26 rad of the spawn's, drawn at reset and held for the whole episode, and about 15% of its episodes are stand episodes, an exactly zero command for the whole episode (decision D-D62). On the cruise family the one-sided 95% lower bounds on survival and on success meet the configured bounds: a cruise episode succeeds when it reaches the horizon, settles its initial heading, stays in tolerance on the configured share of its steps after the settle, makes the configured fraction of the commanded speed along the heading and passes the biped gait clause with its flight cap; on the stand family the configured share holds the stop and no episode falls. The families' counts and bounds are exported to summary.json from the admitted command_tracking_report.json (decision D-D61), and the catalog headlines the cruise success bound and the stand success fraction once a result summary is listed. Its course_complete event, the pelvis 7.65 m along the reset's commanded heading (the eight-metre course less 0.35 m) with a foot loaded, is recorded as a cruise episode's success key and is no gate criterion; a stand episode has no course.

[Full documentation →](../environments/allosaurus_fragilis/README.md)
<!-- END GENERATED: SPECIES -->

## Training Results

<!-- BEGIN GENERATED: RESULTS -->
The summaries below are historical experiment records generated from the versioned JSON files under
`results/`. They are not evidence for the current model revision unless provenance is marked both current
and verified. Current stage budgets may therefore differ from the steps reported here.

No result summary is published under `results/` at present.
<!-- END GENERATED: RESULTS -->
