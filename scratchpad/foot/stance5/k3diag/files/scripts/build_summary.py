"""Write k3diag/files/summary.json: the answers (from files only) + summary_data.json as "data".

    python3 -I build_summary.py K3DIAG_FILES_DIR
"""
import json
import os
import sys

ANSWERS = {
    "headline": [
        "[measured] Joint-noise falls in band s42 start at the report-only 7,340,032 slot, not at 8.4M: 0, 0, 0, 9, 5, 9, 13 of 20 at 4.2, 5.2, 6.3, 7.3, 8.4, 10.5M and the final. Over the slots 6.3M, 7.3M, 8.4M, 10.5M and final, band runs fall 43 times and control runs 11 (300 episodes each). Without s42 that is 7 against 2.",
        "[measured] Most falls are forward: the whole-foot CoP runs to the toe line and the foot pitches about it. Band s42 has 26 of 36 forward, 6 backward and 4 unclassified. The 6 backward falls are all tail_contact, all at 10.5M, which is the one checkpoint where the tail was not pinned.",
        "[measured] The tail habit does not track how many falls there are. Falls go 0 -> 9 from 6.3M to 7.3M while panel saturation flags hold at 22 -> 21 of 40. At 10.5M there are 2/40 flags (none on spawns 3042-3061) and still 9 falls. Within a checkpoint, flagged and unflagged spawns fall at similar rates. [inference] The tail sets the direction of a fall, not whether it happens.",
        "[measured] Policy std, deterministic COM offset, leg posture and static CoP margins do not change in step with the falls either. [inference] What changed between 6.3M and 7.3M is the closed-loop response to perturbation, which the files cannot see. Rollouts are needed: per-group ablation on the recurrent spawns 3050, 3054 and 3059.",
    ],
    "q1_what_the_falls_look_like": {
        "counts": "[measured] Joint-noise falls of 20 at 6.3M / 7.3M / 8.4M / 10.5M / final: band s42 0/9/5/9/13; control s42 0/0/0/1/8; band s52 0/0/1/1/0; band s60 0/0/1/1/3; control s52 0/0/0/1/0; control s60 0/0/0/0/1. The own-sigma test (policy space) gives 0 falls in every run at every one of these slots, except band s42 final: 1 (spawn 3046, 'fallen' at 726).",
        "direction": "[measured + rule] Band s42: 7.3M has 9/9 forward, 8.4M 5/5 forward, 10.5M 6 backward (tail_contact) + 1 forward (nosedive) + 2 unclassified, and the final 11 forward + 2 unclassified. That is 26/36 forward in all. Control s42: 6 forward (head_contact 2, nosedive 1, fallen 3), 2 backward (tail_contact), 1 unclassified. Band s60 final: 2 'fallen' with a centred CoP (unclassified: a collapse or a sideways fall, which the files cannot tell apart) and 1 at step 143, before the window. In the files 'fallen' means the pelvis went below 0.70 m. Most forward falls end as 'fallen', not 'nosedive': the body drops before it pitches 0.493 past its natural forward_z [inference].",
        "readouts_by_cause_all_runs": "[measured, window means over steps 200..L-1, 64 joint falls over all slots] 'fallen' (41): whole-foot CoP r +0.52 / l +0.60, pad CoP +0.49 / +0.55; toe margin of the smaller foot 9.0 N m against heel 37.5; COM 0.127 m ahead of MTP III; max sole tilt 15.6 deg; corner lift 37 mm. nosedive (10): CoP +0.72 / +0.63, toe 8.5, heel 41.2, COM 0.149 m. head_contact (3): CoP +0.97 / +0.71, toe margin -1.9 N m (past the toe line), COM 0.168 m. tail_contact (10): CoP -0.44 / -0.16, toe 25.4, heel 10.4, COM 0.008 m.",
        "fallers_vs_survivors_s42_band_final": "[measured] Fallers (13) against survivors (7) against the deterministic panel: CoP mean +0.50 / +0.25 / +0.09. Toe margin (smaller foot) 11.6 / 22.9 / 25.9 N m; heel margin 35.2 / 32.8 / 31.2 N m. COM ahead of MTP III 0.112 / 0.071 / 0.066 m. Max sole tilt 10.6 / 1.0 / 0.39 deg; corner lift 25 / 3 / 1 mm. Right digit share 0.07 / 0.18 / 0.20.",
        "time_to_fall": "[measured] control_dt is 0.01 s. Band s42: median 576 steps, range 282-999, and only 5 of 36 fall before step 400. Two finals end at step 999. Control s42: median 464, range 352-854. Spawn 3061 is the early exception: in control s60 it falls at 201-248 at four checkpoints, and in band s60 final at 143. So the s42 falls come 1-8 s after the settle window opens, not at spawn.",
        "contacts_lift_slip": "[measured] The terminations seen in the noise tests are fallen, nosedive, tail_contact and head_contact. There is no knee, torso or body_contact termination in any of the six runs. Every band s42 faller reaches a max sole tilt of 3.7 deg or more and a corner lift of 10 mm or more, while no survivor exceeds 3.0 deg or 7.8 mm. Spawn 3061 in control s60 has its right foot never loaded in the window: it stands on one leg, the clearest lateral candidate. 4 of 36 band s42 falls have |CoP_l - CoP_r| >= 0.6 (lateral suspect). Foot slip is not in the judge files.",
        "cop_at_the_edge": "[not in files] The judge keeps window means and window maxima, not a per-step trace, so the CoP just before the fall cannot be read. In forward fallers the window-mean CoP already sits at +0.5 to +0.6, with half the survivors' toe margin.",
        "noise_effect_on_survivors": "[measured] In all six runs the joint noise makes survivors rock: whole-foot CoP std per foot is 0.055-0.061 m on a 0.07 m pad half-span, against 0.002-0.009 m deterministic. The window-mean CoP moves forward by +0.08 to +0.29 against the deterministic panel at the same slot, and the COM by -2 to +13 mm. Survivors' pad gate reaches about 0.8, so the K2a/K2b rails are over for most noise episodes in every run.",
    },
    "q2_which_actuator_groups": {
        "status": "[not answerable from files] The judge rolls one sigma vector per test, with no per-group ablation. The rollout part must ablate. What the files constrain:",
        "excess_over_training_noise": "[measured] sigma_ctrl / sigma_own at band s42 final: hip pitch 9.5x, knee 9.1x, ankle 4.7x, hip roll 3.0x, neck/head 1.9x, tail 1.6x. The pattern is the same in every run and grows through training, e.g. hip pitch from 6.9x at 6.3M. The own-sigma test gives 0-1 falls everywhere. [inference] Leg noise, hip pitch and knee first, is the main candidate. Tail and neck noise in the joint test is within about 2x of what each policy trains under.",
        "band_vs_control_sigma": "[measured] The band arm's own hip-pitch sigma is about 30% below control's at the same step: 6.3M 0.020-0.022 vs 0.030, final 0.014 vs 0.019. Knee is about 10% lower; ankle and hip roll are equal. So the joint test's hip-pitch noise is 6.9-10.2x own for band against 4.7-7.9x for control. This holds for all three band seeds, while s52 and s60 band hold. [inference] Hip pitch is the first group to ablate for the band-vs-control gap; it does not by itself explain s42.",
        "tail_in_joint_space": "[arithmetic] Joint space clips the command first, so a tail_3 command pinned at +1 only receives downward kicks. The mean shift is -sigma/sqrt(2 pi) = -0.082, about -0.007 rad of joint by satdiag's map q = -0.093 + 0.086 a. [inference] That is too small to matter as a bias.",
        "common_noise_final": "[measured] Policy-space common noise (sigma mean 0.183; hip pitch 0.167, knee 0.125, ankle 0.110, all above sigma_ctrl's 0.141, 0.079 and 0.067): band s42 31/40, control s42 30/40, the other four runs 1-6/40. [inference] At the final, s42 is fragile to leg-heavy noise in both arms. The band adds fragility only in the joint test (13 vs 8). The common noise uses a different RNG stream, so its spawns are not paired with the joint test's.",
        "foot_rotation": "[measured] Every forward fall ends with the foot pitched about its toe line: tilt 3.7 deg or more, corner lift 10 mm or more, toe margin around 9 N m. [inference] The ankle and leg cannot pull the CoP back inside the support once the COM is about 0.11-0.15 m ahead of MTP III.",
        "ablation_targets": "[measured] Spawns 3050, 3054 and 3059 fall at all four band s42 checkpoints from 7.3M to the final. Spawns 3047, 3051, 3055 and 3061 fall at three. Spawn 3061 falls in all six runs at some checkpoint. The z sequence of a spawn is identical in every run and checkpoint.",
    },
    "q3_band_vs_control_same_noise": {
        "falls": "[measured, paired draw for draw] Band 43 against control 11 over 300 episodes each (6.3M, 7.3M, 8.4M, 10.5M, final). Band has as many falls as control or more at every seed and slot (at s52 10.5M both have 1, on different spawns). s42: 8.4M 5 vs 0, 10.5M 9 vs 1 (1 shared), final 13 vs 8, with all 8 control falls among band's 13 (p = 0.010 if the two were independent).",
        "static_margins_do_not_explain": "[measured] Band policies stand centred and symmetric. Their deterministic per-foot CoP split (max minus min) is 0.04-0.34, against 0.26-0.98 for control at 8.4M and the final. Band toe margins are equal or larger: s42 final deterministic toe/heel 25.9 / 31.2 N m for band against 22.2 / 24.5 for control. Under noise, survivors' medians are toe 23.4 / heel 32.1 N m for band s42 final against 21.4 / 25.7 for control s42. CoP rocking std is 0.058 vs 0.061 m. [inference] The band did not shrink the static or mean-noise margins. Its runs reach the falling state more often.",
        "recovery": "[measured] Neither arm recovers from a large foot rotation. Survivors' max sole tilt is 3.0 deg or less in band runs and 3.7 deg or less in control; fallers are at 3.7 deg or more and 7.1 deg or more. Survivors with a roll-rail excursion (tilt > 2 deg or lift > 4 mm): band s42 4 against 36 falls, control s42 6 against 9, band s60 6 against 5, control s60 1 against 1. [inference] Recovery stops at about 3-4 deg of foot tilt in both arms. The difference is how often noise drives a foot there.",
        "ankle_torque_headroom": "[not in files] The noise rows hold no actuator forces. The eval ledger's actuator_force_frac_max is deterministic, a maximum over 15 actuators, and held by head_pitch or tail_1 per satdiag. It reads 0.27-0.36 in all runs, so it says nothing about the ankle under noise.",
        "band_readouts": "[measured, deterministic panels only; the noise test does not capture the band] The band is nearly idle in deterministic play: band s42 whole-foot out-of-band steps are 0-491 of 64,000 foot-steps, and the most paid is -0.70 per episode (8.4M), against the statue's -1.91. [inference] Under noise, survivors' whole-foot |CoP| averages about 0.8, so the band would pay on many noisy steps. This is not measured.",
    },
    "q4_s42_rise_tail_habit_or_else": {
        "tail_habit_count": "[measured] Band s42 tail_3 window share at |cmd| >= 0.9 (satroll): 0.774 at 4.2M, 0.670 at 5.2M, 0.751 at 6.3M; there is no satroll after 6.3M. Panel saturation flags (any actuator, |a| >= 0.99 on more than 10% of window steps): 24, 13, 22, 21, 24, 2, 20 of 40 at 4.2, 5.2, 6.3, 7.3, 8.4, 10.5M and the final. Joint falls: 0, 0, 0, 9, 5, 9, 13. Falls jump at 6.3 -> 7.3M with no change in flags. At 10.5M (2 flags, none on spawns 3042-3061) there are still 9 falls.",
        "same_spawn": "[measured] Falls among tail-flagged against unflagged spawns (deterministic panel flag, same spawn seed): 7.3M 5/12 vs 4/8; 8.4M 2/10 vs 3/10; final 5/11 vs 8/9.",
        "control_s42": "[measured] Control s42 had 35-40 of 40 flagged from 2.1M to 9.4M, with satdiag tail_3 share at 0.9 of 0.84-1.00, and 0-1 joint falls. Its falls rose to 8 at the final as the share fell to 0.74.",
        "direction_tracks_tail": "[measured] With the tail pinned (7.3M, 8.4M, final), 25 of 27 band s42 falls are forward. With it not pinned (10.5M), 6 of 9 are tail_contact. At 10.5M the deterministic COM was 0.054 m ahead of MTP III, against 0.061-0.068 at the other slots, and the CoP -0.09 against +0.01 to +0.10. [inference] This fits satdiag's finding that a raised tail moves the COM forward by about 10 mm.",
        "policy_std": "[measured] Own sigma mean: 0.0887, 0.0838, 0.0800, 0.0742, 0.0699 at 6.3M, 7.3M, 8.4M, 10.5M and the final. Band s52 and s60 match within 0.003 and stay at 0-3 falls. Leg sigma (hip pitch, knee, ankle mean) is 0.0174 -> 0.0126 for s42 against 0.0173 -> 0.0121 and 0.0179 -> 0.0125. Not s42-specific.",
        "com_offset": "[measured] Deterministic COM ahead of MTP III: 0.068, 0.061, 0.062, 0.054, 0.066 m at 6.3M, 7.3M, 8.4M, 10.5M and the final. It moved back when the falls began. Control s42 was at 0.067-0.069 at its 0-fall checkpoints. Not tracking.",
        "leg_posture": "[measured, eval ledger, 524,288-step bin means] For the bins ending at 6.3, 7.3, 8.4, 10.5 and 12.06M: hip q -2.12, -3.05, -2.51, -3.12, -2.01 deg; knee 0.38, -0.09, 0.07, -0.40, -0.25 deg; ankle 0.85, 1.55, 1.36, 2.63, 1.11 deg. These are moves of 1.8 deg or less, back near the 6.3M values at the final, where falls are highest. Not tracking.",
        "training_and_eval_ledgers": "[measured] Band s42 training episodes (own noise) fall at 0-0.4% per bin through 9.96M. Then 2.1% (10.49M), 3.5%, 2.3% and 5.4% (last bin). tail_contact leads first (7 of 11 and 11 of 19); 'fallen' is 25 of 26 in the last bin. Control s42: 0.2-1.1%, then 2.8% in the last bin. The other four runs stay at 1.7% or less. Deterministic eval falls in band s42 are 0 until the 9.96M bin, then 2, 1, 1, 0, 2. [inference] The joint test exposes the fragility about 3M steps before training noise and deterministic evaluation do.",
        "answer": "[inference] s42's rise is not the tail habit, policy std, COM offset or leg posture as far as the files show: none of them moves with the falls between 6.3M and 7.3M or later. The tail habit changes which way the robot falls. The extra falls come from a change in how the policy reacts to joint perturbation (mostly legs, by sigma ratio), plus a seed-specific fragility that control s42 shares at the final under the common noise (30-31/40 in both arms). Testing this needs rollouts: per-group ablation, and a 6.3M vs 7.3M comparison on spawns 3050, 3054 and 3059.",
    },
}

CAVEATS = [
    "Every readout is a window mean (steps 200..L-1) or a window maximum. The judge files hold no per-step trace, so the CoP just before a fall, the onset time within an episode, foot slip and ankle torque cannot be read.",
    "The files hold no lateral readout. Sideways falls are only suspected, from L/R CoP asymmetry or an unloaded foot.",
    "The direction rule (forward: nosedive/head_contact, or 'fallen' with mean CoP >= +0.3; backward: tail_contact, or <= -0.3) is this analysis's own.",
    "The tail_3 share by checkpoint exists for band s42 only at 4.2, 5.2 and 6.3M (satroll). After that, only the panel's any-actuator saturation flags are available. That tail_3 is the flagged actuator after 6.3M is inferred from the earlier rolls.",
    "n = 20 per noise test, so per-slot counts carry wide intervals: 0 vs 9 of 20 is clear, 5 vs 9 is not. 7,340,032 is a report-only slot.",
    "That the deterministic panel and the noise test start from the same state for a given spawn seed is inferred: both reset with that seed.",
    "The common-noise test uses a different RNG stream (SeedSequence spawn_key (1,)), so its spawns are not paired with the joint test's.",
]


def main(out):
    data = json.load(open(os.path.join(out, "summary_data.json")))
    s = {"schema": "stance5-k3diag-files-summary/v1", "date": "2026-10-22", "report_only": True,
         "scope": "K3 (joint-noise falls), from the judge files and ledgers of stance5_LTSB_s52/s60/s42 (band) and stance4_LTS_s52/s60/s42 (control); no rollouts",
         "answers": ANSWERS, "caveats": CAVEATS,
         "outputs": {"summary": os.path.join(out, "summary.json"), "episodes": os.path.join(out, "noise_episodes.csv.gz"),
                     "ledger_bins": os.path.join(out, "ledger_bins.csv"), "extract": os.path.join(out, "extract.json.gz"),
                     "scripts": [os.path.join(out, "scripts", n) for n in ("extract.py", "analyze.py", "build_summary.py")]},
         "data": data}
    with open(os.path.join(out, "summary.json"), "w") as fh:
        json.dump(s, fh, indent=1)
    print("ok")


if __name__ == "__main__":
    main(sys.argv[1])
