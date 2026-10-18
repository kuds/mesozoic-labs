"""Write shared_fall_report.txt from shared_fall_summary.json (every number below is read from that file)."""
import json
from pathlib import Path

OUT = Path("scratchpad/foot/battery/shared_fall")
S = json.loads((OUT / "shared_fall_summary.json").read_text())
SRC = str(OUT / "shared_fall_summary.json")
E = S["events"]
DT = 0.01


def t(step):
    return "n/a" if step is None else f"step {step} (t={step * DT:.2f} s)"


L = []
w = L.append
w("SHARED FALL REPLAY, seed 7047 -- OFFLINE EVALUATION of uncertified scratch-recipe checkpoints (r9 arm L symmetry")
w("screen, MirrorKLPPO + experiment_* keys, 6,291,456 steps). Report only; certifies nothing; no training.")
w(f"Every number below is read from {SRC} (key path in [brackets]); that file is reduced by scripts/analyze_fall.py")
w("from replays/replay_<role>.json and replays/trace_<role>_y+0.npz (scripts/fall_replay.py). Figure: shared_fall_7047.png.")
w("")
for r in ("s60_6M", "s52_6M"):
    c = S["checkpoints"][r]
    w(f"  {r}: {c['model']} sha256 {c['model_sha256'][:16]}..., vecnorm sha256 {c['vecnorm_sha256'][:16]}...")
w("  statue: zero action (home control) every step.")
w("Rollout: main 2e734e29 stance stage config (options off; arm L's lean term and settled leg target are reward-only),")
w("fresh env per heading, reset(seed=7047), whole-scene SpawnYaw turn, deterministic policy with frozen VecNormalize")
w("(stance_report._load_policy), main's SubstepContactRecorder attached; a read-only probe and a scratch MjData give the")
w("extra telemetry. Conventions: [conventions].")
w("")
a60, a52, ast = E["s60_6M"], E["s52_6M"], E["statue"]
ctx0 = S.get("context_audit_panel_heading0") or {}
w("0. IN PLAIN TERMS")
w(f"  Seed 7047's reset tips the animal forward: both pads start {ast['heel']['sole_pitch_deg_spawn']} deg toe-down with the heels")
w(f"  {ast['heel']['corner_lift_mm_spawn']} mm up and the XCoM {ast['spawn']['xcom_minus_pad_centre_fwd_m']} m ahead of the pad centre,")
w(f"  {round(ast['spawn']['pad_toe_line_minus_pad_centre_m'] - ast['spawn']['xcom_minus_pad_centre_fwd_m'], 4)} m short of the pad toe line"
  + (" (the most toe-down and most forward spawn of the 40 audit seeds, section 7)." if ctx0 else "."))
w(f"  The statue folds: trunk up to {ast['statue_extremes_steps_11_to_1000']['pitch_down_deg_range'][1]} deg nose-down, hips sag "
  f"{ast['joint_deg_vs_home']['at_steps']['21']['r_hip_pitch']} deg into flexion, pelvis slides back, heels come down,")
w(f"  and it rocks between toe and heel edges with the XCoM never closer than {ast['statue_extremes_steps_11_to_1000']['xcom_pad_signed_margin_min_m']} m to the pad boundary.")
w(f"  Both policies hold the hips at or above home, extended relative to the statue's sag (servo targets +{a60['applied_target_deg_vs_home_ctrl']['41']['r_hip_pitch']}/"
  f"+{a60['applied_target_deg_vs_home_ctrl']['41']['l_hip_pitch']} deg s60, +{a52['applied_target_deg_vs_home_ctrl']['41']['r_hip_pitch']}/"
  f"+{a52['applied_target_deg_vs_home_ctrl']['41']['l_hip_pitch']} deg s52 at step 41) and the trunk near level, so the heels never")
w(f"  come down (closest to flat {a60['heel']['sole_pitch_deg_closest_to_flat_steps_11_to_end']} / {a52['heel']['sole_pitch_deg_closest_to_flat_steps_11_to_end']} deg). "
  f"The pad CoP sits on the pad toe edge from step {a60['pad_cop_at_toe_edge_both_feet_first_step']} / {a52['pad_cop_at_toe_edge_both_feet_first_step']},")
w(f"  the XCoM passes the pad toe line at step {a60['xcom_vs_pad_toe_line']['negative_for_good_from_step']} / {a52['xcom_vs_pad_toe_line']['negative_for_good_from_step']} and never returns. With the CoP already at the edge the ankles")
w("  have nothing left to push with; the remaining support is the passive, spring-hinged digits, which take load near their tips")
w(f"  and bend up. The XCoM creeps forward ({a60['creep_steps_31_81']['xcom_fwd_rate_mm_per_step']} / {a52['creep_steps_31_81']['xcom_fwd_rate_mm_per_step']} mm/step) "
  f"and leaves the pad + digit-tip support at step {a60['xcom_vs_pad_plus_digit_tip_hull']['outside_for_good_from_step']} / "
  f"{a52['xcom_vs_pad_plus_digit_tip_hull']['outside_for_good_from_step']}")
w(f"  (t = {a60['xcom_vs_pad_plus_digit_tip_hull']['outside_for_good_from_step'] * DT:.2f} / {a52['xcom_vs_pad_plus_digit_tip_hull']['outside_for_good_from_step'] * DT:.2f} s). "
  f"From there it is an inverted-pendulum topple straight forward over the toes (forward speed growth")
w(f"  {a60['topple']['forward_speed_growth_rate_per_s']} / {a52['topple']['forward_speed_growth_rate_per_s']} 1/s, a little under omega0 {a60['topple']['omega0_mean_per_s']} / {a52['topple']['omega0_mean_per_s']} 1/s), "
  f"ending in head_contact at step {a60['length']} (s60) and nosedive at step {a52['length']} (s52),")
w("  both inside the 200-step settle window.")
w(f"  Nothing saturates: leg actuators peak at {a60['actuators']['max_force_fraction_leg']} / {a52['actuators']['max_force_fraction_leg']} of their force range, "
  f"leg commands at |a| <= {a60['actuators']['max_abs_action_leg']} / {a52['actuators']['max_abs_action_leg']},")
w(f"  feet stay well inside the friction cone (net shear <= {max(a60['friction']['net_shear_over_mu_normal_max_steps_11_to_end'])} / "
  f"{max(a52['friction']['net_shear_over_mu_normal_max_steps_11_to_end'])} of mu N). The only command at its bound is s60's neck pitch")
w(f"  (+1 from step {a60['actuators']['action_abs_ge_0p99_first_step'].get('neck_pitch')}, head driven down to its joint limit), "
  "a late response inside the topple, not its cause.")
w("  The failure is posture and support geometry, not actuator authority: on the worst spawn of the panel the policies keep")
w("  the body tipped over the pad toe edges instead of folding back onto the heels as the statue does.")
w("")
w("1. REPRODUCTION AND HEADINGS [crosscheck_vs_audit_certificates, headings]")
for r in ("s60_6M", "s52_6M"):
    c = S["crosscheck_vs_audit_certificates"][r]
    h = S["headings"][r]
    w(f"  {r}: lengths equal to the audit panel at all six headings (policy and statue): {c['all_lengths_equal']}; "
      f"max |reward diff| {c['max_abs_reward_diff']:.1e}; source of the audit numbers {c['source']}")
    w(f"     replay: length {h[0]['length']}, {h[0]['termination_reason']}, at every heading "
      f"{[x['yaw_deg'] for x in h]}; max |CoM(heading frame) - CoM at heading 0| over the six = "
      f"{max(x['max_abs_com_dev_vs_yaw0_m'] for x in h)} m (6 dp)")
hs = S["headings"]["statue"]
w(f"  statue: length {hs[0]['length']} ({hs[0]['termination_reason']}) at all six headings, same max deviation "
  f"{max(x['max_abs_com_dev_vs_yaw0_m'] for x in hs)} m. The fall is one episode repeated six times; heading 0 stands for all.")
w("")
w("2. THE SPAWN (identical for all three; [events.<role>.spawn], [events.<role>.heel.*spawn])")
sp = E["statue"]["spawn"]
w(f"  CoM {sp['com_minus_pad_centre_fwd_m']} m ahead of the pad centre; pad toe line {sp['pad_toe_line_minus_pad_centre_m']} m;"
  f" XCoM {sp['xcom_minus_pad_centre_fwd_m']} m (s60/s52 {E['s60_6M']['spawn']['xcom_minus_pad_centre_fwd_m']}/"
  f"{E['s52_6M']['spawn']['xcom_minus_pad_centre_fwd_m']} after their first action); CoM forward speed "
  f"{E['s60_6M']['spawn']['com_fwd_velocity_mps']} m/s; omega0 {sp['omega0_per_s']} 1/s.")
w(f"  Both pads spawn pitched toe-down {E['statue']['heel']['sole_pitch_deg_spawn']} deg (R, L), heel corners "
  f"{E['statue']['heel']['corner_lift_mm_spawn']} mm up. The authored home keyframe has the CoM "
  f"{S['home_keyframe_static']['com_minus_pad_centre_fwd_m']} m ahead of the pad centre [home_keyframe_static].")
w(f"  So seed 7047 starts with the XCoM within {round(sp['pad_toe_line_minus_pad_centre_m'] - sp['xcom_minus_pad_centre_fwd_m'], 4)} m of the pad toe line (statue, after step 1) and the animal")
w("  tipped forward on its pad toe edges. The six sagittal leg joints all drew positive reset noise (replays/replay_statue.json:")
w("  episodes[].spawn_joint_offset_from_home).")
w("")
w("3. TIMELINE OF THE TWO FALLS (1-based control steps, 0.01 s each)")
for r in ("s60_6M", "s52_6M"):
    e = E[r]
    w(f"  {r} [events.{r}.*]:")
    w(f"    pad CoP reaches the pad toe edge on both feet at {t(e['pad_cop_at_toe_edge_both_feet_first_step'])} and stays "
      f"there for {e['pad_cop_at_toe_edge_both_feet_fraction_steps_11_to_end']:.1%} of steps 11-end (heel edge "
      f"{e['pad_cop_at_heel_edge_any_foot_fraction_steps_11_to_end']:.1%}).")
    w(f"    pads never flatten: closest to flat {e['heel']['sole_pitch_deg_closest_to_flat_steps_11_to_end']} deg, "
      f"mean steps 11-60 {e['heel']['sole_pitch_deg_mean_steps_11_60']} deg, heel corner lift mean steps 11-60 "
      f"{e['heel']['corner_lift_mm_mean_steps_11_60']} mm, min {e['heel']['corner_lift_mm_min_steps_11_to_end']} mm.")
    x = e["xcom_vs_pad_toe_line"]
    w(f"    XCoM passes the pad toe line at {t(x['first_negative_step'])} and never returns (negative for good from "
      f"{t(x['negative_for_good_from_step'])}; largest margin after step 10 {x['max_m']} m).")
    c = e["com_vs_pad_toe_line"]
    w(f"    the CoM itself passes the pad toe line at {t(c['negative_for_good_from_step'])}.")
    w(f"    support then = pad toe edge + the passive digits, loaded near their tips (load position along the digit, "
      f"0 = MTP, 1 = tip, at step 41: {e['digit_load_position_0mtp_1tip'].get('41')}); whole-foot CoP on the span "
      f"(+1 = pad toe line): {e['whole_foot_cop_norm_heel_m1_toe_p1']}.")
    w(f"    creep: XCoM moves forward {e['creep_steps_31_81']['xcom_fwd_rate_mm_per_step']} mm/step over steps 31-81; "
      f"pelvis over the pad centre (m) {e['pelvis_minus_pad_centre_fwd_m']}.")
    w(f"    the passive digits yield as they take load: MTP joints vs home (deg; negative = digit bent up against the "
      f"metatarsus) {e['mtp_deg_vs_home']}.")
    f_ = e["xcom_vs_pad_plus_digit_tip_hull"]
    w(f"    XCoM leaves the pad + digit-tip hull at {t(f_['outside_for_good_from_step'])} (first outside "
      f"{t(f_['first_outside_step'])}); point of no return with the feet planted.")
    tp = e["topple"]
    w(f"    topple: forward CoM speed {tp['forward_speed_at_hull_exit_mps']} -> {tp['forward_speed_at_end_mps']} m/s, "
      f"growth rate {tp['forward_speed_growth_rate_per_s']} 1/s vs omega0 {tp['omega0_mean_per_s']} 1/s "
      f"(fit over steps {tp['fit_steps']}): an inverted-pendulum topple about the toes.")
    fd = e["fall_direction"]
    w(f"    end: {e['termination_reason']} at step {e['length']}; trunk {fd['pitch_down_deg_end']} deg nose-down, head tip "
      f"{fd['head_tip_z_m_end']} m (head_contact < 0.12), forward_z {fd['forward_z_end']} (nosedive < "
      f"{fd['nosedive_forward_z_threshold']}), heels {e['heel']['corner_lift_mm_end']} mm up, pad pitch "
      f"{e['heel']['sole_pitch_deg_end']} deg, pads unloaded / digits {e['loads_fraction_BW'][str(e['length'])]} x BW.")
    w(f"    direction: CoM moved {fd['com_displacement_fwd_lat_m_step11_to_end']} m (fwd, left) from step 11, "
      f"{fd['direction_deg_from_heading']} deg off the heading; roll {fd['roll_deg_end']} deg, yaw change "
      f"{fd['yaw_change_deg_end']} deg. Straight forward, over the toes.")
w("")
w("4. THE STATUE ON THE SAME SEED [events.statue.*]")
e = E["statue"]
x = e["statue_extremes_steps_11_to_1000"]
w(f"  pads flatten (mean pitch steps 11-60 {e['heel']['sole_pitch_deg_mean_steps_11_60']} deg, corner lift "
  f"{e['heel']['corner_lift_mm_mean_steps_11_60']} mm); pelvis moves back over the pads (m) {e['pelvis_minus_pad_centre_fwd_m']}.")
w(f"  XCoM range {x['xcom_fwd_rel_pad_centre_range_m']} m about the pad centre; min signed margin to the pad hull "
  f"{x['xcom_pad_signed_margin_min_m']} m (toe {x['xcom_pad_toe_margin_min_m']}, heel {x['xcom_pad_heel_margin_min_m']});"
  f" pad CoP spans {x['pad_cop_fore_range']} (it rocks to the heel edge for "
  f"{e['pad_cop_at_heel_edge_any_foot_fraction_steps_11_to_end']:.1%} of steps and back); trunk {x['pitch_down_deg_range']} deg.")
w(f"  posture: trunk drops nose-down to {x['pitch_down_deg_range'][1]} deg while the hips sag into flexion "
  f"(hip pitch vs home at step 21/61: {e['joint_deg_vs_home']['at_steps']['21']['r_hip_pitch']}/"
  f"{e['joint_deg_vs_home']['at_steps']['61']['r_hip_pitch']} deg right).")
w("")
w("5. WHAT IS NOT LIMITING [events.<role>.actuators, .friction, .joint_deg_vs_home]")
for r in ("s60_6M", "s52_6M"):
    a = E[r]["actuators"]
    fr = E[r]["friction"]
    w(f"  {r}: largest leg-actuator |force|/forcerange over the episode {a['max_force_fraction_leg']} (ankles "
      f"{a['max_force_fraction_of_forcerange']['r_ankle']}/{a['max_force_fraction_of_forcerange']['l_ankle']}, hip pitch "
      f"{a['max_force_fraction_of_forcerange']['r_hip_pitch']}/{a['max_force_fraction_of_forcerange']['l_hip_pitch']}); "
      f"no actuator ever at its force limit: {not any(a['any_substep_at_force_limit'].values())}; max |leg action| "
      f"{a['max_abs_action_leg']}; commands at |a|>=0.99: {a['action_abs_ge_0p99_fraction'] or 'none'} "
      f"(first step {a['action_abs_ge_0p99_first_step'] or '-'}).")
    w(f"     friction: net shear / (mu N) per foot max {fr['net_shear_over_mu_normal_max_steps_11_to_end']} "
      f"(steps 11-60 mean {fr['net_shear_over_mu_normal_mean_steps_11_60']}); force-weighted contact cone use max "
      f"{fr['forceweighted_contact_cone_use_max_steps_11_to_end']}; foot-site slide {fr['foot_site_slide_mm_step11_to_end']} mm.")
    w(f"     joints within 0.5 deg of a range limit (fraction of steps): {E[r]['joint_within_0p5deg_of_range_limit_fraction']} "
      f"(statue: {E['statue']['joint_within_0p5deg_of_range_limit_fraction']})")
    w(f"     largest joint excursions from home (deg): legs {dict((k, v) for k, v in E[r]['joint_deg_vs_home']['max_abs_over_episode'].items() if 'hip' in k or 'knee' in k or 'ankle' in k)}")
w(f"  statue for scale: leg force fraction {E['statue']['actuators']['max_force_fraction_leg']}, friction "
  f"{E['statue']['friction']['net_shear_over_mu_normal_max_steps_11_to_end']}.")
w("")
w("6. WHAT THE POLICIES DO INSTEAD [events.<role>.applied_target_deg_vs_home_ctrl, .joint_deg_vs_home, kinematic_attribution_com_fwd_shift_m]")
for r in ("s60_6M", "s52_6M"):
    tg = E[r]["applied_target_deg_vs_home_ctrl"]["41"]
    dq = E[r]["joint_deg_vs_home"]["at_steps"]["61"]
    w(f"  {r}: applied servo targets vs home at step 41 (deg): hip pitch {tg['r_hip_pitch']}/{tg['l_hip_pitch']}, knee "
      f"{tg['r_knee']}/{tg['l_knee']}, ankle {tg['r_ankle']}/{tg['l_ankle']}, neck {tg['neck_pitch']}; joint angles at "
      f"step 61: hip pitch {dq['r_hip_pitch']}/{dq['l_hip_pitch']}, knee {dq['r_knee']}/{dq['l_knee']}, ankle "
      f"{dq['r_ankle']}/{dq['l_ankle']} (statue {E['statue']['joint_deg_vs_home']['at_steps']['61']['r_hip_pitch']}/"
      f"{E['statue']['joint_deg_vs_home']['at_steps']['61']['r_knee']}/{E['statue']['joint_deg_vs_home']['at_steps']['61']['r_ankle']} hip/knee/ankle right).")
ka = S["kinematic_attribution_com_fwd_shift_m"]["by_role_step"]
w(f"  feet-held kinematic attribution (CoM fwd minus CoM with the group at home): all joints s60 step 61 "
  f"{ka['s60_6M']['61']['all_joints']} m, s52 step 61 {ka['s52_6M']['61']['all_joints']} m, statue step 61 "
  f"{ka['statue']['61']['all_joints']} m; neck+head s60 {ka['s60_6M']['61']['neck_head']} m, tail "
  f"{ka['s60_6M']['61']['tail']} m. The joints stay near home; the forward CoM comes from the whole body staying "
  f"tipped forward on the toe edges, not from a joint excursion.")
w("")
ctx = S.get("context_audit_panel_heading0")
if ctx:
    w("7. CONTEXT: the audit panel 7042-7081 at heading 0 [context_audit_panel_heading0]")
    for r, blk in ctx.items():
        rows = blk["rows"]
        falls = [x["seed"] for x in rows if x["length"] < 1000]
        w(f"  {r}: falls {falls}; source {blk['source']}")
    st = {x["seed"]: x for x in ctx["ctx_statue"]["rows"]}
    order = sorted(st.values(), key=lambda x: x["spawn_mean_sole_pitch_deg"])
    rank = [x["seed"] for x in order].index(7047) + 1
    w(f"  spawn pad pitch (mean of feet): 7047 = {st[7047]['spawn_mean_sole_pitch_deg']} deg, rank {rank} of {len(order)} "
      f"most toe-down; panel range {order[0]['spawn_mean_sole_pitch_deg']} .. {order[-1]['spawn_mean_sole_pitch_deg']} deg.")
    order2 = sorted(st.values(), key=lambda x: -x["spawn_xcom_minus_pad_centre_m"])
    rank2 = [x["seed"] for x in order2].index(7047) + 1
    w(f"  spawn XCoM ahead of pad centre: 7047 = {st[7047]['spawn_xcom_minus_pad_centre_m']} m, rank {rank2} of {len(order2)}"
      f" (panel max {order2[0]['spawn_xcom_minus_pad_centre_m']}, min {order2[-1]['spawn_xcom_minus_pad_centre_m']}).")
    for r in ("ctx_s60_6M", "ctx_s52_6M", "ctx_statue"):
        rows = [x for x in ctx[r]["rows"] if x["seed"] != 7047]
        frac = [x["pad_cop_both_toe_edge_frac_steps_11_200"] for x in rows]
        mpt = [x["min_xcom_pad_toe_margin_m_steps_11_200"] for x in rows]
        mfu = [x["min_xcom_full_hull_margin_m_steps_11_200"] for x in rows]
        worst = min(rows, key=lambda x: x["min_xcom_pad_toe_margin_m_steps_11_200"])
        w(f"  {r}: lowest other-seed pad-toe margin on seed {worst['seed']}; all 39 other seeds reach the horizon: "
          f"{all(x['length'] == 1000 for x in rows)}")
        w(f"  {r}, the 39 other seeds, steps 11-200: pad CoP at the toe edge on both feet median "
          f"{sorted(frac)[len(frac) // 2]} (range {min(frac)}..{max(frac)}); min XCoM pad-toe margin median "
          f"{sorted(mpt)[len(mpt) // 2]} m (lowest {min(mpt)}); min XCoM pad+digit hull margin lowest {min(mfu)} m.")
    w("")
w("CAVEATS")
w("  - Offline evaluation of uncertified scratch-recipe checkpoints; one seed; certifies nothing.")
w("  - The loaded-contact hull ([events.*.xcom_vs_loaded_contact_hull]) is a thin sliver when a pad bears on its toe")
w("    edge only, so it reads 'outside' from step 11 even for the statue; the pad and pad+digit-tip hulls carry the story.")
w("  - The digit-tip hull uses the capsule tips of the loaded feet; the digits are passive (MTP springs), so that edge")
w("    is softer than the pad edge. The kinematic attribution holds each foot at its pose and breaks the closed chain.")
w("  - Geometry is the boundary state after each step (scratch MjData forward pass); loads and CoPs are the recorder's")
w("    means over the step's 5 substeps.")
(OUT / "shared_fall_report.txt").write_text("\n".join(L) + "\n")
print("\n".join(L))
