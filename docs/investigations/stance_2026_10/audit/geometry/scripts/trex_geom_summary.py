"""Render geometry_audit.json into a plain-text table against the report's model numbers.

Usage: python -P trex_geom_summary.py <geometry_audit.json> <out.txt>
"""

import json
import sys

R = json.load(open(sys.argv[1]))
H = R["home_keyframe"]
F = R["settle"]["final_state_measure"]
W = R["settle"]["window_means"]
Lh, Lf = H["legs"]["r"], F["legs"]["r"]
Gh, Gs = R["stiffness_groups"]["home_floor"], R["stiffness_groups"]["settled_floor"]
Ga = R["stiffness_groups"]["settled_ankle"]
Gm = R["stiffness_groups"]["settled_mtp3"]
rng = R["ranges_included_angle"]
T1 = R["tail_1_holding_budget_at_ventral_stop"]
dyn = R["balance_authority_dynamic_settles"]
env_auth = R["balance_authority_env_pathway"]

rows = [
    ("total mass (kg)", "85.7", f"{R['masses']['total_kg']:.2f}", "same at settle"),
    ("COM ahead of hip (% femur)", "42%", f"{H['com_ahead_of_hip_pct_femur']:.1f}% ({H['com_ahead_of_hip_m']:.4f} m)",
     f"settled {W['com_ahead_of_hip_pct_femur']['mean']:.1f}% ({W['com_ahead_of_hip_m']['mean']:.4f} m)"),
    ("COM vs hip height", "at hip height, 2% above", f"{H['com_above_hip_pct_femur']:+.1f}% femur ({H['com_above_hip_m']:+.4f} m)",
     f"settled {100*W['com_above_hip_m']['mean']/Lh['femur_hip_to_knee_m']:+.1f}% femur ({W['com_above_hip_m']['mean']:+.4f} m)"),
    ("COM height above floor (m)", "0.88", f"{H['com_height_above_floor_m']:.4f}", f"settled {W['com_height_above_floor_m']['mean']:.4f}"),
    ("COM projection, signed pad statistic", "+0.10", f"{H['com_proj_pad_stat_signed_mean_feet']:+.3f}",
     f"settled {W['com_proj_pad_stat_signed_mean_feet']['mean']:+.3f} (window {W['com_proj_pad_stat_signed_mean_feet']['min']:+.3f}..{W['com_proj_pad_stat_signed_mean_feet']['max']:+.3f})"),
    ("pad-only CoP statistic (gate quantity) at statue", "(not given)", "keyframe instant +1.00 (solver artefact)",
     f"settled {W['pad_cop_stat_signed_r']['mean']:+.3f} (window {W['pad_cop_stat_signed_r']['min']:+.3f}..{W['pad_cop_stat_signed_r']['max']:+.3f})"),
    ("whole-foot CoP, pad half-lengths", "(not given)", "-", f"settled {W['wholefoot_cop_signed_r']['mean']:+.3f}"),
    ("footprint pad rear -> digit III tip (m)", "~0.23", f"{Lh['footprint_pad_rear_to_d3_tip_incl_radius_m']:.3f} (0.200 to capsule end, no cap)", "same"),
    ("footprint / hip height", "0.26", f"{H['footprint_over_hip_height']:.3f} (hip {H['hip_mid_height_m']:.4f} m)", f"settled {F['footprint_over_hip_height']:.3f}"),
    ("pad about digit III MTP (m)", "-0.01 to +0.13",
     f"{Lh['pad_rel_mtp']['d3']['pad_rear_edge_rel_mtp_m']:+.3f} to {Lh['pad_rel_mtp']['d3']['pad_front_edge_rel_mtp_m']:+.3f}", "same"),
    ("mgh about floor (N m/rad)", "~740", f"{H['mgh_floor_Nm_per_rad']:.1f}", f"settled {W['mgh_floor_Nm_per_rad']['mean']:.1f}"),
    ("mgh about ankle axis (N m/rad)", "(not given)", f"{H['mgh_ankle_Nm_per_rad']:.1f}", f"settled {W['mgh_ankle_Nm_per_rad']['mean']:.1f}"),
    ("six digit springs (sum) / mgh", "240, ~1/3", f"{Gh['six_digit_springs_sum']:.0f} = {Gh['six_digit_springs_over_mgh']:.3f}",
     f"settled floor {Gs['six_digit_springs_over_mgh']:.3f}; MTP pivot {Gm['six_digit_springs_over_mgh']:.3f}; ankle pivot {Ga['six_digit_springs_over_mgh']:.3f}"),
    ("two ankle servos kp (sum) / mgh", "1,800, ~2.4x", f"{Gh['two_ankle_servo_kp_sum']:.0f} = {Gh['two_ankle_servo_kp_over_mgh']:.3f}",
     f"settled floor {Gs['two_ankle_servo_kp_over_mgh']:.3f}; ankle pivot {Ga['two_ankle_servo_kp_over_mgh']:.3f}"),
    ("metatarsus / femur", "0.61", f"{Lh['metatarsus_over_femur']:.3f} ({Lh['metatarsus_ankle_to_mtp3_m']:.4f}/{Lh['femur_hip_to_knee_m']:.4f} m)", "same"),
    ("tibia / femur", "1.00", f"{Lh['tibia_over_femur']:.3f}", "same"),
    ("digit share of foot load", "~22-23% at home", f"keyframe instant {H['contacts']['r']['digit_share']*100:.1f}%",
     f"settled {W['digit_share_contacts_r']['mean']*100:.1f}% (window {W['digit_share_contacts_r']['min']*100:.1f}-{W['digit_share_contacts_r']['max']*100:.1f}%)"),
    ("digit load position (0 MTP, 1 tip)", "(not given)", "-",
     f"settled d2 {W['digit_load_pos_frac_r_d2']['mean']:.3f}, d3 {W['digit_load_pos_frac_r_d3']['mean']:.3f}, d4 {W['digit_load_pos_frac_r_d4']['mean']:.3f}; load centre {1000*W['digit_load_centre_ahead_of_mtp3_m_r']['mean']:.1f} mm ahead of MTP III"),
    ("digit spread II-III / III-IV / total (deg)", "~9 / ~8 / ~16", f"{Lh['spread_II_III_deg']:.2f} / {Lh['spread_III_IV_deg']:.2f} / {Lh['spread_total_II_IV_deg']:.2f}", "same"),
    ("segment shares (% body mass)", "head 10.1, neck 5.8, tail 18.2; per side thigh 5.8, shank 4.1, foot 2.1",
     "head {:.1f}, neck {:.1f}, tail {:.1f}, trunk {:.1f}, forelimbs {:.2f}; per side thigh {:.2f}, shank {:.2f}, foot {:.2f}, digits {:.2f}".format(
         R["masses"]["segments"]["head (skull body)"]["pct"], R["masses"]["segments"]["neck"]["pct"],
         R["masses"]["segments"]["tail (tail_1..tail_5)"]["pct"], R["masses"]["segments"]["trunk (pelvis body: torso+ribcage+belly)"]["pct"],
         R["masses"]["segments"]["forelimbs (both)"]["pct"], R["masses"]["per_side"]["r"]["thigh"]["pct"],
         R["masses"]["per_side"]["r"]["shank (tibia)"]["pct"], R["masses"]["per_side"]["r"]["foot (metatarsus+pad+digits)"]["pct"],
         R["masses"]["per_side"]["r"]["digits (d2+d3+d4)"]["pct"]), ""),
    ("tail pitch actuators", "+-0.21 rad (+-12 deg), kp 60-100", "ctrl +-0.2094 rad; kp 100/80/60; forcerange 80/65/50 N m",
     f"tail_1 rests on its -12 deg stop at the statue; full-up command short by {-T1['net_lift_margin_Nm']:.1f} N m"),
]
out = ["Geometry audit, anatomical T. rex, main 2e734e29 (physics r9). Source: geometry_audit.json (same directory).",
       "Columns: quantity | report value | home keyframe (measured) | env settle, noise-free zero-action statue (measured)", ""]
for q, rep, h, s in rows:
    out.append(f"{q}\n    report: {rep}\n    home:   {h}\n    settle: {s}")
out.append("")
out.append("Joint ranges in the included-angle convention (sagittal, 180 = straight; right leg, others at home):")
for k in ("hip", "knee", "ankle (intertarsal)", "MTP / digit III"):
    v = rng[k]
    out.append(f"  {k}: MJCF qpos {v['mjcf_range_deg'][0]:.1f}..{v['mjcf_range_deg'][1]:.1f} (ref {v['ref_deg']:.1f}, springref {v['springref_deg']:.1f})"
               f" -> included {v['included_at_range_lo_deg']:.1f}..{v['included_at_range_hi_deg']:.1f}, home {v['included_at_home_deg']:.1f},"
               f" settled {v['included_at_settled_deg']:.1f}; d(included)/d(qpos) = {v['d_included_d_qpos']:+.0f}")
out.append(f"  hip femur-from-vertical at range ends: {rng['hip']['femur_from_vertical_at_lo_deg']:.1f} .. {rng['hip']['femur_from_vertical_at_hi_deg']:.1f} (home {rng['hip']['femur_from_vertical_at_home_deg']:.1f}, settled {Lf['included_angles_deg']['femur_from_vertical_knee_fwd_positive']:.1f})")
out.append(f"  digit III below horizontal at range ends (metatarsus at home slope): {rng['MTP / digit III']['digit3_below_horizontal_at_lo_deg']:.1f} .. {rng['MTP / digit III']['digit3_below_horizontal_at_hi_deg']:.1f}")
out.append("")
out.append("Balance authority (quasi-static; window means; pad statistic units):")
for k, v in dyn.items():
    out.append(f"  {k}: COM proj {v['window_com_proj_pad_stat']:+.3f} (d {v['delta_com_proj_stat_vs_baseline']:+.3f}), pad CoP {v['window_pad_cop_stat_r']:+.3f}, whole-foot CoP {v['window_wholefoot_cop_r']:+.3f}, fell={v['fell']}, tail_1 {v['final_joint_deg']['tail_1_pitch']:.2f} deg")
for k, v in env_auth.items():
    out.append(f"  ENV {k}: first terminated step {v['first_terminated_step']}, min pelvis z {v['min_pelvis_z']:.3f}")
out.append(f"  tail_1 at ventral stop: gravity {T1['gravity_torque_at_stop_Nm']:.1f} N m vs servo(full up) {T1['servo_torque_full_up_at_stop_Nm']:.1f} + spring {T1['spring_torque_at_stop_Nm']:.1f}")
open(sys.argv[2], "w").write("\n".join(out) + "\n")
print("\n".join(out))
