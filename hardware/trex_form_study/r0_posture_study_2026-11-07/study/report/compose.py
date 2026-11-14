"""Bind the review narrative and tables to the screened model revisions."""
import json
import csv
import hashlib
from pathlib import Path
from xml.sax.saxutils import escape
HERE=Path(__file__).resolve().parent
R=HERE.parent.parent
def read(p):return json.loads((R/p).read_text())
def short(s):return s[:12]
m=read('study/mechanics/metrics.json')
motion=read('study/mechanics/motion.json')
hardware=read('study/hardware/screen.json')
clear=read('study/mechanics/clearance.json')
prop={q['model_file'][:-4]:q for q in read('study/reference/proportions.json')}
led={k:read('study/model/generated/'+k+'.json') for k in m if k not in ('PR662','R3')}
cost={r['model']:r for r in csv.DictReader((HERE/'cover_work.csv').open())}
three=['A10','B10','C10']
for k in led:
    sha=hashlib.sha256((R/f'study/model/generated/{k}.xml').read_bytes()).hexdigest()
    assert sha==led[k]['xml_sha256']==m[k]['xml_sha256']==motion[k]['stand']['xml_sha256']==clear[k]['xml_sha256'],k
    if k in prop:assert sha==prop[k]['model_sha256'],k
pages=[]
def page(sec,title,*items):pages.append(dict(section=sec,title=title,items=list(items)))
def txt(x,y,w,text,size=11):return ['text',x,y,w,text,size]
def blocks(x,y,w,size,*pairs):return ['blocks',x,y,w,size,pairs]
def label(x,y,text,size=9):return ['label',x,y,text,size]
def im(path,x,y,w,h):return ['image',path,x,y,w,h]
def tab(x,y,w,cols,rs,widths,sz=10):return ['table',x,y,w,cols,rs,widths,sz]
def board(name,section):page(section,None,im('study/visuals/plates/'+name+'.png',22,38,856,582))
def torque(k):return abs(m[k]['single_support']['left']['joint_Nm']['left_hip_roll'])
def value(x):return f'{x:.1f}'
def href(url,s):return f'<a href="{url}" color="#276B5D"><u>{s}</u></a>'
pr662=href('https://github.com/kuds/mesozoic-labs/pull/662','PR #662')
pr663=href('https://github.com/kuds/mesozoic-labs/pull/663','PR #663')
source_link=href('https://chatgpt.com/library?file_id=libfile_fde3_f-ab3addefb9c48191998ce0b68aa8e7c1_5a238845a59881919f692be20e4c3c2e&amp;account_id=personal','Original R3 handoff')
source_note=f'The {source_link} is catalogued as a 269,274,463-byte ZIP. The archive could not be downloaded for this study. The candidate source below is independently verifiable and is included in the package.'
page('Design review',None,
 ['text',38,61,480,'Body shape<br/>and posture',45,'#173533','Times-Roman',47],
 txt(40,174,360,'A comparative design study for the intermediate robotic <i>Tyrannosaurus rex</i>.',15),
 im('study/report/hero.png',405,70,463,389),
 blocks(40,275,345,12,('B10 · Deep ribcage','The forward chest, deeper jaw and tapered midsection give the clearest adult T. rex profile. Trim its hip-side chest panel before copying that form to a control model.'),
        ('C10 · Open rib frame','Start short control experiments here. Open ribs preserve the service paths and leave the hip travel clear in the sampled motions.')),
 label(460,455,'B10   /   10° chest-up model, compiled geometry',9),
 ['rule',40,501,860],
 txt(40,522,820,f'The prescribed one-foot hold requires {torque("B10"):.2f} N·m at B10’s left hip roll and {torque("C10"):.2f} N·m at C10’s. The sustainable planning assumption is 6 N·m. Resolve lateral support and moving clearances before a long walking run.',14))
page('Provenance','Three different baselines',tab(34,98,832,['Source','Geometry used here','Evidence'],[
 ['#662 nominal','Compiled locally. The nine candidates inherit this model’s leg kinematics, joints and installed hardware ledger.','Head e103f7ff2609<br/>XML 06019e3c43c4<br/>9.896 kg'],
 ['R2 redesigned compact','Original PR #663 side, silhouette and three-quarter views. This is the body in the published redesign comparison.','XML 857d189b8a0d (published prefix)<br/>599.8 mm · 10.282 kg'],
 ['R3 narrow stance','Published three-quarter image, source record and reported loads. Its soles moved from 214 to 175 mm spacing, hips stayed 214 mm apart, and the fixed arms moved onto chassis rails.','XML 87e36ef3cffc<br/>599.8 mm · 10.370 kg']
 ],[.17,.54,.29],11),
 blocks(35,355,816,11,('Executable provenance',source_note),
 ('Exact revisions','The findings head is b5c18bce1e576542a51532ffb442c238d83b0c9a. It names handoff commit f7d9857791e0b75d388118fd38fac2d2ce2c0794. The published clean-checkout record binds the R3 physical XML, original policy and governed overlay. Those results belong to the original platform.'),
 ('Candidate lineage',f'Every candidate has a separate XML and SHA-256. The supplied PR #662 nominal XML and component ledger match the GitHub blobs at e103f7ff2609fa1d7eb08507fb50ed9bb410b632. The design study ran short mechanical probes only. {pr662} · {pr663}')))
board('published_baselines','R2 and R3   /   Published views')
page('Visual audit',None,im('study/reference/r2_visual_audit.png',23,43,854,570))
page('Geometry diagnosis','Where the bulk comes from',
 tab(34,95,832,['Constraint','What the views and hardware show','Design response'],[
 ['A · Installed parts','Twelve leg drives carry 4.32 kg before mounts. Hip and ankle cylinders project sideways. The flat sole and raised hock lock in a distinctly robotic lower leg.','Keep articulation visible. Give the upper thigh a moving fairing; judge frontal width on the cases as well as on the ribcage.'],
 ['B · Component arrangement','The battery, compute tray, disconnect, camera and expression drives each need local space and access. A straight tray and fixed aft hardware push mass and panel edges behind the hip.','Use stepped rails, separate the fan outlet and hatch, and carry the rear belly upward. Recheck the actual extraction stroke and cable path.'],
 ['C · Changeable surfaces','On R2, a broad lid crosses the hip. Its two low side lobes create an inverted V over the axle. The short rising neck meets the front of the shell abruptly.','Move maximum chest depth forward, recover the belly toward the pelvis, broaden the rear skull and carry a tapered pelvic root around the tail joint.'],
 ['Views and limits','The R2 plate shows side, oblique and black profile. The R3 partner brief adds its distinct oblique view. The front and top audit uses the compiled PR #662 baseline and each generated candidate.','Use the dossier’s four-view sheets and transverse cuts to identify where skin sets the outline and where a joint housing sets it.']
 ],[.175,.43,.395],11.4),
 txt(36,500,826,'The published R2 side view places roughly 60% of the tip-to-snout span behind the hip, but only about 38% behind the visible tail seam. Exposing the proximal transition changes the apparent tail length without adding a heavier distal lever.',13))
page('Anatomical references',None,im('study/reference/reference.png',26,41,848,530),
 txt(43,574,815,href('https://www.fieldmuseum.org/blog/fresh-science-makeover-sue','Field Museum, SUE remount (2018)')+'  ·  '+href('https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0026037','Hutchinson et al. (2011), figs. 2–4, 7 and table 3')+'  ·  '+href('https://www.app.pan.pl/article/item/app48-161.html','Hurum &amp; Sabath (2003), skull comparisons'),9.7))
board('pr663_side_comparison','Exterior comparison   /   Side')
board('family_exteriors','Exterior comparison   /   Three-quarter')
board('silhouettes','Physical scale and normalized height')
board('posture_matrix','Posture matrix   /   Nine models')
for k in three:board(k+'_four_views','Multi-view exterior   /   '+k)
board('hardware_cutaways','Service view   /   Installed components')
page('Transverse sections',None,im('study/sections/transverse_sections.png',20,40,563,575),
 blocks(605,97,248,11,('Chest depth','At the forward section, B carries its shell down the sides of the chest. A recovers earlier. C uses dorsal skin and intermittent ribs around an open flank.'),
        ('Aft of the hip','The battery and drive mounts occupy separate regions. The sparse space between them can remain open or carry a shallow fixed fairing.'),
        ('Reading the cuts','Each column is a vertical plane at the same fore-aft offset from the hip pitch axis. Dashed lines mark the sagittal center and hip height. The cuts show intersections with actual geometry.')))
page('Design parameters','How the 0°, 10° and 20° poses are defined',
 blocks(35,95,400,11.5,('Chest angle','The angle is atan2(Δz, Δx) between rail datums at local x = +65 and +165 mm. Positive values raise the anterior datum. Both datums follow the front bracket. The pelvic hardware remains level.'),
 ('Neck, head and tail','The two-stage cervical rise ends at the supported head-pitch shaft. The skull carries its own pitch offset. The tail has a separate molded drop and yaw shaft. Arms and jaw are fixed.'),
 ('Feet and legs','The bilateral home solution retains the #662 neutral joint coordinates with both sole undersides flat at floor level. The built-in link offsets establish the thigh, knee and hock zigzag. The posture sweep isolates chest position. Transfer tests separately flex, counter-roll and lift the swing leg.')),
 tab(462,95,404,['At 10°','A','B','C'],[
 ['Neck run / rise (mm)']+[f'{led[k]["parameters_mm"]["neck_run"]} / {led[k]["parameters_mm"]["neck_rise"]}' for k in three],
 ['Skull pitch']+[f'{led[k]["head_pitch_deg"]:+.0f}°' for k in three],
 ['Head size L/W/D (mm)']+[' / '.join(str(round(prop[k][s])) for s in ('skull_x_mm','skull_width_mm','skull_depth_mm')) for k in three],
 ['Trunk cover width (mm)']+[str(round(prop[k]['trunk_cover_width_mm'])) for k in three],
 ['Covered side area (cm²)']+[str(round(prop[k]['trunk_cover_side_projected_cm2'])) for k in three],
 ['Area ahead of hip']+[f'{prop[k]["trunk_cover_projected_percent_ahead_of_hip"]:.0f}%' for k in three],
 ['Powered tail length setting (mm)']+[str(led[k]['parameters_mm']['tail_len']) for k in three]
 ],[.36,.214,.213,.213],9.3),
 txt(463,420,398,'Head spans include the modeled assembly. Projected covering area is the side-view union of torso sheets at 0.5 mm resolution. The rigidly installed chest bracket is a separate design for each angle; the robot has no powered torso-pitch joint.',10.4))
def inst(k):
 h=hardware[k]; b=h['battery_routes']['left']; c=h['compute_service']
 return [k,f'{len([q for q in h["shell_contacts"] if q["type"]!="reserve" and q["test"]=="intersection"])}',str(len(b['remaining_after_removing_all_intersecting_covers'])),str(len(c['blockers_after_removing_intersecting_covers'])),str(h['cooling']['rays_open_to_300mm'])+'/100']
page('Installation','Service clearances decide where the skin can close',
 tab(35,95,830,['Part','Installed basis','Opening or access requirement'],[
 ['Leg and expression drives','12 × RMD-X4-P36, Ø55 × 61 mm in Ø60 × 64 mm installation cases. Two XM430-W210-T servos, 28.5 × 46.5 × 34 mm.','Allow harness flex and supported outputs. The augmented geometry retains fixed arms, powered neck pitch and powered tail yaw.'],
 ['Battery','6S 3,300 mAh pack; tolerance-max 143 × 44 × 45 mm before leads. B and C put its long axis across the body.','240 mm continuous lateral extraction with a sleeve and clearance allowance. Remove the intersecting flank pieces and retain disconnected leads.'],
 ['Compute and power','Orin Nano kit limit 103.2 × 90.7 × 35.86 mm with cooler. Separate carrier, converter boards, MCU/CAN, power protection and disconnect reservations.','50 mm lift, full 250 mm extraction, alternate pulls and plug corridors. Fan test samples unobstructed rays; a complete inlet-to-outlet duct needs a thermal bench.'],
 ['Optics','OAK-D Lite 91 × 28 × 17.5 mm. Two VL53L1X carrier boards, 13 × 18 × 2 mm before pins.','Provide the real optical apertures, USB clamp and wire terminations. Retain clearance through head travel and whole-body tilt.']
 ],[.17,.43,.40],10.1),
 tab(35,375,830,['10° model','Part–skin pairs intersecting¹','Battery pull: fixed blockers','Jetson lift: fixed blockers','Upward fan rays clear'],[inst(k) for k in three],[.13,.21,.23,.23,.20],9.8),
 txt(36,525,826,'¹ Includes screened physical case/skin intersections at the neutral pose. Each blocker count is after removing covers that cross the chosen stroke; fixed rails or allocated service reserves may remain. Complete three-dimensional positions, connector tests and alternate removal directions are in study/hardware/installed_layout.csv and hardware/screen.json.',10.5))
table=[]
for k in led:
 row=m[k];home=motion[k]['stand'];tr=motion[k]['transfer']
 table.append([k,f'{row["crown_mm"]:.0f}',f'{row["dimensions_mm"][0]:.0f}',f'{row["mass_kg"]:.3f}',', '.join(value(v) for v in row['com_mm']),f'{row["double_support_hull_margin_mm"]:.1f}',f'{torque(k):.2f}', 'pass' if home['passed'] else 'fail','pass' if tr['passed'] else 'fail'])
page('Mechanical screening','Standing and transfer outcomes for all nine designs',
 tab(34,97,832,['Model','Crown mm','Length mm','Mass kg','CoM x, y, z mm','Two-foot margin mm','Left hip roll N·m','Stand','Transfer'],table,[.072,.075,.077,.076,.208,.138,.15,.10,.104],9.1),
 txt(35,430,820,'All models retain a 360 mm hip height and a 354 mm maximum installed width. +X is forward, +Y is left, and +Z is up from the floor. Both sole undersides are level at home. The margin measures the nearest edge of the two-sole convex hull. A positive value here can coexist with excess torque on one foot.',10.7),
 txt(35,498,820,'Stand uses 2 seconds of free-base actuation. The 4.8-second open-loop transfer probe failed on #662 and all nine candidates; it cannot establish a ranking of learned gaits. A pass requires at least 100 ms on each side with an 8 mm lifted swing sole, ≥80% body weight on the stance foot, ≤5% on the swing foot, and limited tilt. These are candidate screening results; the prior R3 controller has a different plant identity.',10.7))
page('Service access',None,im('study/hardware/service_paths.png',25,52,850,493),
 txt(44,551,810,'Both B10 and C10 use the transverse layout. Their 240 mm battery withdrawal and 250 mm processor extraction clear fixed geometry after named covers are removed. A10 retains ten fixed obstructions on its battery path. The final fan opening clears 60 of 100 sampled vertical rays in each case; that test measures geometry only.',10.9))
page('Equilibrium under load','An upright chest does little for hip-roll demand',
 im('study/mechanics/single_support.png',28,96,842,373),
 txt(36,490,826,'Moving the chest changes the fore–aft and vertical position of the center of mass. In a one-foot hold, the large roll moment comes principally from the lateral offset between the supported mass and the loaded hip. This sweep leaves the leg spacing fixed. The published narrower R3 already exceeded the 6 N·m sustainable assumption at about 9 N·m.',11.6),
 txt(36,555,826,'Allowing pelvis roll, counter-roll and swing-leg adduction reduced the best sampled left-hip demands to 8.52 / 8.64 / 8.52 N·m for A10 / B10 / C10. Each still exceeds 6 N·m. The alternate poses were screened for equilibrium and joint limits; collision and thermal validation remain open.',10.8))
def mechrow(k):
 d=m[k];j=d['single_support']['left']['joint_Nm'];a=d['appendage_loads'];c=clear[k]['summary'];o=d['optics']
 return [k,' / '.join(f'{abs(j["left_"+s]):.2f}' for s in ('hip_pitch','knee','ankle_roll')),f'{abs(a["neck_pitch"]["gravity_axis_Nm"]):.3f}',f'{a["tail_yaw"]["gravity_off_axis_bending_Nm"]:.3f}',f'{min(q["min_tail_ground_mm"] for q in c.values()):.0f}',f'{o["self_occluded_rays"]}/{o["grid_rays"]}',f'{o["lowest_center_column_floor_forward_m"]:.2f}']
page('Motion and sensing','What the short movement tests expose',
 tab(35,95,830,['Model','Single-foot hip pitch / knee / ankle roll N·m','Neck pitch gravity N·m','Tail shaft bending N·m','Min tail–floor mm¹','Camera rays blocked²','Lowest visible floor m²'],[mechrow(k) for k in three],[.075,.25,.15,.15,.12,.13,.125],9.5),
 blocks(36,275,396,10.7,('Swept clearances','In 44 sampled poses, C10 had no measured cover gap under 3 mm. B10 avoided cover penetration at home, in 16 transfer samples and at 9 head/tail positions, but hit a hip drive in 6 of 18 extreme configurations. Both have one extreme leg-to-leg close approach. Finite sampling can miss an intermediate clash.'),
 ('Swing-foot height','A position solver can place the swing foot well above the floor. Under gravity, finite drive response and shifting contact loads, the same mechanism need not achieve that clearance or transfer its weight. The bench records actual sole loading, drift, peak/RMS torque and non-sole support.')),
 blocks(462,275,399,10.7,('Supported shafts','Gravity creates almost zero yaw effort on a vertical tail axis while bending the tail bearings. A head with a horizontal pitch axis sees gravity torque even while looking forward. The complete six-component work and link inertia are supplied for sizing.'),
 ('Sensor coverage','The torso camera and the two downward-ranging center rays move with the installed bracket. Inspect low rails, off-axis obstructions, dark carpet and rear sweep separately. A clear central camera ray or two floor returns cover only a fraction of the footprint.')),
 txt(36,565,826,'¹ Minimum in the prescribed grounded-pelvis pose set, including tail motion.  ² Neutral camera samples and center-column floor distance. Moving optics results and contact witnesses are in mechanics/clearance.json.',9.7))
def csection(k):
 d=cost[k]
 return [k,str(d['removable_pieces']),f'{float(d["area_m2"]):.3f}',f'{float(d["sheet_kg"]):.3f}',f'{float(d["mount_budget_kg"]):.3f}',f'${d["fitup_labor_usd_low"]}–${d["fitup_labor_usd_high"]}']
page('Manufacture','Hollow covers, separate mounts and a cost basis',
 tab(35,96,830,['10° model','Shell pieces','Surface area m²','Sheet mass kg','Mount allowance kg','Fit-up labor scenario'],[csection(k) for k in three],[.11,.14,.18,.17,.19,.21],10.4),
 blocks(36,258,398,11,('Shell construction','Candidate sheets have an inner surface, outer surface and trimmed edges. Mass comes from area × thickness × 1,010 kg/m³ for unfilled PA12. Triangle integration gives each center and central inertia; parallel-axis terms place it on the moving body.'),
 ('Build method','Use removable powder-bed nylon panels on the metal frame. Thickness, per-panel hardware, mount positions and inertia are in each candidate ledger. Actual inserts, retention, edge stiffness and finite tolerances require fabrication CAD and printed coupons.')),
 blocks(461,258,402,11,('Cost comparison','Illustrative bench allowance: 4–8 minutes per separately handled panel at $60–90/hour. This is one fit-up pass. The printed part, finishing and engineering require a supplier quote. C’s thin ribs add assembly operations even when they reduce skin area.'),
 ('Prototype budget','PR #663 estimated $16,447 nominal and $21,817 high for the complete R3 prototype. A new cost total depends on nesting, revised brackets, ducting, harness routing and the drive decision. Its existing cover allowance must be replaced when quotes arrive.')))
page('Downselection','Carry B10’s body form and C10’s service layout forward',
 tab(35,95,830,['Candidate','Recognition and motion cues','Engineering trade and disposition'],[
 ['A10 · Tapered flank','Lightest-looking enclosed profile. The open midsection separates pelvis and chest, but the hip notch, narrow shins and flat foot still dominate in side view.','Useful control for a smaller covering area. Its longitudinal pack still obstructs side extraction. Retain the model and results.'],
 ['B10 · Deep ribcage','Best head-to-chest balance in side, plan and front views. The broadest forward chest and fuller cheek give it an adult T. rex silhouette. Revisit the tail seam and the transition over the hip in detailed CAD.','Primary visual candidate. The transverse battery and full compute lift now clear. The hip drive still penetrates the chest covering in 6 of 18 extreme poses. Relieve that sector and rerun the loaded sweep.'],
 ['C10 · Open rib frame','The ribs make the frame and hardware understandable in three-quarter view. In black side view the battery case and separated ribs interrupt the belly.','First geometry for short control tests. No measured cover gap under 3 mm in 44 sampled poses. One extreme opposite-leg close approach remains. Preserve access while strengthening the ventral chest read.']
 ],[.18,.4,.42],10.5),
 blocks(36,405,824,10.9,('What would change the choice','A successful guarded battery/Jetson service rehearsal and a collision-free loaded transfer would move a candidate into controller tuning. A blind side/front/plan review that consistently favors C, or a substantially smaller build volume quoted for A, would reopen the body selection.'),
 ('Next physical work','Weigh the assembled trays and sample covers; verify neck/tail hubs and cable loops in full motion; measure drive torque, supply sag and thermal duty on one loaded leg; then repeat the transfer with measured contacts, backlash and deflection. Keep the existing R3 controller as the reference for a future port.')))
page('Reproduction','Reopen the models and repeat the checks',
 blocks(35,99,827,11,('Start at the package root','README.md gives a single reproduction command, the output map and paths for opening the models in MuJoCo. The nine MJCFs are self-contained. Each candidate JSON binds the model SHA-256, source SHA-256, panel geometry, inertia and the component ledger. SHA256SUMS checks the distributed files.'),
 ('Verified input','source/pr662/model.xml — SHA-256 06019e3c43c48376d7d048f0e9c4b540df0094459247125b9d7c35f7d9ce6f66. Its component ledger is 6beb0ea7b6d1961b87d4163dcd9f96a2dcfcfb80acaffd3891444b44c08fa077. Source files, source links and the published R2/R3 evidence are under source/.'),
 ('Rebuild nine candidates','python study/model/build.py --source source/pr662/model.xml --ledger source/pr662/components.json'),
 ('Inspect or rerun','python -m mujoco.viewer --mjcf=study/model/generated/B10.xml<br/>The mechanics and hardware subdirectories give individual commands and per-pose JSON results. The PNG plates include side, front, top and three-quarter views of every posture.'),
 ('Original evidence and references',f'{pr662} · {pr663} · {source_link} · {href("https://www.fieldmuseum.org/blog/fresh-science-makeover-sue","Field Museum")} · {href("https://www.app.pan.pl/article/item/app48-161.html","Hurum and Sabath")} · {href("https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0026037","Hutchinson et al.")}. Hardware source links and tolerance assumptions accompany the part list.')))
(HERE/'pages.json').write_text(json.dumps(pages,indent=2,ensure_ascii=False)+'\n')
