"""Build compact tables from the raw mechanical results."""
from pathlib import Path
import json
from screen import BASE

def read(name):
    return json.loads((BASE/name).read_text())

def table(headers,rows):
    return "\n".join(["| "+" | ".join(headers)+" |","|"+"|".join("---" for _ in headers)+"|"]+
                     ["| "+" | ".join(map(str,row))+" |" for row in rows])

def nm(v,key):
    return f'{abs(v[key]):.2f}'

def max_pair(v,keys):
    return max(abs(v[s+'_'+k]) for s in ('left','right') for k in keys)

def main():
    metrics=read('metrics.json');moves=read('motion.json');gaps=read('clearance.json')
    bodies=[];efforts=[];motions=[];optics=[];artic=[];clear=[]
    for name,v in metrics.items():
        xyz=', '.join(f'{n:.1f}' for n in v["com_mm"])
        bodies.append([name,f'{v["mass_kg"]:.3f}', xyz,f'{v["double_support_hull_margin_mm"]:.1f}',f'{v["neutral_sole_corner_spread_mm"]:.2f}'])
        n=v['neutral_load']['joint_Nm'];left=v["single_support"]["left"];right=v["single_support"]["right"]
        s=max([left,right],key=lambda a:max_pair(a["joint_Nm"],["hip_roll"]))
        t=s["joint_Nm"];side='left' if s is left else 'right'
        f=lambda x:f'{x:.2f}'
        efforts.append([name,f(max_pair(n,['hip_pitch','hip_roll'])),f(max_pair(n,['knee'])),f(max_pair(n,['ankle_pitch','ankle_roll'])),
                        f'{abs(left["joint_Nm"]["left_hip_roll"]):.2f} / {abs(right["joint_Nm"]["right_hip_roll"]):.2f}',
                        f'{abs(t[side+"_hip_pitch"]):.2f} / {abs(t[side+"_knee"]):.2f} / {max(abs(t[side+"_ankle_pitch"]),abs(t[side+"_ankle_roll"])):.2f}',
                        f'{min(left["cop_margin_mm"][0],right["cop_margin_mm"][0]):.1f}',f'{min(left["swing_sole_clearance_mm"],right["swing_sole_clearance_mm"]):.1f}'])
        a=v['appendage_loads']
        artic.append([name, *(f'{abs(z):.3f}' for z in (a['neck_pitch']['gravity_axis_Nm'],a['tail_yaw']['gravity_axis_Nm'],a['tail_yaw']['gravity_off_axis_bending_Nm'])),
                      f'{a["neck_pitch"]["reflected_link_inertia_kgm2"]:.4f} / {a["tail_yaw"]["reflected_link_inertia_kgm2"]:.4f}'])
        o=v['optics'];tc=o['tof']; tip=gaps.get(name,{}).get('summary',{}).get('articulation',{}).get('min_tail_ground_mm')
        optics.append([name,f'{o["self_occluded_rays"]}/187',o['camera_center_hit'][0] or 'clear',f'{o["lowest_center_column_floor_forward_m"]:.2f}' if o["lowest_center_column_floor_forward_m"]!=None else 'above horizon',
                       ' / '.join((tc[s]['hit'] or 'clear') for s in ('left','right')),f'{tip:.0f}' if tip!=None else '—'])
        if name in moves:
            stand=moves[name]["stand"]; x=moves[name]["transfer"]; longest=x["longest_loaded_single_support_s"]
            motions.append([name,('pass' if stand['passed'] else 'fail')+f' ({stand["max_pelvis_tilt_deg"]:.1f}°)', 'pass' if x['passed'] else 'fail',f'{x["max_pelvis_tilt_deg"]:.1f}°',
                            f'{longest[0]:.2f} / {longest[1]:.2f}',f'{max_pair(x["peak_joint_Nm"],["hip_roll"]):.2f}',x['first_nonsole_support']['geom'] if x['first_nonsole_support'] else 'none'])
        g=gaps.get(name,{}).get('summary')
        if g:
            clr=[name]
            for family in ['home','transfer','articulation','joint_box']:
                cases=[case for label,case in gaps[name]['cases'].items() if label==family or label.startswith(family+'_')]
                count=sum(any(w["mm"]<0 or w.get("transverse_triangle_intersection",False) for w in case["witnesses_under_3mm"]) for case in cases)
                clr.append(f'{count}/{len(cases)}')
            clear.append(clr)
    body="""# Mechanical screen

Distances are in the model world frame, measured in millimetres unless stated. +X is forward, +Y is left and +Z is up. The origin in the horizontal plane is the neutral root. The model’s assigned inertials include each installed component and modeled cover. The floor is level. Each table uses the model identified by its filename and SHA-256 in `metrics.json`.

## Standing support

The hull is the convex hull of both rectangular sole undersides. The margin is the signed horizontal distance from the system center of mass to its nearest edge. A positive double-support margin establishes geometric support for this pose. It cannot establish centering over one foot.

"""
    body+=table(['Model','Mass kg','COM x, y, z','Two-sole margin','Corner height spread'],bodies)
    body+="""

## Static joint loads

Gravity and the support reactions balance all six root forces and moments. For two-foot support each vertical reaction acts at COM.x on a sole centerline, with the load split solving lateral moment balance. Single-foot screening counter-rolls hips and ankles in 0.005 rad increments from 0 through 0.45 rad subject to the model limits, with the swing hip/knee/ankle adding −0.25/+0.50/−0.25 rad. The standing sole is translated to the floor. The row selects the largest signed center-of-pressure margin on each side, using the published platform flat-foot search convention. Marks are prescribed gravity loads in N·m, without friction, acceleration or impact. The hip-roll column retains both sides. The single-support hip pitch, knee and ankle trio is for the side with the larger hip-roll load. Ankle is the larger of roll/pitch. Raw files include each signed joint torque and margin for each foot. A negative margin describes an infeasible gravity-only equilibrium.

"""
    body+=table(['Model','2-foot hip','2-foot knee','2-foot ankle','1-foot hip roll L/R','1-foot hip pitch/knee/ankle','Worst 1-foot margin','Minimum swing gap'],efforts)
    if (BASE/'pelvis_probe.json').exists():
        alt=read('pelvis_probe.json')
        probe_rows=[]
        for k,v in alt.items():
            if k not in metrics:continue
            assert v.get('xml_sha256',metrics[k]['xml_sha256'])==metrics[k]['xml_sha256'],k
            a,b=[v[s]['min_hip_roll'] for s in ('left','right')]
            probe_rows.append([k,f"{a['hip_roll_Nm']:.2f} / {b['hip_roll_Nm']:.2f}",f"{a['pelvis_roll_deg']:+.0f}° / {b['pelvis_roll_deg']:+.0f}°",f"{a['sole_margin_mm']:.1f} / {b['sole_margin_mm']:.1f}"])
        body+="\n\nAn additional stance search permits five independent pelvis roll angles (−10° through +10°), thirteen support counter-roll angles and three levels of swing-leg adduction. It ranks joint-limit-compliant gravity poses with sole margin ≥3 mm, flat support and swing gap ≥8 mm by hip-roll effort. The minima below precede interference screening. None reaches the 6 N·m sustained assumption.\n\n"
        body+=table(['Model','Best 1-foot hip roll L/R N·m','Pelvis roll L/R','Sole margin L/R mm'],probe_rows)
    body+="""

## Actuated head and tail

The tail yaw shaft is vertical in neutral, while tail weight creates a bending moment on its support. The inertia is the sum of the moving bodies’ central tensors rotated to the pivot frame plus their parallel-axis contributions. Values exclude motor rotor inertia, friction and acceleration commands.

"""
    body+=table(['Model','Neck gravity N·m','Tail yaw gravity N·m','Tail off-axis N·m','Neck / tail inertia kg·m²'],artic)
    body+="""

## Free-base motion

Each run uses the installed drive at 22.2 V, retained delay/deadband, torque-speed limit and rolling effort derating. Two seconds of neutral hold and a separate 4.8-second trial with two prescribed lateral transfer cycles are run from home on the model floor. The latter simultaneously commands hip/ankle counter-roll up to 0.24 rad, swing knee up to 0.50 rad, hip and ankle pitch compensation, neck pitch ±0.15 rad and tail yaw ±0.20 rad. A valid loaded single-foot interval needs ≥80% body weight on the stance sole, ≤5% on the other sole, swing gap ≥8 mm and no other floor support over 1 N. Both tests require tilt <12° and sole-only support. Stand requires drift <25 mm. Transfer requires ≥0.10 s loaded on each side. Trajectories are open-loop probes. `motion.json` includes per-axis peak, RMS, time above continuous assumptions and the lowest tail surface sampled at 50 Hz.

"""
    body+=table(['Model','Stand (tilt)','Transfer','Transfer max tilt','Longest L/R s','Peak hip roll N·m','Other floor support'],motions)
    body+="""

## Visibility and tail

Neutral camera ray tests use the 17×11 grid of a 640×400 image at the installed vertical field of view. The ideal floor distance is measured horizontally ahead of the camera at the bottom of its center column on a level floor. Physical self-occlusion uses MuJoCo rays through visible geometry. Each ToF center ray is tested against the model. The raw clearance results repeat this test for nine joint-limit samples of head pitch and tail yaw. Sensor calibration, depth validity, side/rear coverage, detection and tolerance require physical tests.

"""
    body+=table(['Model','Blocked image rays','Center ray','Ideal floor distance m','ToF hits L/R','Lowest tail through articulation mm'],optics)
    body+="""

## Surface interference

Counts below are poses with confirmed penetration of a modeled hardware primitive or a transverse intersection between two moving panels. The families are the home pose, 16 coupled transfer samples, nine head/tail limit samples and 18 seeded full joint-box samples. The sampled motion paths cover a subset of the full continuous envelope. Direct signed checks use opposing motor cases, bearings, forks, shafts, rails, feet and spars on different welded bodies. Same-body packaging is audited separately. The signed-distance sample is an upper bound on the true minimum. Each triangle is subdivided with maximum edge 5 mm. A positive sampled clearance below 8 mm cannot certify a 3 mm manufacturing gap. Shell-to-shell samples report unsigned separation. For pairs flagged below 3 mm, an explicit edge-versus-triangle test detects transverse crossings. Coplanar contact and unsampled close approaches remain unresolved. The JSON retains each pose, pair and measured signed or unsigned witness under 3 mm. It also includes native pairwise distance between opposing leg cases, links and feet.

"""
    body+=table(['Model','Neutral','Transfer','Head/tail range','Joint-box extremes'],clear)
    for n in metrics:
        if n in moves: assert all(r['xml_sha256']==metrics[n]['xml_sha256'] for r in moves[n].values()),n
        if 'xml_sha256' in gaps.get(n,{}): assert gaps[n]['xml_sha256']==metrics[n]['xml_sha256'],n
    (BASE/'SCREENING.md').write_text(body+'\n')

if __name__=='__main__':
    main()
