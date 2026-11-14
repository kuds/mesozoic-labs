"""A small alternate stance family to test hip-roll conclusions with pelvis and legs independent."""
import argparse,json,hashlib
from pathlib import Path
import numpy as np
import mujoco as mj
from screen import BASE,SIDES,convert,fresh,home_command,model_key,sole_corners,static_load


def search(m):
    d=fresh(m)
    home=home_command(m)
    ids={m.joint(j).name:i for i,j in enumerate(m.actuator_trnid[:,0])}
    limits=m.jnt_range[m.actuator_trnid[:,0]]
    best={}
    for stance,other,sign in [('left','right',1),('right','left',-1)]:
        attempts=valid=0;minimum=None
        for roll_deg in [-10,-5,0,5,10]:
            phi=np.radians(roll_deg)*sign
            for offset in np.linspace(.12,.36,13):
                for swing_adduction in [-.08,0,.08]:
                    q=home.copy()
                    for s in SIDES:
                        hip_roll= -sign*offset+(sign*swing_adduction if s==other else 0)
                        q[ids[s+'_hip_roll']]+=hip_roll
                        q[ids[s+'_ankle_roll']]+=-phi-hip_roll
                    for axis,t in [('hip_pitch',-.25),('knee',.5),('ankle_pitch',-.25)]:
                        q[ids[other+'_'+axis]]+=t
                    if any(q<limits[:,0]-1e-9) or any(q>limits[:,1]+1e-9):continue
                    attempts+=1
                    d.qpos[:]=m.key_qpos[0]
                    d.qpos[3:7]=np.array([np.cos(phi/2),np.sin(phi/2),0,0])
                    d.qpos[m.jnt_qposadr[m.actuator_trnid[:,0]]]=q
                    mj.mj_forward(m,d)
                    d.qpos[2]-=sole_corners(m,d,stance)[:,2].min()
                    mj.mj_forward(m,d)
                    corners=sole_corners(m,d,stance)
                    clearance=sole_corners(m,d,other)[:,2].min()*1000
                    if np.ptp(corners[:,2])*1000>.5 or clearance<8:continue
                    row=static_load(m,d,[stance])
                    if row['cop_margin_mm'][0]<3:continue
                    valid+=1
                    value=abs(row['joint_Nm'][stance+'_hip_roll'])
                    if minimum is None or value<minimum['hip_roll_Nm']:
                        minimum=dict(hip_roll_Nm=value,pelvis_roll_deg=roll_deg*sign,stance_roll_offset_rad=-sign*offset,swing_extra_roll_rad=sign*swing_adduction,
                                     sole_margin_mm=row['cop_margin_mm'][0], swing_clearance_mm=clearance,joint_Nm=row["joint_Nm"])
        best[stance]=dict(admissible_limit_samples=attempts,feasible_support_samples=valid,min_hip_roll=minimum)
    return best


if __name__=='__main__':
    p=argparse.ArgumentParser(__doc__);p.add_argument('models',nargs='+',type=Path);a=p.parse_args()
    rows={}
    for path in a.models:
        result=search(mj.MjModel.from_xml_path(str(path.resolve())))
        result['xml_sha256']=hashlib.sha256(path.read_bytes()).hexdigest()
        rows[model_key(path)]=result
    (BASE/'pelvis_probe.json').write_text(json.dumps(rows,default=convert,indent=2)+'\n')
    for k,v in rows.items(): print(k,[(s,round(r['min_hip_roll']['hip_roll_Nm'],2)) for s,r in v.items() if s in SIDES])
