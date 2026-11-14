"""Short free-base stand and weight-transfer tests. Requires the source InstalledDrive."""
from __future__ import annotations
import argparse
import importlib.util
import hashlib
import json
from pathlib import Path

import mujoco as mj
import numpy as np

from screen import BASE, SIDES, fresh, home_command, sole_corners, convert, model_key, bounds, physical, descendants


def targets(m, condition, t):
    q = home_command(m)
    if condition == "stand":
        return q
    index = {m.joint(j).name:i for i,j in enumerate(m.actuator_trnid[:,0])}
    # Fixed ref for every model. Two lateral cycles in 4 s, preceded by settling.
    phase = np.clip((t-.5)/4,0.,1.)*4*np.pi
    wave = np.sin(phase)
    for sign,side in zip((1.,-1.),SIDES):
        lift = max(0.,-sign*wave)**2
        hip = -.26*lift + .08*sign*wave**2*np.cos(phase)
        knee = .50*lift
        for name,angle in (("hip_roll",-.24*wave),("ankle_roll",.24*wave),
                           ("hip_pitch",hip),("knee",knee),("ankle_pitch",-hip-knee)):
            q[index[side+"_"+name]] += angle
    q[index["neck_pitch"]] += .15*wave
    q[index["tail_yaw"]] -= .20*wave
    return np.clip(q,m.actuator_ctrlrange[:,0],m.actuator_ctrlrange[:,1])


def simulate(path,condition,Drive):
    m=mj.MjModel.from_xml_path(str(path.resolve()))
    d=fresh(m)
    drive=Drive(m,voltage=22.2)
    drive.reset(d)
    dt=m.opt.timestep
    hz=2. if condition=="stand" else 4.8
    chronological=np.array([targets(m,condition,t) for t in np.arange(round(hz/dt))*dt])
    pelvis=m.body("pelvis").id
    weight=m.body_mass.sum()*-m.opt.gravity[2]
    initial=d.xpos[pelvis,:2].copy()
    loaded_run=np.zeros(2);longest=np.zeros(2)
    floor=m.geom("floor").id
    soles=[m.geom(side+"_sole").id for side in SIDES]
    max_tilt=0.;max_drift=0.;nonsole=None
    tail_bodies=descendants(m,m.body("tail").id)
    tail_geoms=[g for g in range(m.ngeom) if physical(m,g) and m.geom_bodyid[g] in tail_bodies]
    min_tail=np.inf
    stride=max(1,round(.02/dt))
    max_clear=np.zeros(2);squared=np.zeros(m.nu);peak=np.zeros(m.nu);over=np.zeros(m.nu)
    ids=[m.joint(j).name for j in m.actuator_trnid[:,0]]
    reason=None
    for step,target in enumerate(chronological):
        drive.step(d,target)
        if step%stride==0:
            min_tail=min(min_tail,min(bounds(m,d,g)[0,2] for g in tail_geoms))
        w=np.empty(6);loads=np.zeros(2);other=False
        for k,c in enumerate(d.contact):
            if floor not in c.geom:
                continue
            g=(c.geom2 if c.geom1==floor else c.geom1)
            mj.mj_contactForce(m,d,k,w)
            vertical=(c.frame.reshape(3,3).T@w[:3])[2]*(1 if c.geom1==floor else -1)
            if g in soles:
                loads[soles.index(g)]+=vertical
            elif vertical>1:
                other=True
                if nonsole is None:
                    nonsole={"geom":m.geom(g).name,"at_s":step*dt,"load_N":vertical}
        heights=np.array([sole_corners(m,d,s)[:,2].min() for s in SIDES])
        valid=(loads >= .80*weight) & (loads[::-1]<=.05*weight) & (heights[::-1]>=.008) & (not other)
        loaded_run=np.where(valid,loaded_run+dt,0)
        longest=np.maximum(longest,loaded_run)
        max_clear=np.maximum(max_clear,heights)
        tilt=np.degrees(np.arccos(np.clip(d.xmat[pelvis,8],-1,1)))
        max_tilt=max(max_tilt,tilt)
        max_drift=max(max_drift,np.linalg.norm(d.xpos[pelvis,:2]-initial))
        squared+=d.actuator_force**2
        peak=np.maximum(peak,np.abs(d.actuator_force))
        over+=dt*(np.abs(d.actuator_force)>drive.cont)
        if not np.all(np.isfinite(d.qpos)) or tilt>55 or d.xpos[pelvis,2]<.16:
            reason="fall_or_nonfinite"
            break
    completed=(reason is None)
    quality=completed and max_tilt<12 and nonsole is None
    if condition=="stand":
        quality=quality and max_drift<.025
    else:
        quality=quality and np.all(longest>=.1)
    return dict(xml_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),passed=bool(quality),completed=completed,duration_s=(step+1)*dt,
                stop=reason,max_pelvis_tilt_deg=max_tilt,max_drift_mm=max_drift*1000,
                first_nonsole_support=nonsole,min_tail_floor_mm_50Hz=min_tail*1000, longest_loaded_single_support_s=longest,
                max_swing_bottom_mm=max_clear*1000,peak_joint_Nm=dict(zip(ids,peak)),
                rms_joint_Nm=dict(zip(ids,np.sqrt(squared/(step+1)))),above_continuous_s=dict(zip(ids,over)))


def main():
    p=argparse.ArgumentParser(__doc__)
    p.add_argument("models",nargs="+",type=Path)
    p.add_argument("--drive",type=Path,required=True,help="handoff environments/shared/intermediate_drive.py")
    p.add_argument("--out",type=Path,default=BASE/"motion.json")
    a=p.parse_args()
    spec=importlib.util.spec_from_file_location("installed_drive",a.drive)
    drive=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(drive)
    rows={}
    for path in a.models:
        rows[model_key(path)]={case:simulate(path,case,drive.InstalledDrive) for case in ("stand","transfer")}
        print(path.stem,[(n,v['passed'],round(v['max_pelvis_tilt_deg'],1)) for n,v in rows[model_key(path)].items()],flush=True)
    a.out.write_text(json.dumps(rows,indent=2,default=convert)+"\n")

if __name__=="__main__":
    main()
