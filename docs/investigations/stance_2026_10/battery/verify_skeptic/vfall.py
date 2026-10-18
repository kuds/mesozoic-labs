import json, numpy as np, math
S="scratchpad/foot/battery/shared_fall"
meta={r:json.load(open(f"{S}/replays/replay_{r}.json")) for r in ("s60_6M","s52_6M","statue")}
N=meta['statue']['names']
ACT=[a.replace('_act','') for a in N['actuator_names']]
print(ACT)
print('pad half',N['pad_half_size'],'heel/toe',N['heel_toe_line_from_ankle_m'], 'BW',N['body_weight_n'])
for r in meta:
    z=np.load(f"{S}/replays/trace_{r}_y+0.npz")
    T=len(z['reward_f64'])
    sc=z['rec_sole_cop'][:,:,0]
    bt=np.all(sc>=0.99,1)
    first=int(np.flatnonzero(bt[2:])[0])+2+1 if bt[2:].any() else None
    frac=bt[10:].mean()
    heel=np.any(sc[10:]<=-0.99,1).mean()
    mp=z['m_xcom_pad']; mf=z['m_xcom_full']; mc=z['m_com_pad']
    def stays(mask,start=10):
        mask=np.asarray(mask,bool)
        if not mask[-1]: return None
        bad=np.flatnonzero(~mask[start:]); return (start if not bad.size else int(bad[-1])+1+start)+1
    # independent: toe line from pad_toe_hf and xcom
    xf=z['xcom_hf'][:,0]; toe=z['pad_toe_hf'][:,:,0].mean(1); pc=z['pad_center_hf'].mean(1)[:,0]
    # the 'm_xcom_pad' toe margin is hull chord; compare with simple toe line
    simple=toe-xf
    leg=[ACT.index(a) for a in ["r_hip_pitch","r_hip_roll","r_knee","r_ankle","l_hip_pitch","l_hip_roll","l_knee","l_ankle"]]
    afl=np.nanmax(z['afrac_max'][:,leg])
    act=z['action'];
    print(r,'T',T,'sum reward',z['reward_f64'].sum(),'first both-toe',first,'frac',round(frac,3),'heel frac',round(heel,3),
          'xcom toe-neg-for-good',stays(mp[:,1]<0),'simple toe line neg for good',stays(simple<0),
          'com neg',stays(mc[:,1]<0),'full hull out',stays(mf[:,0]<0),'min full',np.nanmin(mf[10:,0]),
          'afrac leg',round(float(afl),3),'max|a| leg',round(float(np.abs(act[:,leg]).max()),3),
          'step1 xcom-pc',round(float(xf[0]-pc[0]),4),'toe-pc',round(float(toe[0]-pc[0]),4),
          'sole pitch step1',z['rec_sole_pitch_deg'][0], 'lift',z['rec_sole_corner_lift'][0]*1000,
          'head end',z['head_tip_z'][-1],'fz end',z['forward_z'][-1], 'pitch range', float(z['pitch_down_deg'][10:].min()), float(z['pitch_down_deg'][10:].max()))
    # topple fit
    vf=z['vcom_hf'][:,0]; t=(np.arange(T)+1)*0.01
    i=stays(mf[:,0]<0)
    if i:
        i0=i-1; seg=slice(i0,T-3); lam=np.polyfit(t[seg],np.log(np.maximum(vf[seg],1e-6)),1)[0]; print('  lam',lam,'omega0',z['omega0'][seg].mean(),'v exit',vf[i0],'v end',vf[-1])
    # hip
    adr=np.array(N['actuator_qposadr_minus7']); home=np.array(N['home_qpos_j']); hc=np.array(N['home_ctrl'])
    hp=[ACT.index('r_hip_pitch'),ACT.index('l_hip_pitch')]
    print('  hip joint vs home step41', np.degrees(z['qpos_j'][40,adr[hp]]-home[adr[hp]]), 'hip ctrl vs home ctrl step41', np.degrees(z['ctrl'][40,hp]-hc[hp]), 'statue-min hip joint', np.degrees((z['qpos_j'][:,adr[hp]]-home[adr[hp]]).min(0)))
    print('  sole pitch closest to flat 11..end', z['rec_sole_pitch_deg'][10:].max(0), 'mean 11-60', z['rec_sole_pitch_deg'][10:60].mean(0))
    print('  spawn joint offset', np.round(np.degrees(np.array(meta[r]['episodes'][2]['spawn_joint_offset_from_home'])),2).tolist())
print(N['joint_names'])
