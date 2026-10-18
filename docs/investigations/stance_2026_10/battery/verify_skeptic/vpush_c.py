import json, numpy as np
P="scratchpad/foot/battery/push"
d=json.load(open(P+"/push_summary.json"))
print("== margin arithmetic: lenient (+-1 grid step) vs strict bracket, spearman, ratio")
for role in ("s60_6M","s52_6M","s42_6M","statue"):
    for dd in ("fwd","back","left","right"):
        m=d['roles'][role]['directions'][dd]['margin_arithmetic']; ps=m['per_seed']
        strict=sum(1 for p in ps if p['first_fall_mps'] is not None and p['observed_max_survived_mps']<=p['pred_dv_cop_pad']<=p['first_fall_mps'])
        print(role,dd,'b_pad',m['b_m_median']['cop_pad'],'b_full',m['b_m_median']['cop_full'],'pred',m['pred_dv_mps_median']['cop_pad'],'ratio',m['ratio_observed_max_survived_over_pred_cop_pad_median'],'lenient',m['seeds_pred_cop_pad_within_one_grid_step_of_bracket'],'strict',strict,'rho_cop',m['spearman_observed_max_survived_vs_pred_cop_pad'],'rho_xcom',m['spearman_observed_max_survived_vs_pred_xcom_pad'],'w0',m['omega0_median'],'h',m['com_height_m_median'])
print("== CoP excursion at >=8/16 envelope speed and s42 fwd 0.15/0.2")
for role in ("s60_6M","s52_6M","s42_6M","statue"):
    for dd in ("fwd","back","left","right"):
        D=d['roles'][role]['directions'][dd]; e8=D['envelope_ge50pct_monotone_mps']
        for sp,s in D['speeds'].items():
            if abs(float(sp)-e8)<1e-9 or (role=='s42_6M' and dd=='fwd' and sp in ('0.15','0.2')):
                print(role,dd,sp,'env8',e8,'surv',s['survived'],'copdisp',s['win_cop_peak_disp_along_push_m_survivors_median'],'frac_b',s['win_cop_peak_fraction_of_b_pad_survivors_median'],'padcop',s['win_pad_cop_fore_aft_max_survivors_median'],'rec',s['recovery_time_s_adaptive_median'],s['recovery_time_s_adaptive_max'])
sag_f=sag_x=sag_f05=sag_x05=surv_lift=surv_step=sag_lift=lat_f=lat_l=lat_lf=0; maxdv=0; latmax=0
for role in ("s60_6M","s52_6M","s42_6M","statue"):
    t=json.load(open(f"{P}/telemetry/push_{role}.json"))
    for r in t['rows']:
        U={"fwd":(1,0),"back":(-1,0),"left":(0,1),"right":(0,-1),"control":(0,0)}[r['direction']]
        for e in r['episodes']:
            if e.get('dv_applied') is not None: maxdv=max(maxdv,float(np.abs(np.asarray(e['dv_applied'])[:2]-np.array(U)*r['speed']).max()))
            lift=sum(e.get('liftoff_steps_per_foot') or [0])>0
            if e['survived']: surv_lift+=int(lift); surv_step+=int(bool(e.get('stepped')))
            if r['direction'] in ('fwd','back'):
                sag_lift+=int(lift)
                if not e['survived']:
                    x=int((e['xcom_outside_pad_steps'] or 0)>0); sag_f+=1; sag_x+=x
                    if r['speed']<=0.5+1e-9: sag_f05+=1; sag_x05+=x
            if r['direction'] in ('left','right'):
                if e['survived'] and e.get('win_cop_peak_fraction_of_b_pad') is not None: latmax=max(latmax,e['win_cop_peak_fraction_of_b_pad'])
                if r['speed']<=0.5+1e-9 and not e['survived']: lat_f+=1; lat_l+=int(lift); lat_lf+=int(lift and e['reason']=='fallen')
print('sagittal fallers all speeds',sag_f,'xcom outside',sag_x,'| <=0.5',sag_f05,sag_x05,'| survivors liftoff',surv_lift,'stepped',surv_step,'| sagittal liftoff',sag_lift,'| lateral fallers<=0.5',lat_f,'liftoff',lat_l,'liftoff&fallen',lat_lf,'| max |dv_applied-commanded|',maxdv,'| lateral survivor max CoP/b',latmax)
z=np.load(P+"/telemetry/push_series_s60_6M.npz")
for k in ("control_0","fwd_0.2","back_0.3","left_0.35","fwd_2"): print('s60 vh first post-impulse step median',k,float(np.nanmedian(z[k+"_vh"][:,0])))
