import json, numpy as np, glob, os, csv, statistics, collections
S="scratchpad"
O=S+"/foot/battery/own_sigma/panels"; H=S+"/foot/next_signals/howclose"
certdir={"s60_6M":"cert_r9armL_s60_6291456_optoff","s52_6M":"cert_r9armL_s52_6291456_optoff","s42_6M":"cert_r9armL_s42_6291456_optoff",
         "s60_rb524288":"cert_r9armL_s60_robustbest_optoff","s52_rb917504":"cert_r9armL_s52_robustbest_optoff","s42_rb655360":"cert_r9armL_s42_robustbest_optoff"}
def load(lbl,mode): return json.load(open(f"{O}/{lbl}_{mode}/panel.json"))
print(list(load("s60_6M","stochastic")['episodes'][0].keys()))
for lbl,cd in certdir.items():
    rows=[r for r in csv.DictReader(open(f"{H}/{cd}/stance_heading_panel_selected.csv")) if r['yaw_deg']=='0.0']
    pol={int(r['panel_seed']):r for r in rows if r['role']=='policy'}
    out={}
    for mode in ("deterministic","stochastic"):
        p=load(lbl,mode); eps=p['episodes']
        cop=[e['max_sole_cop_fore_aft'] for e in eps if e['reached_horizon']]
        signed=[e['pad']['signed_pad_cop_gate_foot'] for e in eps if e['reached_horizon']]
        ext=[e['pad']['signed_pad_cop_extreme_foot'] for e in eps if e['reached_horizon']]
        unclean={e['seed']:(round(e['max_sole_cop_fore_aft'],4), [x.split(':')[0] for x in (e.get('reasons') or [])]) for e in eps if not e['clean']}
        rc=all(e['pad']['recompute_matches_gate_metric'] for e in eps)
        fails=collections.Counter(x.split(':')[0] for e in eps for x in (e.get('reasons') or []))
        spk=[e.get('settle_peak_floor_force_bw') for e in eps]
        out[mode]=dict(nclean=sum(e['clean'] for e in eps), med=round(statistics.median(cop),4), worst=round(max(cop),4), past=sum(c>0.8 for c in cop),
                       smed=round(statistics.median(signed),4), sext=max(ext,key=abs), heel=sum(s<0 for s in signed), rew=round(np.mean([e['reward'] for e in eps]),1),
                       unclean=unclean, recompute=rc, fails=dict(fails), spk_med=round(statistics.median(spk),3), spk_max=round(max(spk),3))
        if mode=="deterministic":
            mism=0
            for e in eps:
                r=pol[e['seed']]
                if not (int(r['length'])==e['length'] and abs(float(r['reward'])-e['reward'])<1e-6 and (r['clean']=='True')==bool(e['clean']) and
                        (abs(float(r['max_sole_cop_fore_aft'])-e['max_sole_cop_fore_aft'])<1e-9 if e['max_sole_cop_fore_aft'] is not None else True)):
                    mism+=1
            out[mode]['cert_mismatch']=mism; out[mode]['max_rew_diff']=max(abs(float(pol[e['seed']]['reward'])-e['reward']) for e in eps)
    print(lbl, json.dumps(out))
    # noise lost/gained
    d={e['seed']:e['clean'] for e in load(lbl,'deterministic')['episodes']}; s={e['seed']:e['clean'] for e in load(lbl,'stochastic')['episodes']}
    print('  lost',[k for k in d if d[k] and not s[k]],'gained',[k for k in d if s[k] and not d[k]])
    pd={e['seed']:e['max_sole_cop_fore_aft'] for e in load(lbl,'deterministic')['episodes']}; ps={e['seed']:e['max_sole_cop_fore_aft'] for e in load(lbl,'stochastic')['episodes']}
    print('  paired diff median', round(statistics.median(ps[k]-pd[k] for k in pd),4))
    st=load(lbl,'stochastic'); v=st['v2_result']; print('  v2 keys', {k:v[k] for k in v if k in ('n_clean','clean_lcb','passed','failures','hop_or_fall_episodes')})
# statue
st=load("statue","statue"); eps=st['episodes']
cop=[e['max_sole_cop_fore_aft'] for e in eps if e['reached_horizon']]
print('statue', [ (e['seed'],e['length'],e['termination_reason']) for e in eps if not e['reached_horizon']], statistics.median(cop), max(cop), statistics.median([e['pad']['signed_pad_cop_gate_foot'] for e in eps if e['reached_horizon']]))
rows=[r for r in csv.DictReader(open(f"{H}/cert_r9armL_s60_6291456_optoff/stance_heading_panel_selected.csv")) if r['yaw_deg']=='0.0' and r['role']=='statue']
sr={int(r['panel_seed']):r for r in rows}
print('statue vs cert mism', sum(1 for e in eps if int(sr[e['seed']]['length'])!=e['length'] or abs(float(sr[e['seed']]['reward'])-e['reward'])>1e-6))
print(json.dumps(st['statue_reference'])[:400])
