import csv, json, collections, glob, sys
from pathlib import Path
import environments
assert "main_ro" in environments.__file__
from environments.shared.curriculum.recovery_gate import binomial_lcb, binomial_ucb
S=Path("scratchpad")
F=S/"foot/battery/fresh_seeds"; H=S/"foot/next_signals/howclose"
def rd(d):
    rows=list(csv.DictReader(open(d/"stance_heading_panel_selected.csv")))
    pol=[r for r in rows if r['role']=='policy']; st=[r for r in rows if r['role']!='policy']
    return rows,pol,st
def f(x):
    try: return float(x)
    except: return None
out={}
dirs={}
for s in (60,52):
  for k in ("6291456","robustbest"):
    for p in (3042,5042,7042,9042):
      d = H/f"cert_r9armL_s{s}_{k}_optoff" if p==3042 else (H/f"audit7042_r9armL_s{s}_{k}_optoff" if p==7042 else F/f"cert{p}_r9armL_s{s}_{k}_optoff")
      if d.exists(): dirs[(s,k,p)]=d
for (s,k,p),d in dirs.items():
    rows,pol,st=rd(d)
    roles=collections.Counter(r['role'] for r in rows)
    yaws=sorted({float(r['yaw_deg']) for r in pol})
    seeds=sorted({int(r['panel_seed']) for r in pol})
    lab=collections.defaultdict(set); unc=collections.defaultdict(list)
    for r in pol:
        c = r['clean'] in ('True','true','1')
        lab[int(r['panel_seed'])].add(c)
        if not c: unc[int(r['panel_seed'])].append((float(r['yaw_deg']), r['reasons'][:200], f(r['max_sole_cop_fore_aft'])))
    nclean=sum(1 for sd,v in lab.items() if v=={True})
    mixed=[sd for sd,v in lab.items() if len(v)>1]
    nfh=sum(1 for r in pol if r['reached_horizon'] in ('True','true','1'))
    full=[r for r in pol if r['reached_horizon'] in ('True','true','1')]
    w=max(full,key=lambda r:f(r['max_sole_cop_fore_aft']))
    stfall=sorted({int(r['panel_seed']) for r in st if r['reached_horizon'] not in ('True','true','1')})
    import statistics
    rew=statistics.mean(f(r['reward']) for r in pol if float(r['yaw_deg'])==-90.0)
    srew=statistics.mean(f(r['reward']) for r in st if float(r['yaw_deg'])==-90.0)
    ck=sorted({r['checkpoint_sha256'][:16] for r in pol}); ts=sorted({r['task_sha256'][:20] for r in pol}); gs=sorted({r['gate_sha256'][:20] for r in pol})
    seedstart=sorted({r['seed_start'] for r in pol})
    out[f"s{s}_{k}_{p}"]=dict(roles=dict(roles),yaws=yaws,seeds=(seeds[0],seeds[-1],len(seeds)),nclean=nclean,lcb=binomial_lcb(nclean,40),mixed=mixed,
       full_horizon=nfh, worst_cop=(f(w['max_sole_cop_fore_aft']),w['panel_seed'],w['yaw_deg']),
       unclean={sd:(len(v),sorted({x[1].split(':')[0] for x in v}),max(x[2] for x in v if x[2] is not None)) for sd,v in unc.items()},
       statue_fall=stfall, reward_m90=rew, statue_reward_m90=srew, ratio=rew/srew, ck=ck, task=ts, gate=gs, seed_start=seedstart)
for kk,v in out.items(): print(kk, json.dumps(v,default=str))
# min passing count
for n in range(34,41): print(n, binomial_lcb(n,40))
# pooled
def cp2(k,n): return (binomial_lcb(k,n,alpha=0.025), binomial_ucb(k,n,alpha=0.025), binomial_lcb(k,n))
for lab,s,k,ps in (("s60_6M",60,"6291456",(3042,5042,7042,9042)),("s52_6M",52,"6291456",(3042,5042,7042,9042)),("s60_rb",60,"robustbest",(3042,5042,9042)),("s52_rb",52,"robustbest",(3042,5042,9042))):
    kk=sum(out[f"s{s}_{k}_{p}"]['nclean'] for p in ps); n=40*len(ps)
    print(lab,kk,n,kk/n,cp2(kk,n))
    kf=sum(out[f"s{s}_{k}_{p}"]['nclean'] for p in (5042,9042)); print('  fresh',kf,80,cp2(kf,80))
