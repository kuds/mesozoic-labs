import pickle, json, numpy as np, collections
P="scratchpad/foot/battery/push"
for role in ("s60_6M","s52_6M","s42_6M","statue"):
    reps=pickle.load(open(f"{P}/repo_probe/sweep_{role}.pkl","rb"))
    meta=json.load(open(f"{P}/repo_probe/sweep_{role}_meta.json"))
    tel=json.load(open(f"{P}/telemetry/push_{role}.json"))
    print(role, meta.get('model_sha256') and meta['model_sha256'][:8], meta.get('vecnorm_sha256') and meta['vecnorm_sha256'][:8], 'main_ro' in meta['environments_file'], meta['speeds'], meta['impulse_step'], meta['episodes_per_row'], meta['first_seed'], tel.get('model_sha256') and tel['model_sha256'][:8], 'main_ro' in tel['environments_file'], tel['first_seed'], tel['episodes_per_row'], tel['impulse_step'])
    # telemetry rows index
    trow={}
    for r in tel['rows']:
        trow[(r['direction'], round(r['speed'],3))]=r
    mapdir={"lateral +y":"left","lateral -y":"right","forward +x":"fwd","backward -x":"back","none (control)":"control"}
    out=[]; neq=0; ncmp=0; mx=0
    for rep in reps:
        imp=rep['impulse']; lab=imp['axis_label']; sp=round(imp['speed'],3)
        m=rep['metrics']
        n=rep.get('episodes');
        fh=m['full_horizon_fraction']
        # per-episode
        ev=rep.get('episode_evidence') or []
        lens=[e['length'] for e in ev]; rews=[e.get('reward') for e in ev]; seeds=[e.get('seed') for e in ev]
        out.append((lab,sp,round(fh*8),len(ev)))
        key=(mapdir[lab], sp) if lab!="none (control)" else None
        tr=None
        for (dname,s),r in trow.items():
            if (lab=="none (control)" and dname in ("control","none")) or (dname==mapdir[lab] and abs(s-sp)<1e-9): tr=r
        if tr is not None and ev:
            te={e['seed']:e for e in tr['episodes']}
            for e in ev:
                t=te.get(e['seed'])
                if t is None: continue
                ncmp+=1; neq+= (t['length']==e['length']); mx=max(mx,abs(t['reward']-e['reward']))
    print('  rows',out)
    print('  crosscheck',ncmp,neq,mx)
    print('  telemetry row keys', sorted(set(d for d,s in trow)), sorted(set(s for d,s in trow)))
