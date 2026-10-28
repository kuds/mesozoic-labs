import json, hashlib
ROOT='/tmp/claude-0/-home-user-mesozoic-labs/0fdf74a4-2989-56c9-a773-92101004d7de/scratchpad/foot/stance6/fetch/F2'
def find(o,key,path=''):
    out=[]
    if isinstance(o,dict):
        for k,v in o.items():
            p=f'{path}.{k}' if path else k
            if k==key: out.append((p,v))
            out+=find(v,key,p)
    elif isinstance(o,list):
        for i,v in enumerate(o): out+=find(v,key,f'{path}[{i}]')
    return out
for r in ('s60','s42','s80'):
    p=f'{ROOT}/stance6_r10LTSB_{r}/copies/01_stance/stage_config.json.copy1'
    b=open(p,'rb').read(); h=hashlib.sha256(b).hexdigest()
    man=json.load(open(f'{ROOT}/stance6_r10LTSB_{r}/copies/artifact_manifest.json.copy1'))
    ents=man.get('entries') or man.get('files')
    want=[e for e in ents if e.get('path')=='01_stance/stage_config.json'] if isinstance(ents,list) else [ents.get('01_stance/stage_config.json')]
    d=json.loads(b)
    print(r,h[:8],'manifest',str(want[0])[:120])
    for k in ('ent_coef','ent_coef_end','ent_coef_decay_timesteps','log_std_init','symmetry_sigma_floor','action_saturation_weight','action_saturation_threshold','learning_rate','learning_rate_end','net_arch','use_sde','action_penalty_source'):
        print('   ',k,find(d,k))
