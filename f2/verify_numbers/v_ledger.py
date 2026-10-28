# Independent re-derivation of D-1 ledger counts from F2 eval_episodes.csv copies (manifest-checked).
import csv, json, hashlib, sys, math
ROOT='/tmp/claude-0/-home-user-mesozoic-labs/0fdf74a4-2989-56c9-a773-92101004d7de/scratchpad/foot/stance6/fetch/F2'
TAIL={'tail_1_pitch_act','tail_1_yaw_act','tail_2_pitch_act','tail_3_pitch_act'}
W=[(0,4194304),(4194304,8388608),(8388608,12001281)]
out={}
for r in ('s60','s42','s80'):
    rid='stance6_r10LTSB_'+r
    man=json.load(open(f'{ROOT}/{rid}/copies/artifact_manifest.json.copy1'))
    ents=man.get('entries') or man.get('files')
    # find entry for eval_episodes
    want=None
    if isinstance(ents,dict):
        e=ents.get('01_stance/eval_episodes.csv'); want=e.get('sha256') if isinstance(e,dict) else e
    else:
        for e in ents:
            if e.get('path')=='01_stance/eval_episodes.csv': want=e.get('sha256')
    p=f'{ROOT}/{rid}/copies/01_stance/eval_episodes.csv.copy1'
    h=hashlib.sha256(open(p,'rb').read()).hexdigest()
    ok=(want is not None and want.replace('sha256:','')==h)
    rows=list(csv.DictReader(open(p,newline='')))
    ev={}
    for row in rows:
        t=int(float(row['timesteps']))
        ev.setdefault(t,[]).append(row)
    steps=sorted(ev)
    def f(x):
        try: return float(x)
        except: return float('nan')
    per={}
    overbar={}
    tailover_rows=[]; any_rows=[]
    for t in steps:
        anyc=0; tailc=0; unres=0; anyact={}
        for row in ev[t]:
            frac=f(row['saturated_actuator_fraction']); act=row['saturated_actuator']
            scr=row['screen_v2_clean']
            measured = scr not in ('','nan','NaN')
            if measured:
                anyover = 'max_actuator_saturation_fraction' in (row['screen_v2_reasons'] or '')
            else:
                anyover = (frac>0.10)
            if anyover: anyc+=1
            if frac>0.10:
                ps=f(row['saturated_actuator_positive_share'])
                sgn='+' if ps>=0.5 else '-'
                overbar[act+sgn]=overbar.get(act+sgn,0)+1
                if act in TAIL: tailc+=1; tailover_rows.append(t)
                else:
                    ties=f(row['saturated_actuator_ties']); ns=f(row['actuators_saturated'])
                    if (ties>=1) or (ns>=2): unres+=1
                any_rows.append(t)
        per[t]=dict(n=len(ev[t]),any=anyc,tail=tailc,unres=unres)
    anyfl=[t for t in steps if per[t]['any']>=5]
    tailfl=[t for t in steps if per[t]['tail']>=5]
    def alarms(fl):
        s=set(fl); res=[]
        for i,t in enumerate(steps):
            if i>0 and t in s and steps[i-1] in s: res.append(t)
        return res
    def spells(fl):
        s=set(fl); sp=[]; cur=None
        for i,t in enumerate(steps):
            if t in s:
                if cur is None: cur=[t,t,1]
                else: cur[1]=t; cur[2]+=1
            else:
                if cur: sp.append(cur); cur=None
        if cur: sp.append(cur)
        return sp
    # window shares
    ws=[]
    for a,b in W:
        es=[t for t in steps if a<=t<b]
        nrows=sum(per[t]['n'] for t in es)
        tov=sum(per[t]['tail'] for t in es)
        oov=sum(1 for t in any_rows if a<=t<b) - tov
        ws.append(dict(evals=len(es),rows=nrows,tail=tov,tailshare=round(tov/nrows,4),other=oov,othershare=round(oov/nrows,4)))
    ent=[]
    for a,b in [(0,7000000),(7000000,12001281)]:
        es=[t for t in steps if a<=t<b]
        nrows=sum(per[t]['n'] for t in es)
        ov=sum(1 for t in any_rows if a<=t<b)
        ent.append(dict(evals=len(es),rows=nrows,overbar=ov,anyfl=len([t for t in anyfl if a<=t<b]),anyal=len([t for t in alarms(anyfl) if a<=t<b])))
    out[r]=dict(sha_ok=ok,sha=h[:12],n_rows=len(rows),n_evals=len(steps),last_step=steps[-1],overbar=overbar,
        first_overbar=min(any_rows) if any_rows else None, last_overbar=max(any_rows) if any_rows else None,
        first_tail_overbar=min(tailover_rows) if tailover_rows else None,
        first_tail_flag=tailfl[0] if tailfl else None, n_tail_fl=len(tailfl), n_any_fl=len(anyfl),
        tail_alarms=alarms(tailfl), any_alarms=alarms(anyfl), tail_spells=spells(tailfl), any_spells=spells(anyfl),
        max_per_eval=max(per[t]['tail']+0 for t in steps), unres=sum(per[t]['unres'] for t in steps),
        windows=ws, entropy_split=ent,
        alarm_pair_counts=[per[steps[steps.index(t)-1]]['any']+per[t]['any'] for t in alarms(anyfl)])
for r,v in out.items():
    print(r, json.dumps(v))
