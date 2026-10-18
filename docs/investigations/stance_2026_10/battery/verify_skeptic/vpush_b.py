import json, numpy as np
P="scratchpad/foot/battery/push"
summ=json.load(open(P+"/push_summary.json"))
for role in ("s60_6M","s52_6M","s42_6M","statue"):
    tel=json.load(open(f"{P}/telemetry/push_{role}.json"))
    grid={}
    for r in tel['rows']:
        eps=r['episodes']
        assert [e['seed'] for e in eps]==list(range(3042,3058)), [e['seed'] for e in eps]
        grid[(r['direction'],round(r['speed'],3))]=sum(1 for e in eps if e['survived'])
    print(role,'control',grid.get(('control',0.0)))
    for d in ("fwd","back","left","right"):
        sp=sorted(s for (dd,s) in grid if dd==d)
        surv=[grid[(d,s)] for s in sp]
        def env(th):
            best=0.0
            for s,v in zip(sp,surv):
                if v>=th: best=s
                else: break
            return best
        sd=summ['roles'][role]['directions'][d]
        print(f"  {d}: {dict(zip(sp,surv))} env16={env(16)} env8={env(8)} | summary {sd.get('envelope_full_recovery_monotone_mps')} {sd.get('envelope_ge50pct_monotone_mps')}")
    # pad CoP parked
    print('  parked', summ['roles'][role]['control'].get('pad_cop_fore_aft_signed_median_per_foot_steps_200_499'), summ['roles'][role]['control'].get('adaptive_settle_threshold_mps'))
