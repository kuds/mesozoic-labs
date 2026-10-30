import json, sys, numpy as np, collections
R=json.load(open(sys.argv[1]))
by=collections.defaultdict(list)
for r in R: by[r["label"]].append(r)
st={r["seed"]:r for r in by["statue"]}
full=lambda r: r["n"]>=1000
print("statue ret per seed", {str(k):round(v["ret"],1) for k,v in st.items()}, "ends", {str(k):v["end"] for k,v in st.items()})
okseeds=[s for s,r in st.items() if full(r)]
print("%-24s %5s %8s %7s %6s %6s %7s %6s  %s"%("label","stand","dRet","dTilt","dZmm","dCOMx","qpin","frac","top term deltas, steps 200-1000 (pts/episode)"))
labels=[l for l in by if l!="statue" and not l.startswith(("UNWIND","REPIN"))]
for lab in labels:
    rs=by[lab]
    a=lab.split(":")[1] if lab.startswith("pin:") else None
    standing=[r for r in rs if full(r) and r["seed"] in okseeds]
    stand=f"{sum(full(r) for r in rs)}/{len(rs)}"
    if not standing:
        print("%-24s %5s   ends=%s n=%s"%(lab,stand,[r['end'] for r in rs],[r['n'] for r in rs])); continue
    dret=np.mean([r["ret"]-st[r["seed"]]["ret"] for r in standing])
    tsum=collections.defaultdict(float)
    for r in standing:
        s=st[r["seed"]]
        for w in ("w200_400","w400_700","w700_1000"):
            for k,v in r[w].items(): tsum[k]+= (v-s[w][k])/len(standing)
    top=sorted(((k,v) for k,v in tsum.items() if k!="_total"), key=lambda kv: kv[1])[:5]
    pos=sorted(((k,v) for k,v in tsum.items() if k!="_total" and v>0.5), key=lambda kv: -kv[1])[:2]
    dt=np.mean([r["ss"]["tilt_deg"]-st[r["seed"]]["ss"]["tilt_deg"] for r in standing])
    dz=np.mean([1e3*(r["ss"]["z"]-st[r["seed"]]["ss"]["z"]) for r in standing])
    dcx=np.mean([r["ss"]["comx_mm"]-st[r["seed"]]["ss"]["comx_mm"] for r in standing])
    if a:
        qp=np.mean([r["ss"]["q_deg"][a] for r in standing]); fr=np.mean([r["ss"]["force_frac"][a] for r in standing]); stop=any(r["ss"]["at_stop"][a] for r in standing)
    else:
        qp=fr=float('nan'); stop=False
    fails=[(r['seed'],r['end'],r['n']) for r in rs if not full(r)]
    print("%-24s %5s %8.1f %7.2f %6.1f %6.1f %7.2f %6.2f%s %s%s%s"%(lab,stand,dret,dt,dz,dcx,qp,fr,"S" if stop else " "," ".join("%s %+.1f"%(k.replace("reward_",""),v) for k,v in top), (" | gains: "+" ".join("%s %+.1f"%(k.replace("reward_",""),v) for k,v in pos)) if pos else "", (" | FAILS "+str(fails)) if fails else ""))
