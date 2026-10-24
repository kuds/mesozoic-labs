import numpy as np, glob, json, os, collections
D="/tmp/claude-0/-home-user-mesozoic-labs/0fdf74a4-2989-56c9-a773-92101004d7de/scratchpad/foot/stance5/k3diag/rollouts/npz"
GROUPS = {"legs":[3,4,5,6,7,8,9,10],"tail":[11,12,13,14],"neckhead":[0,1,2],"hip_pitch":[3,7],"knee":[5,9],"legs_no_hippitch":[4,5,6,8,9,10],"legs_no_knee":[3,4,6,7,8,10],"hp_sym":[3,7],"hp_anti":[3,7],"full":list(range(15)),"noise0":[]}
res=collections.defaultdict(dict)
for f in sorted(glob.glob(D+"/*.npz")):
    b=os.path.basename(f)[:-4]; run,st,cond=b.split("__")
    z=np.load(f,allow_pickle=False)
    L=z["length"]; meta=json.loads(str(z["meta_json"]))
    seeds=z["seed"]
    assert list(seeds)==list(range(3042,3062)), (b, seeds)
    mask=np.array(meta["mask"]); 
    if cond in GROUPS:
        exp=np.zeros(15); exp[GROUPS[cond]]=1
        assert np.array_equal(mask,exp),(b,mask)
    falls=int((L<1000).sum())
    v=meta.get("verified")
    res[(run,st)][cond]=(falls, [int(s) for s,l in zip(seeds,L) if l<1000], v and (v["lengths_terms_equal"], v["judge_falls"]))
order=["full","noise0","legs","tail","neckhead","notail","hip_pitch","hip_roll","knee","ankle","legs_no_hippitch","legs_no_hiproll","legs_no_knee","legs_no_ankle","hp_sym","hp_anti"]
print("run,steps,"+",".join(order))
for k in sorted(res):
    print(k[0],k[1],",".join(str(res[k][c][0]) if c in res[k] else "-" for c in order), "verif:",res[k].get("full",(0,0,None))[2])
print(len(res))
for k in sorted(res):
    for c in ("hip_pitch","hp_anti","hp_sym","legs_no_hippitch","knee","legs_no_knee","full"):
        if c in res[k]: print(k,c,res[k][c][0],res[k][c][1])
