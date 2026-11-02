import json,sys
d=json.load(open(sys.argv[1]))
def walk(x,pre=""):
    if isinstance(x,dict):
        for k,v in x.items(): walk(v,pre+"."+k)
    elif isinstance(x,list):
        if len(x)<4:
            for i,v in enumerate(x): walk(v,pre+f"[{i}]")
        else: print(pre,"list",len(x))
    else:
        s=str(x)
        print(pre,"=",s[:160])
walk(d)
