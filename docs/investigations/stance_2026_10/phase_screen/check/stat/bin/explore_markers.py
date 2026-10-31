import sys, json
def walk(o, pre="", depth=0, maxd=6):
    if depth > maxd:
        return
    if isinstance(o, dict):
        for k, v in o.items():
            if isinstance(v, (dict, list)):
                n = len(v)
                print(f"{pre}{k}: {type(v).__name__}[{n}]")
                if isinstance(v, dict) or (isinstance(v, list) and v and isinstance(v[0], dict)):
                    walk(v if isinstance(v, dict) else v[0], pre + "  ", depth + 1, maxd)
            else:
                s = repr(v)
                print(f"{pre}{k}: {s[:100]}")
p = sys.argv[1]
m = json.load(open(p))
walk(m, maxd=int(sys.argv[2]) if len(sys.argv) > 2 else 3)
