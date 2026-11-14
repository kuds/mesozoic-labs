"""Planning arithmetic for the assembly burden of individually removable panels."""
import argparse
import csv
import json
from pathlib import Path

def find_panels(obj):
    if isinstance(obj, list) and obj and isinstance(obj[0],dict) and {"wall_mm","mass_kg","area_m2"}<=obj[0].keys():
        return obj
    if isinstance(obj,dict):
        for val in obj.values():
            result=find_panels(val)
            if result:return result
    return []

def summary(f):
    data=json.loads(f.read_text())
    covers=find_panels(data)
    if not covers:raise ValueError("Expected per-panel ledger in "+str(f))
    mass=sum(p["mass_kg"] for p in covers)
    mounts=sum(p["fastener_mass_kg"] for p in covers)
    area=sum(p["area_m2"] for p in covers)
    n=len(covers)
    return dict(model=f.stem,removable_pieces=n,area_m2=round(area,4),sheet_kg=round(mass,4),
                mount_budget_kg=round(mounts,4),workshop_labor_hours_low=round(n*4/60,2),
                workshop_labor_hours_high=round(n*8/60,2),fitup_labor_usd_low=n*4,fitup_labor_usd_high=n*12)

if __name__=="__main__":
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("ledger",nargs="+",type=Path)
    ap.add_argument("--out",type=Path,default=Path(__file__).with_name("cover_work.csv"))
    a=ap.parse_args();rows=[summary(p) for p in a.ledger]
    with a.out.open("w") as h:
        w=csv.DictWriter(h,fieldnames=rows[0]);w.writeheader();w.writerows(rows)
