"""Rebuild the candidates and review book from the included, pinned study inputs."""
import argparse
import subprocess
import sys
from pathlib import Path

ROOT=Path(__file__).resolve().parent
def run(tool,*args):
    subprocess.run([sys.executable,str(ROOT/tool),*(str(a) for a in args)],cwd=ROOT,check=True)

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--only-models',action='store_true',help='Rebuild only the nine MJCFs and mass ledgers.')
    args=p.parse_args()
    base=Path('source/pr662/model.xml')
    gen=Path('study/model/generated')
    run('study/model/build.py','--source',base,'--ledger','source/pr662/components.json','--out',gen)
    if args.only_models:return
    models=[gen/f'{fam}{angle}.xml' for fam in 'ABC' for angle in ['00','10','20']]
    preferred=[gen/f'{fam}10.xml' for fam in 'ABC']
    run('study/mechanics/screen.py',base,*models)
    run('study/mechanics/motion.py',base,*models,'--drive','study/mechanics/upstream/intermediate_drive.py')
    run('study/mechanics/clearance.py',*models)
    run('study/mechanics/pelvis_probe.py',base,*preferred)
    run('study/mechanics/tabulate.py')
    run('study/mechanics/load_figure.py',preferred[-1])
    run('study/hardware/screen.py',base,*models)
    run('study/hardware/report.py')
    run('study/hardware/service_figure.py')
    run('study/visuals/plates.py','--pr662',base,'--models',gen)
    run('study/visuals/viewer.py','--pr662',base,'--models',gen)
    run('study/reference/proportions.py','--out','study/reference/proportions.json',*models)
    run('study/sections/cut_planes.py','--model','PR 662='+str(base),*[s for file in preferred for s in ['--model',file.stem+'='+str(file)]])
    run('study/report/cover_costs.py',*[f.with_suffix('.json') for f in preferred])
    run('study/report/hero.py')
    run('study/report/compose.py')
    run('study/report/make_review.py','--output','T_rex_body_study.pdf')

if __name__=='__main__':main()
