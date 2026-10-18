import sys, pickletools
for p in sys.argv[1:]:
    data = open(p,'rb').read()
    strs = []; globs=set()
    for op, arg, pos in pickletools.genops(data):
        if op.name in ('SHORT_BINUNICODE','BINUNICODE','UNICODE','BINUNICODE8'):
            strs.append(arg)
        elif op.name == 'STACK_GLOBAL':
            globs.add(f"{strs[-2]}.{strs[-1]}")
        elif op.name in ('GLOBAL','INST'):
            globs.add(arg.replace(' ','.'))
    print(p.split('/pilot_runs/')[1], sorted(globs))
