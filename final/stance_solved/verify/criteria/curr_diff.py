"""Read-only: compare the [curriculum] tables (keys and values, comments stripped) of BR's and MAIN's T. rex stance.toml."""
import sys, tomllib
a = tomllib.load(open(sys.argv[1], 'rb')).get('curriculum', {})
b = tomllib.load(open(sys.argv[2], 'rb')).get('curriculum', {})
for k in sorted(set(a) | set(b)):
    if a.get(k) != b.get(k):
        print('DIFF', k, 'BR=', a.get(k), 'MAIN=', b.get(k))
print('keys BR', len(a), 'keys MAIN', len(b))
