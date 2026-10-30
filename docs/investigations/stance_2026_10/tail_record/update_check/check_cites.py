"""Map ledger ids cited in the record to sections; compare with where_in_record. Read only."""
import csv, re, sys
rec, led = sys.argv[1], sys.argv[2]
lines = open(rec, encoding='utf-8').read().split('\n')
rows = {r['id']: r for r in csv.DictReader(open(led, newline='', encoding='utf-8'))}
sec = 'status'
cites = {}
def sec_of(h):
    m = re.match(r'#+\s+(\d+(?:\.\d+)*)', h)
    return '§' + m.group(1) if m else None
cur_major = None
for i, ln in enumerate(lines, 1):
    if ln.startswith('#'):
        s = sec_of(ln)
        if s: sec = s
        continue
    for m in re.finditer(r'L(\d{3})(?:\s*[-–]\s*L(\d{3}))?', ln):
        a = int(m.group(1)); b = int(m.group(2)) if m.group(2) else a
        for k in range(a, b + 1):
            cites.setdefault('L%03d' % k, set()).add(sec)
def norm(s):
    # reduce §4.11 -> §4.11 ; keep as-is
    return s
missing = [i for i in rows if i not in cites]
print('rows never cited:', missing)
unknown = [i for i in cites if i not in rows]
print('cited ids not in ledger:', unknown)
ids = sys.argv[3:] if len(sys.argv) > 3 else sorted(rows)
for i in ids:
    w = set(x.strip() for x in rows[i]['where_in_record'].split(';') if x.strip())
    c = cites.get(i, set())
    # collapse subsections to how ledger names them: ledger may use §3.6, §4.8, §2, etc.
    def collapse(s):
        out=set()
        for x in s:
            out.add(x)
        return out
    print(i, 'where:', sorted(w), '| cited in:', sorted(c))
