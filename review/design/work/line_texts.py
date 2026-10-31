import importlib.util, re, sys
sys.dont_write_bytecode = True
PS = "/home/user/work/phase_screen"
crit = " ".join(open(PS + "/criteria.txt", encoding="utf-8").read().split())
spec = importlib.util.spec_from_file_location("sbx", PS + "/tools/score_bs.py")
SB = importlib.util.module_from_spec(spec); spec.loader.exec_module(SB)
for k, t in SB.LINE_TEXT.items():
    print(k, '"' + t + '"' in crit)
for a in (SB.ADDITION_AS_COUNTED, SB.ADDITION_NOT_INFORMATIVE, SB.SEC_SUPPORTED, SB.SEC_BONFERRONI, SB.SEC_GUARD, SB.SEC_UNDER_LINE1):
    print(repr(a[:50]), ('"' + a + '"') in crit or a in crit)
