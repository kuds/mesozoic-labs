"""Proof that final/criteria_ecs_final.txt differs from v3/criteria_ecs_v3.txt only inside v3's PINS block (l297-l301)."""
import difflib
import hashlib
import sys

ECS = "/tmp/claude-0/-home-user-mesozoic-labs/184ee67c-2802-5235-8e52-6dc7daa72cd3/scratchpad/r10round/ecs"
a = open(f"{ECS}/v3/criteria_ecs_v3.txt").read().splitlines(keepends=True)
b = open(f"{ECS}/final/criteria_ecs_final.txt").read().splitlines(keepends=True)
start = next(i for i, x in enumerate(a) if x.startswith("PINS ("))
end = next(i for i, x in enumerate(a) if x.startswith("END (draft v3"))
print(f"v3 {len(a)} lines; PINS block l{start + 1}-l{end} (heading to the line before END); final {len(b)} lines")
ops = [op for op in difflib.SequenceMatcher(None, a, b, autojunk=False).get_opcodes() if op[0] != "equal"]
for op in ops:
    print("  change:", op[0], f"v3 l{op[1] + 1}-l{op[2]}", f"final l{op[3] + 1}-l{op[4]}")
inside = all(start <= op[1] and op[2] <= end for op in ops)
head = a[:start] == b[:start]
tail = a[end:] == b[len(b) - (len(a) - end):]
print(f"  lines before PINS identical ({start}): {head}; lines from END on identical ({len(a) - end}): {tail}; "
      f"every change inside v3's PINS block: {inside}; changes: {len(ops)}")
print("  v3 sha256", hashlib.sha256("".join(a).encode()).hexdigest())
print("  final sha256", hashlib.sha256("".join(b).encode()).hexdigest())
ok = head and tail and inside and len(ops) == 1
print("PINS ONLY:", "PASS" if ok else "FAIL")
sys.exit(0 if ok else 1)
