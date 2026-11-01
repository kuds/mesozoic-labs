"""usage: python -I -B compare_failed.py BASE_failed.txt AFTER_failed.txt -- the failure sets' difference."""
import sys

def load(path):
    lines = open(path).read().splitlines()
    return lines[0], {line for line in lines[1:] if line.strip()}

(ca, a), (cb, b) = load(sys.argv[1]), load(sys.argv[2])
print("base ", ca)
print("after", cb)
print(f"failing in both: {len(a & b)}")
print(f"new failures (after only): {len(b - a)}")
for line in sorted(b - a):
    print("  NEW", line)
print(f"fixed (base only): {len(a - b)}")
for line in sorted(a - b):
    print("  GONE", line)
