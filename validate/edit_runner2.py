"""PS build: run_waves_bs.sh, the test-only scorer path override (LB's BS_SCORER, its hunks #52, #53, #57 reduced to
the path; the pinned name stays score_bs.py), run once."""
import sys
from pathlib import Path

p = Path(sys.argv[1])
s = p.read_text()


def rep(old, new):
    global s
    assert s.count(old) == 1, (s.count(old), old[:90])
    s = s.replace(old, new)


rep('''LAUNCH_OVERRIDE=""
# Test hook''', '''LAUNCH_OVERRIDE=""
SCORER_OVERRIDE=""        # PS (LB's BS_SCORER): the scorer file the APPROVED pin is checked against, test mode only
# Test hook''')
rep('''  LAUNCH_OVERRIDE=${BS_LAUNCH_DISK_BYTES:-}
fi''', '''  LAUNCH_OVERRIDE=${BS_LAUNCH_DISK_BYTES:-}
  SCORER_OVERRIDE=${BS_SCORER:-}
fi''')
rep('''SCORER_PIN=$("$PY" -I -B - "$BS/APPROVED" "$TOOLS/score_bs.py" <<''',
    '''SCORER_PIN=$("$PY" -I -B - "$BS/APPROVED" "${SCORER_OVERRIDE:-$TOOLS/score_bs.py}" <<''')
p.write_text(s)
print("ok")
