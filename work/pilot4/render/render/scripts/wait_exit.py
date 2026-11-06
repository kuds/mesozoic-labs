"""Block until process PID exits (a pidfd becomes readable when it does) or TIMEOUT seconds pass; one wait, no loop."""
import os, select, sys
pid, timeout = int(sys.argv[1]), float(sys.argv[2])
try:
    fd = os.pidfd_open(pid)
except ProcessLookupError:
    print(f"pid {pid}: already gone"); sys.exit(0)
ready, _, _ = select.select([fd], [], [], timeout)
print(f"pid {pid}: {'exited' if ready else 'still running after the wait'}")
sys.exit(0 if ready else 1)
