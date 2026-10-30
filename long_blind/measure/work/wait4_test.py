import os, subprocess, sys, time
busy = "x = 0\nfor i in range(30000000):\n    x += i\n"
code = "import subprocess, sys\nsubprocess.run([sys.executable, '-c', %r])\n" % busy
t0 = time.monotonic()
p = subprocess.Popen([sys.executable, "-c", code])
pid, status, ru = os.wait4(p.pid, 0)
p.returncode = os.waitstatus_to_exitcode(status)
print("rc", p.returncode, "wall", round(time.monotonic() - t0, 2), "utime", round(ru.ru_utime, 2), "stime", round(ru.ru_stime, 2))
