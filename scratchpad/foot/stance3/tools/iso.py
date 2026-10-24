"""Run a module (-m NAME args), a script (PATH args) or pytest (--pytest args) with `environments` imported from the
ISO_WT clone only (editable finder removed, main checkout dropped), no bytecode, BLAS threads 1."""
import os
import runpy
import sys

os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[k] = "1"
sys.dont_write_bytecode = True
MAIN = "/home/user/mesozoic-labs"
WT = os.path.realpath(os.environ["ISO_WT"])  # stance3 copy: ISO_WT is required (no default clone)
sys.meta_path[:] = [f for f in sys.meta_path if "__editable__" not in (getattr(f, "__module__", "") or type(f).__module__)]
sys.path_hooks[:] = [h for h in sys.path_hooks if "__editable__" not in getattr(h, "__module__", "")]
sys.path[:] = [p for p in sys.path if "__editable__" not in p and os.path.realpath(p or ".") != MAIN
               and os.path.realpath(p or ".") != os.path.dirname(os.path.abspath(__file__))]
sys.path_importer_cache.clear()
sys.path.insert(0, WT)
os.environ["PYTHONPATH"] = WT + (os.pathsep + os.environ["PYTHONPATH"] if os.environ.get("PYTHONPATH") else "")
import environments  # noqa: E402

assert os.path.realpath(environments.__file__).startswith(WT + "/"), environments.__file__
print(f"[iso] environments = {os.path.realpath(environments.__file__)}", file=sys.stderr, flush=True)
try:  # stance3 copy: one torch thread in every process
    import torch  # noqa: E402

    torch.set_num_threads(1)
    print(f"[iso] torch threads = {torch.get_num_threads()}", file=sys.stderr, flush=True)
except ImportError:
    pass


def _check():
    bad = [n for n, m in list(sys.modules.items()) if (n == "environments" or n.startswith("environments."))
           and getattr(m, "__file__", None) and not os.path.realpath(m.__file__).startswith(WT + "/")]
    if bad:
        print("[iso] MODULES OUTSIDE THE CLONE:", bad[:5], file=sys.stderr)
        return 5
    return 0


args = sys.argv[1:]
rc = 0
try:
    if args and args[0] == "--pytest":
        import pytest
        rc = pytest.main(["-p", "no:cacheprovider", *args[1:]])
    elif args and args[0] == "-m":
        sys.argv = [args[1], *args[2:]]
        runpy.run_module(args[1], run_name="__main__", alter_sys=True)
    else:
        sys.argv = list(args)
        runpy.run_path(args[0], run_name="__main__")
except SystemExit as e:
    rc = e.code if isinstance(e.code, int) else (0 if e.code is None else 1)
rc = rc or _check()
sys.exit(rc)
