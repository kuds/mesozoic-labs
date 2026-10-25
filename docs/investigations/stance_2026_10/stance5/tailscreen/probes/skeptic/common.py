import os, sys, importlib.util
for _k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[_k] = "1"
os.environ["PYTHONDONTWRITEBYTECODE"] = "1"; sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.realpath(__file__))
def setup(zp, pkl):
    spec = importlib.util.spec_from_file_location("judge_stance5", os.path.join(HERE, "judge_stance5.py"))
    J = importlib.util.module_from_spec(spec); sys.modules["judge_stance5"] = J; spec.loader.exec_module(J)
    J.isolate(os.path.join(HERE, "code"))
    import torch; torch.set_num_threads(1)
    ctx = J.Ctx("LTSB"); J.modules_under(os.path.join(HERE, "code"))
    model, norm = J.load_checkpoint(ctx, zp, pkl)
    return J, ctx, model, norm
