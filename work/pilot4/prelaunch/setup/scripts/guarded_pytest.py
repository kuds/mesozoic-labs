"""Run pytest on the pilot-4 worktree with every training entry point blocked.

Usage (cwd = the worktree):  python scripts/guarded_pytest.py <pytest args...>

Before pytest starts, this patches, to raise TrainingBlocked:
  * every stable_baselines3 algorithm's learn() and train() (BaseAlgorithm, OnPolicyAlgorithm,
    OffPolicyAlgorithm, PPO, SAC, A2C, TD3, DDPG, DQN where present);
  * the step() of every torch.optim.Optimizer subclass in torch.optim;
  * torch.Tensor.backward and torch.autograd.backward;
  * subprocess.Popen when it starts a Python process whose arguments name training
    ('train', 'curriculum', 'learn', 'notebook', 'nbconvert', 'papermill', 'jupyter').
A test that reaches any of them fails with TrainingBlocked before any update happens; the
static deselection list (deselect.txt) is the primary filter, this is the safety net.
It also puts the worktree first on sys.path and refuses to run if 'environments' would be
imported from anywhere else.
"""

from __future__ import annotations

import os
import subprocess
import sys

WORKTREE = os.environ.get(
    "GUARDED_WORKTREE", "/tmp/claude-0/-home-user-mesozoic-labs/aff94cce-91ba-5515-b81b-a9f018e73dc9/scratchpad/wt_p4"
)


class TrainingBlocked(RuntimeError):
    pass


def _blocked(name):
    def _raise(*args, **kwargs):
        raise TrainingBlocked(f"training entry point blocked by the pilot-4 build guard: {name}")

    return _raise


def install_guard() -> list[str]:
    patched: list[str] = []
    import torch
    import torch.optim as optim

    for attr in dir(optim):
        cls = getattr(optim, attr)
        if isinstance(cls, type) and issubclass(cls, optim.Optimizer) and "step" in cls.__dict__:
            cls.step = _blocked(f"torch.optim.{attr}.step")
            patched.append(f"torch.optim.{attr}.step")
    torch.Tensor.backward = _blocked("torch.Tensor.backward")
    torch.autograd.backward = _blocked("torch.autograd.backward")
    patched += ["torch.Tensor.backward", "torch.autograd.backward"]

    import stable_baselines3 as sb3
    from stable_baselines3.common import base_class, off_policy_algorithm, on_policy_algorithm

    classes = [
        base_class.BaseAlgorithm,
        on_policy_algorithm.OnPolicyAlgorithm,
        off_policy_algorithm.OffPolicyAlgorithm,
    ]
    for name in ("PPO", "SAC", "A2C", "TD3", "DDPG", "DQN"):
        if hasattr(sb3, name):
            classes.append(getattr(sb3, name))
    for cls in classes:
        for method in ("learn", "train"):
            if method in cls.__dict__:
                setattr(cls, method, _blocked(f"{cls.__module__}.{cls.__name__}.{method}"))
                patched.append(f"{cls.__name__}.{method}")

    original_init = subprocess.Popen.__init__
    words = ("train", "curriculum", "learn", "notebook", "nbconvert", "papermill", "jupyter")

    def guarded_init(self, args, *a, **k):
        argv = [str(x) for x in (args if isinstance(args, (list, tuple)) else [args])]
        joined = " ".join(argv).lower()
        if "python" in joined and any(word in joined for word in words):
            raise TrainingBlocked(f"python subprocess blocked by the pilot-4 build guard: {argv[:6]}")
        return original_init(self, args, *a, **k)

    subprocess.Popen.__init__ = guarded_init
    patched.append("subprocess.Popen.__init__ (python + training words)")
    return patched


class DeselectPlugin:
    """Deselect exactly the listed tests: an id equal to a listed base id, or that id plus a parametrization."""

    def __init__(self, bases: list[str], report_path: str | None) -> None:
        self.bases = bases
        self.report_path = report_path

    def pytest_collection_modifyitems(self, config, items):
        kept, dropped = [], []
        for item in items:
            if any(item.nodeid == base or item.nodeid.startswith(base + "[") for base in self.bases):
                dropped.append(item)
            else:
                kept.append(item)
        if dropped:
            config.hook.pytest_deselected(items=dropped)
            items[:] = kept
        if self.report_path:
            with open(self.report_path, "w", encoding="utf-8") as stream:
                for item in dropped:
                    stream.write(item.nodeid + "\n")


def main() -> int:
    if os.getcwd() != WORKTREE:
        print(f"guarded_pytest: run from {WORKTREE}, not {os.getcwd()}", file=sys.stderr)
        return 3
    args = list(sys.argv[1:])
    bases: list[str] = []
    report = None
    if "--deselect-file" in args:
        index = args.index("--deselect-file")
        path = args[index + 1]
        del args[index : index + 2]
        with open(path, encoding="utf-8") as stream:
            bases = [line.split("\t")[0].strip() for line in stream if line.strip() and not line.startswith("#")]
    if "--deselected-report" in args:
        index = args.index("--deselected-report")
        report = args[index + 1]
        del args[index : index + 2]
    sys.path.insert(0, WORKTREE)
    import environments

    if not os.path.realpath(environments.__file__).startswith(WORKTREE + os.sep):
        print(f"guarded_pytest: environments imports from {environments.__file__}", file=sys.stderr)
        return 3
    patched = install_guard()
    print(
        f"guarded_pytest: environments from {environments.__file__}; blocked {len(patched)} entry points; "
        f"{len(bases)} base ids to deselect"
    )
    import pytest

    return int(pytest.main(args, plugins=[DeselectPlugin(bases, report)]))


if __name__ == "__main__":
    raise SystemExit(main())
