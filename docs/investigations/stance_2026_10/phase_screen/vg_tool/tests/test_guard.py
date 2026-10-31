"""The refusal guard on synthetic roots inside vg_tool (pytest --basetemp under vg_tool/tmp): no score.json, status
incomplete or PROVISIONAL, an undecided entry, a queue mismatch, another root, no --root (no default); and, while it
refuses, nothing under <root>/runs or <root>/reads is opened, listed or statted.  Also run_root's wiring with the
simulation stubbed (the critic form per arm, settings, SCREEN.json against APPROVED)."""
import builtins
import io
import json
import os
import shutil
from pathlib import Path

import pytest
import vg_sigma as T

PS_SCREEN = Path("/home/user/work/phase_screen/SCREEN.json")      # read only (allowed: the pinned queue)
QUEUE = [tuple(x) for x in json.loads(PS_SCREEN.read_text())["queue"]]


def _classes(cls_of=None, T_of=None):
    out = []
    for arm, seed in QUEUE:
        name = f"{arm}_{seed}"
        cls = (cls_of or {}).get(name, "COUNTED")
        out.append({"name": name, "arm": arm, "seed": seed, "cls": cls,
                    "T": (T_of or {}).get(name, T.C.E if cls == "COUNTED" else None), "cause": None})
    return out


def make_root(base, *, score=True, status="final", classes=None, root_field=None):
    root = Path(base) / "root"
    (root / "score").mkdir(parents=True)
    shutil.copyfile(PS_SCREEN, root / "SCREEN.json")
    for arm, seed in QUEUE:                       # canaries a refusing guard must never touch
        d = root / "runs" / f"{arm}_{seed}" / "01_stance" / "models"
        d.mkdir(parents=True)
        (d / "stage1_final.zip").write_bytes(b"canary")
        (root / "reads" / f"{arm}_{seed}").mkdir(parents=True)
        (root / "reads" / f"{arm}_{seed}" / "markers.json").write_text("{}")
    if score:
        d = {"version": "synthetic", "root": str(root.resolve()) if root_field is None else root_field,
             "status": status, "line": 3, "line_key": "NO REDUCTION DETECTED",
             "classes": _classes() if classes is None else classes}
        (root / "score" / "score.json").write_text(json.dumps(d))
    return root


@pytest.fixture(autouse=True)
def no_simulation(monkeypatch):
    """A guard that wrongly lets a root through fails here at once, before any simulation."""
    def boom(*a, **k):
        raise AssertionError("the guard let the root through to the simulation")
    monkeypatch.setattr(T, "setup_sim", boom)
    monkeypatch.setattr(T, "measure_pair", boom)


@pytest.fixture
def watch(monkeypatch):
    """Record every open / stat / listing of a path under <root>/runs or <root>/reads."""
    hits = []

    def bad(p):
        s = os.fspath(p) if isinstance(p, (str, bytes, os.PathLike)) else ""
        s = s.decode() if isinstance(s, bytes) else s
        return "/root/runs" in s or "/root/reads" in s

    def wrap(fn, name):
        def inner(p, *a, **k):
            if bad(p):
                hits.append((name, os.fspath(p)))
            return fn(p, *a, **k)
        return inner

    for mod, name in ((builtins, "open"), (io, "open"), (os, "open"), (os, "stat"), (os, "lstat"),
                      (os, "listdir"), (os, "scandir")):
        monkeypatch.setattr(mod, name, wrap(getattr(mod, name), f"{mod.__name__}.{name}"))
    return hits


def _main(root, out, *extra):
    return T.main(["--root", str(root), "--out", str(out), *extra])


def test_refuses_without_score_json(tmp_path, watch, capsys):
    root = make_root(tmp_path, score=False)
    watch.clear()                                   # the fixture's own canary writes
    out = tmp_path / "out"
    assert _main(root, out) == T.EXIT_REFUSED
    assert "score.json missing" in capsys.readouterr().out
    assert not out.exists() and watch == []


@pytest.mark.parametrize("status", ["incomplete", "PROVISIONAL", "Final", None])
def test_refuses_unless_status_final(tmp_path, watch, capsys, status):
    classes = [{"name": f"{a}_{s}", "arm": a, "seed": s, "decided": False} for a, s in QUEUE] \
        if status == "incomplete" else None
    root = make_root(tmp_path, status=status, classes=classes)
    watch.clear()                                   # the fixture's own canary writes
    out = tmp_path / "out"
    assert _main(root, out) == T.EXIT_REFUSED
    assert "is not 'final'" in capsys.readouterr().out
    assert not out.exists() and watch == []


def test_refuses_an_undecided_entry(tmp_path, watch, capsys):
    name = f"{QUEUE[3][0]}_{QUEUE[3][1]}"
    root = make_root(tmp_path, classes=_classes({name: "UNDECIDED"}))
    watch.clear()                                   # the fixture's own canary writes
    out = tmp_path / "out"
    assert _main(root, out) == T.EXIT_REFUSED
    assert "is not decided" in capsys.readouterr().out
    assert not out.exists() and watch == []


def test_refuses_queue_mismatch_and_bad_T_and_other_root(tmp_path, watch):
    swapped = _classes()
    swapped[0], swapped[1] = swapped[1], swapped[0]
    name = f"{QUEUE[0][0]}_{QUEUE[0][1]}"
    roots = {"oa": make_root(tmp_path / "a", classes=swapped),                       # queue order differs
             "ob": make_root(tmp_path / "b", classes=_classes()[:7]),                # an entry missing
             "oc": make_root(tmp_path / "c", classes=_classes(T_of={name: T.C.E - 8192})),   # COUNTED, T != E
             "od": make_root(tmp_path / "d", root_field="/home/user/work/phase_screen"),     # another root
             "oe": make_root(tmp_path / "e", classes=_classes({f"{a}_{s}": "LOST" for a, s in QUEUE})),  # none
             "of": make_root(tmp_path / "f")}
    (roots["of"] / "score" / "score.json").write_text("{not json")
    watch.clear()                                   # the fixture's own canary writes
    for out, root in roots.items():
        assert _main(root, tmp_path / out) == T.EXIT_REFUSED, out
    assert watch == []
    assert not any((tmp_path / x).exists() for x in ("oa", "ob", "oc", "od", "oe", "of"))


def test_refuses_out_inside_the_root_or_ps(tmp_path, watch):
    root = make_root(tmp_path)
    watch.clear()                                   # the fixture's own canary writes
    assert _main(root, root / "vg") == T.EXIT_REFUSED
    assert _main(root, "/home/user/work/phase_screen/vg_report") == T.EXIT_REFUSED
    assert not (root / "vg").exists() and not Path("/home/user/work/phase_screen/vg_report").exists()
    assert watch == []


def test_root_mode_refuses_pair_options(tmp_path, watch):
    root = make_root(tmp_path)
    watch.clear()                                   # the fixture's own canary writes
    for extra in (["--episodes", "8"], ["--seed", "1"], ["--form", "plain"], ["--phase-start"]):
        assert _main(root, tmp_path / "o", *extra) == T.EXIT_REFUSED
    assert watch == []


def test_plan_passes_and_reads_only_score_and_screen(tmp_path, watch, capsys):
    stopped = f"{QUEUE[2][0]}_{QUEUE[2][1]}"
    lost = f"{QUEUE[5][0]}_{QUEUE[5][1]}"
    root = make_root(tmp_path, classes=_classes({stopped: "STOPPED", lost: "LOST"}, {stopped: T.C.E - 131072}))
    watch.clear()                                   # the fixture's own canary writes
    out = tmp_path / "out"
    assert _main(root, out, "--plan") == T.EXIT_OK
    plan = json.loads(capsys.readouterr().out)
    counted = [c["name"] for c in plan["guard"]["counted"]]
    assert counted == [f"{a}_{s}" for a, s in QUEUE if f"{a}_{s}" not in (stopped, lost)]
    assert plan["would_read"][counted[0]]["final"][0].endswith(f"runs/{counted[0]}/01_stance/models/stage1_final.zip")
    assert len(plan["would_read"][counted[0]]["saved_zips"]) == 5
    assert watch == [] and not out.exists()


def test_guard_returns_counted_in_queue_order(tmp_path, watch):
    root = make_root(tmp_path)
    watch.clear()                                   # the fixture's own canary writes
    plan = T.guard(root)
    assert [(c["arm"], c["seed"]) for c in plan["counted"]] == QUEUE
    assert plan["score_status"] == "final" and len(plan["score_sha256"]) == 64
    assert watch == []


# ------------------------------------------------------------------------------------------------ fixer additions
def test_root_is_required_no_default(tmp_path, monkeypatch, capsys):
    """No --root and no --pair refuses before anything is opened: guard / run_root / run_pair are stubbed to fail, so
    this test can never open <PS>/score (outcome-bearing while the screen runs), whatever the code does."""
    def boom(*a, **k):
        raise AssertionError("reached the guard / a mode without an explicit --root or --pair")
    for name in ("guard", "run_root", "run_pair"):
        monkeypatch.setattr(T, name, boom)
    out = tmp_path / "o"
    for extra in ([], ["--plan"]):
        assert T.main(["--out", str(out), *extra]) == T.EXIT_REFUSED
        assert "--root is required" in capsys.readouterr().out
    root = make_root(tmp_path)
    for extra in (["--root", str(root)], ["--plan"]):                       # root-mode options with --pair
        assert T.main(["--pair", "a.zip", "a.pkl", "--form", "plain", "--out", str(out), *extra]) == T.EXIT_REFUSED
    assert not out.exists()


def _tiny_zip(path, seed):
    import zipfile

    import torch

    g = torch.Generator().manual_seed(seed)
    buf = io.BytesIO()
    torch.save({"log_std": -1.7 + 0.1 * torch.randn(15, generator=g)}, buf)
    with zipfile.ZipFile(path, "w") as zf:
        zf.writestr("policy.pth", buf.getvalue())


def make_full_root(base, classes, approved_screen_sha=None):
    """A synthetic root with every input run_root reads: tiny final / saved / start-copy zips (policy.pth log_std
    only), a pkl stand-in, markers with the loop row at E in the arm's critic form; APPROVED copied from <PS> (read
    only), its screen_sha256 replaced when approved_screen_sha is given."""
    import hashlib

    root = Path(base) / "root"
    (root / "score").mkdir(parents=True)
    shutil.copyfile(PS_SCREEN, root / "SCREEN.json")
    appr = json.loads(Path("/home/user/work/phase_screen/APPROVED").read_text())
    if approved_screen_sha is not None:
        appr["screen_sha256"] = approved_screen_sha
    (root / "APPROVED").write_text(json.dumps(appr))
    for i, (arm, seed) in enumerate(QUEUE):
        name = f"{arm}_{seed}"
        m = root / "runs" / name / T.C.STAGE_DIR / "models"
        m.mkdir(parents=True)
        _tiny_zip(m / T.C.FINAL_ZIP, 100 + i)
        (m / T.C.FINAL_PKL).write_bytes(b"pkl " + name.encode())
        _tiny_zip(m / T.C.periodic_names(T.C.S)[0], 1)
        for j, s in enumerate(T.C.SAVED_STEPS):
            _tiny_zip(m / T.C.periodic_names(s)[0], 1000 * i + j)
        sha = {p: hashlib.sha256((m / p).read_bytes()).hexdigest() for p in (T.C.FINAL_ZIP, T.C.FINAL_PKL)}
        (root / "reads" / name).mkdir(parents=True)
        (root / "reads" / name / "markers.json").write_text(json.dumps({"loop": [
            {"steps": T.C.E, "zip_sha256": sha[T.C.FINAL_ZIP], "pkl_sha256": sha[T.C.FINAL_PKL],
             "critic_form": {"control": "plain", "critic_phase": "phase+raw+settled"}[arm]}]}))
    (root / "score" / "score.json").write_text(json.dumps({"version": "synthetic", "root": str(root.resolve()),
                                                           "status": "final", "classes": classes}))
    return root


def _stub_simulation(monkeypatch):
    """setup_sim and measure_pair stubbed (no CPU): the stub records each pair's requested forms and settings."""
    calls = []

    def fake_measure(zip_path, pkl_path, *, forms, seed, n_episodes, label, expect_steps=None, phase_start=False,
                     max_vec_steps=None):
        import hashlib

        calls.append({"label": label, "forms": tuple(forms), "seed": seed, "n": n_episodes, "expect": expect_steps,
                      "phase_start": phase_start})
        rows = [{"bucket": b, "lo": 0, "hi": 0, "n_episodes": 1, "n_states": 1, "mean_err": 0.01, "se_err": None,
                 "mean_abs_err": 0.01, "se_abs_err": None, "pooled_err": 0.01, "pooled_abs_err": 0.01, "mean_V": 0.0,
                 "mean_G": 0.0} for b in T.BUCKET_NAMES]
        rec = {"label": label, "zip": str(zip_path), "pkl": str(pkl_path),
               "zip_sha256": hashlib.sha256(Path(zip_path).read_bytes()).hexdigest(),
               "pkl_sha256": hashlib.sha256(Path(pkl_path).read_bytes()).hexdigest(), "num_timesteps": expect_steps,
               "env_seeds": [seed + r for r in range(4)], "units": {"sigma_r": 1.0, "gamma": T.GAMMA},
               "episodes": {"n": n_episodes, "truncated": n_episodes, "terminated": 0},
               "forms": {f: {"buckets": rows} for f in forms}, "sigma": T.sigma_record(T.read_log_std(zip_path))}
        return rec, [], {"label": label}

    monkeypatch.setattr(T, "setup_sim", lambda tmp: "stub")
    monkeypatch.setattr(T, "measure_pair", fake_measure)
    return calls


def test_root_mode_wiring_forms_per_arm(tmp_path, monkeypatch):
    """run_root end to end with the simulation stubbed: S once in both forms with the start transform, then each
    COUNTED run in queue order in its arm's critic form (control plain, critic_phase phase; literal here, not
    ARM_FORM), never phase_start, at E, the pre-registered seed 6271 and 32 episodes; STOPPED / LOST not measured."""
    calls = _stub_simulation(monkeypatch)
    stopped, lost = f"{QUEUE[3][0]}_{QUEUE[3][1]}", f"{QUEUE[4][0]}_{QUEUE[4][1]}"
    root = make_full_root(tmp_path, _classes({stopped: "STOPPED", lost: "LOST"}, {stopped: T.C.E - 131072}))
    out = tmp_path / "out"
    code = T.main(["--root", str(root), "--out", str(out)])
    res = json.loads((out / "vg_sigma.json").read_text())
    counted = [f"{a}_{s}" for a, s in QUEUE if f"{a}_{s}" not in (stopped, lost)]
    want_forms = {"control": ("plain",), "critic_phase": ("phase",)}
    assert calls[0]["expect"] == T.C.S and calls[0]["forms"] == ("plain", "phase") and calls[0]["phase_start"] is True
    assert [c["label"] for c in calls[1:]] == counted
    for c in calls[1:]:
        arm = c["label"].rsplit("_", 1)[0]
        assert c["forms"] == want_forms[arm] and c["phase_start"] is False and c["expect"] == T.C.E, c
    assert all(c["seed"] == 6271 == T.PROD_SEED and c["n"] == 32 == T.N_EPISODES for c in calls)
    assert sorted({r["run"] for r in res["sigma_table"]}) == sorted(counted)
    assert all(p.get("markers_match") is not False for p in res["pairs"][1:])
    assert not any("SCREEN.json" in p for p in res["problems"])
    assert code == (T.EXIT_PROBLEM if res["problems"] else T.EXIT_OK)


def test_root_mode_flags_screen_sha_against_approved(tmp_path, monkeypatch):
    _stub_simulation(monkeypatch)
    root = make_full_root(tmp_path, _classes(), approved_screen_sha="0" * 64)
    out = tmp_path / "out"
    assert T.main(["--root", str(root), "--out", str(out)]) == T.EXIT_PROBLEM
    res = json.loads((out / "vg_sigma.json").read_text())
    assert any(p.startswith("SCREEN.json: sha256") for p in res["problems"])
