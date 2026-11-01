"""PC manipulation check (pc/manip_check.py): tests on real checkpoints with the HEAD measuring copy (skipped when
ST/tools/dry/copy_head or the smoke pair is absent).  The bank is built once per module (two deterministic episodes,
first 500 observations each).

    cd /tmp && PY -I -B -m pytest -q -p no:cacheprovider ST/tools/tests/test_manip_pc.py
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

T = Path(__file__).resolve().parent.parent
COPY = T / "dry" / "copy_head"
SMOKE = T / "dry" / "smoke_train" / "run" / "models"
S1 = Path("/tmp/claude-0/-home-user-mesozoic-labs/0fdf74a4-2989-56c9-a773-92101004d7de/scratchpad")
S6 = S1 / "foot/stance6/fetch/F2/stance6_r10LTSB_s80/copies/01_stance/models"
pytestmark = pytest.mark.skipif(not (COPY.is_dir() and SMOKE.is_dir()), reason="measuring copy or smoke pair absent")


def _mc():
    spec = importlib.util.spec_from_file_location("manip_check", T / "pc" / "manip_check.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["manip_check"] = mod
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def env_and_loader():
    sys.path.insert(0, str(COPY))
    from environments.shared.config import load_all_stages
    from environments.shared.model_id import ModelId
    from environments.shared.plant_contract import current_plant_identity
    from environments.shared.reporting.episode_trace import load_trace_policy
    from environments.shared.species_registry import env_class as model_env_class
    from environments.shared.stage_manifest import load_stage_manifest

    model_id = ModelId("tyrannosaurus_rex", "anatomical")
    stage = load_stage_manifest(model_id).resolve_behavior("stance").reference
    kw = dict(load_all_stages(model_id)[stage]["env_kwargs"])
    cls = model_env_class(model_id)

    def load(zp, pp):
        return load_trace_policy(str(zp), str(pp), lambda: cls(**kw), plant_identity=current_plant_identity(model_id))

    return cls, kw, load


@pytest.fixture(scope="module")
def smoke(env_and_loader):
    cls, kw, load = env_and_loader
    MC = _mc()
    tp = load(SMOKE / "stage1_final.zip", SMOKE / "stage1_final_vecnorm.pkl")
    env = cls(**kw)
    try:
        bank, ks, per = MC.collect_bank(tp, env)
    finally:
        env.close()
    return MC, tp, bank, ks, per


def test_bank_is_two_episodes_of_500_with_the_envs_counter(smoke):
    _MC, _tp, bank, ks, per = smoke
    assert bank.shape == (1000, 63) and str(bank.dtype) == "float32"
    assert list(ks[:3]) == [0, 1, 2] and ks[499] == 499 and ks[500] == 0 and ks[-1] == 499
    assert [p["seed"] for p in per] == [3042, 3043]


def test_smoke_pair_is_in_force(smoke):
    MC, tp, bank, ks, _ = smoke
    r = MC.manipulation_checks(tp.model.policy, bank, ks)
    assert r["verdict"] == "IN FORCE" and r["failing"] == []
    c = r["checks"]
    assert all(v > 0 for v in c["(i) phase in force"]["max_abs_dV_by_perturbation"].values())
    assert all(v == 0.0 for v in c["(ii) actor independent"]["max_abs_dmu_by_perturbation"].values())
    assert c["(ii) actor independent"]["actor_first_layer_inputs"] == 63


def test_zeroed_phase_columns_are_not_in_force_and_feed_equals_stock(smoke):
    import torch as th

    MC, tp, bank, ks, _ = smoke
    pol = tp.model.policy
    w = pol.mlp_extractor.value_net[0].weight
    saved = w.detach().clone()
    try:
        with th.no_grad():
            w[:, 63:] = 0.0
        r = MC.manipulation_checks(pol, bank, ks)
        assert r["verdict"] == "NOT IN FORCE" and r["failing"] == ["(i)"]
        assert r["checks"]["(i) phase in force"]["phase_weights_nonzero"] == 0
    finally:
        with th.no_grad():
            w.copy_(saved)


def test_critic_blind_to_obs6_fails_iii(smoke):
    import torch as th

    MC, tp, bank, ks, _ = smoke
    w = tp.model.policy.mlp_extractor.value_net[0].weight
    saved = w.detach().clone()
    try:
        with th.no_grad():
            w[:, 6] = 0.0
        r = MC.manipulation_checks(tp.model.policy, bank, ks)
        assert r["verdict"] == "NOT IN FORCE" and r["failing"] == ["(iii)"]
    finally:
        with th.no_grad():
            w.copy_(saved)


def test_effective_and_settled_feeds_differ_on_a_phase_critic(smoke):
    import numpy as np

    MC, tp, bank, ks, _ = smoke
    v_eff = MC.critic_values(tp.model.policy, bank, ks, mode="effective")
    v_set = MC.critic_values(tp.model.policy, bank, ks, mode="settled")
    sel = ks >= 200                                 # phi(k) = phi(1000) = (1, 1, 1) for k >= 200
    assert np.array_equal(v_eff[sel], v_set[sel]) and not np.array_equal(v_eff[~sel], v_set[~sel])


@pytest.mark.skipif(not S6.is_dir(), reason="STANCE6 F2 copies absent")
def test_widened_stance6_pair(env_and_loader):
    import numpy as np

    cls, kw, load = env_and_loader
    MC = _mc()
    tp = load(S6 / "stage1_final.zip.copy1", S6 / "stage1_final_vecnorm.pkl.copy1")
    env = cls(**kw)
    try:
        bank, ks, _ = MC.collect_bank(tp, env, limit=50)
    finally:
        env.close()
    stock = tp.model.policy
    assert MC.manipulation_checks(stock, bank, ks)["verdict"] == "NOT IN FORCE"
    wide = MC.widen_stock_policy(stock, 200, 0.98)
    r = MC.manipulation_checks(wide, bank, ks)
    assert r["verdict"] == "NOT IN FORCE" and r["failing"] == ["(i)"]
    assert np.array_equal(MC.critic_values(wide, bank, ks), MC.critic_values(stock, bank))
    assert np.array_equal(MC.actor_means(wide, bank, MC.phase_matrix(wide.critic_phase, ks)), MC.actor_means(stock, bank))
