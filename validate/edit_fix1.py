"""PS build: fixes after review of my own edits (bs_hooks: the mirror map built before any mutation; tests: the
no_learning mutant, the forward assertion, the real-env test's truncation step)."""
import sys
from pathlib import Path

tools = Path(sys.argv[1])


def edit(name, pairs):
    p = tools / name
    s = p.read_text()
    for old, new in pairs:
        assert s.count(old) == 1, (name, s.count(old), old[:90])
        s = s.replace(old, new)
    p.write_text(s)


edit("bs_hooks.py", [
    ('''    if type(ext) is not MlpExtractor:
        raise HookError(f"the policy's mlp_extractor is {type(ext).__name__}, not MlpExtractor")
    rec: dict[str, Any] = {''', '''    if type(ext) is not MlpExtractor:
        raise HookError(f"the policy's mlp_extractor is {type(ext).__name__}, not MlpExtractor")
    new_map = phase_mirror_map(state.mirror_map)      # built (and checked) before anything is changed
    rec: dict[str, Any] = {'''),
    ('''    state.mirror_map = phase_mirror_map(state.mirror_map)
    if getattr(state, "tensors", None):''', '''    state.mirror_map = new_map
    if getattr(state, "tensors", None):'''),
])
edit("tests/test_bs_hooks.py", [
    ('''        if j >= 1 and mutate != "no_learning":            # an "update": the critic's phase columns move
            with torch.no_grad():
                p.mlp_extractor.value_net[0].weight[:, SL] += 0.01 * (j + 1)
            m._n_updates += 7''', '''        if j >= 1:                                        # an "update": the critic's phase columns move
            m._n_updates += 7
            if mutate != "no_learning":
                with torch.no_grad():
                    p.mlp_extractor.value_net[0].weight[:, SL] += 0.01 * (j + 1)'''),
    ('''            assert torch.equal(p.forward(y, deterministic=True)[0], mu_load.clamp(-1e9, 1e9))''',
     '''            assert torch.equal(p.forward(y, deterministic=True)[0], mu_load)'''),
    ('''    zero = np.zeros(a.action_space.shape, dtype=np.float32)
    seen = {}
    for _ in range(1000):
        oa, ra, ta, tra, _ = a.step(zero)
        ob, rb, tb, trb, _ = b.step(zero)
        k = a._step_count
        assert np.array_equal(oa[:60], ob[:60]) and ra == rb and (ta, tra) == (tb, trb)
        assert oa[SL].tolist() == [float(np.float32(x)) for x in H.phase_tuple(k)]
        seen[k] = oa[SL].tolist()
        if ta or tra:
            break
    assert k == 1000 and tra and seen[1000] == [1.0, 1.0, 1.0]       # the time-limit's terminal observation
    assert seen[199] == [float(np.float32(0.995)), 0.0, float(np.float32(0.98 ** 1))] and seen[200] == [1.0, 1.0, 1.0]''',
     '''    zero = np.zeros(a.action_space.shape, dtype=np.float32)
    seen = {}

    def lockstep(n):
        for _ in range(n):
            oa, ra, ta, tra, _ = a.step(zero)
            ob, rb, tb, trb, _ = b.step(zero)
            k = a._step_count
            assert k == b._step_count and np.array_equal(oa[:60], ob[:60]) and ra == rb and (ta, tra) == (tb, trb)
            assert oa[SL].tolist() == [float(np.float32(x)) for x in H.phase_tuple(k)] and np.all(ob[SL] == 0.0)
            seen[k] = (oa[SL].tolist(), ta, tra)
            if ta or tra:
                return
    lockstep(210)                                       # across the gate: k = 199 / 200 / 201
    assert seen[199][0] == [float(np.float32(0.995)), 0.0, float(np.float32(0.98))]
    assert seen[200][0] == [1.0, 1.0, 1.0] and seen[201][0] == [1.0, 1.0, 1.0] and 1 in seen
    a._step_count = b._step_count = 998                 # the time limit: the terminal observation at k = 1000
    lockstep(2)
    assert seen[1000] == ([1.0, 1.0, 1.0], False, True)'''),
])
print("ok")
