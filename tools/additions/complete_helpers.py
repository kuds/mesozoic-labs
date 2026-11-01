# ------------------------------------------------------------------------------------------------- PC: the switch
def critic_phase_fact(name, sc, pins) -> dict:
    """PC launch-check fact (8) and C4 (p): stage_config.json records the critic-phase switch as the pin
    critic_phase_switch states it ({"path": dotted path, "value": the value as the file writes it}; a boolean pin is
    met only by the JSON boolean itself).  ok None when the pin is unfilled or the file is not read."""
    want = C.pin(pins, "critic_phase_switch")
    if sc is None:
        return fact(name, None, None, "not computed: stage_config.json not read")
    if not isinstance(want, dict) or "path" not in want or "value" not in want:
        return fact(name, None, None, "pin critic_phase_switch unfilled")
    got = C.dig(sc, want["path"])
    if isinstance(want["value"], bool):
        ok = isinstance(got, bool) and got is want["value"]
    else:
        ok = got == want["value"]
    return fact(name, ok, dict(path=want["path"], got=got, want=want["value"]))


