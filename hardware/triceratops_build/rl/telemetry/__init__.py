"""Training and hardware telemetry for legged-robot simulation runs (decision D21; plan PLAN_STEERING_AND_TELEMETRY.md).

Robot-agnostic by construction: nothing here names a species, a build or a joint. Each module takes what it needs
(a Stable-Baselines3 model, a MuJoCo model, a drive table, a callable that runs one episode) from the caller, so a
T. rex or a larger Triceratops env can use it unchanged. Simulation only.
"""
