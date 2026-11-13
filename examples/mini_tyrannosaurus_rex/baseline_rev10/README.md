# Mini revision-10 reference

This is the previous brown/tan 800 g model from commit
`25f47bb55e3d070e3fc456e1a595611e6726744b`, retained as the mechanism and controller
reference for the silhouette-C adaptation. Its two XML include paths now point
one directory up to the unchanged shared assets. Apart from those paths, the
model and component ledger match that commit exactly.

The adjacent validation, tail, walking and static-support JSON files preserve
historical evidence unchanged. Their hashes refer to the original model paths
and original runtime, not the relocated XML bytes. The original 29.616-second
wagging result was a failure, and is not a current simulation result. See the
[updated walking report](../walking_validation.json) for the same-runtime
revision-10 wag replay and the silhouette-C trials. No walking policy is qualified.

View the original with MuJoCo's viewer, or use the existing renderer:

```bash
python -m examples.mini_tyrannosaurus_rex.render_design --model examples/mini_tyrannosaurus_rex/baseline_rev10/model.xml --output-dir artifacts/mini-rev10
```

The selected default is [silhouette C](../README.md). Running `head_design`
regenerates that default from this retained mechanism without scaling motors,
changing joint pivots or raising torque limits.
