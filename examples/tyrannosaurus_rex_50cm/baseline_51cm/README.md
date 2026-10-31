# Original 51.5 cm reference

This preserves the original 7.617 kg, 515 mm, 18-axis model used by the
parts-budget study. Its geometry and assigned masses are unchanged; the XML
now locates the shared meshes in `../assets`.

The [selected default](../README.md) uses silhouette C at 583.6 mm and 7.957 kg.
Its 340 g net mass addition is recorded separately, retaining the 600 g reserve.
The baseline remains useful for reproducing comparisons and reconciling the
original parts budget. Neither configuration has demonstrated walking or
physical hardware viability.

- [Reference MJCF](model.xml)
- [Reference component ledger](components.json)

```bash
python -m examples.tyrannosaurus_rex_50cm.build_model --baseline
python -m examples.tyrannosaurus_rex_50cm.demo --view \
  --model examples/tyrannosaurus_rex_50cm/baseline_51cm/model.xml
```

Reset to the `home` keyframe when loading the XML directly. The example viewer
does this automatically. The main generator without `--baseline` writes the
selected silhouette C, leaving this reference separate.
