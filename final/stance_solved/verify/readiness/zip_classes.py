"""Read only: the JSON 'data' member of an SB3 zip; print the module/qualname of the pickled classes (no unpickle)."""
import base64, json, sys, zipfile, pickletools, io
for p in sys.argv[1:]:
    with zipfile.ZipFile(p) as z:
        d = json.loads(z.read("data"))
    out = {}
    for key in ("policy_class", "rollout_buffer_class"):
        raw = base64.b64decode(d[key][":serialized:"])
        strs = [arg for op, arg, pos in pickletools.genops(io.BytesIO(raw)) if isinstance(arg, str)]
        out[key] = strs
    pk = d.get("policy_kwargs", {})
    print(p.split("/")[-4], p.split("/")[-1], out, "observation_space shape:", d.get("observation_space", {}).get("_shape"),
          "mesozoic_critic_phase" in d, {k: v for k, v in d.items() if "critic_phase" in k})
