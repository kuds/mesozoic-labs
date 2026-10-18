import zipfile, io, torch, numpy as np, json
S="scratchpad"
R=S+"/foot/symmetry/screen/runs"
O=S+"/foot/battery/own_sigma/panels"
paths={"s60_6M":R+"/sym_lean_settled_s60/resume6m/models/stage1_6291456_steps.zip","s52_6M":R+"/sym_lean_settled_s52/resume6m/models/stage1_6291456_steps.zip",
"s42_6M":R+"/sym_lean_settled_s42/resume6m/models/stage1_6291456_steps.zip","s60_rb524288":R+"/sym_lean_settled_s60/models/robust_best_model.zip",
"s52_rb917504":R+"/sym_lean_settled_s52/models/robust_best_model.zip","s42_rb655360":R+"/sym_lean_settled_s42/models/robust_best_model.zip"}
for k,p in paths.items():
    z=zipfile.ZipFile(p)
    sd=torch.load(io.BytesIO(z.read("policy.pth")),map_location="cpu",weights_only=True)
    ls=sd['log_std'].numpy().astype(np.float64)
    m=json.load(open(f"{O}/{k}_stochastic/panel.json"))['meta']
    print(k, np.round(np.exp(ls),4).tolist(), float(np.abs(ls-np.array(m['log_std'])).max()))
