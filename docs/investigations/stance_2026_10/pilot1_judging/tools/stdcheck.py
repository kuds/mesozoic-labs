import sys, zipfile, io, torch, math
for p in sys.argv[1:]:
    with zipfile.ZipFile(p) as z:
        sd = torch.load(io.BytesIO(z.read("policy.pth")), map_location="cpu", weights_only=True)
    ls = sd["log_std"].numpy()
    std = [math.exp(v) for v in ls]
    print(f"{p.split('scratchpad/')[-1]:<95} mean std {sum(std)/len(std):.3f}  min {min(std):.3f}  max {max(std):.3f}")
