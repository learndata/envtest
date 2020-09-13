# update for feature1
import torch

if torch.cuda.is_available():
    device = torch.device("cuda")
    x = torch.rand(10000, 256, device=device)
    y = x.to(device)
    print(x[0:5, 0:5])
    print(y.to("cpu", torch.double)[0:5, 0:5])