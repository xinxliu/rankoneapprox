import torch
import matplotlib.pyplot as plt
if __name__ == "__main__":

    loss = torch.load("./logs/cfg32.loss")
    ep = []
    lo = []
    for i, l in loss:
        lo.append(l.avg)
        ep.append(i)
    plt.plot(ep,lo)
    plt.savefig("./figs/cfg32.png")