import os
import math
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.distributions import MultivariateNormal
from typing import Optional

def test_maf(model, train, test_loader):
    model.eval()
    test_loss = []
    _, _ = model.forward(train)
    with torch.no_grad():
        for batch in test_loader:
            u, log_det = model.forward(batch.float())

            negloglik_loss = 0.5 * (u ** 2).sum(dim=1)
            negloglik_loss += 0.5 * batch.shape[1] * np.log(2 * math.pi)
            negloglik_loss -= log_det

            test_loss.extend(negloglik_loss)
    N = len(test_loss)
    print(
        "Test loss: {:.4f} +/- {:.4f}".format(
            np.mean(test_loss), 2 + np.std(test_loss) / np.sqrt(N)
        )
    )


def test_made(model, test_loader, cuda_device: Optional[int]=None):
    if cuda_device != None:
        torch.cuda.set_device(cuda_device)
    model.eval()
    neglogP = []
    with torch.no_grad():
        for batch in test_loader:
            if cuda_device == None:
                input = batch.float()
                out = model.forward(input)
                mu, logp = torch.chunk(out, 2, dim=1)
                u = (input - mu) * torch.exp(0.5 * logp)

                negloglik_loss = 0.5 * (u ** 2).sum(dim=1)
                negloglik_loss += 0.5 * input.shape[1] * np.log(2 * math.pi)
                negloglik_loss -= 0.5 * torch.sum(logp, dim=1)

                neglogP.extend(negloglik_loss)
            else:
                input = batch.float().cuda()
                out = model.forward(input)
                mu, logp = torch.chunk(out, 2, dim=1)
                u = (input - mu) * torch.exp(0.5 * logp).cuda()

                negloglik_loss = 0.5 * (u ** 2).sum(dim=1)
                negloglik_loss += 0.5 * input.shape[1] * np.log(2 * math.pi)
                negloglik_loss -= 0.5 * torch.sum(logp, dim=1)
                neglogP.extend(negloglik_loss.cpu())
    
    print(len(neglogP))
    return neglogP


