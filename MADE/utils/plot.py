import os
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from torch.distributions import MultivariateNormal


def sample_digits_maf(model, epoch, random_order=False, seed=None, test=False):
    model.eval()
    n_samples = 80
    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)
    if random_order is True:
        np.random.seed(seed)
        order = np.random.permutation(784)
    else:
        order = np.arange(784)

    u = torch.zeros(n_samples, 784).normal_(0, 1)
    mvn = MultivariateNormal(torch.zeros(28 * 28), torch.eye(28 * 28))
    log_prob = mvn.log_prob(u)
    samples, log_det = model.backward(u)

    # log_det = log_prob - log_det
    # log_det = log_det[np.logical_not(np.isnan(log_det.detach().numpy()))]
    # idx = np.argsort(log_det.detach().numpy())
    # samples = samples[idx].flip(dims=(0,))
    # samples = samples[80 : 80 + n_samples]

    samples = (torch.sigmoid(samples) - 1e-6) / (1 - 2e-6)
    samples = samples.detach().cpu().view(n_samples, 28, 28)

    fig, axes = plt.subplots(ncols=10, nrows=8)
    ax = axes.ravel()
    for i in range(n_samples):
        ax[i].imshow(
            np.transpose(samples[i], (0, 1)), cmap="gray", interpolation="none"
        )
        ax[i].axis("off")
        ax[i].set_xticklabels([])
        ax[i].set_yticklabels([])
        ax[i].set_frame_on(False)

    if not os.path.exists("gif_results"):
        os.makedirs("gif_results")
    if test is False:
        save_path = "gif_results/samples_gaussian_" + str(epoch) + ".png"
    else:
        save_path = "figs/samples_gaussian_" + str(epoch) + ".png"

    fig.subplots_adjust(wspace=-0.35, hspace=0.065)
    plt.gca().set_axis_off()
    plt.savefig(
        save_path, dpi=300, bbox_inches="tight", pad_inches=0,
    )
    plt.close()


def plot_losses(epochs, train_losses, val_losses, title=None):
    sns.set(style="white")
    fig, axes = plt.subplots(
        ncols=1, nrows=1, figsize=[10, 5], sharey=True, sharex=True, dpi=400
    )

    train = pd.Series(train_losses).astype(float)
    val = pd.Series(val_losses).astype(float)
    train.index += 1
    val.index += 1

    axes = sns.lineplot(data=train, color="gray", label="Training loss")
    axes = sns.lineplot(data=val, color="orange", label="Validation loss")

    axes.set_ylabel("Negative log-likelihood")
    axes.legend(
        frameon=False,
        prop={"size": 14},
        fancybox=False,
        handletextpad=0.5,
        handlelength=1,
    )
    axes.set_ylim(1250, 1600)
    axes.set_xlim(0, 50)
    axes.set_title(title) if title is not None else axes.set_title(None)
    if not os.path.exists("plots"):
        os.makedirs("plots")
    save_path = "plots/train_plots" + str(epochs[-1]) + ".pdf"
    plt.savefig(
        save_path, dpi=300, bbox_inches="tight", pad_inches=0,
    )
    plt.close()
