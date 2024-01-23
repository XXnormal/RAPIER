from .gen_model import GEN, MLP
from .made import MADE
import torch
import numpy as np 
import torch.nn as nn 
import torch.nn.functional as F
import sys
import os 
from sklearn.datasets import make_blobs
import math

# train 3 GANs for data augmentation 
def main(feat_dir, model_dir, made_dir, TRAIN, cuda_device):

    train_type_be = 'be_' + TRAIN
    train_type_ma = 'ma_' + TRAIN
    be = np.load(os.path.join(feat_dir, train_type_be + '.npy'))[:, :32]
    ma = np.load(os.path.join(feat_dir, train_type_ma + '.npy'))[:, :32]

    input_size = 2
    output_size = be.shape[1]
    hiddens = [8, 16]
    device = int(cuda_device) if cuda_device != 'None' else None
    model_name = 'made' # 'MAF' or 'MADE'
    dataset_name = 'myData'
    batch_size = 500
    hidden_dims = [512]
    epochs = 500 # Modified
    lr = 5e-3

    load_name_be = f"{model_name}_{dataset_name}_{train_type_be}_{'_'.join(str(d) for d in hidden_dims)}.pt"
    load_name_ma = f"{model_name}_{dataset_name}_{train_type_ma}_{'_'.join(str(d) for d in hidden_dims)}.pt"
    save_name_be = f"gen_GAN_{train_type_be}_{'_'.join(str(d) for d in hiddens)}.pt"
    save_name_ma1 = f"gen1_GAN_{train_type_ma}_{'_'.join(str(d) for d in hiddens)}.pt"
    save_name_ma2 = f"gen2_GAN_{train_type_ma}_{'_'.join(str(d) for d in hiddens)}.pt"

    NLogP_be = []
    NLogP_ma = []
    NLogP_be_sort = []
    NLogP_ma_sort = []

    with open(os.path.join(made_dir, '%s_%sMADE'%(train_type_be, train_type_be)), 'r') as fp:
        for line in fp:
            s = float(line.strip())
            NLogP_be.append(s)
            NLogP_be_sort.append(s)

    with open(os.path.join(made_dir, '%s_%sMADE'%(train_type_ma, train_type_ma)), 'r') as fp:
        for line in fp:
            s = float(line.strip())
            NLogP_ma.append(s)
            NLogP_ma_sort.append(s)

    NLogP_be_sort.sort()
    NLogP_ma_sort.sort()
    NLogP_be = np.array(NLogP_be)
    NLogP_ma = np.array(NLogP_ma)

    be_MIN_ratio = 0.7
    be_MAX_ratio = 0.8
    be_min_ratio = 0.8
    be_max_ratio = 0.9
    ma_max_ratio = 0.95

    be_MIN = NLogP_be_sort[int(be_MIN_ratio * len(NLogP_be))]
    be_MAX = NLogP_be_sort[int(be_MAX_ratio * len(NLogP_be))]
    be_min = NLogP_be_sort[int(be_min_ratio * len(NLogP_be))]
    be_max = NLogP_be_sort[int(be_max_ratio * len(NLogP_be))]
    ma_max = NLogP_ma_sort[int(ma_max_ratio * len(NLogP_ma))]

    MaGenModel_1 = GEN(input_size, hiddens, output_size, device)
    if device != None:
        torch.cuda.set_device(device)
        MaGenModel_1 = MaGenModel_1.cuda()

    MaGenModel_2 = GEN(input_size, hiddens, output_size, device)
    if device != None:
        torch.cuda.set_device(device)
        MaGenModel_2 = MaGenModel_2.cuda()

    BeGenModel = GEN(input_size, hiddens, output_size, device)
    if device != None:
        torch.cuda.set_device(device)
        BeGenModel = BeGenModel.cuda()

    BeMADE = torch.load(os.path.join(model_dir, load_name_be))
    if device != None:
        torch.cuda.set_device(device)
        BeMADE = BeMADE.cuda()
        
    MaMADE = torch.load(os.path.join(model_dir, load_name_ma))
    if device != None:
        torch.cuda.set_device(device)
        MaMADE = MaMADE.cuda()

    optimizer_be = torch.optim.Adam(BeGenModel.parameters(), lr=lr, weight_decay=1e-6)
    optimizer_ma1 = torch.optim.Adam(MaGenModel_1.parameters(), lr=lr, weight_decay=1e-6)
    optimizer_ma2 = torch.optim.Adam(MaGenModel_2.parameters(), lr=lr, weight_decay=1e-6)

    D = MLP(input_size=output_size, hiddens=[16, 8], output_size=2, device=device)
    if device != None:
        D.to_cuda(device)
        D = D.cuda()
    optimizer_D = torch.optim.Adam(D.parameters(), lr=lr)
    save_name_D = f"dis_{'_'.join(str(d) for d in hiddens)}.pt"

    be_mean = torch.Tensor(np.mean(be, axis=0))
    be_std = torch.Tensor(np.std(be, axis=0))
    ma_mean = torch.Tensor(np.mean(ma, axis=0))
    ma_std = torch.Tensor(np.std(ma, axis=0))

    be = torch.Tensor(be)
    ma = torch.Tensor(ma)

    print(be.shape, NLogP_be.shape)
    be_in_MINMAX = be[(NLogP_be - be_MAX) * (NLogP_be - be_MIN) < 0]
    be_in_minmax = be[(NLogP_be - be_max) * (NLogP_be - be_min) < 0]
    ma_in_minmax = ma[NLogP_ma > ma_max]

    def Entropy(GenModel, batch_size, seed):
        X, _ = make_blobs(n_samples=batch_size, centers=[[0, 0]], n_features=2, random_state=seed)
        X = torch.Tensor(X)
        batch = GenModel.forward(X)
        N = batch_size
        L = torch.linalg.norm(batch, dim=1)
        S = batch / L.view(-1, 1)
        H = (torch.sum(torch.matmul(S, S.t()), dim=(1, 0)) - N) / (N * (N - 1))
        return batch, H

    def get_NLogP(batch, MADE, Print=False):
        input = batch.float().cuda()
        out = MADE.forward(input)
        mu, logp = torch.chunk(out, 2, dim=1)
        logp = 20 - F.relu(20 - logp)
        u = (input - mu) * torch.exp(0.5 * logp).cuda()
        negloglik_loss = 0.5 * (u ** 2).sum(dim=1)
        negloglik_loss += 0.5 * input.shape[1] * np.log(2 * math.pi)
        negloglik_loss -= 0.5 * torch.sum(logp, dim=1)
        return negloglik_loss

    def save_model(GenModel, save_name):
        GenModel = GenModel.cpu()
        GenModel.to_cpu()
        torch.save(GenModel, os.path.join(model_dir, save_name))
        GenModel.to_cuda(device)
        GenModel = GenModel.cuda()

    for epoch in range(epochs):
        # training Generator
        
        # random generate samples
        batch_be, H_be = Entropy(BeGenModel, batch_size, epoch * 378 + 1782)
        batch_ma1, H_ma1 = Entropy(MaGenModel_1, batch_size, epoch * 263 + 3467)
        batch_ma2, H_ma2 = Entropy(MaGenModel_2, batch_size, epoch * 255 + 3353)
        
        # calculate samples' density
        NLogP_be_beMADE = get_NLogP(batch_be, BeMADE)
        NLogP_be_maMADE = get_NLogP((batch_be * be_std.cuda() + be_mean.cuda() - ma_mean.cuda()) / ma_std.cuda(), MaMADE)
        NLogP_ma1_beMADE = get_NLogP(batch_ma1, BeMADE)
        NLogP_ma1_maMADE = get_NLogP((batch_ma1 * be_std.cuda() + be_mean.cuda() - ma_mean.cuda()) / ma_std.cuda(), MaMADE, Print=True)
        NLogP_ma2_beMADE = get_NLogP((batch_ma2 * ma_std.cuda() + ma_mean.cuda() - be_mean.cuda()) / be_std.cuda(), BeMADE)
        NLogP_ma2_maMADE = get_NLogP(batch_ma2, MaMADE)
        
        # loss function for MB
        E1_ma1 = -torch.mean(NLogP_ma1_beMADE * NLogP_ma1_maMADE.ge(ma_max) * NLogP_ma1_beMADE.lt(be_min))
        E2_ma1 =  torch.mean(NLogP_ma1_beMADE * NLogP_ma1_maMADE.ge(ma_max) * NLogP_ma1_beMADE.gt(be_max))
        E3_ma1 = -torch.mean(NLogP_ma1_maMADE * NLogP_ma1_maMADE.lt(ma_max))
        fm_ma1 = torch.linalg.norm(
            torch.mean(D.f(batch_ma1 * be_std.cuda() + be_mean.cuda())) - 
            torch.mean(D.f(be_in_minmax))
        )
        loss_ma1 = H_ma1 + E1_ma1 + E2_ma1 + E3_ma1 + fm_ma1

        # loss function for MO
        E1_ma2 = -torch.mean(NLogP_ma2_maMADE * NLogP_ma2_beMADE.ge(be_max) * NLogP_ma2_maMADE.lt(ma_max))
        E2_ma2 = -torch.mean(NLogP_ma2_beMADE * NLogP_ma2_beMADE.lt(be_max))
        fm_ma2 = torch.linalg.norm(
            torch.mean(D.f(batch_ma2 * ma_std.cuda() + ma_mean.cuda())) - 
            torch.mean(D.f(ma_in_minmax))
        )
        loss_ma2 = H_ma2 + E1_ma2 + E2_ma2 + fm_ma2

        # loss function for NB
        E1_be = -torch.mean(NLogP_be_beMADE * NLogP_be_maMADE.ge(ma_max) * NLogP_be_beMADE.lt(be_MIN))
        E2_be =  torch.mean(NLogP_be_beMADE * NLogP_be_maMADE.ge(ma_max) * NLogP_be_beMADE.gt(be_MAX))
        E3_be = -torch.mean(NLogP_be_maMADE * NLogP_be_maMADE.lt(ma_max))
        fm_be = torch.linalg.norm(
            torch.mean(D.f(batch_be * be_std.cuda() + be_mean.cuda())) - 
            torch.mean(D.f(be_in_MINMAX))
        )
        loss_be = H_be + E1_be + E2_be + E3_be + fm_be
        print('epoch: %d, loss_be: %5f, loss_ma1: %5f, loss_ma2: %5f' % (epoch, loss_be, loss_ma1, loss_ma2))

        optimizer_be.zero_grad()
        loss_be.backward()
        optimizer_be.step()

        optimizer_ma1.zero_grad()
        loss_ma1.backward()
        optimizer_ma1.step()

        optimizer_ma2.zero_grad()
        loss_ma2.backward()
        optimizer_ma2.step()

        if epoch % 10 == 9:
            # training discriminator
            
            save_model(BeGenModel, save_name_be)
            save_model(MaGenModel_1, save_name_ma1)
            save_model(MaGenModel_2, save_name_ma2)
            
            reminder = 1e-3
            for epoch_D in range(10):
                Gbe, Hbe = Entropy(BeGenModel, batch_size, epoch * 356 + 32342)
                Gma1, Hma1 = Entropy(MaGenModel_1, batch_size, epoch * 356 + 32142)
                Gma2, Hma2 = Entropy(MaGenModel_2, batch_size, epoch * 242 + 24279)
                
                D_be = F.softmax(D(be), dim=1)[:, 0]
                E1_D = torch.mean(torch.log(D_be + reminder * D_be.lt(reminder)))
                
                D_Gma1 = F.softmax(D(Gma1), dim=1)[:, 0]
                E2_D = torch.mean(torch.log(1 - D_Gma1 + reminder * (1 - D_Gma1).lt(reminder)))

                D_Gma2 = F.softmax(D(Gma2), dim=1)[:, 0]
                E3_D = torch.mean(torch.log(1 - D_Gma2 + reminder * (1 - D_Gma2).lt(reminder)))

                E4_D = torch.mean(D_be * torch.log(D_be + reminder * D_be.lt(reminder)))

                D_ma = F.softmax(D(ma), dim=1)[:, 0]
                E5_D = torch.mean(torch.log(1 - D_ma + reminder * (1 - D_ma).lt(reminder)))

                D_Gbe = F.softmax(D(Gbe), dim=1)[:, 0]
                E6_D = torch.mean(torch.log(D_Gbe + reminder * D_Gbe.lt(reminder)))
                
                # loss for D
                loss_D = E1_D + E2_D + E3_D + E4_D + E5_D + E6_D

                optimizer_D.zero_grad()
                loss_D.backward()
                optimizer_D.step()

            save_model(D, save_name_D)
