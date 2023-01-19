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

def main(white_type, black_type, TRAIN, cuda_device):

    print('train_gen_GAN', white_type, black_type, TRAIN, cuda_device)

    root_dir = os.path.join('../data/', 'white_' + white_type + '_black_' + black_type, 'source')
    model_dir = os.path.join(root_dir, 'model')
    feat_dir = os.path.join(root_dir, 'feat')
    made_dir = os.path.join(root_dir, 'made')

    train_type_w = 'w_' + TRAIN
    train_type_b = 'b_' + TRAIN
    w = np.load(os.path.join(feat_dir, train_type_w + '.npy'))[:, :-1]
    b = np.load(os.path.join(feat_dir, train_type_b + '.npy'))[:, :-1]

    input_size = 2
    output_size = w.shape[1]
    print(output_size)
    hiddens = [8, 16]
    device = int(cuda_device) if cuda_device != 'None' else None
    model_name = 'made' # 'MAF' or 'MADE'
    dataset_name = 'myData'
    batch_size = 500
    hidden_dims = [512]
    epochs = 500 # Modified
    lr = 5e-3
    i = 0
    patience = 30

    load_name_w = f"{model_name}_{dataset_name}_{train_type_w}_{'_'.join(str(d) for d in hidden_dims)}.pt"
    load_name_b = f"{model_name}_{dataset_name}_{train_type_b}_{'_'.join(str(d) for d in hidden_dims)}.pt"
    save_name_w = f"gen_GAN_{train_type_w}_{'_'.join(str(d) for d in hiddens)}.pt"
    save_name_b1 = f"gen1_GAN_{train_type_b}_{'_'.join(str(d) for d in hiddens)}.pt"
    save_name_b2 = f"gen2_GAN_{train_type_b}_{'_'.join(str(d) for d in hiddens)}.pt"

    NLogP_w = []
    NLogP_b = []
    NLogP_w_sort = []
    NLogP_b_sort = []

    with open(os.path.join(made_dir, train_type_w + '_in_' + train_type_w), 'r') as fp:
        for line in fp:
            s = float(line.strip())
            NLogP_w.append(s)
            NLogP_w_sort.append(s)

    with open(os.path.join(made_dir, train_type_b + '_in_' + train_type_b), 'r') as fp:
        for line in fp:
            s = float(line.strip())
            NLogP_b.append(s)
            NLogP_b_sort.append(s)

    NLogP_w_sort.sort()
    NLogP_b_sort.sort()
    NLogP_w = np.array(NLogP_w)
    NLogP_b = np.array(NLogP_b)

    w_MIN_ratio = 0.7
    w_MAX_ratio = 0.8
    w_min_ratio = 0.8
    w_max_ratio = 0.9
    b_max_ratio = 0.95

    w_MIN = NLogP_w_sort[int(w_MIN_ratio * len(NLogP_w))]
    w_MAX = NLogP_w_sort[int(w_MAX_ratio * len(NLogP_w))]
    w_min = NLogP_w_sort[int(w_min_ratio * len(NLogP_w))]
    w_max = NLogP_w_sort[int(w_max_ratio * len(NLogP_w))]
    b_max = NLogP_b_sort[int(b_max_ratio * len(NLogP_b))]

    print(w_MAX, w_MIN)
    print(w_max, w_min)
    print(b_max)

    BGenModel_1 = GEN(input_size, hiddens, output_size, device)
    if device != None:
        torch.cuda.set_device(device)
        BGenModel_1 = BGenModel_1.cuda()

    BGenModel_2 = GEN(input_size, hiddens, output_size, device)
    if device != None:
        torch.cuda.set_device(device)
        BGenModel_2 = BGenModel_2.cuda()

    WGenModel = GEN(input_size, hiddens, output_size, device)
    if device != None:
        torch.cuda.set_device(device)
        WGenModel = WGenModel.cuda()

    WMADE = torch.load(os.path.join(model_dir, load_name_w))
    if device != None:
        torch.cuda.set_device(device)
        WMADE = WMADE.cuda()
        
    BMADE = torch.load(os.path.join(model_dir, load_name_b))
    if device != None:
        torch.cuda.set_device(device)
        BMADE = BMADE.cuda()

    optimizer_w = torch.optim.Adam(WGenModel.parameters(), lr=lr, weight_decay=1e-6)
    optimizer_b1 = torch.optim.Adam(BGenModel_1.parameters(), lr=lr, weight_decay=1e-6)
    optimizer_b2 = torch.optim.Adam(BGenModel_2.parameters(), lr=lr, weight_decay=1e-6)

    D = MLP(input_size=output_size, hiddens=[16, 8], output_size=2, device=device)
    if device != None:
        D.to_cuda(device)
        D = D.cuda()
    optimizer_D = torch.optim.Adam(D.parameters(), lr=lr)
    save_name_D = f"dis_{'_'.join(str(d) for d in hiddens)}.pt"

    print(w.shape, b.shape)

    w_mean = torch.Tensor(np.mean(w, axis=0))
    w_std = torch.Tensor(np.std(w, axis=0))
    b_mean = torch.Tensor(np.mean(b, axis=0))
    b_std = torch.Tensor(np.std(b, axis=0))

    w = torch.Tensor(w)
    b = torch.Tensor(b)

    w_in_MINMAX = w[(NLogP_w - w_MAX) * (NLogP_w - w_MIN) < 0]
    w_in_minmax = w[(NLogP_w - w_max) * (NLogP_w - w_min) < 0]
    b_in_minmax = b[NLogP_b > b_max]

    def Entropy(GenModel, batch_size, seed):
        X, _ = make_blobs(n_samples=batch_size, centers=[[0, 0]], n_features=2, random_state=seed)
        X = torch.Tensor(X)
        batch = GenModel.forward(X)
        N = batch_size
        L = torch.linalg.norm(batch, dim=1)
        S = batch / L.view(-1, 1)
        H = (torch.sum(torch.matmul(S, S.t()), dim=(1, 0)) - N) / (N * (N - 1))
        return batch, H

    def get_NLogP(batch, MADE):
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

    #  二分类mlp + 概率校正 构建 pb 函数
    def accuracy(logit, target):
        """Computes the precision@k for the specified values of k"""
        output = F.softmax(logit, dim=1)
        batch_size = target.size(0)

        _, pred = output.topk(1, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        correct_k = correct[:1].view(-1).float().sum(0, keepdim=True)
        return correct_k.mul_(100.0 / batch_size)

    def train_mlp(train_loader, epoch, model, optimizer):
        train_total = 0
        train_correct = 0

        for i, data_labels in enumerate(train_loader):

            # Forward + Backward + Optimize
            feats = data_labels[:, :-1].to(dtype=torch.float32)
            labels = data_labels[:, -1].to(dtype=int)

            if device != None:
                torch.cuda.set_device(device)
                feats = feats.cuda()
                labels = labels.cuda()

            logits = model(feats)
            prec = accuracy(logits, labels)
            train_total += 1
            train_correct += prec

            loss = F.cross_entropy(logits, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        train_acc = float(train_correct) / float(train_total)
        return train_acc

    for epoch in range(epochs):

        batch_w, H_w = Entropy(WGenModel, batch_size, epoch * 378 + 1782)
        batch_b1, H_b1 = Entropy(BGenModel_1, batch_size, epoch * 263 + 3467)
        batch_b2, H_b2 = Entropy(BGenModel_2, batch_size, epoch * 255 + 3353)
        
        NLogP_ww = get_NLogP(batch_w, WMADE)
        NLogP_wb = get_NLogP((batch_w * w_std.cuda() + w_mean.cuda() - b_mean.cuda()) / b_std.cuda(), BMADE)
        NLogP_b1w = get_NLogP(batch_b1, WMADE)
        NLogP_b1b = get_NLogP((batch_b1 * w_std.cuda() + w_mean.cuda() - b_mean.cuda()) / b_std.cuda(), BMADE)
        NLogP_b2w = get_NLogP((batch_b2 * b_std.cuda() + b_mean.cuda() - w_mean.cuda()) / w_std.cuda(), WMADE)
        NLogP_b2b = get_NLogP(batch_b2, BMADE)
        
        E1_b1 = -torch.mean(NLogP_b1w * NLogP_b1b.ge(b_max) * NLogP_b1w.lt(w_min))
        E2_b1 =  torch.mean(NLogP_b1w * NLogP_b1b.ge(b_max) * NLogP_b1w.gt(w_max))
        E3_b1 = -torch.mean(NLogP_b1b * NLogP_b1b.lt(b_max))
        fm_b1 = torch.linalg.norm(
            torch.mean(D.f(batch_b1 * w_std.cuda() + w_mean.cuda())) - 
            torch.mean(D.f(w_in_minmax))
        )
        loss_b1 = H_b1 + E1_b1 + E2_b1 + E3_b1 + fm_b1

        E1_b2 = -torch.mean(NLogP_b2b * NLogP_b2w.ge(w_max) * NLogP_b2b.lt(b_max))
        E2_b2 = -torch.mean(NLogP_b2w * NLogP_b2w.lt(w_max))
        fm_b2 = torch.linalg.norm(
            torch.mean(D.f(batch_b2 * b_std.cuda() + b_mean.cuda())) - 
            torch.mean(D.f(b_in_minmax))
        )
        loss_b2 = H_b2 + E1_b2 + E2_b2 + fm_b2

        E1_w = -torch.mean(NLogP_ww * NLogP_wb.ge(b_max) * NLogP_ww.lt(w_MIN))
        E2_w =  torch.mean(NLogP_ww * NLogP_wb.ge(b_max) * NLogP_ww.gt(w_MAX))
        E3_w = -torch.mean(NLogP_wb * NLogP_wb.lt(b_max))
        fm_w = torch.linalg.norm(
            torch.mean(D.f(batch_w * w_std.cuda() + w_mean.cuda())) - 
            torch.mean(D.f(w_in_MINMAX))
        )
        loss_w = H_w + E1_w + E2_w + E3_w + fm_w

        print('epoch: %d, loss_w: %5f, loss_b1: %5f, loss_b2: %5f' % (epoch, loss_w, loss_b1, loss_b2))

        optimizer_w.zero_grad()
        loss_w.backward()
        optimizer_w.step()

        optimizer_b1.zero_grad()
        loss_b1.backward()
        optimizer_b1.step()

        optimizer_b2.zero_grad()
        loss_b2.backward()
        optimizer_b2.step()

        if epoch % 10 == 9:
            save_model(WGenModel, save_name_w)
            save_model(BGenModel_1, save_name_b1)
            save_model(BGenModel_2, save_name_b2)
            
            accs_ori = []
            accs_gen = []
            reminder = 1e-3
            for epoch_D in range(10):
                Gw, Hw = Entropy(WGenModel, batch_size, epoch * 356 + 32342)
                Gb1, Hb1 = Entropy(BGenModel_1, batch_size, epoch * 356 + 32142)
                Gb2, Hb2 = Entropy(BGenModel_2, batch_size, epoch * 242 + 24279)
                
                D_w = F.softmax(D(w), dim=1)[:, 0]
                E1_D = torch.mean(torch.log(D_w + reminder * D_w.lt(reminder)))
                
                D_Gb1 = F.softmax(D(Gb1), dim=1)[:, 0]
                E2_D = torch.mean(torch.log(1 - D_Gb1 + reminder * (1 - D_Gb1).lt(reminder)))

                D_Gb2 = F.softmax(D(Gb2), dim=1)[:, 0]
                E3_D = torch.mean(torch.log(1 - D_Gb2 + reminder * (1 - D_Gb2).lt(reminder)))

                E4_D = torch.mean(D_w * torch.log(D_w + reminder * D_w.lt(reminder)))

                D_b = F.softmax(D(b), dim=1)[:, 0]
                E5_D = torch.mean(torch.log(1 - D_b + reminder * (1 - D_b).lt(reminder)))

                D_Gw = F.softmax(D(Gw), dim=1)[:, 0]
                E6_D = torch.mean(torch.log(D_Gw + reminder * D_Gw.lt(reminder)))
                
                loss_D = E1_D + E2_D + E3_D + E4_D + E5_D + E6_D

                optimizer_D.zero_grad()
                loss_D.backward()
                optimizer_D.step()

                correct_num = D_w.lt(0.5).sum() + D_b.gt(0.5).sum()
                total_num = len(D_w) + len(D_b)
                accs_ori.append((correct_num / total_num).cpu().numpy())
                
                correct_num = correct_num + D_Gb1.gt(0.5).sum() + D_Gb2.gt(0.5).sum() + D_Gw.lt(0.5).sum()
                total_num = total_num + len(D_Gb1) + len(D_Gb2) + len(Gw)
                accs_gen.append((correct_num / total_num).cpu().numpy())

            print('Acc_ori of Discriminator: %.5f.' % (np.mean(accs_ori)))
            print('Acc_gen of Discriminator: %.5f.' % (np.mean(accs_gen)))

            save_model(D, save_name_D)
            
if __name__ == '__main__':
    _, white_type, black_type, TRAIN, cuda_device = sys.argv
    main(white_type, black_type, TRAIN, cuda_device)
