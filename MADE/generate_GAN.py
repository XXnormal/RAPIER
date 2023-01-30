from .gen_model import GEN
from .made import MADE
import torch
import numpy as np 
import torch.nn as nn 
import torch.nn.functional as F
import sys
import os 
from sklearn.datasets import make_blobs
import math

def main(feat_dir, model_dir, TRAIN, index, cuda_device):

    w = np.load(os.path.join(feat_dir, 'w_%s.npy'%(TRAIN)))
    b = np.load(os.path.join(feat_dir, 'b_%s.npy'%(TRAIN)))

    output_size = w.shape[1]
    hiddens = [8, 16]
    device = int(cuda_device) if cuda_device != 'None' else None
    train_type_w = 'w_' + TRAIN
    train_type_b = 'b_' + TRAIN

    load_name_w = f"gen_GAN_{train_type_w}_{'_'.join(str(d) for d in hiddens)}.pt"
    load_name_b1 = f"gen1_GAN_{train_type_b}_{'_'.join(str(d) for d in hiddens)}.pt"
    load_name_b2 = f"gen2_GAN_{train_type_b}_{'_'.join(str(d) for d in hiddens)}.pt"
    WGenModel = torch.load(os.path.join(model_dir, load_name_w))
    BGenModel_1 = torch.load(os.path.join(model_dir, load_name_b1))
    BGenModel_2 = torch.load(os.path.join(model_dir, load_name_b2))

    if device != None:
        torch.cuda.set_device(device)
        WGenModel.to_cuda(device)
        WGenModel = WGenModel.cuda()
        BGenModel_1.to_cuda(device)
        BGenModel_1 = BGenModel_1.cuda()
        BGenModel_2.to_cuda(device)
        BGenModel_2 = BGenModel_2.cuda()

    def generate(train_type, GenModel, total_size, seed):

        data_train = np.load(os.path.join(feat_dir, train_type + '.npy'))[:, :output_size]
        mu_train = torch.Tensor(data_train.mean(axis=0))
        s_train = torch.Tensor(data_train.std(axis=0))

        if device != None:
            mu_train = mu_train.cuda()
            s_train = s_train.cuda()

        X, _ = make_blobs(n_samples=total_size, centers=[[0, 0]], n_features=2, random_state=seed)
        X = torch.Tensor(X)
        batch = GenModel.forward(X)
        batch1 = batch * s_train + mu_train
        gen_data = batch1.detach().cpu()
        
        return np.array(gen_data)

    gen_data_w = generate(train_type_w, WGenModel, int(w.shape[0]), np.random.randint(1000))
    gen_data_b1 = generate(train_type_w, BGenModel_1, int(b.shape[0]), np.random.randint(1000))
    gen_data_b2 = generate(train_type_b, BGenModel_2, int(b.shape[0]), np.random.randint(1000))

    np.save(os.path.join(feat_dir, 'w_%s_generated_GAN_%s.npy'%(TRAIN, index)), gen_data_w)
    np.save(os.path.join(feat_dir, 'b_%s_generated_GAN_1_%s.npy'%(TRAIN, index)), gen_data_b1)
    np.save(os.path.join(feat_dir, 'b_%s_generated_GAN_2_%s.npy'%(TRAIN, index)), gen_data_b2)
