import torch
import os
import numpy as np
from .made import MADE
from .datasets.data_loaders import get_data, get_data_loaders
from .utils.test import test_made
import sys
import os

# using MADE calculate each sample's density estimation
def main(feat_dir, model_dir, made_dir, TRAIN, TEST, DEVICE):

    # --------- SET PARAMETERS ----------
    model_name = 'made' # 'MAF' or 'MADE'
    dataset_name = 'myData'
    train_type = TRAIN
    test_type = TEST
    batch_size = 1024
    hidden_dims = [512]
    cuda_device = int(DEVICE) if DEVICE != 'None' else None
    # -----------------------------------

    # Get dataset.=
    data = get_data(dataset_name, feat_dir, train_type, test_type)
    train = torch.from_numpy(data.train.x)
    # Get data loaders.
    train_loader, val_loader, test_loader = get_data_loaders(data, batch_size)
    # Get model.
    save_name = f"{model_name}_{dataset_name}_{train_type}_{'_'.join(str(d) for d in hidden_dims)}.pt"

    model = torch.load(os.path.join(model_dir, save_name))

    if cuda_device != None:
        torch.cuda.set_device(cuda_device)
        model = model.cuda()

    neglogP = test_made(model, test_loader, cuda_device)

    with open(os.path.join(made_dir, '%s_%sMADE'%(test_type, train_type)), 'w') as fp:
        for neglogp in neglogP:
            fp.write(str(float(neglogp)) + '\n')
