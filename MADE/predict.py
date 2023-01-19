import torch
import os
import numpy as np
from .made import MADE
from .datasets.data_loaders import get_data, get_data_loaders
from .utils.test import test_made
import sys
import os

def main(white_type, black_type, TRAIN, TEST, DEVICE, train_with_label=True, test_with_label=True):
    print('predict', white_type, black_type, TRAIN, TEST, DEVICE)
    root_dir = os.path.join('../data/', 'white_' + white_type + '_black_' + black_type, 'source')
    data_dir = os.path.join(root_dir, 'data')
    model_dir = os.path.join(root_dir, 'model')
    feat_dir = os.path.join(root_dir, 'feat')
    made_dir = os.path.join(root_dir, 'made')

    # --------- SET PARAMETERS ----------
    model_name = 'made' # 'MAF' or 'MADE'
    dataset_name = 'myData'
    train_type = TRAIN
    test_type = TEST
    batch_size = 1024
    n_mades = 5
    hidden_dims = [512]
    lr = 1e-4
    random_order = False
    patience = 30  # For early stopping
    seed = 290713
    cuda_device = int(DEVICE) if DEVICE != 'None' else None
    # -----------------------------------

    # Get dataset.=
    data = get_data(dataset_name, feat_dir, train_type, test_type, train_with_label=train_with_label, test_with_label=test_with_label)
    train = torch.from_numpy(data.train.x)
    # Get data loaders.
    train_loader, val_loader, test_loader = get_data_loaders(data, batch_size)
    # Get model.
    n_in = data.n_dims
    # Format name of model save file.
    save_name = f"{model_name}_{dataset_name}_{train_type}_{'_'.join(str(d) for d in hidden_dims)}.pt"

    model = torch.load(os.path.join(model_dir, save_name))

    if cuda_device != None:
        torch.cuda.set_device(cuda_device)
        model = model.cuda()

    neglogP = test_made(model, test_loader, cuda_device)

    with open(os.path.join(made_dir, test_type + '_in_' + train_type), 'w') as fp:
        for neglogp in neglogP:
            fp.write(str(float(neglogp)) + '\n')

if __name__ == '__main__':
    _, white_type, black_type, TRAIN, TEST, DEVICE = sys.argv
    main(white_type, black_type, TRAIN, TEST, DEVICE)