import torch
import numpy as np
from .made import MADE
from .datasets.data_loaders import get_data, get_data_loaders
from .utils.train import train_one_epoch_maf, train_one_epoch_made
from .utils.validation import val_maf, val_made
from .utils.test import test_maf, test_made
from .utils.plot import sample_digits_maf, plot_losses
import sys
import os
from .predict_epochs import predict_epochs
import re

def main(white_type, black_type, TRAIN, TEST, DEVICE, MINLOSS):

    print('train_epochs', white_type, black_type, TRAIN, TEST, DEVICE, MINLOSS)

    root_dir = os.path.join('../data/', 'white_' + white_type + '_black_' + black_type, 'source')
    data_dir = os.path.join(root_dir, 'data')
    model_dir = os.path.join(root_dir, 'model')
    feat_dir = os.path.join(root_dir, 'feat')
    made_dir = os.path.join(root_dir, 'made')

    # --------- SET PARAMETERS ----------
    model_name = 'made'  # 'MAF' or 'MADE'
    dataset_name = 'myData'
    train_type = TRAIN
    test_type = TEST
    batch_size = 128
    n_mades = 5
    hidden_dims = [512]
    lr = 1e-4
    random_order = False
    patience = 50  # For early stopping
    min_loss = int(MINLOSS)
    seed = 290713
    cuda_device = int(DEVICE) if DEVICE != 'None' else None
    plot = True
    max_epochs = 2000
    # -----------------------------------

    for filename in os.listdir(made_dir):
        if re.match('w' + str(TRAIN)[1:] + '_in_' + str(TRAIN) + '_\d+', filename):
            os.system('rm ' + os.path.join(made_dir, filename))
        if re.match('b' + str(TRAIN)[1:] + '_in_' + str(TRAIN) + '_\d+', filename):
            os.system('rm ' + os.path.join(made_dir, filename))
            
    # Get dataset.=
    data = get_data(dataset_name, feat_dir, train_type, test_type, True)
    train = torch.from_numpy(data.train.x)
    # Get data loaders.
    train_loader, val_loader, test_loader = get_data_loaders(data, batch_size)
    # Get model.
    n_in = data.n_dims
    model = MADE(n_in, hidden_dims, random_order=random_order, seed=seed, gaussian=True, cuda_device=cuda_device)

    # Get optimiser.
    optimiser = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-6)

    if cuda_device != None:
        torch.cuda.set_device(cuda_device)
        model = model.cuda()

    # Format name of model save file.
    save_name = f"{model_name}_{dataset_name}_{train_type}_{'_'.join(str(d) for d in hidden_dims)}.pt"
    # Initialise list for plotting.
    epochs_list = []
    train_losses = []
    val_losses = []
    # Initialiise early stopping.
    i = 0
    max_loss = np.inf
    # Training loop.
    for epoch in range(1, max_epochs):
        if model_name == "maf":
            train_loss = train_one_epoch_maf(model, epoch, optimiser, train_loader)
            val_loss = val_maf(model, train, val_loader)
        elif model_name == "made":
            train_loss = train_one_epoch_made(model, epoch, optimiser, train_loader, cuda_device)
            val_loss = val_made(model, val_loader, cuda_device)

        epochs_list.append(epoch)
        train_losses.append(train_loss)
        val_losses.append(val_loss)

        if epoch % 10 == 0:
            model = model.cpu()
            torch.save(
                model, os.path.join(model_dir, 'epochs_' + save_name)
            )  # Will print a UserWarning 1st epoch.
            if cuda_device != None:
                model = model.cuda()

            if TRAIN == 'w_all_new':
                predict_epochs(white_type, black_type, TRAIN, 'w%s_new'%(TRAIN[1:]), DEVICE, epoch)
                predict_epochs(white_type, black_type, TRAIN, 'b%s_new'%(TRAIN[1:]), DEVICE, epoch)
            else:
                predict_epochs(white_type, black_type, TRAIN, 'w%s'%(TRAIN[1:]), DEVICE, epoch)
                predict_epochs(white_type, black_type, TRAIN, 'b%s'%(TRAIN[1:]), DEVICE, epoch)

        # Early stopping. Save model on each epoch with improvement.
        if val_loss < max_loss and train_loss > min_loss:
            i = 0
            max_loss = val_loss
            model = model.cpu()
            torch.save(
                model, os.path.join(model_dir, save_name)
            )  # Will print a UserWarning 1st epoch.
            if cuda_device != None:
                model = model.cuda()
            
        else:
            i += 1

        if i < patience:
            print("Patience counter: {}/{}".format(i, patience))
        else:
            print("Patience counter: {}/{}\n Terminate training!".format(i, patience))
            break

if __name__ == '__main__':
    _, white_type, black_type, TRAIN, TEST, DEVICE, MINLOSS = sys.argv
    main(white_type, black_type, TRAIN, TEST, DEVICE, MINLOSS)