from .model import LSTM_AE_GMM
import numpy as np
import torch
import torch.nn as nn
import sys
import os

batch_size = 128
Max_epochs = 500

def main(data_dir, model_dir, mode, device):

    # get raw time-series data of training traffic data
    train_data_be = np.load(os.path.join(data_dir, 'be.npy'))
    train_data_ma = np.load(os.path.join(data_dir, 'ma.npy'))
    
    assert(train_data_be.shape[1] == 51 and train_data_ma.shape[1] == 51)

    train_data = np.concatenate([train_data_be[:, :50], train_data_ma[:, :50]], axis=0)
    np.random.shuffle(train_data)
    
    total_size, input_size = train_data.shape
    max_epochs = Max_epochs

    device_id = int(device)
    torch.cuda.set_device(device_id)
    if mode == 'continue':
        dagmm = torch.load(os.path.join(model_dir, 'gru_ae.pkl'))
        dagmm.to_cuda(device_id)
        dagmm = dagmm.cuda()

    else:
        dagmm = LSTM_AE_GMM(
            input_size=input_size,
            max_len=2000,
            emb_dim=32,
            hidden_size=8,
            dropout=0.2,
            est_hidden_size=64,
            est_output_size=8,
            device=device_id,
        ).cuda()

    dagmm.train_mode()
    optimizer = torch.optim.Adam(dagmm.parameters(), lr=1e-2)
    for epoch in range(max_epochs):
        sum_loss = 0
        for batch in range(total_size // batch_size + 1):
            if batch * batch_size >= total_size:
                break
            optimizer.zero_grad()
            input = train_data[batch_size * batch : batch_size * (batch + 1)]
            loss = dagmm.loss(torch.Tensor(input).long().cuda())
            loss.backward()
            optimizer.step()
            sum_loss += loss.detach().cpu().numpy()
        print('epoch:', epoch, 'loss:', sum_loss)
        
    dagmm.to_cpu()
    dagmm = dagmm.cpu()
    torch.save(dagmm, os.path.join(model_dir, 'gru_ae.pkl'))
