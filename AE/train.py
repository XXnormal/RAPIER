from .model import LSTM_AE_GMM
import numpy as np
import torch
import torch.nn as nn
import sys
import os

batch_size = 128
Max_epochs = 1000

def main(data_dir, model_dir, mode, device):

    train_data_w = np.load(os.path.join(data_dir, 'w.npy'))
    train_data_b = np.load(os.path.join(data_dir, 'b.npy'))

    assert(train_data_w.shape[1] == train_data_b.shape[1])

    train_data = np.concatenate([train_data_w, train_data_b], axis=0)
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
        for batch in range(total_size // batch_size + 1):
            if batch * batch_size >= total_size:
                break
            optimizer.zero_grad()
            input = train_data[batch_size * batch : batch_size * (batch + 1)]
            loss = dagmm.loss(torch.Tensor(input).long().cuda())
            loss.backward()
            optimizer.step()
        
    dagmm.to_cpu()
    dagmm = dagmm.cpu()
    torch.save(dagmm, os.path.join(model_dir, 'gru_ae.pkl'))

if __name__ == '__main__':
    
    _, white_type, black_type, data_type, mode, device = sys.argv
    main(white_type, black_type, data_type, mode, device)