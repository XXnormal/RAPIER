import sys
from .model import LSTM_AE_GMM
import numpy as np
import torch
import torch.nn as nn
import sys
import os

batch_size = 128

def main(data_dir, model_dir, feat_dir, data_type, device):

    device = int(device) if device != 'None' else None
    # get raw testing traffic time-series data
    test_data_label = np.load(os.path.join(data_dir, data_type + '.npy'))
    
    test_data = test_data_label[:, :50]
    test_label = test_data_label[:, -1]
    
    total_size, _ = test_data.shape

    device_id = int(device)
    torch.cuda.set_device(device_id)
    dagmm = torch.load(os.path.join(model_dir, 'gru_ae.pkl'))
    dagmm.to_cuda(device_id)
    dagmm = dagmm.cuda()
    dagmm.test_mode()
    
    feature = []
    for batch in range(total_size // batch_size + 1):
        if batch * batch_size >= total_size:
            break
        input = test_data[batch_size * batch : batch_size * (batch + 1)]
        output = dagmm.feature(torch.Tensor(input).long().cuda())
        feature.append(output.detach().cpu())

    feature = torch.cat(feature, dim=0).numpy()
    feature = np.concatenate([feature, test_label[:, None]], axis=1)
    np.save(os.path.join(feat_dir, data_type + '.npy'), feature)