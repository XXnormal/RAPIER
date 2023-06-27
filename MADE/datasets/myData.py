import numpy as np
import os

class MyDataset:
    class Data:
        def __init__(self, data):
            self.x = data.astype(np.float32)
            self.N = self.x.shape[0]
    
    def __init__(self, feat_dir, train_type, test_type):
        
        train_file = os.path.join(feat_dir, train_type + '.npy')
        test_file = os.path.join(feat_dir, test_type + '.npy')

        train, valid, test = load_data_normalized(train_file, test_file)

        self.train = self.Data(train)
        self.val = self.Data(valid)
        self.test = self.Data(test)

        self.n_dims = self.train.x.shape[1]

def load_data(root_path, is_train):

    data = np.load(root_path)[:, :32]
    
    if is_train is True:
        N_validate = int(0.1 * data.shape[0])
        data_validate = data[-N_validate:]
        data_train = data
        return data_train, data_validate
    else:
        return data

def load_data_normalized(train_path, test_path):
    
    data_train, data_validate = load_data(train_path, True)
    data_test = load_data(test_path, False)
    data = data_train

    mu = data.mean(axis=0)
    s = data.std(axis=0)

    for i, score in enumerate(s):
        if score == 0:
            s[i] = 1

    data_train = (data_train - mu) / s
    data_validate = (data_validate - mu) / s
    data_test = (data_test - mu) / s

    return data_train, data_validate, data_test
