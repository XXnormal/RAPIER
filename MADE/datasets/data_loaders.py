import torch
from .myData import MyDataset

def get_data(dataset: str, feat_dir=None, train_type=None, test_type=None):
    if dataset == 'myData':
        return MyDataset(feat_dir, train_type, test_type)

    raise ValueError(
        f"Unknown dataset '{dataset}'. Please choose either 'mnist', 'power', or 'hepmass'."
    )

def get_data_loaders(data, batch_size: int = 1024):
    train = torch.from_numpy(data.train.x)
    val = torch.from_numpy(data.val.x)
    test = torch.from_numpy(data.test.x)

    train_loader = torch.utils.data.DataLoader(train, batch_size=batch_size,)
    val_loader = torch.utils.data.DataLoader(val, batch_size=batch_size,)
    test_loader = torch.utils.data.DataLoader(test, batch_size=batch_size,)

    return train_loader, val_loader, test_loader
