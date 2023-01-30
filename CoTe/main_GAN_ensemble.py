# -*- coding:utf-8 -*-
import torch 
import torch.nn as nn
import torch.nn.functional as F

from .model import MLP
import sys, os
import numpy as np
from tqdm import tqdm

from .loss import loss_coteaching

# Hyper Parameters
batch_size = 128
learning_rate = 1e-3
epochs = 500
num_gradual = 10
forget_rate = 0.1
exponent = 1
rate_schedule = np.ones(epochs) * forget_rate
rate_schedule[:num_gradual] = np.linspace(0, forget_rate ** exponent, num_gradual)

def accuracy(logit, target):
    """Computes the precision@k for the specified values of k"""
    output = F.softmax(logit, dim=1)
    batch_size = target.size(0)

    _, pred = output.topk(1, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    correct_k = correct[:1].view(-1).float().sum(0, keepdim=True)
    return correct_k.mul_(100.0 / batch_size)

# Train the Model
def train(train_loader, epoch, model1, optimizer1, model2, optimizer2, device):
    # print('Training...')
    
    train_total1=0
    train_correct1=0 
    train_total2=0
    train_correct2=0 

    for i, data_labels in enumerate(train_loader):
        
        # Forward + Backward + Optimize
        feats = data_labels[:, :-1].to(dtype=torch.float32)
        labels = data_labels[:, -1].to(dtype=int)
        if device != None:
            torch.cuda.set_device(device)
            feats = feats.cuda()
            labels = labels.cuda()
    
        logits1 = model1(feats)
        prec1 = accuracy(logits1, labels)
        train_total1 += 1
        train_correct1 += prec1

        logits2 = model2(feats)
        prec2 = accuracy(logits2, labels)
        train_total2 += 1
        train_correct2 += prec2
        loss_1, loss_2 = loss_coteaching(logits1, logits2, labels, rate_schedule[epoch])

        optimizer1.zero_grad()
        loss_1.backward()
        optimizer1.step()
        optimizer2.zero_grad()
        loss_2.backward()
        optimizer2.step()

    train_acc1=float(train_correct1)/float(train_total1)
    train_acc2=float(train_correct2)/float(train_total2)
    return train_acc1, train_acc2

# Evaluate the Model
# Test the Model
def test(test_loader, model, device, alpha=0.5):

    preds = []
    for i, data in enumerate(test_loader):
        
        # Forward + Backward + Optimize
        feats = data.to(dtype=torch.float32)
        
        if device != None:
            torch.cuda.set_device(device)
            feats = feats.cuda()
            labels = labels.cuda()
        
        logits = model(feats)
        outputs = F.softmax(logits, dim=1)
        preds.append((outputs[:, 1] > alpha).detach().cpu().numpy())

    #return test_acc, np.concatenate(right, axis=0), np.concatenate(wrong, axis=0)
    return np.concatenate(preds, axis=0)

def main(feat_dir, model_dir, result_dir, TRAIN, cuda_device, parallel=5):
    
    cuda_device = int(cuda_device)

    w = np.load(os.path.join(feat_dir, 'w.npy'))
    b = np.load(os.path.join(feat_dir, 'b.npy'))
    wshape = w.shape[0]
    bshape = b.shape[0]

    for index in range(parallel):

        w_gen = np.load(os.path.join(feat_dir, 'w_%s_generated_GAN_%d.npy' % (TRAIN, index)))
        b_gen1 = np.load(os.path.join(feat_dir, 'b_%s_generated_GAN_1_%d.npy' % (TRAIN, index)))
        b_gen2 = np.load(os.path.join(feat_dir, 'b_%s_generated_GAN_2_%d.npy' % (TRAIN, index)))
        np.random.shuffle(w_gen)
        np.random.shuffle(b_gen1)
        np.random.shuffle(b_gen2)
        w = np.concatenate([
            w, 
            w_gen[:wshape // (parallel)], 
        ], axis=0)
        
        b = np.concatenate([
            b,
            b_gen1[:bshape // (parallel)],
            b_gen2[:bshape // (parallel)],
        ], axis=0)

    train_data = np.concatenate([w, b], axis=0)
    train_label = np.concatenate([np.zeros(w.shape[0]), np.ones(b.shape[0])], axis=0)
    train_dataset = np.concatenate((train_data, train_label[:, None]), axis=1)

    test_data = np.load(os.path.join(feat_dir, 'test.npy'))

    device = int(cuda_device) if cuda_device != 'None' else None
    # define drop rate schedule

    if device != None:
        torch.cuda.set_device(device)
    # Data Loader (Input Pipeline)
    print('loading dataset...')
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    # Define models
    print('building model...')
    mlp1 = MLP(input_size=32, hiddens=[16, 8], output_size=2, device=device)
    if device != None:
        mlp1.to_cuda(device)
        mlp1 = mlp1.cuda()
    optimizer1 = torch.optim.Adam(mlp1.parameters(), lr=learning_rate)
    
    mlp2 = MLP(input_size=32, hiddens=[16, 8], output_size=2, device=device)
    if device != None:
        mlp2.to_cuda(device)
        mlp2 = mlp2.cuda()
    optimizer2 = torch.optim.Adam(mlp2.parameters(), lr=learning_rate)

    epoch=0
    mlp1.train()
    mlp2.train()
    for epoch in tqdm(range(epochs)):
        train(train_loader, epoch, mlp1, optimizer1, mlp2, optimizer2, device)
        
    mlp1.eval()

    test_loader = torch.utils.data.DataLoader(dataset=test_data, batch_size=batch_size, shuffle=False)
    preds = test(test_loader, mlp1, device)
    np.save(os.path.join(result_dir, 'prediction.npy'), preds)

    mlp1 = mlp1.cpu()
    mlp1.to_cpu()
    torch.save(mlp1, os.path.join(model_dir, 'Detection_Model.pkl'))

