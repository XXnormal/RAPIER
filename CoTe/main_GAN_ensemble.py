# -*- coding:utf-8 -*-
import os
import torch 
import torch.nn as nn
import torch.nn.functional as F

from sklearn.metrics import f1_score
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
        # print ('Epoch [%03d/%03d], Iter [%02d/%02d] Accuracy1: %.4F, Accuracy2: %.4f, Loss1: %.10f, Loss2: %.10f' \
            # % (epoch+1, epochs, i+1, train_dataset.shape[0]//batch_size+1, prec1, prec2, loss_1.data.item(), loss_2.data.item()))

    train_acc1=float(train_correct1)/float(train_total1)
    train_acc2=float(train_correct2)/float(train_total2)
    return train_acc1, train_acc2

# Evaluate the Model
# Test the Model
def test(test_loader, model, device, alpha=0.5):
    test_total=0
    test_correct=0
    #wrong = []
    #right = []
    #index = 0
    for i, data_labels in enumerate(test_loader):
        
        # Forward + Backward + Optimize
        feats = data_labels[:, :-1].to(dtype=torch.float32)
        labels = data_labels[:, -1].to(dtype=int)
        
        if device != None:
            torch.cuda.set_device(device)
            feats = feats.cuda()
            labels = labels.cuda()
        
        logits = model(feats)
        outputs = F.softmax(logits, dim=1)
        pred = (outputs[:, 1] > alpha)
        
        #right.append(np.arange(feats.shape[0])[(pred == labels).cpu().detach().numpy()] + index)
        #wrong.append(np.arange(feats.shape[0])[(pred != labels).cpu().detach().numpy()] + index)
        #index += feats.shape[0]
        test_total += labels.shape[0]
        test_correct += (pred == labels).sum()

    test_acc = 100*float(test_correct) / float(test_total)
    #return test_acc, np.concatenate(right, axis=0), np.concatenate(wrong, axis=0)
    return test_acc, None, None

def main(white_type, black_type, TRAIN, multi, cuda_device, parallel=5, alpha=0.5):
    
    print('main_GAN', white_type, black_type, TRAIN, multi, cuda_device)
    multi = float(multi)
    parallel = int(parallel)
    cuda_device = int(cuda_device)
    root_dir = os.path.join('../data/', 'white_' + white_type + '_black_' + black_type, 'source')
    model_dir = os.path.join(root_dir, 'model')
    data_dir = os.path.join(root_dir, 'data')
    feat_dir = os.path.join(root_dir, 'feat')
    result_dir = os.path.join(root_dir, 'result')

    try:
        os.mkdir(result_dir)
    except:
        pass
    w = np.load(os.path.join(feat_dir, 'w_%s.npy'%(TRAIN)))[:, :32]
    b = np.load(os.path.join(feat_dir, 'b_%s.npy'%(TRAIN)))[:, :32]
    wshape = w.shape[0]
    bshape = b.shape[0]
    for index in range(parallel):

        w_gen = np.load(os.path.join(feat_dir, 'w_%s_generated_GAN_%.2f_%d.npy' % (TRAIN, multi, index)))[:, :32]
        b_gen1 = np.load(os.path.join(feat_dir, 'b_%s_generated_GAN_1_%.2f_%d.npy' % (TRAIN, multi, index)))[:, :32]
        b_gen2 = np.load(os.path.join(feat_dir, 'b_%s_generated_GAN_2_%.2f_%d.npy' % (TRAIN, multi, index)))[:, :32]
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

    print(w.shape[0], b.shape[0])
    train_data = np.concatenate([w, b], axis=0)
    train_label = np.concatenate([np.zeros(w.shape[0]), np.ones(b.shape[0])], axis=0)
    train_dataset = np.concatenate((train_data, train_label[:, None]), axis=1)

    #w_origin = np.load(os.path.join(data_dir, 'wtest.npy'))
    w = np.load(os.path.join(feat_dir, 'wtest_all.npy'))[:, :32]
    b = np.load(os.path.join(feat_dir, 'btest_all.npy'))[:, :32]
    w = np.concatenate([w, np.zeros(w.shape[0])[:, None]], axis=1)
    b = np.concatenate([b, np.ones(b.shape[0])[:, None]], axis=1)
    wtest_num = w.shape[0]
    btest_num = b.shape[0]

    if wtest_num > btest_num * 10:
        btest_num //= 2
        wtest_num = btest_num * 10
    else:
        wtest_num //= 2
        btest_num = wtest_num // 10
    
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
    # training
    os.system('rm %s' % (os.path.join(result_dir, 'test_GAN_with_generated_%d.txt'%parallel)))
    for epoch in tqdm(range(epochs)):
        # train models
        mlp1.train()
        mlp2.train()
        train(train_loader, epoch, mlp1, optimizer1, mlp2, optimizer2, device)
        if (epoch + 1) % 100 == 0:
            mlp1.eval()
            test_acc_w = []
            test_acc_b = []
            for _ in range(5):
                np.random.shuffle(w)
                np.random.shuffle(b)
                test_loader_w = torch.utils.data.DataLoader(dataset=w[:wtest_num], batch_size=batch_size, shuffle=False)
                test_loader_b = torch.utils.data.DataLoader(dataset=b[:btest_num], batch_size=batch_size, shuffle=False)
                acc_w, _, _ = test(test_loader_w, mlp1, device, alpha=alpha)
                acc_b, _, _ = test(test_loader_b, mlp1, device, alpha=alpha)
                test_acc_w.append(acc_w)
                test_acc_b.append(acc_b)
            with open(os.path.join(result_dir, 'test_GAN_with_generated_%d.txt'%parallel), 'a') as fp:
                fp.write('%d: w_acc: %.5f, b_acc: %.5f\n' % (epoch+1, np.mean(test_acc_w), np.mean(test_acc_b)))

    mlp1 = mlp1.cpu()
    mlp1.to_cpu()
    torch.save(mlp1, os.path.join(model_dir, 'CoTeaching_GAN_generate_%d.pkl'%parallel))

def main_para(white_type, black_type, TRAIN, multi, cuda_device):
    
    print('main_GAN', white_type, black_type, TRAIN, multi, cuda_device)
    multi = float(multi)
    cuda_device = int(cuda_device)

    root_dir = os.path.join('../data/', 'white_' + white_type + '_black_' + black_type, 'source')
    model_dir = os.path.join(root_dir, 'model')
    feat_dir = os.path.join(root_dir, 'feat')
    made_dir = os.path.join(root_dir, 'made')
    result_dir = os.path.join(root_dir, 'result')

    try:
        os.mkdir(result_dir)
    except:
        pass

    w = np.load(os.path.join(feat_dir, 'w_%s.npy'%(TRAIN)))[:, :32]
    b = np.load(os.path.join(feat_dir, 'b_%s.npy'%(TRAIN)))[:, :32]

    for index in range(10):

        w_gen = np.load(os.path.join(feat_dir, 'w_%s_generated_GAN_%.2f_%d.npy' % (TRAIN, multi, index)))[:, :32]
        b_gen1 = np.load(os.path.join(feat_dir, 'b_%s_generated_GAN_1_%.2f_%d.npy' % (TRAIN, multi, index)))[:, :32]
        b_gen2 = np.load(os.path.join(feat_dir, 'b_%s_generated_GAN_2_%.2f_%d.npy' % (TRAIN, multi, index)))[:, :32]
        np.random.shuffle(b_gen1)
        np.random.shuffle(b_gen2)
        w = np.concatenate([
            w, 
            w_gen[:w_gen.shape[0]], 
        ], axis=0)
        
        b = np.concatenate([
            b,
            b_gen1[:b_gen1.shape[0]],
            b_gen2[:b_gen2.shape[0]],
        ], axis=0)

    print(w.shape[0], b.shape[0])
    train_data = np.concatenate([w, b], axis=0)
    train_label = np.concatenate([np.zeros(w.shape[0]), np.ones(b.shape[0])], axis=0)
    train_dataset = np.concatenate((train_data, train_label[:, None]), axis=1)

    w = np.load(os.path.join(feat_dir, 'wtest_all.npy'))[:, :32]
    b = np.load(os.path.join(feat_dir, 'btest_all.npy'))[:, :32]
    np.random.shuffle(w)
    np.random.shuffle(b)

    w = np.concatenate([w, np.zeros(w.shape[0])[:, None]], axis=1)[:100000, :]
    b = np.concatenate([b, np.ones(b.shape[0])[:, None]], axis=1)[:100000, :]
    wtest_num = w.shape[0]
    btest_num = b.shape[0]

    device = int(cuda_device) if cuda_device != 'None' else None
    # define drop rate schedule

    if device != None:
        torch.cuda.set_device(device)
    # Data Loader (Input Pipeline)
    print('loading dataset...')
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    test_loader_w = torch.utils.data.DataLoader(dataset=w, batch_size=batch_size, shuffle=False)
    test_loader_b = torch.utils.data.DataLoader(dataset=b, batch_size=batch_size, shuffle=False)
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
    # training
    os.system('rm %s' % (os.path.join(result_dir, 'test_GAN_with_generated_%.2f.txt'%multi)))
    for epoch in tqdm(range(epochs)):
        # train models
        mlp1.train()
        mlp2.train()
        train(train_loader, epoch, mlp1, optimizer1, mlp2, optimizer2, device)
        # print('epoch %d: train_acc1 %.5f, train_acc2 %.5f' % (epoch+1, train_acc1, train_acc2))
        if (epoch + 1) % 100 == 0:
            mlp1.eval()
            test_acc_w, wrong_white = test(test_loader_w, mlp1, device)
            test_acc_b, wrong_black = test(test_loader_b, mlp1, device)
            TP = test_acc_b * btest_num / 100
            FP = (100 - test_acc_w) * wtest_num / 100
            FN = (100 - test_acc_b) * btest_num / 100
            TN = test_acc_w * wtest_num / 100
            prec = TP / (TP + FP)
            recall = TP / (TP + FN)
            F1_score = 2 * prec * recall / (prec + recall)
            with open(os.path.join(result_dir, 'test_GAN_with_generated_%.2f.txt'%multi), 'a') as fp:
                fp.write('%d: prec: %.5f, recall: %.5f, f1: %.5f, w_acc: %.5f, b_acc: %.5f\n' % (epoch+1, prec, recall, F1_score, test_acc_w, test_acc_b))
            np.save(os.path.join(feat_dir, 'wtest_wrong.npy'), wrong_white)
            np.save(os.path.join(feat_dir, 'btest_wrong.npy'), wrong_black)
    mlp1 = mlp1.cpu()
    mlp1.to_cpu()
    torch.save(mlp1, os.path.join(model_dir, 'CoTeaching_GAN_generate_%.2f.pkl'%multi))

def main_dataset(dataset, white_type, black_type, TRAIN, multi, cuda_device):
    
    print('main_GAN', dataset, white_type, black_type, TRAIN, multi, cuda_device)
    multi = float(multi)
    cuda_device = int(cuda_device)

    root_dir = os.path.join('../data/', 'white_' + white_type + '_black_' + black_type, 'source')
    model_dir = os.path.join(root_dir, 'model')
    feat_dir = os.path.join(root_dir, 'feat')
    made_dir = os.path.join(root_dir, 'made')
    result_dir = os.path.join(root_dir, 'result')

    try:
        os.mkdir(result_dir)
    except:
        pass

    w = np.load(os.path.join(feat_dir, 'w_%s.npy'%(TRAIN)))[:, :32]
    b = np.load(os.path.join(feat_dir, 'b_%s.npy'%(TRAIN)))[:, :32]

    for index in range(10):

        w_gen = np.load(os.path.join(feat_dir, 'w_%s_generated_GAN_%.2f_%d.npy' % (TRAIN, multi, index)))[:, :32]
        b_gen1 = np.load(os.path.join(feat_dir, 'b_%s_generated_GAN_1_%.2f_%d.npy' % (TRAIN, multi, index)))[:, :32]
        b_gen2 = np.load(os.path.join(feat_dir, 'b_%s_generated_GAN_2_%.2f_%d.npy' % (TRAIN, multi, index)))[:, :32]
        np.random.shuffle(b_gen1)
        np.random.shuffle(b_gen2)
        w = np.concatenate([
            w, 
            w_gen[:w_gen.shape[0] // 20], 
        ], axis=0)
        
        b = np.concatenate([
            b,
            b_gen1[:b_gen1.shape[0] // 20],
            b_gen2[:b_gen2.shape[0] // 20],
        ], axis=0)

    print(w.shape[0], b.shape[0])
    train_data = np.concatenate([w, b], axis=0)
    train_label = np.concatenate([np.zeros(w.shape[0]), np.ones(b.shape[0])], axis=0)
    train_dataset = np.concatenate((train_data, train_label[:, None]), axis=1)

    w = np.load(os.path.join(feat_dir, 'wtest_%s_all.npy'%dataset))[:, :32]
    b = np.load(os.path.join(feat_dir, 'btest_%s_all.npy'%dataset))[:, :32]
    np.random.shuffle(w)
    np.random.shuffle(b)

    w = np.concatenate([w, np.zeros(w.shape[0])[:, None]], axis=1)
    b = np.concatenate([b, np.ones(b.shape[0])[:, None]], axis=1)
    wtest_num = w.shape[0]
    btest_num = b.shape[0]

    device = int(cuda_device) if cuda_device != 'None' else None
    # define drop rate schedule

    if device != None:
        torch.cuda.set_device(device)
    # Data Loader (Input Pipeline)
    print('loading dataset...')
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    test_loader_w = torch.utils.data.DataLoader(dataset=w, batch_size=batch_size, shuffle=False)
    test_loader_b = torch.utils.data.DataLoader(dataset=b, batch_size=batch_size, shuffle=False)
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
    # training
    os.system('rm %s' % (os.path.join(result_dir, 'test_%s_GAN_with_generated_%.2f.txt'%(dataset,multi))))
    for epoch in tqdm(range(epochs)):
        # train models
        mlp1.train()
        mlp2.train()
        train(train_loader, epoch, mlp1, optimizer1, mlp2, optimizer2, device)
        # print('epoch %d: train_acc1 %.5f, train_acc2 %.5f' % (epoch+1, train_acc1, train_acc2))
        if (epoch + 1) % 100 == 0:
            mlp1.eval()
            test_acc_w, wrong_white = test(test_loader_w, mlp1, device)
            test_acc_b, wrong_black = test(test_loader_b, mlp1, device)
            TP = test_acc_b * btest_num / 100
            FP = (100 - test_acc_w) * wtest_num / 100
            FN = (100 - test_acc_b) * btest_num / 100
            TN = test_acc_w * wtest_num / 100
            prec = TP / (TP + FP)
            recall = TP / (TP + FN)
            F1_score = 2 * prec * recall / (prec + recall)
            with open(os.path.join(result_dir, 'test_%s_GAN_with_generated_%.2f.txt'%(dataset,multi)), 'a') as fp:
                fp.write('%d: prec: %.5f, recall: %.5f, f1: %.5f, w_acc: %.5f, b_acc: %.5f\n' % (epoch+1, prec, recall, F1_score, test_acc_w, test_acc_b))
            np.save(os.path.join(feat_dir, 'wtest_wrong.npy'), wrong_white)
            np.save(os.path.join(feat_dir, 'btest_wrong.npy'), wrong_black)
    mlp1 = mlp1.cpu()
    mlp1.to_cpu()
    torch.save(mlp1, os.path.join(model_dir, 'CoTeaching_%s_GAN_generate_%.2f.pkl'%(dataset, multi)))

def main_new_dataset(testdataset, white_type, black_type, TRAIN, multi, cuda_device):
    
    print('main_GAN', testdataset, white_type, black_type, TRAIN, multi, cuda_device)
    multi = float(multi)
    cuda_device = int(cuda_device)

    root_dir = os.path.join('../data/', 'white_' + white_type + '_black_' + black_type, 'source')
    model_dir = os.path.join(root_dir, 'model')
    feat_dir = os.path.join(root_dir, 'feat')
    made_dir = os.path.join(root_dir, 'made')
    result_dir = os.path.join(root_dir, 'result')

    try:
        os.mkdir(result_dir)
    except:
        pass

    w = np.load(os.path.join(feat_dir, 'w_%s.npy'%(TRAIN)))[:, :32]
    b = np.load(os.path.join(feat_dir, 'b_%s.npy'%(TRAIN)))[:, :32]

    for index in range(10):

        w_gen = np.load(os.path.join(feat_dir, 'w_%s_generated_GAN_%.2f_%d.npy' % (TRAIN, multi, index)))[:, :32]
        b_gen1 = np.load(os.path.join(feat_dir, 'b_%s_generated_GAN_1_%.2f_%d.npy' % (TRAIN, multi, index)))[:, :32]
        b_gen2 = np.load(os.path.join(feat_dir, 'b_%s_generated_GAN_2_%.2f_%d.npy' % (TRAIN, multi, index)))[:, :32]
        np.random.shuffle(b_gen1)
        np.random.shuffle(b_gen2)
        w = np.concatenate([
            w, 
            w_gen[:w_gen.shape[0]], 
        ], axis=0)
        
        b = np.concatenate([
            b,
            b_gen1[:b_gen1.shape[0]],
            b_gen2[:b_gen2.shape[0]],
        ], axis=0)

    print(w.shape[0], b.shape[0])
    train_data = np.concatenate([w, b], axis=0)
    train_label = np.concatenate([np.zeros(w.shape[0]), np.ones(b.shape[0])], axis=0)
    train_dataset = np.concatenate((train_data, train_label[:, None]), axis=1)

    w = np.load(os.path.join(feat_dir, 'wtest_%s_new_all.npy'%testdataset))[:, :32]
    b = np.load(os.path.join(feat_dir, 'btest_%s_new_all.npy'%testdataset))[:, :32]
    np.random.shuffle(w)
    np.random.shuffle(b)

    w = np.concatenate([w, np.zeros(w.shape[0])[:, None]], axis=1)
    b = np.concatenate([b, np.ones(b.shape[0])[:, None]], axis=1)
    wtest_num = w.shape[0]
    btest_num = b.shape[0]

    device = int(cuda_device) if cuda_device != 'None' else None
    # define drop rate schedule

    if device != None:
        torch.cuda.set_device(device)
    # Data Loader (Input Pipeline)
    print('loading dataset...')
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    test_loader_w = torch.utils.data.DataLoader(dataset=w, batch_size=batch_size, shuffle=False)
    test_loader_b = torch.utils.data.DataLoader(dataset=b, batch_size=batch_size, shuffle=False)
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
    # training
    os.system('rm %s' % (os.path.join(result_dir, 'test_%s_new_GAN_with_generated_%.2f.txt'%(testdataset,multi))))
    for epoch in tqdm(range(epochs)):
        # train models
        mlp1.train()
        mlp2.train()
        train(train_loader, epoch, mlp1, optimizer1, mlp2, optimizer2, device)
        # print('epoch %d: train_acc1 %.5f, train_acc2 %.5f' % (epoch+1, train_acc1, train_acc2))
        if (epoch + 1) % 100 == 0:
            mlp1.eval()
            test_acc_w, wrong_white = test(test_loader_w, mlp1, device)
            test_acc_b, wrong_black = test(test_loader_b, mlp1, device)
            TP = test_acc_b * btest_num / 100
            FP = (100 - test_acc_w) * wtest_num / 100
            FN = (100 - test_acc_b) * btest_num / 100
            TN = test_acc_w * wtest_num / 100
            prec = TP / (TP + FP)
            recall = TP / (TP + FN)
            F1_score = 2 * prec * recall / (prec + recall)
            with open(os.path.join(result_dir, 'test_%s_new_GAN_with_generated_%.2f.txt'%(testdataset,multi)), 'a') as fp:
                fp.write('%d: prec: %.5f, recall: %.5f, f1: %.5f, w_acc: %.5f, b_acc: %.5f\n' % (epoch+1, prec, recall, F1_score, test_acc_w, test_acc_b))
            np.save(os.path.join(feat_dir, 'wtest_wrong.npy'), wrong_white)
            np.save(os.path.join(feat_dir, 'btest_wrong.npy'), wrong_black)
    mlp1 = mlp1.cpu()
    mlp1.to_cpu()
    torch.save(mlp1, os.path.join(model_dir, 'CoTeaching_%s_GAN_generate_%.2f.pkl'%(testdataset, multi)))

if __name__=='__main__':
    _, white_type, black_type, TRAIN, multi, cuda_device = sys.argv
    main(white_type, black_type, TRAIN, multi, cuda_device)

