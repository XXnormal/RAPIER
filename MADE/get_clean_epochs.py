from matplotlib import pyplot as plt
import numpy as np
import sys
import os
import re
from sklearn.cluster import DBSCAN

def main(feat_dir, made_dir, alpha, TRAIN):
    
    # According to MADE-density, select true benign samples
    alpha = float(alpha)
    be = np.load(os.path.join(feat_dir, 'be.npy'))  
    ma = np.load(os.path.join(feat_dir, 'ma.npy'))
    feats = np.concatenate((be, ma), axis=0)
    print(feats.shape)
    be_number, be_shape = be.shape
    ma_number, ma_shape = ma.shape
    assert(be_shape == ma_shape)
    NLogP = [0 for _ in range(be_number + ma_number)] 
    nlogp_lst = [[] for _ in range(be_number + ma_number)]

    epochs = 0
    for filename in os.listdir(made_dir):
        if re.match('be_%s_\d+'%(TRAIN), filename):
            epochs = epochs + 1

    for i in range(epochs // 2, epochs): 
        epoch = (i + 1) * 10
        with open(os.path.join(made_dir, 'be_%sMADE_%d'%(TRAIN, epoch)), 'r') as fp:
            for i, line in enumerate(fp):
                s = float(line.strip())
                if s > 10000:
                    s = 10000
                NLogP[i] = NLogP[i] + s  
                nlogp_lst[i].append(s)

        with open(os.path.join(made_dir, 'ma_%sMADE_%d'%(TRAIN, epoch)), 'r') as fp:
            for i, line in enumerate(fp):
                s = float(line.strip())
                if s > 10000:
                    s = 10000
                NLogP[i + be_number] = NLogP[i + be_number] + s
                nlogp_lst[i + be_number].append(s)

    seq = list(range(len(NLogP)))
    seq.sort(key = lambda x: NLogP[x]) 

    be_extract = []
    be_extract_lossline = []
    extract_range = int(alpha * (be_number + ma_number))
    for i in range(extract_range): 
        be_extract.append(feats[seq[i]])
        be_extract_lossline.append(nlogp_lst[seq[i]])

    # According to the distances between pairs of samples, filter for more confidential benign samples.
    be_extract = np.array(be_extract)

    def gaussian(feat, target_set):
        ro = 0
        sigma = 5
        toBe = np.sort(np.linalg.norm(feat[None, :32].repeat(target_set.shape[0], axis=0) - target_set[:, :32], axis=1))
        num = target_set.shape[0] // 2
        for i in range(num):
            dis = toBe[i]
            ro += np.exp(-(dis ** 2 / 2 / sigma ** 2))
        return ro / num

    toBes = []
    toBesort = []
    for feat in be_extract:
        gauss = gaussian(feat, be_extract)
        toBes.append(gauss)
        toBesort.append(gauss)
    toBesort.sort()
    dom = toBesort[int(len(toBesort) * 0.5)]

    be_clean = []
    be_clean_lossline = []
    remain_index = []
    for i, toBe in enumerate(toBes):
        if toBe >= dom:
            be_clean.append(be_extract[i])
            be_clean_lossline.append(be_extract_lossline[i])
        else:
            remain_index.append(seq[i])

    remain_index += seq[extract_range:]

    # According to the distances between remaining samples and true benign samples, filter for more confidential malicious samples
    
    remain_index.sort(key = lambda x: -NLogP[x])  
    ma_extract = [feats[index] for index in remain_index]
    ma_extract_lossline = [nlogp_lst[index] for index in remain_index]

    ma_extract = np.array(ma_extract)
    be_clean = np.array(be_clean)
    ma_clean = []
    ma_clean_lossline = []
    
    be_unknown = []
    ma_unknown = []
    unknown_index = []

    be_unknown_lossline = []
    ma_unknown_lossline = []

    toBes = []
    for feat in ma_extract:
        toBe = np.sort(np.linalg.norm(feat[None, :32].repeat(be_clean.shape[0], axis=0) - be_clean[:, :32], axis=1))
        toBes.append(toBe[:].mean())
    toMas = []
    for feat in ma_extract:
        toMa = np.sort(np.linalg.norm(feat[None, :32].repeat(ma_extract.shape[0], axis=0) - ma_extract[:, :32], axis=1))
        toMas.append(toMa[1:].mean())

    relative_dis = [(toMa - toBe) for toMa, toBe in zip(toMas, toBes)]
    relative_dis.sort()
    dom = relative_dis[int(len(be_clean) * 1)]
    
    for i, (toMa, toBe, feat, lossline, index) in \
        enumerate(zip(toMas, toBes, ma_extract, ma_extract_lossline, remain_index)):
        
        if toMas[i] - toBes[i] < dom or np.isnan(dom) or np.isinf(dom):
            ma_clean.append(feat)
            ma_clean_lossline.append(lossline)
        else:
            unknown_index.append(index)
            if index < be_number:
                be_unknown.append(feat)
                be_unknown_lossline.append(nlogp_lst[index])
            else:
                ma_unknown.append(feat)
                ma_unknown_lossline.append(nlogp_lst[index])

    be_num = 0
    ma_num = 0
    for feat in be_clean:
        if int(feat[-1]) == 0:
            be_num += 1
        else:
            ma_num += 1
    print('be_clean: {} be + {} ma.'.format(be_num, ma_num))

    be_num = 0
    ma_num = 0
    for feat in ma_clean:
        if int(feat[-1]) == 0:
            be_num += 1
        else:
            ma_num += 1
    print('ma_clean: {} be + {} ma.'.format(be_num, ma_num))
    
    np.save(os.path.join(feat_dir, 'be_groundtruth.npy'), np.array(be_clean))
    np.save(os.path.join(feat_dir, 'ma_groundtruth.npy'), np.array(ma_clean))
    np.save(os.path.join(feat_dir, 'be_unknown.npy'), np.array(be_unknown))
    np.save(os.path.join(feat_dir, 'ma_unknown.npy'), np.array(ma_unknown))

    print(len(be_clean), len(ma_clean), len(be_unknown), len(ma_unknown))
