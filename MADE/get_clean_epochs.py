from matplotlib import pyplot as plt
import numpy as np
import sys
import os
import re
from sklearn.cluster import DBSCAN

def main(feat_dir, made_dir, alpha, TRAIN):
    
    # 1 提取纯净白样本 ------------------------------------------------------------------------

    # 1.1 根据 made 训练后半段 epoch 的 概率密度累加和 初步筛选白样本
    alpha = float(alpha)
    be = np.load(os.path.join(feat_dir, 'be.npy'))  # 白标签数据（含噪声）特征
    ma = np.load(os.path.join(feat_dir, 'ma.npy'))  # 黑标签数据（含噪声）特征
    feats = np.concatenate((be, ma), axis=0)
    be_number, be_shape = be.shape
    ma_number, ma_shape = ma.shape
    assert(be_shape == ma_shape)
    NLogP = [0 for _ in range(be_number + ma_number)]  # 全体样本的概率密度累加和
    nlogp_lst = [[] for _ in range(be_number + ma_number)]

    epochs = 0
    for filename in os.listdir(made_dir):
        if re.match('be_%s_\d+'%(TRAIN), filename):
            epochs = epochs + 1

    for i in range(epochs // 2, epochs):  # 统计各样本在后半段 epoch 中的概率密度累加和
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
    seq.sort(key = lambda x: NLogP[x])  # 对全体样本在 NLogP 中的索引按照对应值排序

    be_extract = []
    be_extract_lossline = []
    extract_range = int(alpha * (be_number + ma_number))
    for i in range(extract_range):  # 初步抽取前 alpha 的样本 be_extract
        print(seq[i])
        be_extract.append(feats[seq[i]])
        be_extract_lossline.append(nlogp_lst[seq[i]])

    # 1.2 基于距离衡量算法，进一步剔除 be_extract 中的噪声 得到 be_clean
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
    remain_index = []  # 剩余样本索引值列表
    for i, toBe in enumerate(toBes):
        if toBe >= dom:
            be_clean.append(be_extract[i])
            be_clean_lossline.append(be_extract_lossline[i])
        else:
            remain_index.append(seq[i])

    remain_index += seq[extract_range:]

    # 2 在剩余样本索引值列表 remain_index 中继续纯净黑样本 ----------------------------------------

    remain_index.sort(key = lambda x: -NLogP[x])  # 对剩余样本在 ressum 中的索引按照对应值排序

    ma_extract = [feats[index] for index in remain_index]
    ma_extract_lossline = [nlogp_lst[index] for index in remain_index]

    # 2.2 基于距离衡量算法，进一步剔除 black_extract 中的噪声 得到 black_clean
    ma_extract = np.array(ma_extract)
    be_clean = np.array(be_clean)
    ma_clean = []
    ma_clean_lossline = []
    
    be_unknown = []
    ma_unknown = []
    unknown_index = []

    be_unknown_lossline = []
    ma_unknown_lossline = []

    # 根据目标样本到 black_extract 和 纯净白样本 的相对距离排序，取前一部分的样本
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

    # 3 存储提取结果 -------------------------------------------------------------------------

    np.save(os.path.join(feat_dir, 'be_groundtruth.npy'), np.array(be_clean))
    np.save(os.path.join(feat_dir, 'ma_groundtruth.npy'), np.array(ma_clean))
    np.save(os.path.join(feat_dir, 'be_unknown.npy'), np.array(be_unknown))
    np.save(os.path.join(feat_dir, 'ma_unknown.npy'), np.array(ma_unknown))
    
    print(np.array(be_unknown).shape)

