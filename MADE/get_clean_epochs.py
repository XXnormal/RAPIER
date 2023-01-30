from matplotlib import pyplot as plt
import numpy as np
import sys
import os
import re
from sklearn.cluster import DBSCAN

def main(feat_dir, made_dir, white_ratio, TRAIN):
    
    white_ratio = float(white_ratio)

    white_length = 0
    black_length = 0
    with open(os.path.join(made_dir, 'w_in_' + str(TRAIN) + '_10'), 'r') as fp:
        for i, line in enumerate(fp):
            white_length = white_length + 1
    with open(os.path.join(made_dir, 'b_in_' + str(TRAIN) + '_10'), 'r') as fp:
        for i, line in enumerate(fp):
            black_length = black_length + 1

    # 1 提取纯净白样本 ------------------------------------------------------------------------

    # 1.1 根据 made 训练后半段 epoch 的 概率密度累加和 初步筛选白样本
    w = np.load(os.path.join(feat_dir, 'w.npy'))  # 白标签数据（含噪声）特征
    b = np.load(os.path.join(feat_dir, 'b.npy'))  # 黑标签数据（含噪声）特征

    NLogP = []  # 全体样本的概率密度累加和
    nlogp_lst = []
    for i in range(white_length):  # 前一半为白标签数据（含噪声）
        NLogP.append(0)
        nlogp_lst.append([])
    for i in range(black_length):  # 后一半为黑标签数据（含噪声）
        NLogP.append(0)
        nlogp_lst.append([])

    epochs = 0
    for filename in os.listdir(made_dir):
        if re.match('w_in_' + str(TRAIN) + '_\d+', filename):
            epochs = epochs + 1
    print(epochs)

    for i in range(epochs // 2, epochs):  # 统计各样本在后半段 epoch 中的概率密度累加和
        epoch = (i + 1) * 10
        with open(os.path.join(made_dir, 'w_in_' + str(TRAIN) + '_' + str(epoch)), 'r') as fp:
            for i, line in enumerate(fp):
                s = float(line.strip())
                if s > 10000:
                    s = 10000
                NLogP[i] = NLogP[i] + s  # w[i] 对应 NlogP[i]
                nlogp_lst[i].append(s)

        with open(os.path.join(made_dir, 'b_in_' + str(TRAIN) + '_' + str(epoch)), 'r') as fp:
            for i, line in enumerate(fp):
                s = float(line.strip())
                if s > 10000:
                    s = 10000
                NLogP[i + white_length] = NLogP[i + white_length] + s  # b[i] 对应 NlogP[i + white_length]
                nlogp_lst[i + white_length].append(s)

    total = len(NLogP)
    seq = list(range(total))
    seq.sort(key = lambda x: NLogP[x])  # 对全体样本在 NLogP 中的索引按照对应值排序

    white_extract = []
    white_extract_lossline = []
    for i in range(int(white_ratio * white_length)):  # 初步抽取前 white_ratio 的样本 white_extract
        index = seq[i]
        if index < white_length:
            feat = w[index]
        else:
            feat = b[index - white_length]
        white_extract.append(feat)
        white_extract_lossline.append(nlogp_lst[index])

    # 1.2 基于距离衡量算法，进一步剔除 white_extract 中的噪声 得到 white_clean
    white_extract = np.array(white_extract)

    def gaussian(feat, target_set):
        ro = 0
        sigma = 5
        toW = np.sort(np.linalg.norm(feat[None, :].repeat(target_set.shape[0], axis=0) - target_set, axis=1))
        num = target_set.shape[0] // 2
        for i in range(num):
            dis = toW[i]
            ro += np.exp(-(dis ** 2 / 2 / sigma ** 2))
        return ro / num

    toWs = []
    toWsort = []
    for feat in white_extract:
        gauss = gaussian(feat, white_extract)
        toWs.append(gauss)
        toWsort.append(gauss)
    toWsort.sort()
    dom = toWsort[int(len(toWsort) * 0.5)]

    white_clean = []
    white_clean_lossline = []
    remain_index = []  # 剩余样本索引值列表
    i = 0
    for feat, toW in zip(white_extract, toWs):
        if toW >= dom:
            white_clean.append(feat)
            white_clean_lossline.append(white_extract_lossline[i])
        else:
            index = seq[i]
            remain_index.append(index)
        i = i + 1

    for i in range(int(white_ratio * white_length), white_length + black_length):
        index = seq[i]
        remain_index.append(index)

    # 2 在剩余样本索引值列表 remain_index 中继续纯净黑样本 ----------------------------------------

    # 2.1 根据 made 训练后半段 epoch 的 概率密度累加和 初步筛选黑样本
    NlogP = []  # 全体样本的 概率密度累加和
    nlogp_lst = []  # 全体样本的 概率密度序列
    for i in range(white_length):
        NlogP.append(0)
        nlogp_lst.append([])
    for i in range(black_length):
        NlogP.append(0)
        nlogp_lst.append([])

    epochs = 0
    for filename in os.listdir(made_dir):
        if re.match('b_in_' + str(TRAIN) + '_\d+', filename):
            epochs = epochs + 1

    for i in range(epochs // 2, epochs):  # 统计各样本在整个训练中的 概率密度绝对差值累加和
        epoch = (i + 1) * 10
        with open(os.path.join(made_dir, 'w_in_' + str(TRAIN) + '_' + str(epoch)), 'r') as fp:
            for i, line in enumerate(fp):
                s = float(line.strip())
                if s > 10000:
                    s = 10000
                NlogP[i] = NlogP[i] + s # abs(s - nlogp_lastepoch[i])
                nlogp_lst[i].append(s)

        with open(os.path.join(made_dir, 'b_in_' + str(TRAIN) + '_' + str(epoch)), 'r') as fp:
            for i, line in enumerate(fp):
                s = float(line.strip())
                if s > 10000:
                    s = 10000
                NlogP[i + white_length] = NlogP[i + white_length] + s  # abs(s - nlogp_lastepoch[i + white_length])
                nlogp_lst[i + white_length].append(s)

    remain_index.sort(key = lambda x: -NlogP[x])  # 对剩余样本在 ressum 中的索引按照对应值排序

    black_extract = []
    black_extract_lossline = []
    extract_range = len(remain_index)

    for i in range(extract_range):  # 初步抽取前 ratio 的样本 black_extract
        index = remain_index[i]
        if index < white_length:
            feat = w[index]
        else:
            feat = b[index - white_length]
        black_extract.append(feat)

        black_extract_lossline.append(nlogp_lst[index])

    # 2.2 基于距离衡量算法，进一步剔除 black_extract 中的噪声 得到 black_clean
    black_extract = np.array(black_extract)

    black_clean = []
    white_unknown = []
    black_unknown = []
    unknown_index = []

    black_clean_lossline = []
    white_unknown_lossline = []
    black_unknown_lossline = []

    # 根据目标样本到 black_extract 和 纯净白样本 的相对距离排序，取前一部分的样本
    toWs = []
    for feat in black_extract:
        toW = np.sort(np.linalg.norm(feat[None, :].repeat(np.array(white_clean).shape[0], axis=0) - np.array(white_clean), axis=1))
        toWs.append(toW[:].mean())
    toBs = []
    for feat in black_extract:
        toB = np.sort(np.linalg.norm(feat[None, :].repeat(black_extract.shape[0], axis=0) - black_extract, axis=1))
        toBs.append(toB[1:].mean())

    reltive_dis = []
    for i in range(len(toWs)):
        reltive_dis.append(toBs[i] - toWs[i])
    reltive_dis.sort()
    dom = reltive_dis[int(len(white_clean) * 1)]
    
    i = 0
    yan = []
    for feat in black_extract:
        if toBs[i] - toWs[i] < dom or np.isnan(dom) or np.isinf(dom):
            black_clean.append(feat)
            yan.append(i)
            black_clean_lossline.append(black_extract_lossline[i])
        else:
            index = remain_index[i]
            unknown_index.append(index)
            if index < white_length:
                white_unknown.append(w[index])
                white_unknown_lossline.append(nlogp_lst[index])
            else:
                black_unknown.append(b[index - white_length])
                black_unknown_lossline.append(nlogp_lst[index])
        i = i + 1

    for i in range(extract_range, len(remain_index)):
        index = remain_index[i]
        unknown_index.append(index)
        if index < white_length:
            white_unknown.append(w[index])
            white_unknown_lossline.append(nlogp_lst[index])
        else:
            black_unknown.append(b[index - white_length])
            black_unknown_lossline.append(nlogp_lst[index])

    # 3 存储提取结果 -------------------------------------------------------------------------

    np.save(os.path.join(feat_dir, 'w_groundtruth.npy'), np.array(white_clean))
    np.save(os.path.join(feat_dir, 'b_groundtruth.npy'), np.array(black_clean))
    if len(white_unknown) > 0:
        np.save(os.path.join(feat_dir, 'w_unknown.npy'), np.array(white_unknown))
    if len(black_unknown) > 0:
        np.save(os.path.join(feat_dir, 'b_unknown.npy'), np.array(black_unknown))

