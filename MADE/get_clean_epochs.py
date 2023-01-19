from matplotlib import pyplot as plt
import numpy as np
import sys
import os
import re
from sklearn.cluster import DBSCAN

def main(white_type, black_type, white_ratio, TRAIN):
    
    print('get_clean_epochs', white_type, black_type, white_ratio, TRAIN)
    white_ratio = float(white_ratio)
    
    root_dir = os.path.join('../data/', 'white_' + white_type + '_black_' + black_type, 'source')
    feat_dir = os.path.join(root_dir, 'feat')
    made_dir = os.path.join(root_dir, 'made')
    plot_dir = os.path.join(root_dir, 'plot')

    white_length = 0
    black_length = 0
    with open(os.path.join(made_dir, 'w' + str(TRAIN)[1:] + '_in_' + str(TRAIN) + '_10'), 'r') as fp:
        for i, line in enumerate(fp):
            white_length = white_length + 1
    with open(os.path.join(made_dir, 'b' + str(TRAIN)[1:] + '_in_' + str(TRAIN) + '_10'), 'r') as fp:
        for i, line in enumerate(fp):
            black_length = black_length + 1

    # 1 提取纯净白样本 ------------------------------------------------------------------------

    # 1.1 根据 made 训练后半段 epoch 的 概率密度累加和 初步筛选白样本
    w = np.load(os.path.join(feat_dir, 'w%s.npy'%(TRAIN[1:])))  # 白标签数据（含噪声）特征
    b = np.load(os.path.join(feat_dir, 'b%s.npy'%(TRAIN[1:])))  # 黑标签数据（含噪声）特征

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
        if re.match('w' + str(TRAIN)[1:] + '_in_' + str(TRAIN) + '_\d+', filename):
            epochs = epochs + 1
    print(epochs)

    for i in range(epochs // 2, epochs):  # 统计各样本在后半段 epoch 中的概率密度累加和
        epoch = (i + 1) * 10
        with open(os.path.join(made_dir, 'w' + str(TRAIN)[1:] + '_in_' + str(TRAIN) + '_' + str(epoch)), 'r') as fp:
            for i, line in enumerate(fp):
                s = float(line.strip())
                if s > 10000:
                    s = 10000
                NLogP[i] = NLogP[i] + s  # w[i] 对应 NlogP[i]
                nlogp_lst[i].append(s)

        with open(os.path.join(made_dir, 'b' + str(TRAIN)[1:] + '_in_' + str(TRAIN) + '_' + str(epoch)), 'r') as fp:
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
        toW = np.sort(np.linalg.norm(feat[None, :-1].repeat(target_set.shape[0], axis=0) - target_set[:, :-1], axis=1))
        num = target_set.shape[0] // 2
        for i in range(num):
            dis = toW[i]
            ro += np.exp(-(dis ** 2 / 2 / sigma ** 2))
        return ro / num

    toWs = []
    toWsort = []
    toWs_w = []
    toWs_b = []
    for feat in white_extract:
        toW = np.sort(np.linalg.norm(feat[None, :-1].repeat(white_extract.shape[0], axis=0) - white_extract[:, :-1], axis=1))
        gauss = gaussian(feat, white_extract)
        if feat[-1] == 0:
            # toWs_w.append(gaussian(feat, np.array(w.tolist() + b.tolist())))
            toWs_w.append(gauss)
            # toWs_w.append(toW[1: 11].mean())
        else:
            # toWs_b.append(gaussian(feat, np.array(w.tolist() + b.tolist())))
            toWs_b.append(gauss)
            # toWs_b.append(toW[1: 11].mean())
        toWs.append(gauss)
        toWsort.append(gauss)
        # toWs.append(toW[1: 11].mean())
    toWsort.sort()
    dom = toWsort[int(len(toWsort) * 0.5)]

    # data = np.array(toWs).reshape(-1, 1)
    # dbscan = DBSCAN(eps=0.01)
    # dbscan.fit(data)
    # label_pred = dbscan.labels_ #获取聚类标签
    # x0 = data[label_pred == 0]
    # dom = x0[np.argmax(x0)]
    # print(label_pred)
    # # plt.scatter(data[:,0], data[:,1],s=s*10,c=dbscan.labels_)
    # x0 = data[label_pred == 0]
    # x1 = data[label_pred == 1]
    # x2 = data[label_pred == -1]
    #
    # X = []
    # for i in range(len(x0)):
    #     X.append(i)
    # plt.scatter(x0, X, alpha=0.5, label='x0')
    # X = []
    # for i in range(len(x1)):
    #     X.append(i + len(x0))
    # plt.scatter(x1, X, alpha=0.5, label='x1')
    # X = []
    # for i in range(len(x2)):
    #     X.append(i + len(x0) + len(x1))
    # plt.scatter(x2, X, alpha=0.5, label='x2')
    # plt.legend()
    # plt.xlabel('dis_to_white_extract')
    # plt.title('In WMADE')
    # plt.show()

    # toWs_w = np.array(toWs_w)
    # toWs_b = np.array(toWs_b)
    # X = []
    # for i in range(len(toWs_w)):
    #     X.append(i)
    # plt.scatter(toWs_w, X, alpha=0.5, label='white')
    # X = []
    # for i in range(len(toWs_b)):
    #     X.append(i + len(toWs_w))
    # plt.scatter(toWs_b, X, alpha=0.5, label='black')
    # plt.vlines(x=dom, ymin=0, ymax=200, linestyles='--', color='black', alpha=0.5)
    # plt.legend()
    # plt.xlabel('dis_to_white_extract')
    # plt.title('In WMADE')
    # plt.show()


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

    # 1.3 输出纯净白样本提取效果
    white_num = 0
    black_num = 0
    for feat in white_extract:
        if int(feat[-1]) == 0:
            white_num = white_num + 1
        else:
            black_num = black_num + 1
    print('white_extract: {} white + {} black.'.format(white_num, black_num))

    white_num = 0
    black_num = 0
    for feat in white_clean:
        if int(feat[-1]) == 0:
            white_num = white_num + 1
        else:
            black_num = black_num + 1
    print('white_clean: {} white + {} black.'.format(white_num, black_num))

    # 2 在剩余样本索引值列表 remain_index 中继续纯净黑样本 ----------------------------------------

    # 2.1 根据 made 训练后半段 epoch 的 概率密度累加和 初步筛选黑样本
    NlogP = []  # 全体样本的 概率密度累加和
    nlogp_lst = []  # 全体样本的 概率密度序列
    for i in range(white_length):  # 前一半为白标签数据（含噪声）
        # nlogp_lastepoch.append(0)
        NlogP.append(0)
        nlogp_lst.append([])
    for i in range(black_length):  # 后一半为黑标签数据（含噪声）
        # nlogp_lastepoch.append(0)
        NlogP.append(0)
        nlogp_lst.append([])

    epochs = 0
    for filename in os.listdir(made_dir):
        if re.match('b' + str(TRAIN)[1:] + '_in_' + str(TRAIN) + '_\d+', filename):
            epochs = epochs + 1

    for i in range(epochs // 2, epochs):  # 统计各样本在整个训练中的 概率密度绝对差值累加和
        epoch = (i + 1) * 10
        with open(os.path.join(made_dir, 'w' + str(TRAIN)[1:] + '_in_' + str(TRAIN) + '_' + str(epoch)), 'r') as fp:
            for i, line in enumerate(fp):
                s = float(line.strip())
                if s > 10000:
                    s = 10000
                NlogP[i] = NlogP[i] + s # abs(s - nlogp_lastepoch[i])
                nlogp_lst[i].append(s)

        with open(os.path.join(made_dir, 'b' + str(TRAIN)[1:] + '_in_' + str(TRAIN) + '_' + str(epoch)), 'r') as fp:
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

    def dis(feat):
        distances = []
        for dest_feat in white_clean:
            distances.append(np.linalg.norm(feat[:-1] - dest_feat[:-1]))
        distances.sort()
        # ret = 0
        # for i in range(len(distances)):
        #     ret += distances[i]
        # return ret
        return np.array(distances)[:].mean()

    black_clean = []
    white_unknown = []
    black_unknown = []
    unknown_index = []
    unknown_reldis = []

    black_clean_lossline = []
    white_unknown_lossline = []
    black_unknown_lossline = []

    # 根据目标样本到 black_extract 和 纯净白样本 的相对距离排序，取前一部分的样本
    toWs = []
    for feat in black_extract:
        toW = np.sort(np.linalg.norm(feat[None, :32].repeat(np.array(white_clean).shape[0], axis=0) - np.array(white_clean)[:, :32], axis=1))
        toWs.append(toW[:].mean())
    toBs = []
    for feat in black_extract:
        toB = np.sort(np.linalg.norm(feat[None, :32].repeat(black_extract.shape[0], axis=0) - black_extract[:, :32], axis=1))
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

    # 2.3 输出纯净黑样本提取效果
    white_num = 0
    black_num = 0
    for feat in black_extract:
        if int(feat[-1]) == 0:
            white_num = white_num + 1
        else:
            black_num = black_num + 1
    print('black_extract: {} white + {} black.'.format(white_num, black_num))

    white_num = 0
    black_num = 0
    for feat in black_clean:
        if int(feat[-1]) == 0:
            white_num = white_num + 1
        else:
            black_num = black_num + 1
    print('black_clean: {} white + {} black.'.format(white_num, black_num))

    # 3 存储提取结果 -------------------------------------------------------------------------

    np.save(os.path.join(feat_dir, 'w%s_groundtruth_%.2f.npy'%(TRAIN[1:], white_ratio)), np.array(white_clean))
    np.save(os.path.join(feat_dir, 'b%s_groundtruth_%.2f.npy'%(TRAIN[1:], white_ratio)), np.array(black_clean))
    if len(white_unknown) > 0:
        np.save(os.path.join(feat_dir, 'w%s_unknown_%.2f.npy'%(TRAIN[1:], white_ratio)), np.array(white_unknown))
    if len(black_unknown) > 0:
        np.save(os.path.join(feat_dir, 'b%s_unknown_%.2f.npy'%(TRAIN[1:], white_ratio)), np.array(black_unknown))

if __name__ == '__main__':
    _, white_type, black_type, white_ratio, TRAIN = sys.argv
    main(white_type, black_type, white_ratio, TRAIN)
