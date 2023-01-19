from genericpath import isfile
import os
import sys
import numpy as np
import xgboost
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis

def main(white_type, black_type, white_ratio, TRAIN):
    
    print('final predict', white_type, black_type, white_ratio, TRAIN)
    white_ratio = float(white_ratio)
    root_dir = os.path.join('../data/', 'white_' + white_type + '_black_' + black_type, 'source')
    feat_dir = os.path.join(root_dir, 'feat')
    result_dir = os.path.join(root_dir, 'result')

    w_g = np.load(os.path.join(feat_dir, 'w%s_groundtruth_%.2f.npy'%(TRAIN[1:], white_ratio)))
    b_g = np.load(os.path.join(feat_dir, 'b%s_groundtruth_%.2f.npy'%(TRAIN[1:], white_ratio)))
    w_u = None 
    b_u = None
    if os.path.isfile(os.path.join(feat_dir, 'w%s_unknown_%.2f.npy'%(TRAIN[1:], white_ratio))):
        w_u = np.load(os.path.join(feat_dir, 'w%s_unknown_%.2f.npy'%(TRAIN[1:], white_ratio)))
    
    if os.path.isfile(os.path.join(feat_dir, 'b%s_unknown_%.2f.npy'%(TRAIN[1:], white_ratio))):
        b_u = np.load(os.path.join(feat_dir, 'b%s_unknown_%.2f.npy'%(TRAIN[1:], white_ratio)))

    if (w_u is None and b_u is None) or (w_u.shape[0] == 0 and b_u.shape[0] == 0):
        all_feat = np.concatenate([w_g[:, :-1], b_g[:, :-1]], axis=0)
        X_test = np.concatenate([w_g[:, :-1], b_g[:, :-1]], axis=0)
        Y_test = np.concatenate([w_g[:, -1], b_g[:, -1]], axis=0)
    else:
        if w_u.shape[0] == 0:
            u = b_u
        elif b_u.shape[0] == 0:
            u = w_u
        else:
            u = np.concatenate([w_u, b_u], axis=0)

        all_feat = np.concatenate([w_g[:, :-1], b_g[:, :-1], u[:, :-1]], axis=0)

        X_train = np.concatenate([w_g[:, :-1], b_g[:, :-1]], axis=0)
        Y_train = np.concatenate([np.zeros(w_g.shape[0]), np.ones(b_g.shape[0])], axis=0)

        X_test = np.concatenate([w_g[:, :-1], b_g[:, :-1], u[:, :-1]], axis=0)
        Y_test = np.concatenate([w_g[:, -1], b_g[:, -1], u[:, -1]], axis=0)

    dtrain = xgboost.DMatrix(X_train, label=Y_train)
    dtest = xgboost.DMatrix(X_test, label=Y_test)
    params = {}

    # GaussianNB
    Gaussiannb = GaussianNB()
    Gaussiannb.fit(X_train, Y_train)
    possibility = Gaussiannb.predict(X_test)
    y_pred = possibility > 0.5
    print('GaussianNB: %.5lf' % (accuracy_score(y_pred=y_pred, y_true=Y_test)))

    ensemble = y_pred.astype(int)
    ensemble_pos = possibility

    # xgboost
    bst = xgboost.train(params, dtrain)
    possibility = bst.predict(dtest)
    y_pred = possibility > 0.5
    # print(y_pred)
    print('xgboost: %.5lf' % (accuracy_score(y_pred=y_pred, y_true=Y_test)))

    ensemble = ensemble + y_pred.astype(int)
    ensemble_pos = ensemble_pos + possibility

    # AdaBoost
    AdaBoost = AdaBoostClassifier()
    AdaBoost.fit(X_train, Y_train)
    possibility = AdaBoost.predict(X_test)
    y_pred = possibility > 0.5
    print('AdaBoost: %.5lf' % (accuracy_score(y_pred=y_pred, y_true=Y_test)))

    ensemble = ensemble + y_pred.astype(int)
    ensemble_pos = ensemble_pos + possibility

    # Linear Discriminant Analysis
    LDA = LinearDiscriminantAnalysis()
    LDA.fit(X_train, Y_train)
    possibility = LDA.predict(X_test)
    y_pred = possibility > 0.5
    print('LDA: %.5lf' % (accuracy_score(y_pred=y_pred, y_true=Y_test)))

    ensemble = ensemble + y_pred.astype(int)
    ensemble_pos = ensemble_pos + possibility

    # SVM
    svm = SVC(kernel = 'rbf', probability=True)
    svm.fit(X_train, Y_train)
    possibility = svm.predict(X_test)
    y_pred = possibility > 0.5
    print('svm: %.5lf' % (accuracy_score(y_pred=y_pred, y_true=Y_test)))

    ensemble = ensemble + y_pred.astype(int)
    ensemble_pos = ensemble_pos + possibility

    # random forest
    rf = RandomForestClassifier()
    rf.fit(X_train, Y_train)
    possibility = rf.predict(X_test)
    y_pred = possibility > 0.5
    print('randomforest: %.5lf' % (accuracy_score(y_pred=y_pred, y_true=Y_test)))

    ensemble = ensemble + y_pred.astype(int)
    ensemble_pos = ensemble_pos + possibility

    # logistic regression
    logistic = LogisticRegression(penalty='l2')
    logistic.fit(X_train, Y_train)
    possibility = logistic.predict(X_test)
    y_pred = possibility > 0.5
    print('logistic: %.5lf' % (accuracy_score(y_pred=y_pred, y_true=Y_test)))

    ensemble = ensemble + y_pred.astype(int)
    ensemble_pos = ensemble_pos + possibility

    ensemble_pred = []
    ensemble_test = []
    wnum = 0
    bnum = 0
    w_all_final = []
    b_all_final = []
    for i in range(len(ensemble)):
        feat_label = np.append(X_test[i], Y_test[i])
        if ensemble[i] >= 4:
            ensemble_pred.append(True)
            ensemble_test.append(Y_test[i])
            bnum = bnum + 1
            b_all_final.append(feat_label)

        else:
            ensemble_pred.append(False)
            ensemble_test.append(Y_test[i])
            wnum = wnum + 1
            w_all_final.append(feat_label)

    print('ensemble: %.5lf' % (accuracy_score(y_pred=ensemble_pred, y_true=ensemble_test)))
    print('white num: {} , black num : {}'.format(wnum, bnum))

    np.random.shuffle(w_all_final)
    np.random.shuffle(b_all_final)
    print(len(w_all_final), len(b_all_final))
    np.save(os.path.join(feat_dir, 'w%s_corrected_%.2f.npy'%(TRAIN[1:], white_ratio)), np.array(w_all_final))
    np.save(os.path.join(feat_dir, 'b%s_corrected_%.2f.npy'%(TRAIN[1:], white_ratio)), np.array(b_all_final))

    with open(os.path.join(result_dir, 'label%s_correction_result_%.2f.txt'%(TRAIN[1:], white_ratio)), 'w') as fp:
        white_num = 0
        black_num = 0
        for feat in w_all_final:
            if int(feat[-1]) == 0:
                white_num = white_num + 1
            else:
                black_num = black_num + 1
        fp.write('w_all_final: {} white + {} black.\n'.format(white_num, black_num))

        noise = black_num
        tot = black_num + white_num

        white_num = 0
        black_num = 0
        for feat in b_all_final:
            if int(feat[-1]) == 0:
                white_num = white_num + 1
            else:
                black_num = black_num + 1
        fp.write('b_all_final: {} white + {} black.\n'.format(white_num, black_num))

        noise += white_num
        tot += white_num + black_num
        fp.write('all noise ratio: %f\n' % (noise / tot * 100))
        fp.write('------------------------------------------------\n')

if __name__ == '__name__':
    _, white_type, black_type, white_ratio, TRAIN = sys.argv
    main(white_type, black_type, white_ratio, TRAIN)