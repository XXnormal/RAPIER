import os
import numpy as np
import xgboost
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

# Using an Ensemble of ML to correct remaining samples label.
def main(feat_dir):

    be_g = np.load(os.path.join(feat_dir, 'be_groundtruth.npy'))
    ma_g = np.load(os.path.join(feat_dir, 'ma_groundtruth.npy'))
    be_u = np.load(os.path.join(feat_dir, 'be_unknown.npy'))
    ma_u = np.load(os.path.join(feat_dir, 'ma_unknown.npy'))

    X_train = np.concatenate([be_g, ma_g], axis=0)
    Y_train = np.concatenate([np.zeros(be_g.shape[0]), np.ones(ma_g.shape[0])], axis=0)

    X_test = np.concatenate([be_g, ma_g, be_u, ma_u], axis=0)
    Y_test = np.zeros(X_test.shape[0])

    dtrain = xgboost.DMatrix(X_train, label=Y_train)
    dtest = xgboost.DMatrix(X_test, label=Y_test)
    params = {}

    # GaussianNB
    Gaussiannb = GaussianNB()
    Gaussiannb.fit(X_train, Y_train)
    possibility = Gaussiannb.predict(X_test)
    y_pred = possibility > 0.5

    ensemble = y_pred.astype(int)
    ensemble_pos = possibility

    # xgboost
    bst = xgboost.train(params, dtrain)
    possibility = bst.predict(dtest)
    y_pred = possibility > 0.5

    ensemble = ensemble + y_pred.astype(int)
    ensemble_pos = ensemble_pos + possibility

    # AdaBoost
    AdaBoost = AdaBoostClassifier()
    AdaBoost.fit(X_train, Y_train)
    possibility = AdaBoost.predict(X_test)
    y_pred = possibility > 0.5

    ensemble = ensemble + y_pred.astype(int)
    ensemble_pos = ensemble_pos + possibility

    # Linear Discriminant Analysis
    LDA = LinearDiscriminantAnalysis()
    LDA.fit(X_train, Y_train)
    possibility = LDA.predict(X_test)
    y_pred = possibility > 0.5

    ensemble = ensemble + y_pred.astype(int)
    ensemble_pos = ensemble_pos + possibility

    # SVM
    svm = SVC(kernel = 'rbf', probability=True)
    svm.fit(X_train, Y_train)
    possibility = svm.predict(X_test)
    y_pred = possibility > 0.5

    ensemble = ensemble + y_pred.astype(int)
    ensemble_pos = ensemble_pos + possibility

    # random forest
    rf = RandomForestClassifier()
    rf.fit(X_train, Y_train)
    possibility = rf.predict(X_test)
    y_pred = possibility > 0.5

    ensemble = ensemble + y_pred.astype(int)
    ensemble_pos = ensemble_pos + possibility

    # logistic regression
    logistic = LogisticRegression(penalty='l2')
    logistic.fit(X_train, Y_train)
    possibility = logistic.predict(X_test)
    y_pred = possibility > 0.5

    ensemble = ensemble + y_pred.astype(int)
    ensemble_pos = ensemble_pos + possibility

    ensemble_pred = []
    ensemble_test = []
    be_num = 0
    ma_num = 0
    be_all_final = []
    ma_all_final = []
    for i in range(len(ensemble)):
        if ensemble[i] >= 4:
            ensemble_pred.append(True)
            ensemble_test.append(Y_test[i])
            ma_num = ma_num + 1
            ma_all_final.append(X_test[i])

        else:
            ensemble_pred.append(False)
            ensemble_test.append(Y_test[i])
            be_num = be_num + 1
            be_all_final.append(X_test[i])

    be_all_final = np.array(be_all_final)
    ma_all_final = np.array(ma_all_final)
    np.random.shuffle(be_all_final)
    np.random.shuffle(ma_all_final)
    np.save(os.path.join(feat_dir, 'be_corrected.npy'), be_all_final)
    np.save(os.path.join(feat_dir, 'ma_corrected.npy'), ma_all_final)
    
    wrong_be = be_all_final[:, -1].sum()
    wrong_ma = ma_all_final.shape[0] - ma_all_final[:, -1].sum()
    print('malicious in benign set: %d/%d'%(be_all_final.shape[0], wrong_be))
    print('benign in malicious set: %d/%d'%(ma_all_final.shape[0], wrong_ma))
    
    with open('../data/result/label_correction.txt', 'w') as fp:
        fp.write('malicious in benign set: %d(%d)\n'%(wrong_be, be_all_final.shape[0]))
        fp.write('benign in malicious set: %d(%d)\n'%(wrong_ma, ma_all_final.shape[0]))
        fp.write('Remaining noise ratio: %.2f%%\n'%(100 * (wrong_be + wrong_ma) / (be_all_final.shape[0] + ma_all_final.shape[0])))
