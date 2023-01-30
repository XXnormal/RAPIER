import os
import numpy as np
import xgboost
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

def main(feat_dir):

    w_g = np.load(os.path.join(feat_dir, 'w_groundtruth.npy'))
    b_g = np.load(os.path.join(feat_dir, 'b_groundtruth.npy'))
    w_u = None 
    b_u = None
    if os.path.isfile(os.path.join(feat_dir, 'w_unknown.npy')):
        w_u = np.load(os.path.join(feat_dir, 'w_unknown.npy'))
    
    if os.path.isfile(os.path.join(feat_dir, 'b_unknown.npy')):
        b_u = np.load(os.path.join(feat_dir, 'b_unknown.npy'))

    if (w_u is None and b_u is None) or (w_u.shape[0] == 0 and b_u.shape[0] == 0):
        X_test = np.concatenate([w_g, b_g], axis=0)
    else:
        u = []
        if w_u:
            u.append(w_u)
        if b_u:
            u.append(b_u)
        u = np.concatenate(u, axis=0)

        X_train = np.concatenate([w_g, b_g], axis=0)
        Y_train = np.concatenate([np.zeros(w_g.shape[0]), np.ones(b_g.shape[0])], axis=0)

        X_test = np.concatenate([w_g, b_g, u], axis=0)
    
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

    np.random.shuffle(w_all_final)
    np.random.shuffle(b_all_final)
    print(len(w_all_final), len(b_all_final))
    np.save(os.path.join(feat_dir, 'w_corrected.npy'), np.array(w_all_final))
    np.save(os.path.join(feat_dir, 'b_corrected.npy'), np.array(b_all_final))
