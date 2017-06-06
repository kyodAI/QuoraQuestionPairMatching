from keras.optimizers import SGD
from scipy import sparse as ssp
from sklearn.cross_validation import train_test_split
from sklearn.grid_search import GridSearchCV
from sklearn.cross_validation import StratifiedShuffleSplit
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.svm import SVC, LinearSVC, SVR
import numpy as np
from xgboost import XGBClassifier, XGBRegressor
from sklearn.metrics import log_loss
import pandas as pd
import seaborn.apionly as sns
import matplotlib.pyplot as plt
import xgboost

from keras.models import Sequential
from keras.layers import Embedding, Dense, Dropout, Reshape, Merge, BatchNormalization, TimeDistributed, Lambda, Input, \
    Bidirectional, Conv1D, Flatten
from keras.layers.pooling import MaxPooling1D
from keras.models import Model
from keras.layers.recurrent import LSTM, GRU
from sklearn.utils.class_weight import compute_class_weight
from sklearn.preprocessing import PolynomialFeatures

plt.style.use('ggplot')
colors = sns.color_palette()


def load_dataset(ttype='train', clean=False):
    if ttype not in ['train', 'test', 'train_extracted_features',
                     'test_part1',
                     'test_part2',
                     'test_part3',
                     'test_part4',
                     'test_part5',
                     'test_part6',
                     'test_part7',
                     'test_part8',
                     'test_part9',
                     'test_part10',
                     'test_part11',
                     'test_part12',
                     'test_part13',
                     'test_part14',
                     'test_part15',
                     'test_part16',
                     'test_part17',
                     'test_part18',
                     'test_part19',
                     'test_part20',
                     'test_part1_extracted_features',
                     'test_part2_extracted_features',
                     'test_part3_extracted_features',
                     'test_part4_extracted_features',
                     'test_part5_extracted_features',
                     'test_part6_extracted_features',
                     'test_part7_extracted_features',
                     'test_part8_extracted_features',
                     'test_part9_extracted_features',
                     'test_part10_extracted_features',
                     'test_part11_extracted_features',
                     'test_part12_extracted_features',
                     'test_part13_extracted_features',
                     'test_part14_extracted_features',
                     'test_part15_extracted_features',
                     'test_part16_extracted_features',
                     'test_part17_extracted_features',
                     'test_part18_extracted_features',
                     'test_part19_extracted_features',
                     'test_part20_extracted_features',
                     ]:
        raise AssertionError("need  to use train or test keyword")
    cleanfile = '_clean' if clean else ''
    df = pd.read_csv(
        '/media/nacim/a13ff970-2886-43fb-b752-f1a813a76a12/data/{0}{1}.csv'.format(ttype, cleanfile)).replace(np.nan,
                                                                                                              '')
    return df


from datetime import datetime
from csv import DictReader
from math import exp, log, sqrt, pow
import itertools
import math
from random import random, shuffle, uniform, seed
from sklearn.datasets import dump_svmlight_file, load_svmlight_file
import pickle
import sys


def oversample(X_ot, y, p=0.165):
    pos_ot = X_ot[y == 1]
    neg_ot = X_ot[y == 0]
    # p = 0.165
    scale = ((pos_ot.shape[0] * 1.0 / (pos_ot.shape[0] + neg_ot.shape[0])) / p) - 1
    while scale > 1:
        neg_ot = ssp.vstack([neg_ot, neg_ot]).tocsr()
        scale -= 1
    neg_ot = ssp.vstack([neg_ot, neg_ot[:int(scale * neg_ot.shape[0])]]).tocsr()
    ot = ssp.vstack([pos_ot, neg_ot]).tocsr()
    y = np.zeros(ot.shape[0])
    y[:pos_ot.shape[0]] = 1.0
    return ot, y


def naive_method_classifier(crossval=True, xpname='', learningmethod='rf', oncleaned_data=False, maxfeature=100):
    print('Loading Training Dataset')
    df = load_dataset('train', clean=oncleaned_data)
    first = df[['question1']]
    second = df[['question2']]
    first.columns = ["question"]
    second.columns = ["question"]
    dfq = pd.concat([first, second], axis=0, ignore_index=True).fillna('')
    tfidf = TfidfVectorizer(max_features=maxfeature, stop_words='english').fit_transform(dfq['question'].values)

    N = len(df)
    X_tfidf = (np.abs(tfidf[:N] - tfidf[N:])).toarray()
    y = df['is_duplicate'].values
    if crossval:
        sss = StratifiedShuffleSplit(y=y, n_iter=5, test_size=0.2, )

        result = pd.DataFrame()
        for train, test in sss:

            xtrain, xtest, ytrain, ytest = X_tfidf[train], X_tfidf[test], y[train], y[test]
            s = pd.Series()
            for learningmethod in [
                # 'svm_linear',
                'rf',
                'xgboost']:
                print(learningmethod)
                if learningmethod == 'rf':
                    estimator = RandomForestClassifier(n_jobs=4)
                elif learningmethod == 'svm_rbf':
                    estimator = SVC(kernel='rbf')
                elif learningmethod == 'svm_linear':
                    estimator = SVR(kernel='linear')
                elif learningmethod == 'xgboost':
                    estimator = XGBClassifier(nthread=4, n_estimators=350, max_depth=4)

                estimator.fit(xtrain, ytrain)

                ypred = estimator.predict_proba(xtest)
                loss = log_loss(ytest, ypred)
                print loss

                s[learningmethod] = loss
            result = result.append(s.T, ignore_index=True)
        result.to_csv('{0}_clean{1}.csv'.format(xpname, oncleaned_data))
        final = pd.DataFrame()
        for c in result.columns:
            tmp = pd.DataFrame()
            tmp['logloss'] = result[c]
            tmp['classifier'] = c
            final = final.append(tmp, ignore_index=True)
        sns.factorplot(x='classifier', y='logloss', data=final)
        sns.plt.savefig('{0}_clean{1}.pdf'.format(xpname, oncleaned_data), bbox_inches='tight')

    else:
        print 'Test submission'

        print('Load  Test Dataset')
        df = load_dataset('test', clean=oncleaned_data)

        first = df[['question1']]
        second = df[['question2']]
        first.columns = ["question"]
        second.columns = ["question"]
        dfq = pd.concat([first, second], axis=0, ignore_index=True).fillna('')
        tfidf = TfidfVectorizer(max_features=maxfeature, stop_words='english').fit_transform(dfq['question'].values)

        N = len(df)
        Xtest = (np.abs(tfidf[:N] - tfidf[N:])).toarray()

        if learningmethod == 'rf':
            estimator = RandomForestClassifier(n_jobs=4)
        elif learningmethod == 'svm_rbf':
            estimator = SVC(kernel='rbf')
        elif learningmethod == 'svm_linear':
            estimator = SVR(kernel='linear')
        elif learningmethod == 'xgboost':
            estimator = XGBClassifier(nthread=4, n_estimators=300, max_depth=4)
        print 'fitting a %s classifier' % learningmethod
        estimator.fit(X_tfidf, y)

        ypred = estimator.predict_proba(Xtest)
        result = pd.DataFrame()
        result['test_id'] = df['test_id']
        result['is_duplicate'] = ypred[:, 1]
        result.to_csv(xpname + '_clean{0}_{1}.csv'.format(oncleaned_data, learningmethod), index=False)


def extracted_features_method_classifier(crossval=True, xpname='cross_validation_classifier', oncleaned_data=False,
                                         learningmethod='xgboost'):
    print('Loading Training Dataset')
    df = load_dataset('train', clean=oncleaned_data)
    Xtrain = load_dataset('train_extracted_features').replace('', 0).replace(np.nan, 0).replace(np.inf, 0).replace(
        -np.inf, 0).astype(np.float32)

    Xtrain = np.array(Xtrain)
    y = df['is_duplicate'].values
    del df

    if crossval:
        sss = StratifiedShuffleSplit(y=y, n_iter=5, test_size=0.2, )

        result = pd.DataFrame()
        for train, test in sss:

            xtrain, xtest, ytrain, ytest = Xtrain[train], Xtrain[test], y[train], y[test]
            s = pd.Series()
            for learningmethod in [
                # 'svm_linear',
                # 'svm_rbf',
                'xgboost',
                'rf',
            ]:
                print(learningmethod)
                if learningmethod == 'rf':
                    estimator = RandomForestClassifier(n_jobs=4, n_estimators=100)
                    estimator.fit(xtrain, ytrain)
                    ypred = estimator.predict_proba(xtest)
                    loss = log_loss(ytest, ypred)
                    print loss
                    s[learningmethod] = loss
                elif learningmethod == 'svm_rbf':
                    estimator = SVC(kernel='rbf')
                elif learningmethod == 'svm_linear':
                    estimator = LinearSVC()
                elif learningmethod == 'xgboost':
                    estimator = XGBClassifier(nthread=4, n_estimators=350, max_depth=4)
                    gpu_params = {
                        'objective': 'binary:logistic',
                        'eval_metric': 'logloss',
                        'eta': 0.01,
                        'max_depth': 9,
                        'min_child_weight': 1,
                        # 'updater': 'grow_gpu',
                        'n_estimators': 1000,
                        'scale_pos_weight': 1
                    }
                    D_training = xgboost.DMatrix(xtrain, label=ytrain)
                    D_validation = xgboost.DMatrix(xtest, label=ytest)
                    watchlist = [(D_training, 'training'), (D_validation, 'validation')]

                    bst = xgboost.train(gpu_params, D_training, 50000, watchlist, early_stopping_rounds=10000,
                                        verbose_eval=50)
                    ypred = bst.predict(D_validation)
                    print ypred
                    exit()

                    #

            result = result.append(s.T, ignore_index=True)
        result.to_csv('{0}_clean{1}.csv'.format(xpname, oncleaned_data))
        final = pd.DataFrame()
        for c in result.columns:
            tmp = pd.DataFrame()
            tmp['logloss'] = result[c]
            tmp['classifier'] = c
            final = final.append(tmp, ignore_index=True)
        sns.factorplot(x='classifier', y='logloss', data=final)
        sns.plt.savefig('{0}_clean{1}.pdf'.format(xpname, oncleaned_data), bbox_inches='tight')

    else:
        print 'Test submission'
        if learningmethod == 'rf':
            estimator = RandomForestClassifier(n_jobs=5, n_estimators=150)
        elif learningmethod == 'svm_rbf':
            estimator = SVC(kernel='rbf')
        elif learningmethod == 'svm_linear':
            estimator = SVR(kernel='linear')
        elif learningmethod == 'xgboost':
            estimator = XGBClassifier(nthread=4, n_estimators=300, max_depth=4)
        print 'fitting a %s classifier' % learningmethod
        estimator.fit(Xtrain, y)

        result = pd.DataFrame()
        for i in range(1, 21):
            print 'read Test %s' % i
            df = load_dataset('test_part%s_extracted_features' % i, clean=oncleaned_data).replace('', 0).replace(np.nan,
                                                                                                                 0).replace(
                np.inf, 0).replace(-np.inf, 0).astype(np.float32)
            dtmp = load_dataset('test_part%s' % i, clean=oncleaned_data)
            test_ids = dtmp['test_id']
            del dtmp
            Xtest = np.array(df)
            del df
            ypred = estimator.predict_proba(Xtest)
            resulttmp = pd.DataFrame()
            resulttmp['test_id'] = test_ids
            resulttmp['is_duplicate'] = ypred[:, 1]

            result = result.append(resulttmp, ignore_index=True)
        result.to_csv(xpname + '_clean{0}_{1}.csv'.format(oncleaned_data, learningmethod), index=False)


def from_question_representation_method_classifier(crossval=True, xpname='cross_validation_classifier',
                                                   learningmethod='xgboost', merge_method='concat'):
    df = load_dataset('train', clean=False)
    y = df['is_duplicate'].values
    del df
    dataset = 'train'
    Xtrain = load_questions_and_merge(dataset, method=merge_method)

    if crossval:
        sss = StratifiedShuffleSplit(y=y, n_iter=1, test_size=0.2, )

        result = pd.DataFrame()
        for train, test in sss:

            xtrain, xtest, ytrain, ytest = Xtrain[train], Xtrain[test], y[train], y[test]
            s = pd.Series()
            for learningmethod in [
                'rf',
                'xgboost']:
                print(learningmethod)
                if learningmethod == 'rf':
                    estimator = RandomForestClassifier(n_jobs=6, n_estimators=150)
                elif learningmethod == 'svm_rbf':
                    estimator = SVC(kernel='rbf')
                elif learningmethod == 'svm_linear':
                    estimator = LinearSVC()
                elif learningmethod == 'xgboost':
                    estimator = XGBClassifier(nthread=6, n_estimators=300, max_depth=4)

                estimator.fit(xtrain, ytrain)

                ypred = estimator.predict_proba(xtest)
                loss = log_loss(ytest, ypred)
                print loss

                s[learningmethod] = loss
            result = result.append(s.T, ignore_index=True)
        result.to_csv('{0}_merge{1}.csv'.format(xpname, merge_method))
        final = pd.DataFrame()
        for c in result.columns:
            tmp = pd.DataFrame()
            tmp['logloss'] = result[c]
            tmp['classifier'] = c
            final = final.append(tmp, ignore_index=True)
        sns.factorplot(x='classifier', y='logloss', data=final)
        sns.plt.savefig('{0}_merge{1}.pdf'.format(xpname, merge_method), bbox_inches='tight')

    else:
        print 'Test submission'
        if learningmethod == 'rf':
            estimator = RandomForestClassifier(n_jobs=6, n_estimators=150)
        elif learningmethod == 'svm_rbf':
            estimator = SVC(kernel='rbf')
        elif learningmethod == 'svm_linear':
            estimator = SVR(kernel='linear')
        elif learningmethod == 'xgboost':
            estimator = XGBClassifier(nthread=6, n_estimators=300, max_depth=4)
        print 'fitting a %s classifier' % learningmethod
        estimator.fit(Xtrain, y)

        result = pd.DataFrame()
        for i in range(1, 21):
            print 'read Test %s' % i
            Xtest = load_questions_and_merge('test_part%s' % (i), method=merge_method)
            dtmp = load_dataset('test_part%s' % i, clean=False)
            test_ids = dtmp['test_id']
            del dtmp
            ypred = estimator.predict_proba(Xtest)
            resulttmp = pd.DataFrame()
            resulttmp['test_id'] = test_ids
            resulttmp['is_duplicate'] = ypred[:, 1]
            result = result.append(resulttmp, ignore_index=True)
        result.to_csv(xpname + '_merge{0}_{1}.csv'.format(merge_method, learningmethod), index=False)


def load_questions_and_merge(dataset, method='concat'):
    question1 = pd.read_csv(
        '/media/nacim/a13ff970-2886-43fb-b752-f1a813a76a12/data/{0}_question1.csv'.format(dataset)).replace(np.nan,
                                                                                                            0).replace(
        np.inf, 0).replace(-np.inf, 0).astype(np.float32)
    question2 = pd.read_csv(
        '/media/nacim/a13ff970-2886-43fb-b752-f1a813a76a12/data/{0}_question2.csv'.format(dataset)).replace(np.nan,
                                                                                                            0).replace(
        np.inf, 0).replace(-np.inf, 0).astype(np.float32)

    if method == 'concat':
        questions = pd.concat([question1, question2], axis=1)
        print questions.shape
    elif method == 'substract':
        questions = np.abs(question1 - question2)
    elif method == 'add':
        questions = np.abs(question1 + question2)
    elif method == 'multiply':
        questions = np.abs(question1 - question2)
    return np.array(questions)


def extracted_features_method_classifier_polynomial_features(crossval=True, xpname='cross_validation_classifier',
                                                             oncleaned_data=False,
                                                             learningmethod='xgboost'):
    pol = PolynomialFeatures()
    print('Loading Training Dataset')
    col = [u'euclidean_distance',
           u'fuzz_token_sort_ratio',
           u'fuzz_partial_token_set_ratio', u'canberra_distance', u'skew_q1vec',
           u'kur_q1vec', u'norm_wmd_train', u'wmd_train',
           u'tfidf_word_match_train', u'fuzz_token_set_ratio',
           u'braycurtis_distance', u'fuzz_partial_ratio', u'minkowski_distance',
           u'fuzz_qratio', u'fuzz_wratio', u'cosine_distance',
           u'fuzz_partial_token_sort_ratio', u'jaccard_distance',
           u'word_match_train',
           # u'skew_q2vec', u'kur_q2vec'
           ]
    # col = [
    #     u'euclidean_distance',
    #        u'fuzz_token_sort_ratio',
    #        u'fuzz_partial_token_set_ratio', u'canberra_distance', u'skew_q1vec',
    #        u'kur_q1vec', u'norm_wmd_train', u'wmd_train',
    #        u'tfidf_word_match_train', u'fuzz_token_set_ratio',
    #        u'braycurtis_distance', u'fuzz_partial_ratio', u'minkowski_distance',
    #        u'fuzz_qratio', u'fuzz_wratio', u'cosine_distance',
    #        u'fuzz_partial_token_sort_ratio', u'jaccard_distance',
    #        u'word_match_train', u'skew_q2vec', u'kur_q2vec']
    # df = load_dataset('train', clean=oncleaned_data)
    df = pd.read_csv('/home/nacim/DATASET_KAGGLE/quora/train.csv')
    Xtrain = load_dataset('train_extracted_features').replace('', 0).replace(np.nan, 0).replace(np.inf, 0).replace(
        -np.inf, 0).astype(np.float32)

    y = df['is_duplicate'].values
    Xtrain = np.array(Xtrain[col])
    # Xtrain = poly.fit_transform(Xtrain)
    Xtrain = pol.fit_transform(Xtrain)


    del df

    if crossval:
        sss = StratifiedShuffleSplit(y=y, n_iter=5, test_size=0.2, )

        result = pd.DataFrame()
        for train, test in sss:
            xtrain, xtest, ytrain, ytest = Xtrain[train], Xtrain[test], y[train], y[test]

            xtrain, ytrain = oversample(ssp.csr_matrix(xtrain), ytrain, p=0.165)
            # xtest = ssp.csr_matrix(xtest)

            # dump_svmlight_file(xtrain,ytrain,path='/')
            s = pd.Series()
            for learningmethod in [
                # 'svm_linear',
                # 'svm_rbf',
                'xgboost',
                'rf',
            ]:
                print(learningmethod)
                if learningmethod == 'rf':
                    estimator = RandomForestClassifier(n_jobs=6, n_estimators=100)
                    estimator.fit(xtrain, ytrain)
                    ypred = estimator.predict_proba(xtest)
                    loss = log_loss(ytest, ypred)
                    print loss
                    exit()
                    s[learningmethod] = loss
                elif learningmethod == 'svm_rbf':
                    estimator = SVC(kernel='rbf')
                elif learningmethod == 'svm_linear':
                    estimator = LinearSVC()
                elif learningmethod == 'xgboost':
                    estimator = XGBClassifier(nthread=4, n_estimators=350, max_depth=4)
                    gpu_params = {
                        'objective': 'binary:logistic',
                        'eval_metric': 'logloss',
                        'eta': 0.02,
                        'max_depth': 4,
                        'min_child_weight': 1,
                        'subsample': 0.8,
                        'colsample_bytree': 0.8,
                        # 'updater': 'grow_gpu',
                        'n_estimators': 300,
                        'scale_pos_weight': 1
                    }
                    D_training = xgboost.DMatrix(xtrain, label=ytrain)
                    D_validation = xgboost.DMatrix(xtest, label=ytest)
                    watchlist = [(D_training, 'training'), (D_validation, 'validation')]

                    bst = xgboost.train(gpu_params, D_training, 50000, watchlist, early_stopping_rounds=10000)
                    ypred = bst.predict(D_validation)
                    print ypred
                    exit()

                    #

            result = result.append(s.T, ignore_index=True)
        result.to_csv('{0}_clean{1}.csv'.format(xpname, oncleaned_data))
        final = pd.DataFrame()
        for c in result.columns:
            tmp = pd.DataFrame()
            tmp['logloss'] = result[c]
            tmp['classifier'] = c
            final = final.append(tmp, ignore_index=True)
        sns.factorplot(x='classifier', y='logloss', data=final)
        sns.plt.savefig('{0}_clean{1}.pdf'.format(xpname, oncleaned_data), bbox_inches='tight')

    else:
        print 'Test submission'
        if learningmethod == 'rf':
            estimator = RandomForestClassifier(n_jobs=5, n_estimators=150)
        elif learningmethod == 'xgboost':
            estimator = XGBClassifier(nthread=4, n_estimators=400, max_depth=5)
        # print 'fitting a %s classifier' % learningmethod
        # estimator.fit(Xtrain, y)

        if learningmethod == 'DL':
            model = Sequential()
            model.add(Dense(1024, input_dim=Xtrain.shape[1],kernel_initializer='normal', activation='sigmoid',
                            bias_initializer='random_normal'
                            ))
            # model.add(Dropout(0.3))
            # model.add(Dense(512, kernel_initializer='normal', activation='relu'))
            model.add(Dense(1, kernel_initializer='normal', activation='sigmoid'))
            # sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)

            model.compile(loss='binary_crossentropy', optimizer='nadam', metrics=['accuracy'])

            estimator = model
            y, Xtrain, _ = unison_shuffled_copies(y, Xtrain, np.zeros(len(y)))
            hist = estimator.fit(Xtrain, y, batch_size=1024, epochs=100, validation_split=0.2, shuffle=True,
                                 class_weight={0: 0.79264156344230219, 1: 1.3542873987525375},
                                 verbose=1,
                                 )
            plot_training(hist, 'training_NN_extracted_features___relu')

        result = pd.DataFrame()
        for i in range(1, 21):
            print 'read Test %s' % i
            df = load_dataset('test_part%s_extracted_features' % i, clean=oncleaned_data).replace('', 0).replace(np.nan,
                                                                                                                 0).replace(
                np.inf, 0).replace(-np.inf, 0).astype(np.float32)
            dtmp = load_dataset('test_part%s' % i, clean=oncleaned_data)
            test_ids = dtmp['test_id']
            del dtmp
            Xtest = np.array(df[col])
            Xtest = pol.transform(Xtest)
            del df
            ypred = estimator.predict_proba(Xtest)
            # print ypred
            # exit()
            resulttmp = pd.DataFrame()
            resulttmp['test_id'] = test_ids
            if learningmethod == 'DL':
                resulttmp['is_duplicate'] = ypred.ravel()
            else:
                resulttmp['is_duplicate'] = ypred[:, 1]

            result = result.append(resulttmp, ignore_index=True)
        result.to_csv(xpname + '_clean_____relu{0}_{1}.csv'.format(oncleaned_data, learningmethod), index=False)

def unison_shuffled_copies(y, q1,q2):
    assert len(y) == len(q1) == len(q2)
    p = np.random.permutation(len(y))
    return y[p],q1[p],q2[p]
def plot_training(hist, model_weights_file):
    plt.plot(hist.history['acc'], label='accuracy')
    plt.plot(hist.history['val_acc'], label='val accurayc')
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    # summarize history for loss
    plt.plot(hist.history['loss'], label='loss')
    plt.plot(hist.history['val_loss'], label='val loss')
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend()
    plt.savefig(model_weights_file + '.pdf', bbox_inches='tight')


if __name__ == '__main__':
    # naive_method_classifier(crossval=True,xpname='cross_validation_classifier',oncleaned_data=False,learningmethod='xgboost')
    # naive_method_classifier(crossval=False, xpname='results/submission_classifier300features', oncleaned_data=False, learningmethod='xgboost',maxfeature=300)
    # naive_method_classifier(crossval=False, xpname='results/submission_classifier300features', oncleaned_data=True, learningmethod='xgboost',maxfeature=300)
    # naive_method_classifier(crossval=True,xpname='cross_validation_classifier',oncleaned_data=True,learningmethod='xgboost',maxfeature=100)

    # extracted_features_method_classifier(crossval=False,xpname='results/submission_classifier_extracted_features',learningmethod='xgboost')
    # extracted_features_method_classifier(crossval=False,xpname='results/submission_classifier_extracted_features',learningmethod='rf')
    # from_question_representation_method_classifier(crossval=True,xpname='cross_validation_classifier_representation',merge_method='concat')
    # from_question_representation_method_classifier(crossval=True,xpname='cross_validation_classifier_representation',merge_method='substract')
    # from_question_representation_method_classifier(crossval=True,xpname='cross_validation_classifier_representation',merge_method='add')
    # from_question_representation_method_classifier(crossval=True,xpname='cross_validation_classifier_representation',merge_method='multiply')

    # print '======================================'
    # print 'SUBMISSION'
    # from_question_representation_method_classifier(crossval=False,xpname='results/submission_classifier_representation',learningmethod='rf',merge_method='concat')
    # from_question_representation_method_classifier(crossval=False,xpname='results/submission_classifier_representation',learningmethod='rf',merge_method='substract')
    # from_question_representation_method_classifier(crossval=False,xpname='results/submission_classifier_representation',learningmethod='rf',merge_method='multiply')
    # from_question_representation_method_classifier(crossval=False,xpname='results/submission_classifier_representation',learningmethod='xgboost',merge_method='concat')
    # from_question_representation_method_classifier(crossval=False,xpname='results/submission_classifier_representation',learningmethod='xgboost',merge_method='substract')
    # from_question_representation_method_classifier(crossval=False,xpname='results/submission_classifier_representation',learningmethod='xgboost',merge_method='multiply')


    extracted_features_method_classifier_polynomial_features(crossval=False,
                                                             xpname='cross_validation_classifier_extracted_features_final_DL_deepreluthensig',
                                                             learningmethod='xgboosT')
