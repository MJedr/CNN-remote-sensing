import numpy as np
import pandas as pd
import copy
import random
import os
import matplotlib.pyplot as plt
import statystyki
import pickle
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
import assistCNN
import csv
import time
dane= 'training.pickle'

dane_test=assistCNN.podzial_klas_na_2(dane)
x_trening=dane_test[0]['ekstrakcja']
x_trening=assistCNN.np.array(x_trening)
x_test=dane_test[1]['ekstrakcja']
# x_wal=dane_test[2]['ekstrakcja']
# x_test=assistCNN.np.array(x_test)
# x_wal=assistCNN.np.array(x_wal)
y_trening=dane_test[0]['klasa']
y_test=dane_test[1]['klasa']
#y_wal=dane_test[2]['klasa']

#wymiary zależą od liczby kanałów!!! MNF-30, pełne dane -430
X_trening=np.zeros((x_trening.shape[0], 430, ))
X_test=np.zeros((x_test.shape[0], 430, ))
# X_wal=np.zeros((x_wal.shape[0], 430, ))

for nr, tabela in enumerate(x_trening):
    tabela = np.array(tabela)
    X_trening[nr]=tabela

for nr, tabela in enumerate(x_test):
    tabela = np.array(tabela)
    X_test[nr]=tabela
#
# for nr, tabela in enumerate(x_wal):
#     tabela = np.array(tabela)
#     X_wal[nr]=tabela

labels=[y_trening, y_test]
labels_en=assistCNN.label_encod(labels)
klasy=np.unique(y_trening)

Ytr = labels_en[0]
Yte = labels_en[1]
#Ywa = labels_en[2]
data_to_predict=[X_test]
###########klasyfikacja - opracowanie modelu
#należy nazwać model SVM
# nazwa_modelu_svm='SVM_RBF'
# nazwa_modelu_RF='RF_500_OOB'

#RF
# rf=RandomForestClassifier(n_estimators=500, oob_score=True)
# clf_rf= rf.fit(X_trening, y_trening_en)

#SVM grid search
# gamma_range=[0.001, 0.01, 0.1, 1]
# parameters = {'kernel':('poly', 'rbf'), 'C':[0.01, 0.1, 1, 10], 'gamma':[0.001, 0.01, 0.1, 1]}
# clf = GridSearchCV(svm.SVC(), parameters, cv=2)
# print('Parameters found.')
#clf = svm.SVC(gamma = 0.000001, kernel = 'rbf', C = 150)
czasy=[]
for i in range(10):
    dane_test = assistCNN.podzial_klas_na_2(dane)
    x_trening = dane_test[0]['ekstrakcja']
    x_trening = assistCNN.np.array(x_trening)
    x_test = dane_test[1]['ekstrakcja']
    # x_wal=dane_test[2]['ekstrakcja']
    # x_test=assistCNN.np.array(x_test)
    # x_wal=assistCNN.np.array(x_wal)
    y_trening = dane_test[0]['klasa']
    y_test = dane_test[1]['klasa']
    # y_wal=dane_test[2]['klasa']

    # wymiary zależą od liczby kanałów!!! MNF-30, pełne dane -430
    X_trening = np.zeros((x_trening.shape[0], 430,))
    X_test = np.zeros((x_test.shape[0], 430,))
    # X_wal=np.zeros((x_wal.shape[0], 430, ))

    for nr, tabela in enumerate(x_trening):
        tabela = np.array(tabela)
        X_trening[nr] = tabela

    for nr, tabela in enumerate(x_test):
        tabela = np.array(tabela)
        X_test[nr] = tabela
    #
    # for nr, tabela in enumerate(x_wal):
    #     tabela = np.array(tabela)
    #     X_wal[nr]=tabela

    labels = [y_trening, y_test]
    labels_en = assistCNN.label_encod(labels)
    klasy = np.unique(y_trening)

    Ytr = labels_en[0]
    Yte = labels_en[1]
    # Ywa = labels_en[2]
    data_to_predict = [X_test]
    ###########klasyfikacja - opracowanie modelu
    # należy nazwać model SVM
    # nazwa_modelu_svm='SVM_RBF'
    # nazwa_modelu_RF='RF_500_OOB'

    # RF
    # rf=RandomForestClassifier(n_estimators=500, oob_score=True)
    # clf_rf= rf.fit(X_trening, y_trening_en)

    # SVM grid search
    # gamma_range=[0.001, 0.01, 0.1, 1]
    # parameters = {'kernel':('poly', 'rbf'), 'C':[0.01, 0.1, 1, 10], 'gamma':[0.001, 0.01, 0.1, 1]}
    # clf = GridSearchCV(svm.SVC(), parameters, cv=2)
    # print('Parameters found.')
    # clf = svm.SVC(gamma = 0.000001, kernel = 'rbf', C = 150)

    clf = svm.SVC(random_state= 1, kernel='rbf', gamma = 0.00000001, C=150)
    # random_params = {'kernel':np.array(['rbf', 'linear']),
    #                  'C':np.arange(100, 200, 5),
    #                  'gamma':np.array([0.0000001, 0.000001, 0.000001])}
    # rand_search = RandomizedSearchCV(clf, param_distributions = random_params, n_iter = 100, n_jobs = 4, cv = 3)
    # print('parameters found')
    # clf_svm = rand_search.fit(X_trening, Ytr)
    start=time.time()
    clf_svm=clf.fit(X_trening, Ytr)
    end=time.time()
    delta=end-start
    print('Delta time = {}'.format(delta))
    czasy.append(delta)
    # print('Best Penalty:', clf_svm.best_estimator_.get_params()['penalty'])
    # print('Best C:', clf_svm.best_estimator_.get_params()['C'])
    print('Model fit.')
    # grid_search_best=clf.cv_results_
    # print(grid_search_best)
    # Prediction on

    #zapis modelu

    filename_rf = 'rf_model' + str(i) + '.sav'
    filename_svm='svm_model' + str(i) + '.sav'
    pickle.dump(clf_svm, open(filename_svm, 'wb'))

    for nr, X in enumerate(data_to_predict):
        Ypred = clf_svm.predict(X)
        y= nr +1
        statystyki.macierz_bledow_wykres(labels_en[y], Ypred, 17, klasy, 'macierzSVM_'+ str(nr)+'_'+str(i))
        statystyki.podstawowe_statystyki(labels_en[y], Ypred, klasy, 'statystykiSVM' + str(nr)+'_'+str(i))
        statystyki.raport(labels_en[y], Ypred, klasy, 'raportSVM'+str(nr)+'_'+str(i))
        statystyki.macierz_bledow(labels_en[y], Ypred, klasy, 'SVM_macierz'+str(nr)+'_'+str(i))

with open('time.csv', "w") as f:
    writer=csv.writer(f, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
    for c in czasy:
        writer(c)