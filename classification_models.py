import csv
import pandas as pd
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, cohen_kappa_score, f1_score
import pickle
from sklearn import svm
import time

def SVM_classifier(X_training, y_training, kernel='rbf',
                   gamma='0.00000001', C=150, degree=1):

    print('Starts fitting model ...')
    if kernel == 'rbf':
        clf = svm.SVC(random_state=1, kernel=kernel, gamma=gamma, C=C)
    elif kernel == 'poly':
        clf = svm.SVC(random_state=1, kernel=kernel, gamma=gamma, C=C, degree=degree)
    else:
        clf = svm.LinearSVC(C=C)
    start = time.time()
    clf_svm = clf.fit(X_training, y_training)
    end = time.time()
    delta = end - start
    print('Model fitted. Fitting time:', delta)
    print('Predicting values ...')

def RF_classifier(X_training, y_training,
                  n_estimators=150, oob_score=True, bootstrap=True):

    print('Starts fitting model ...')
    clf = RandomForestClassifier(n_estimators=n_estimators, oob_score=oob_score,
                                 bootstrap=bootstrap)
    start = time.time()
    clf_svm = clf.fit(X_training, y_training)
    end = time.time()
    delta = end - start
    print('Model fitted. Fitting time:', delta)
    print('Predicting values ...')

def classify(model, X_test, y_test,
                   model_name='model',
                   conf_matrix=False, accuracy_report=False):

    model_name = model_name + '_' + type(model).__name__
    while os.path.isfile(model_name + ".sav"):
        model_name = model_name + str(1)
    start_prediction = time.time()
    y_pred = model.predict(X_test)
    end_prediction = time.time()
    delta_prediction = end_prediction - start_prediction
    print('Test set predicted ...')

    if conf_matrix:
        cm_filename = model_name + '_cm.csv'
        cm = pd.DataFrame(confusion_matrix(y_test, y_pred))
        cm.to_csv(cm_filename)

    if accuracy_report:
        raport_filename = model_name + '_report.csv'
        report = classification_report(y_test, y_pred)
        with open(raport_filename, 'w') as acc_report:
            acc_report.write(report)

    filename_svm = model_name + '.sav'
    pickle.dump(model, open(filename_svm, 'wb'))

    with open(model_name + '.csv', 'a', newline='') as history:
        writer = csv.writer(history, delimiter=';')
        writer.writerow([model_name, accuracy_score(y_test, y_pred),
                      cohen_kappa_score(y_test, y_pred), f1_score(y_test, y_pred, average='weighted'),
                      delta_prediction])

    return y_pred


