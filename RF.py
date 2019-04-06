import numpy as np
import pickle
from sklearn.preprocessing import LabelEncoder
import assistCNN
import statystyki
from sklearn.ensemble import RandomForestClassifier
import time
import csv
# filename='rf_model.sav'
# model = pickle.load(open(filename, 'rb'))
dane= 'training.pickle'

czasy=[]
for i in range(1):
    dane_test=assistCNN.podzial_klas_na_2(dane)
    x_trening=dane_test[0]['ekstrakcja']
    x_trening=assistCNN.np.array(x_trening)
    x_test=dane_test[1]['ekstrakcja']
    #x_wal=dane_test[2]['ekstrakcja']
    x_test=assistCNN.np.array(x_test)
    #x_wal=assistCNN.np.array(x_wal)
    y_trening=dane_test[0]['klasa']
    y_test=dane_test[1]['klasa']
    #y_wal=dane_test[2]['klasa']

    X_trening=np.zeros((x_trening.shape[0], 430, ))
    X_test=np.zeros((x_test.shape[0], 430, ))
    #X_wal=np.zeros((x_wal.shape[0], 430, ))

    for nr, tabela in enumerate(x_trening):
        tabela = np.array(tabela)
        X_trening[nr]=tabela

    for nr, tabela in enumerate(x_test):
        tabela = np.array(tabela)
        X_test[nr]=tabela

    #for nr, tabela in enumerate(x_wal):
    #    tabela = np.array(tabela)
    #             X_wal[nr]=tabela

    labels=[y_trening, y_test]
    labels_en=assistCNN.label_encod(labels)
    klasy=np.unique(y_trening)

    Ytr = labels_en[0]
    Yte = labels_en[1]
    #Ywa = labels_en[2]
    data_to_predict=[X_test]

    rf=RandomForestClassifier(n_estimators=150, oob_score=True)
    start=time.time()
    clf_rf= rf.fit(X_trening, Ytr)
    end=time.time()
    delta=end-start
    czasy.append(delta)
    print('Delta time = {}'.format(delta))
    #numer modelu
    nr = 13

    for nr, X in enumerate(data_to_predict):
        Ypred = clf_rf.predict(X)
        y= nr + 1
        statystyki.macierz_bledow_wykres(labels_en[y], Ypred, 17, klasy, 'macierzRF_'+ str(nr)+ str(nr)+'_'+str(i))
        statystyki.podstawowe_statystyki(labels_en[y], Ypred, klasy, 'statystykiRF' + str(nr)+ str(nr)+'_'+str(i))
        statystyki.raport(labels_en[y], Ypred, klasy, 'raportRF'+str(nr)+ str(nr)+'_'+str(i))
        statystyki.macierz_bledow(labels_en[y], Ypred, klasy, 'RF_macierz'+str(nr)+ str(nr)+'_'+str(i))
    filename_rf = ('rf_model{0}{1}.sav').format(nr, i)
    pickle.dump(clf_rf, open(filename_rf, 'wb'))

with open('time.csv', "w") as f:
    writer=csv.writer(f, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
    for c in czasy:
        writer(c)