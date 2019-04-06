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
from sklearn.model_selection import GridSearchCV

# wczytanie danych do pandas df
dane_pd = pd.read_pickle('ekstrakcja_przyklad_clean.pickle')
ekstrakcja = dane_pd['ekstrakcja']
klasy = dane_pd['klasa']
indeksy = dane_pd['indeks']
klasy_naglowki = np.unique(klasy)
print(list(dane_pd))


trening = []
test = []
walidacja = []
indeksy_unique = np.unique(dane_pd['indeks'])
poligony_klasy = []

print(dane_pd.klasa)
print(dane_pd[dane_pd.klasa == 'zabudowa'])

# podział na klasy zbiorów uczących (do stratified random sampling)
#jaka_klasa = [[] for klasa in klasy_naglowki]
jaka_klasa = np.zeros((len(klasy_naglowki), dane_pd.shape[1], dane_pd.shape[0]))
print(jaka_klasa.shape)
for n, nazwa in enumerate(klasy_naglowki):
    print(nazwa)
    dane_select = (dane_pd[dane_pd.klasa == nazwa])
    for m, row in dane_select.iterrows():
        print(row)
        # jaka_klasa[n][m] = row
    # for m, col in dane_pd[1]:
    #     dane_dla_klasy = dane_pd[dane_pd['klasa'] == nazwa]
    #     jaka_klasa[1] = (dane_dla_klasy)


print([[list(y) for y in x] for x in jaka_klasa])

# podział zbioru uczącego na 3 części - trening, test, walidacje
for n, element in enumerate(jaka_klasa):
    print(len(element))
    unikalne = np.unique(element[3])
    kopia = copy.deepcopy(unikalne)
    print('kopia \n \n \n', kopia)
    slice = int(len(kopia) / 3)
    random.shuffle(kopia)
    trening.append(kopia[:slice])
    test.append((kopia[slice:(2 * slice)]))
    walidacja.append((kopia[(2 * slice):]))
f = open('test.txt', "w")
f.write(('trening:', trening, '\n', 'test:', test, '\n', 'walidacja:', walidacja))
tr_te_wal = []
tr_te_wal.append(trening)
tr_te_wal.append(test)
tr_te_wal.append(walidacja)

# lista na wylosowane zestawy treningowe, testowe i uczące
dane_do_treningu = [[], [], []]

for nr, element in enumerate(tr_te_wal):
    for i in element:
        for el in i.flatten():
            dane_do_treningu[nr].append(el)
print(len(dane_do_treningu[0]), dane_do_treningu[1], dane_do_treningu[2])

# creating a boolean vector to evaluate if data from df is in training/test/validation index list
dane_trening_bool = dane_pd['indeks'].isin(dane_do_treningu[0])
dane_test_bool = dane_pd['indeks'].isin(dane_do_treningu[1])
dane_walidacja_bool = dane_pd['indeks'].isin(dane_do_treningu[2])
print(dane_trening_bool.value_counts())

#subsetting by boolean series
dane_trening=dane_pd[dane_trening_bool]
print('Ilość wzorców w danych treningowych:', len(dane_trening.index))
print(dane_trening)
dane_test=dane_pd[dane_test_bool]
print(dane_test)
print('Ilość wzorców w danych testowych:', len(dane_test.index))
dane_waldiacja=dane_pd[dane_walidacja_bool]
print('Ilość wzorców w danych walidacyjnych:', len(dane_waldiacja.index))
print(dane_waldiacja)

#utworzenie obiektu label encoder
le=LabelEncoder()

#zakodowanie nazw klas do postaci liczbowej
klasy_nagl_kod=le.fit(klasy_naglowki)

#przekształcenie danych treningowej do macierzy wielowymiarowej (aby mogła być wejściem do klasyfikacji)
X_trening=np.array([dane_trening['ekstrakcja']])
X_trening=np.array(X_trening).reshape((X_trening.shape[1], X_trening.shape[2]))

#przekształcenie listy etykiet danych treningowych do macierzy numpy array
y_trening=np.array(dane_trening['klasa'])
#zakodowanie nazw treningowych do formy liczbowej
y_trening_en=le.transform(y_trening)

#przekształcenie danych testowych do macierzy wielowymiarowej (aby mogła być wejściem do klasyfikacji)
X_test=np.array([dane_test['ekstrakcja']])
X_test=np.array([dane_test['ekstrakcja']]).reshape((X_test.shape[1], X_test.shape[2]))

#przekształcenie listy etykiet danych testowych do macierzy numpy array
y_test=dane_test['klasa']
#zakodowanie nazw treningowych do formy liczbowej
y_test_en=le.transform(y_test).reshape(-1, 1)

#przekształcenie danych weryfikacyjnych do macierzy wielowymiarowej (aby mogła być wejściem do klasyfikacji)
X_walidacja=np.array([dane_waldiacja['ekstrakcja']])
X_walidacja=np.array([dane_waldiacja['ekstrakcja']]).reshape((X_walidacja.shape[1], X_walidacja.shape[2]))

#przekształcenie listy etykiet danych weryfikacyjnych do macierzy numpy array
y_walidacja=dane_waldiacja['klasa']
#zakodowanie nazw treningowych do formy liczbowej
y_walidacja_en=le.transform(y_walidacja).reshape(-1, 1)

###########klasyfikacja - opracowanie modelu
#należy nazwać model SVM
nazwa_modelu_svm='SVM_CV'
nazwa_modelu_RF='RF_500_OOB'

#RF
# rf=RandomForestClassifier(n_estimators=500, oob_score=True)
# clf_rf= rf.fit(X_trening, y_trening_en)

#SVM grid search
gamma_range=[0.001, 0.01, 0.1, 1]
parameters = {'kernel':('poly', 'rbf'), 'C':[0.01, 0.1, 1, 10], 'gamma':[0.001, 0.01, 0.1, 1]}
clf = GridSearchCV(svm.SVC(), parameters, cv=2)
print('Parameters found.')
grid_search_best=clf.cv_results_
print(grid_search_best)
clf.fit(X_trening, y_trening_en)
print('Model fit.')

#plot.grid_search(grid_search_best, change=('n_estimators', 'criterion'),
                # subset={'max_features': 'sqrt'})

# Prediction on
y_pred=clf.predict(X_test)

#zapis modelu

filename_rf = 'rf_model.sav'
filename_svm='svm_model.sav'
pickle.dump(clf_rf, open(filename_svm, 'wb'))

#Opracowanie danych; wszystkie statystyki zostaną wygenerowane w folderze roboczym

#eksport raportu, zawierającego podstawowe statystyki dotyczące klasyfikacji dla każdej z klas
raport = (statystyki.raport(y_test_en, y_pred, klasy_naglowki, nazwa_modelu_RF))

#eksport raportu, zawierającego podstawowe statystyki dotyczące klasyfikacji
podstawowe_statystyki = (statystyki.podstawowe_statystyki(y_test_en, y_pred, klasy_naglowki, nazwa_modelu_RF))

#eksport macierzy błędów txt
statystyki.macierz_bledow(y_test_en, y_pred, klasy_naglowki, nazwa_modelu_RF)

#eksport macierzy błędów w formie graficznej
statystyki.macierz_bledow_wykres(y_test_en, y_pred, 5, klasy_naglowki, nazwa_modelu_RF)

#eksport wykresu z przebiegiem procesu uczenia
statystyki.learning_curve(clf_rf, X_trening, y_trening, nazwa_modelu_RF)

#wykres rozrzutu danych dla dwóch dowolnych kanałów
statystyki.rozrzut_danych(X_trening, y_trening_en, nazwa_modelu_RF, 8, 29)



