# -*- coding: utf-8 -*-
"""
Created on Sun Sep 30 19:46:33 2018
@author: cysia
"""

import numpy as np
import pandas as pd
import pickle
import copy
import random
import os
from sklearn.preprocessing import LabelEncoder
#from keras.utils import to_categorical


def podzial_klas_na_3(dane, field_name='indeks', class_field = 'klasa'):
    # porządkowanie wczytanych danych
    # wczytanie danych do pandas df
    dane_pd = pd.read_pickle(dane)
    trening = []
    test = []
    walidacja = []
    indeksy_unique = np.unique(dane_pd[field_name])
    klasy_naglowki = np.unique(dane_pd[class_field])

    # podział na klasy zbiorów uczących (do stratified random sampling)
    jaka_klasa = []
    for nazwa in klasy_naglowki:
        dane_dla_klasy = dane_pd[dane_pd[class_field] == nazwa]
        jaka_klasa.append(dane_dla_klasy)
    print(np.unique(jaka_klasa[1][field_name]))

    # podział zbioru uczącego na 3 części - trening, test, walidacje
    for n, element in enumerate(jaka_klasa):
        unikalne = np.unique(jaka_klasa[n][field_name])
        kopia = copy.deepcopy(unikalne)
        slice = int(len(kopia) / 3)
        random.shuffle(kopia)
        trening.append(kopia[:slice])
        test.append((kopia[slice:(2 * slice)]))
        walidacja.append((kopia[(2 * slice):]))

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

    # creating a boolean vector to evaluate if data from df is in training/test/validation index list
    dane_trening_bool = dane_pd[field_name].isin(dane_do_treningu[0])
    dane_test_bool = dane_pd[field_name].isin(dane_do_treningu[1])
    dane_walidacja_bool = dane_pd[field_name].isin(dane_do_treningu[2])

    # subsetting by boolean series
    dane_trening = dane_pd[dane_trening_bool]
    print('Ilość wzorców w danych treningowych:', len(dane_trening.index))
    dane_test = dane_pd[dane_test_bool]
    print('Ilość wzorców w danych testowych:', len(dane_test.index))
    dane_walidacja = dane_pd[dane_walidacja_bool]
    print('Ilość wzorców w danych walidacyjnych:', len(dane_walidacja.index))

    zestaw_danych = [dane_trening, dane_test, dane_walidacja]
    return zestaw_danych

def podzial_klas_na_2(dane, field_name='indeks', class_field = 'klasa'):
    # porządkowanie wczytanych danych
    # wczytanie danych do pandas df
    dane_pd = pd.read_pickle(dane)
    trening = []
    test = []
    walidacja = []
    indeksy_unique = np.unique(dane_pd[field_name])
    klasy_naglowki = np.unique(dane_pd[class_field])

    # podział na klasy zbiorów uczących (do stratified random sampling)
    jaka_klasa = []
    for nazwa in klasy_naglowki:
        dane_dla_klasy = dane_pd[dane_pd[class_field] == nazwa]
        jaka_klasa.append(dane_dla_klasy)
    print(np.unique(jaka_klasa[1][field_name]))

    # podział zbioru uczącego na 3 części - trening, test, walidacje
    for n, element in enumerate(jaka_klasa):
        unikalne = np.unique(jaka_klasa[n][field_name])
        kopia = copy.deepcopy(unikalne)
        slice = int(len(kopia) / 3)
        random.shuffle(kopia)
        trening.append(kopia[:2*slice])
        test.append((kopia[2 * slice: ]))

        tr_te = []
        tr_te.append(trening)
        tr_te.append(test)
        tr_te.append(walidacja)

    # lista na wylosowane zestawy treningowe, testowe i uczące
    dane_do_treningu = [[], []]

    for nr, element in enumerate(tr_te):
        for i in element:
            for el in i.flatten():
                dane_do_treningu[nr].append(el)

    # creating a boolean vector to evaluate if data from df is in training/test/validation index list
    dane_trening_bool = dane_pd[field_name].isin(dane_do_treningu[0])
    dane_test_bool = dane_pd[field_name].isin(dane_do_treningu[1])

    # subsetting by boolean series
    dane_trening = dane_pd[dane_trening_bool]
    print('Ilość wzorców w danych treningowych:', len(dane_trening.index))
    dane_test = dane_pd[dane_test_bool]
    print('Ilość wzorców w danych testowych:', len(dane_test.index))


    zestaw_danych = [dane_trening, dane_test]
    return zestaw_danych

def dane_do_klasyfikacji(dane_do_przeksztalcenia):
    macierz = np.array(dane_do_przeksztalcenia)
    return macierz


def label_encod(dane_do_zakod):
    '''
    zakodowuje etykiety danych wejsciowych
    dane_do_rozkod jako lista etykiet dla kazdego zestawu danych
    '''
    le = LabelEncoder()
    print(le.get_params())
    dane_macierz = np.array(dane_do_zakod)
    naglowki = np.unique(dane_macierz[0])
    naglowki_kod = le.fit(naglowki)
    zakodowane = []

    for nr, i in enumerate(dane_do_zakod):
        y_zakodowane = le.transform(dane_macierz[nr])
        zakodowane.append(y_zakodowane)

    le_name_mapping = dict(zip(le.classes_, le.transform(le.classes_)))
    print(le_name_mapping)

    return zakodowane

# def ohe(dane_do_zakod):
#     '''
#     zakodowuje etykiety danych wejsciowych One Hot Encoder
#     dane_do_rozkod jako lista etykiet dla kazdego zestawu danych
#     '''
#     integer_encoded=label_encod(dane_do_zakod)
#     print(integer_encoded)
#     zakodowane=[]
#     for nr, element in enumerate(integer_encoded):
#         encoded = to_categorical(integer_encoded[nr])
#         print(encoded)
#         zakodowane.append(encoded)
#     print(zakodowane)
#     return zakodowane

