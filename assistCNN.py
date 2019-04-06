# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import copy
import random
import os
from sklearn.preprocessing import LabelEncoder
#from keras.utils import to_categorical

def spatial_sample(data_to_split, field, training_data_size):
    unique_index = pd.unique(data_to_split[field])
    training = np.random.choice(unique_index, np.ceil(unique_index.shape[0]*training_data_size).astype('int_'),
                                replace=False)
    test = unique_index[np.isin(unique_index, training)==False]
    return training, test

def spatial_stratified_sample(data, field, class_field_name='klasa', training_data_size = 0.7, save=False):
    print('dividing data ...')
    training_index = np.array([-1])
    test_index = np.array([-1])
    unique_class = pd.unique(data[class_field_name])
    for class_name in unique_class:
        subset = data[data[class_field_name] == class_name]
        sampling = spatial_sample(subset, field, 0.7)
        training_index = np.append(training_index, sampling[0])
        test_index = np.append(test_index, sampling[1])
    training_df = data[np.isin(data[field], training_index)]
    test_df = data[np.isin(data[field], test_index)]

    if save:
        pd.to_pickle(training_df, 'training.pickle')
        pd.to_pickle(test_df, 'test.pickle')

    return training_df, test_df


def dane_do_klasyfikacji(dane_do_przeksztalcenia):
    macierz = np.array(dane_do_przeksztalcenia)
    return macierz


def label_encod(data_to_encode):
    print('encoding data ...')
    le = LabelEncoder()
    data_array = np.array(data_to_encode)
    le_fit = le.fit(np.unique(data_array[0]))
    encoded = []
    print(encoded)

    for nb, i in enumerate(data_to_encode):
        y_encoded = le.transform(data_array[nb])
        encoded.append(y_encoded)

    le_name_mapping = dict(zip(le.classes_, le.transform(le.classes_)))
    print(le_name_mapping)

    return np.array(encoded)

def read_data_CNN(data, value_field, class_field, spatial_index_field):

    df = pd.read_pickle(data)
    dane_test = spatial_stratified_sample(df, spatial_index_field)

    x_trening = dane_test[0][value_field]
    x_test = dane_test[1][value_field]
    y_trening = dane_test[0][class_field]
    y_test = dane_test[1][class_field]

    print('reshaping data ...')
    feature_dim = len(x_trening.values[0])
    X_trening = np.array([np.array(x, dtype='int_').reshape(feature_dim, 1) for x in x_trening.values])
    X_test = np.array([np.array(x, dtype='int_').reshape(feature_dim, 1) for x in x_test.values])

    labels_en = label_encod([y_trening, y_test])
    y_tr = labels_en[0]
    y_te = labels_en[1]

    return X_trening, X_test, y_tr, y_te

def read_data(data, value_field, class_field, spatial_index_field):
    df = pd.read_pickle(data)
    dane_test = spatial_stratified_sample(df, spatial_index_field)

    x_trening = dane_test[0][value_field]
    x_test = dane_test[1][value_field]
    y_trening = dane_test[0][class_field]
    y_test = dane_test[0][class_field]

    X_trening = np.array([np.array(x, dtype='int_') for x in x_trening.values])
    X_test = np.array([np.array(x, dtype='int_') for x in x_test.values])

    labels_en = label_encod([y_trening, y_test])
    y_tr = labels_en[0]
    y_te = labels_en[1]

    return X_trening, X_test, y_tr, y_te