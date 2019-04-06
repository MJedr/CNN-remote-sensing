
# -*- coding: utf-8 -*-
"""
Created on Sun Sep 30 11:10:57 2018

@author: cysia
"""
import numpy as np
import pandas as pd
import os

from keras.engine.saving import model_from_json

import assistCNN
from keras.models import Sequential
from keras.layers.advanced_activations import LeakyReLU
from keras.layers import Conv1D, Conv2D, Flatten, Dense, Embedding, GlobalMaxPool1D, Dropout, LocallyConnected1D, MaxPooling1D, BatchNormalization
import matplotlib.pyplot as plt
from keras.optimizers import SGD, Adam, Adadelta, RMSprop
from keras.callbacks import Callback
from keras.regularizers import l1, l2, l1_l2
from keras import regularizers
from keras.metrics import categorical_accuracy
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score
from keras.initializers import he_uniform, he_normal
from keras.callbacks import EarlyStopping, CSVLogger, RemoteMonitor, ModelCheckpoint
from keras.losses import categorical_crossentropy
import statystyki
import time


dane= 'data\\trainMNF.pickle'


#callbacks
mcp = ModelCheckpoint('best_weights', monitor='val_loss', save_best_only=True, save_weights_only=False)
csv_logger = CSVLogger('training.log')
rm = RemoteMonitor(root='http://localhost:9000', field='data',
                   headers=None, send_as_json=False)

#dane_obraz=funkcje.open_envi_array(dane_raster)
class PlotLosses(Callback):
    def on_train_begin(self, logs={}):
        self.i = 0
        self.x = []
        self.losses = []
        self.val_losses = []

        self.fig = plt.figure()

        self.logs = []

    def on_epoch_end(self, epoch, logs={}):
        self.logs.append(logs)
        self.x.append(self.i)
        self.losses.append(logs.get('loss'))
        self.val_losses.append(logs.get('val_loss'))
        self.i += 1


        plt.plot(self.x, self.losses, label="loss")
        plt.plot(self.x, self.val_losses, label="val_loss")
        plt.legend()
        plt.show()


plot_losses = PlotLosses()

#solution by Thong Nguyen
class Metrics(Callback):
    def on_train_begin(self, logs={}):
        self.val_f1s = []
        self.val_recalls = []
        self.val_precisions = []


def on_epoch_end(self, epoch, logs={}):
    val_predict = (np.asarray(self.model.predict(self.model.validation_data[0]))).round()
    val_targ = self.model.validation_data[1]
    _val_f1 = f1_score(val_targ, val_predict)
    _val_recall = recall_score(val_targ, val_predict)
    _val_precision = precision_score(val_targ, val_predict)
    self.val_f1s.append(_val_f1)
    self.val_recalls.append(_val_recall)
    self.val_precisions.append(_val_precision)
    print('  - val_f1: % f _val_precision: % f _val_recall % f' % (_val_f1, _val_precision, _val_recall))
    return

metrics = Metrics()

earlystop = EarlyStopping(monitor='val_acc', patience=10,
                          mode = 'auto')


dane_test=assistCNN.podzial_klas_na_2(dane)

dt1=dane_test[0][dane_test[0].klasa!='6510']
dt2=dane_test[1][dane_test[1].klasa!='6510']

x_trening=dt1['ekstrakcja']
x_trening=assistCNN.np.array(x_trening)
x_test=dt2['ekstrakcja']
y_trening=dt1['klasa']
y_test=dt2['klasa']

#UWAGA - zależne od wymiaru
X_trening=np.zeros((x_trening.shape[0], 30, 1))
X_test=np.zeros((x_test.shape[0], 30, 1))

for nr, tabela in enumerate(x_trening):
    tabela = np.array(tabela).reshape(30,1)
    X_trening[nr]=tabela

for nr, tabela in enumerate(x_test):
    tabela = np.array(tabela).reshape(30,1)
    X_test[nr]=tabela

labels=[y_trening, y_test]
labels_en=assistCNN.label_encod(labels)
klasy=np.unique(y_trening)

Ytr = labels_en[0]
Yte = labels_en[1]


sgd = SGD(lr=0.001, nesterov=True, decay=1e-6, momentum=0.9)
adam =Adam(lr=0.00005, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
adadelta = Adadelta(lr=1.0, rho=0.95, epsilon=None, decay=0.0)
rms = RMSprop(lr=0.001, rho=0.9, epsilon=None, decay=0.0)

json_file = open('wyniki\\\CNN\\29\\model_cnn_11.json',  encoding="utf8")
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights('wyniki\\\CNN\\29\\model_11.h5')
print("Loaded model from disk")
loaded_model.compile(optimizer = adam, loss = 'sparse_categorical_crossentropy',
              metrics =['accuracy'])

start=time.time()
history = loaded_model.fit(X_trening, Ytr, batch_size = 512, epochs = 100,
          validation_data=(X_test, Yte), callbacks = [metrics, earlystop, csv_logger, rm])

loaded_model.predict(X_test)
end=time.time()
delta=end-start
# save model to JSON
model_json = loaded_model.to_json()

Ypred = loaded_model.predict_classes(X_test)
# save model to JSON
model_json = loaded_model.to_json()
# serialize weights to HDF5
loaded_model.save_weights("model_11.h5")
print("Saved model to disk")

#numer modelu
nr = 1
statystyki.macierz_bledow_wykres(Yte, Ypred, 17, klasy, 'macierz_'+ str(nr))
statystyki.podstawowe_statystyki(Yte, Ypred, klasy, 'statystyki' + str(nr))
statystyki.raport(Yte, Ypred, klasy, 'raport'+str(nr))
statystyki.macierz_bledow(Yte, Ypred, klasy, 'CNN_macierz'+str(nr))

with open("model_cnn_11.json", "w") as json_file:
    json_file.write(model_json)

plt.plot(history.history['acc'])
print(delta)
