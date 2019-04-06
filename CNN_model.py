# -*- coding: utf-8 -*-
"""
Created on Sun Sep 30 11:10:57 2018

@author: cysia
"""


import numpy as np
import pandas as pd
import os
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


dane= r'data\trainingALL.pickle'


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

earlystop = EarlyStopping(monitor='val_acc', patience=5,
                          mode = 'auto')

dense = {
    "dense_1" : [512, 256, 128],
    "dense_2": [512, 256, 128]
}
sgd = SGD(lr=0.001, nesterov=True, decay=1e-6, momentum=0.9)
adam = Adam(lr=0.00001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
adadelta = Adadelta(lr=1.0, rho=0.95, epsilon=None, decay=0.0)
rms = RMSprop(lr=0.001, rho=0.9, epsilon=None, decay=0.0)

hyperparameters = {
    'lr': [0.1, 0.01, 0.001],
    'optimizer': ['adam', sgd, 'adadelta']
}

czasy=[]
for i in range(len(dense.get('dense_1'))):
    for j in range(len(dense.get('dense_2'))):
            dane_test=assistCNN.podzial_klas_na_2(dane)

            dt1=dane_test[0][dane_test[0].klasa!='6510']
            dt2=dane_test[1][dane_test[1].klasa!='6510']

            x_trening=dt1['ekstrakcja']
            x_trening=assistCNN.np.array(x_trening)
            x_test=dt2['ekstrakcja']
            y_trening=dt1['klasa']
            y_test=dt2['klasa']

            #UWAGA - zależne od wymiaru
            X_trening=np.zeros((x_trening.shape[0], 430, 1))
            X_test=np.zeros((x_test.shape[0], 430, 1))

            for nr, tabela in enumerate(x_trening):
                tabela = np.array(tabela).reshape(430,1)
                X_trening[nr]=tabela

            for nr, tabela in enumerate(x_test):
                tabela = np.array(tabela).reshape(430,1)
                X_test[nr]=tabela

            labels=[y_trening, y_test]
            labels_en=assistCNN.label_encod(labels)
            klasy=np.unique(y_trening)

            Ytr = labels_en[0]
            Yte = labels_en[1]


            model = Sequential()
            #First Conv and Pooling lyr
            model.add(Conv1D(32, 3,
                             input_shape=(X_trening.shape[1], 1),
                             kernel_initializer=he_normal(seed=12),
                             activation='relu',
                             W_regularizer=l1_l2(0.01))
                      )
            model.add(BatchNormalization())
            # model.add(MaxPooling1D(5))
            # model.add(Dropout(0.2))
            # #Second Conv and Pooling lyr
            model.add(Conv1D(32,16,
                      activation='relu',
                             W_regularizer=l1_l2(0.01),
                             padding='same'))
           #  model.add(MaxPooling1D(5))
            model.add(BatchNormalization())
            # model.add(Dropout(0.2))
            #
            # # #Third Conv and Pooling lyr
            model.add(Conv1D(32,3,
                      activation='relu',
                             W_regularizer=l1_l2(0.01),
                             padding='same'))
           #  model.add(MaxPooling1D(5))
            model.add(BatchNormalization())
            # model.add(Dropout(0.2))
            # #
            # # #
            # #Fourth Conv and Pooling lyr
            model.add(Conv1D(32,3,
                      activation='relu',
                             W_regularizer=l1_l2(0.01),
                             padding='same'))
            # model.add(MaxPooling1D(5))
            model.add(BatchNormalization())
            # model.add(Dropout(0.2))
            #Fully connected lyrs
            model.add(Flatten())
            model.add(Dense(dense.get('dense_1')[i],activation = 'relu'))
            # model.add(BatchNormalization())
            model.add(Dropout(0.2))
            model.add(Dense(dense.get('dense_2')[j],activation = 'relu'))
            # model.add(BatchNormalization())
            model.add(Dropout(0.2))
            model.add(Dense(16,activation = 'softmax', input_shape=(1,)))
            model.compile(optimizer = adam, loss = 'sparse_categorical_crossentropy',
                          metrics =['accuracy'])

            start=time.time()
            history = model.fit(X_trening, Ytr, batch_size = 256, epochs = 100,
                      validation_data=(X_test, Yte), callbacks = [metrics,earlystop])

            model.predict(X_test)
            end=time.time()
            delta=end-start
            czasy.append(delta)
            # save model to JSON
            model_json = model.to_json()

            Ypred = model.predict_classes(X_test)
            # save model to JSON
            # model_json = model.to_json()
            # serialize weights to HDF5
            model.save_weights(str(i)+"model_11.h5")
            print("Saved model to disk")

            #numer modelu
            nr = 1
            statystyki.macierz_bledow_wykres(Yte, Ypred, 16, klasy, 'macierz_'+ '_d1'+str(i) + '_d2'+str(j))
            statystyki.podstawowe_statystyki(Yte, Ypred, klasy, 'statystyki' + '_d1'+str(i) + '_d2'+str(j))
            statystyki.raport(Yte, Ypred, klasy, 'raport'+str(nr)+ '_'+ '_d1'+str(i) + '_d2'+str(j))
            statystyki.macierz_bledow(Yte, Ypred, klasy, 'CNN_macierz'+'_d1'+str(i) + '_d2'+str(j))
            #
            # with open("model_cnn_11.json", "w") as json_file:
            #     json_file.write(model_json)


    print(delta)

cz=pd.DataFrame(czasy)
cz.to_csv('czasyCNN.csv')
