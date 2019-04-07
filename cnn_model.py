# -*- coding: utf-8 -*-
import csv
import os
from keras.models import Sequential
from keras.layers import Conv1D, Flatten, Dense, Dropout, \
    MaxPooling1D, BatchNormalization
from keras.optimizers import Adam
from keras.regularizers import l1_l2
from keras.initializers import he_normal
from keras.callbacks import EarlyStopping, CSVLogger
from sklearn.metrics import accuracy_score, cohen_kappa_score, f1_score, \
    confusion_matrix, classification_report
import pandas as pd
import time
import my_callbacks

def CNN_model(X_training, X_test, y_training, y_test, n_epochs=100, batch_size=256,
              model_name='model', history_file='model_accuracies.csv',
              conf_matrix=False, accuracy_report=False):

    while os.path.isfile(model_name + ".h5"):
        model_name = model_name + str(1)

    csv_logger = CSVLogger('model_training.log')
    plot_losses = my_callbacks.PlotLosses()
    metrics = my_callbacks.Metrics()
    f1_accuracy = my_callbacks.F1Metric()
    earlystop = EarlyStopping(monitor='val_acc', patience=10,
                              mode='auto')
    adam = Adam(lr=0.00001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)

    model = Sequential()
    model.add(Conv1D(32, 9,
                     input_shape=(X_training.shape[1], 1),
                     kernel_initializer=he_normal(seed=12),
                     activation='relu',
                     W_regularizer=l1_l2(0.01)))
    model.add(BatchNormalization())
    model.add(MaxPooling1D(1))
    model.add(Conv1D(32, 3,
                     activation='relu',
                     W_regularizer=l1_l2(0.01),
                     padding='same'))
    model.add(MaxPooling1D(3, padding='same'))
    model.add(BatchNormalization())
    model.add(Conv1D(9, 3,
                     activation='relu',
                     W_regularizer=l1_l2(0.01),
                     padding='same'))
    model.add(MaxPooling1D(3, padding='same'))
    model.add(BatchNormalization())
    model.add(Conv1D(9, 3,
                     activation='relu',
                     W_regularizer=l1_l2(0.01),
                     padding='same'))
    model.add(MaxPooling1D(3, padding='same'))
    model.add(BatchNormalization())
    model.add(Flatten())
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(256, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.2))
    model.add(Dense(17, activation='softmax', input_shape=(1,)))
    model.compile(optimizer=adam, loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    print('starts fitting model ...')
    start = time.time()
    model.fit(X_training, y_training, batch_size=batch_size, epochs=n_epochs,
                        validation_data=(X_test, y_test),
              callbacks=[metrics, csv_logger])
    end = time.time()
    delta = end - start
    print('fitting time: ', delta)

    print('starts predicting model ...')
    start_prediction = time.time()
    model.predict(X_test)
    end_prediction = time.time()
    delta_prediction = end_prediction - start_prediction
    print('prediction time: ', delta_prediction)

    y_pred = model.predict_classes(X_test)

    model.save_weights(model_name + ".h5")
    print('weights saved to disk')

    model_json = model.to_json()
    with open(model_name + '.json', 'w') as json_file:
        json_file.write(model_json)
    print('model saved to disk')

    with open(history_file, 'a', newline='') as history:
        writer = csv.writer(history, delimiter=';')
        writer.writerow([model_name, accuracy_score(y_test, y_pred),
                      cohen_kappa_score(y_test, y_pred), f1_score(y_test, y_pred, average='weighted'),
                      delta, delta_prediction])

    if conf_matrix:
        cm_filename = model_name + '_cm.csv'
        cm = pd.DataFrame(confusion_matrix(y_test, y_pred))
        cm.to_csv(cm_filename)

    if accuracy_report:
        raport_filename = model_name + '_report.csv'
        report = classification_report(y_test, y_pred)
        with open(raport_filename, 'w') as acc_report:
            acc_report.write(report)

    return y_pred