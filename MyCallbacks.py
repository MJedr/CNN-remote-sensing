import time
from keras.callbacks import Callback, ReduceLROnPlateau
from sklearn.metrics import f1_score
import matplotlib.pyplot as plt

class Metrics(Callback):
    def on_train_begin(self, logs={}):
        self.val_f1s = []
        self.val_recalls = []
        self.val_precisions = []

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

class F1Metric(Callback):
    def on_train_begin(self, logs={}):
        self.f1 = []

    def on_train_end(self, logs={}):
        return

    def on_epoch_begin(self, epoch, logs={}):
        return

    def on_epoch_end(self, epoch, logs={}):
        y_pred = self.model.predict(self.model.validation_data[0])
        self.f1.append(f1_score(self.model.validation_data[1], y_pred, average='weighted'))
        return

    def on_batch_begin(self, batch, logs={}):
        return

def learning_rate_reduction(patience):
    ReduceLROnPlateau(monitor='val_acc',
                            patience=patience,
                            verbose=1,
                            factor=0.5,
                            min_lr=0.00001)

class TimeOnBatch(Callback):
    def on_train_begin(self, logs={}):
        self.logs = []

    def on_epoch_begin(self, logs={}):
        self.start = time.time()

    def on_epoch_end(self, logs={}):
        self.logs.append(time.time() - self.starttime)