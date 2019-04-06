import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter

def plot_history(history):
    hist = pd.read_csv(history, sep=",")
    loss_list = hist.loss
    val_loss_list = hist.val_loss
    acc_list = hist.acc
    val_acc_list = hist.val_acc

    if len(loss_list) == 0:
        print('Loss is missing in history')
        return

    epochs = range(0, (hist.epoch.tail(1).values[0].astype(int) + 1))

    fig, ax = plt.subplots(1, 2, figsize=(16, 8))
    ax[0].plot(epochs, loss_list, 'b',
               label='trening')
    ax[0].plot(epochs, val_loss_list, 'g',
               label='walidacja')
    ax[0].set_title('Funkcja kosztu(loss)')
    ax[0].set_xlabel('Epoki')
    ax[0].set_ylabel('Loss')
    ax[0].legend()
    ax[1].plot(epochs, acc_list, 'b',
               label='trening')
    ax[1].plot(epochs, val_acc_list, 'g',
               label='walidacja')
    ax[1].set_title('Dokładność całkowita')
    ax[1].set_xlabel('Epoki')
    ax[1].set_ylabel('OA')
    ax[1].legend()

    fig.savefig('model_history.png')
    plt.show()