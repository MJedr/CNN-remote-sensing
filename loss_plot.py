import re
import matplotlib.pyplot as plt
import pandas as pd
import re

file = 'wyniki\\CNN_MNF_TRAIN\\7\\training.log'
no=int(re.findall("[.0-9]", file)[0])


hist = pd.read_csv(file, sep=",")
hist.epoch[1]
#wersja mnf
print(hist.head())
print(hist.epoch.tail(1).values)
print('test', hist.epoch.tail(1).values[0])

def plot_history(history):
    hist = pd.read_csv(history, sep=",")

    loss_list = hist.loss
    val_loss_list = hist.val_loss
    acc_list = hist.acc


    if len(loss_list) == 0:
        print('Loss is missing in history')
        return

        ## As loss always exists
    epochs = range((0, (hist.epoch.tail(1).values[0].astype(int)+1)))

    ## Loss
    fig, ax = plt.subplots(1, 2, figsize=(16, 8))


    ax[0].plot(epochs, loss_list, 'b',
               label='trening')

    ax[0].plot(epochs, val_loss_list, 'g',
               label='walidacja')
    ax[0].set_title("Funkcja kosztu (loss)")
    ax[0].set_xlabel('Epoki')
    ax[0].set_ylabel('Loss')
    ax[0].legend()

    ## Accuracy
    ax[1].plot(epochs, acc_list, 'b',
               label='trening')

    ax[1].plot(epochs, val_acc_list, 'g',
               label='walidacja')
    ax[1].set_title('Dokładność całkowita')
    ax[1].set_xlabel('Epoki')
    ax[1].set_ylabel('OA')
    ax[1].legend()



    #fig.show()
    fig.savefig('wyniki\\plots\\train_lossMNF{}.png'.format(no))
    plt.show()

fi=plot_history(file)
