from sklearn import metrics
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import scikitplot as skplt
import numpy as np
from random import randint
import os
def podstawowe_statystyki(true_labels, pred_labels, klasy, nazwa_modelu):
    accuracy_dict={}
    accuracy_dict['nazwa modelu'] = (nazwa_modelu)
    accuracy_dict['Dokładność całkowita (Overall Accuracy)'] = metrics.accuracy_score(true_labels, pred_labels)
    accuracy_dict['Współczynnik Kappa']=metrics.cohen_kappa_score(true_labels, pred_labels)
    accuracy_dict['Precision']= metrics.precision_score(true_labels,pred_labels, average='macro')
    accuracy_dict['Recall'] = metrics.recall_score(true_labels, pred_labels, average='macro')
    statystki=pd.DataFrame.from_dict(accuracy_dict, 'index')
    print(statystki)
    path='statystyki_{0}.csv'.format(nazwa_modelu)
    if os.path.isfile(path):
        path='statystyki_{0}_{1}.csv'.format(nazwa_modelu, randint(1,1000))
    out_stat=statystki.to_csv(path, encoding='utf-8')
    return out_stat

def raport(true_labels, pred_labels, names, nazwa_modelu):
    '''
    names - nazwy klas w postaci listy
    '''
    print(nazwa_modelu)
    metryki=(metrics.classification_report(true_labels, pred_labels, target_names=names))
    path='statystyki_{0}.csv'.format(nazwa_modelu)
    if os.path.isfile(path):
        path='statystyki_%s_%d.csv' % (nazwa_modelu, (randint(1,1000)))
    with open(path, 'w') as plk:
        plk.writelines(metryki)
    print(metryki)
    return plk


def macierz_bledow(true_labels, pred_labels, naglowki, nazwa_modelu):

    macierz=metrics.confusion_matrix(true_labels, pred_labels)

    path='macierz_{0}.csv'.format(nazwa_modelu)
    df=pd.DataFrame(macierz, columns=naglowki)
    if os.path.isfile(path):
        path='statystyki_%s_%d.csv' %(nazwa_modelu, randint(1,1000))
    #np.savetxt(path.format(nazwa_modelu), macierz)
    out=df.to_csv(path)
    return out


def macierz_bledow_wykres(true_labels, pred_labels, liczba_klas, names, nazwa_modelu):
    '''
    names - nazwy klas w postaci listy
    '''
    confusion_mat = metrics.confusion_matrix(true_labels, pred_labels)
    fig, ax = plt.subplots(figsize=(liczba_klas, liczba_klas))
    hm=sns.heatmap(confusion_mat, annot=True, annot_kws={"size": 10}, fmt='d',
                cmap='YlOrRd',
                xticklabels=names, yticklabels=names, axes=ax)
    hm.tick_params(labelsize=6)
    plt.xticks(rotation=45.)
    plt.ylabel('Actual')
    plt.xlabel('Predicted')

    path='macierz_wykres_{0}.png'.format(nazwa_modelu)
    if os.path.isfile(path):
        path='macierz_wykres_%s_%d.png'%(nazwa_modelu, randint(1,1000))
    fig=hm.get_figure()
    wykres = fig.savefig(path.format(nazwa_modelu))
    return wykres


def learning_curve(clf, X_trening, y_trening, nazwa_modelu):
    y_trening_en=y_trening
    from scikitplot import estimators
    estimators.plot_learning_curve(clf, X_trening, y_trening_en)
    path='learning_curve_{0}.png'.format(nazwa_modelu)
    if os.path.isfile(path):
        path='learning_curve_%s_%d.png'%(nazwa_modelu, randint(1,1000))
    plot = plt.savefig(path.format(nazwa_modelu))
    return plot

def rozrzut_danych(X_trening, y_trening_en, nazwa_modelu, kanal_1, kanal_2):
    '''
    Rysuje wykres rozrzutu z pogrupowanymi danymi na klasy. Za kanal_1 i kanal_2 należy podstawić wybrane dwa kanały
    '''
    z = []
    path='rozrzut_danych.png'.format(nazwa_modelu)
    if os.path.isfile(path):
        path='rozrzut_danych_%s_%d.png'%(nazwa_modelu, randint(1,1000))

    for element in X_trening:
        z.append([element[kanal_1], element[kanal_2]])
    print(y_trening_en.dtype)
    z = np.array(z).astype('float')
    df = pd.DataFrame(dict(x=z[:, 0], y=z[:, 1], label=y_trening_en))
    fig, ax = plt.subplots()
    grouped = df.groupby('label')
    kolory = ['gold', 'blue', 'green', 'red', 'm', 'k', 'aqua', 'brown', 'coral', 'grey', 'pink','sienna', 'plum', 'tomato', 'wheat', 'khaki', 'cyan']
    klasy=np.unique(y_trening_en)
    ilosc_klas=(klasy.shape[0])
    col=kolory[:ilosc_klas]

    for key, group in grouped:
        group.plot(ax=ax, kind='scatter', x='x', y='y',color=col, legend=True)

    ax.legend()
    plt.xlabel('kanal 1')
    plt.ylabel('kanal 2')
    plt.savefig(path)

def CNN_results(history):
    acc = history.history['acc']
    val_acc = history.history['val_acc']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs = range(1, len(acc) + 1)
    plt.plot(epochs, acc, 'bo', label='Dokładność treningu')
    plt.plot(epochs, val_acc, 'b', label='Dokładność walidacji')
    plt.title('Dokładność treningu i walidacji')
    plt.xlabel('epoki')
    plt.ylabel('dokładność')
    plt.legend()
    plt.figure()
    plt.plot(epochs, loss, 'bo', label='Dokładność treningu')
    plt.plot(epochs, val_loss, 'b', label='Dokładność walidacji')
    plt.title('Dokładność treningu i walidacji')
    plt.legend()
    plt.show()
