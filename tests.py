from classification_preprocessing import spatial_stratified_sample, label_encod, read_data_CNN, read_data
import pandas as pd
from cnn_model import CNN_model

data = r'C:\cysia\cnn_mgr\validation.pickle'


X_trening, X_test, y_trening, y_test = read_data(data, 'ekstrakcja', 'klasa', 'indeks')
print(X_trening.shape)
# print(X_trening.shape, X_test.shape, y_trening.shape, y_test.shape)

# CNN_model(X_trening, X_test, y_trening, y_test, 1, accuracy_report=True)