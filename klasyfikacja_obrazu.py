import numpy as np
from osgeo import gdal, ogr, gdalnumeric
import os
import pickle
import gdal, gdalnumeric, gdalconst
import matplotlib.pyplot as plt

img='F:\\mgr\\NI1_hiper\\NI1_K4_HS_MOZ.dat'
img_hdr='F:\\mgr\\NI1_hiper\\NI1_K4_HS_MOZ.hdr'

#classification model
filename='rf_model.sav'

s=gdal.GetDriverByName('ENVI')
s.Register()

#opens image as array
img_open=gdal.Open(img, gdalconst.GA_ReadOnly)
img_arr=img_open.ReadAsArray()
print('images red')

#opens classification model
model = pickle.load(open(filename, 'rb'))
print('model loaded')

raster_shape = img_arr.shape
print('raster shape:', raster_shape)

#Reshape to 2d np array
new_shape = (img_arr.shape[1] * img_arr.shape[2], img_arr.shape[0])
img_as_array = img_arr.reshape(new_shape)
print('shape after transformation:', img_as_array.shape)
img_as_array_test = img_arr.reshape(new_shape)
img_arr_class = np.zeros((img_arr.shape[1], img_arr.shape[2]))
#Image class prediction
for row in range(1, img_arr.shape[1]+1):
    slice = img_arr[:, :, row-1]
    pred = model.predict(slice)
    print(pred)


class_prediction = model.predict(img_as_array)
print(class_prediction)

# Reshape to oryginal x, y shapes of raster
class_pred=np.zeros((img_arr.shape[1], img_arr.shape[2]))
print('output image shape:', class_pred.shape)

indices=[]
counterx=0
countery=0
for nb, x in enumerate(class_pred[0]):
    counterx += 1
    for nr, y in enumerate(class_pred[1]):
        countery +=1
        index = int(countery)
        print(index)
        print(class_prediction[int(index)])
        (class_pred[int(nr)][int(nb)]) = class_prediction[int(index)]
        print(nr, nb)


plt.imshow(class_pred)
plt.show()


#wizualizacja
import spectral.io.envi as envi
import matplotlib.pyplot as plt
# from spectral import *
#
# #wizualizacja RGB
# image = envi.open(img_hdr, img)
# view = imshow(image, (26,15,10))
# save_rgb('obraz.png', image, bands=(5, 10, 29))
# print(view)

#wizualizacja klasyfikacji
klasyfikacja = plt.imshow(class_prediction, cmap='tab20')
plt.savefig('klasyfikacja.png')
