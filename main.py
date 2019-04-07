# -*- coding: utf-8 -*-
import geospatial_preprocessing
from classification_preprocessing import read_data
from cnn_model import CNN_model

# reference data with training polygons
data_shp = r'esri_shapefile.shp'
# reference image to extract samples and classify
data_raster = r'raster_type_envi.dat'
# output mask file name
mask_raster = 'mask.tiff'
# filenames for extracted values
export_csv = 'extracted.csv'
export_pickle = 'extracted.pickle'


class_names = (geospatial_preprocessing.get_class_names(data_shp))
print('classes to be classified - ', class_names)
class_nb = len(class_names)
# copy of input shp file (to add unique indices for each polygon)
shp_copy = geospatial_preprocessing.create_index_fld(data_shp, class_names)
# rasterization
geospatial_preprocessing.rasterize(data_raster, mask_raster, shp_copy)
# training data extraction
geospatial_preprocessing.extract(data_raster, mask_raster, class_nb, class_names, export_pickle, export_csv)
# splits data
X_trening, X_test, y_trening, y_test = read_data(export_pickle, 'ekstrakcja', 'klasa', 'indeks')
# CNN classification
CNN_model(X_trening, X_test, y_trening, y_test, 1, accuracy_report=True)