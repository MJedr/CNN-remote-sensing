# -*- coding: utf-8 -*-
"""
Created on Mon Aug 13 11:40:07 2018

@author: user
"""


import numpy as np
from osgeo import ogr
import assist

#reference data with training polygons
data_shp=r'C:\cysia\dg\shapefile\KR1\KR1_k4.shp'
#reference image to extract samples and classify
data_raster=r'F:\dg\k4\MOZ\KR1_K4_HS_MOZ.dat'

#class names to extract
class_names=(assist.get_class_names(data_shp))
print('classes to be classified - ', class_names)
#number of classes
class_nb=len(class_names)

#obraz wyjściowy - mmaska ze zrasteryzowanymi klasami
mask_raster=r'C:\cysia\dg\shapefile\KR1\maskaKR4.tiff'

#plik wyjściowy z wartościami pikseli
export_csv=r'C:\cysia\dg\shapefile\KR1\ekstrakcja_przyklad.csv'
export_pickle=r'C:\cysia\dg\shapefile\KR1\ekstrakcja_przyklad.pickle'

#copy of input shp file (to add unique indices for each polygon)
shp_copy = assist.create_index_fld(data_shp, class_names, output_name = r'terningK4')
#shp_copy =  'C:\\cysia\\CNN-to-lc-classification\\terningMNF.shp'
#rasterisation
assist.rasterize(data_raster, mask_raster, shp_copy)
#training data extraction
assist.extract(data_raster, mask_raster, class_nb, class_names, export_pickle, export_csv)





