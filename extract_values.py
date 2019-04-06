# -*- coding: utf-8 -*-
import assist

# reference data with training polygons
data_shp = r'esri_shapefile.shp'
# reference image to extract samples and classify
data_raster = r'raster_type_envi.dat'
# output mask file name
mask_raster = 'mask.tiff'
# filenames for extracted values
export_csv = 'extracted.csv'
export_pickle = 'extracted.pickle'


class_names = (assist.get_class_names(data_shp))
print('classes to be classified - ', class_names)
class_nb = len(class_names)
# copy of input shp file (to add unique indices for each polygon)
shp_copy = assist.create_index_fld(data_shp, class_names)
# rasterization
assist.rasterize(data_raster, mask_raster, shp_copy)
# training data extraction
assist.extract(data_raster, mask_raster, class_nb, class_names, export_pickle, export_csv)