# -*- coding: utf-8 -*-
from gdal import ogr
import os
from osgeo import gdal, gdalnumeric, gdalconst
import numpy as np
import pandas as pd


def open_envi_array(img):
    driver = gdal.GetDriverByName('ENVI')
    driver.Register()
    img_open = gdal.Open(img, gdalconst.GA_ReadOnly)
    img_arr = img_open.ReadAsArray()
    return img_arr


def get_class_names(shp):
    """
    Function gets unique class names from shp file
    shp - input shapefile
    """
    driver_shp = ogr.GetDriverByName('ESRI Shapefile')
    data = driver_shp.Open(shp, 1)
    layer = data.GetLayer()
    feature = layer.GetNextFeature()
    field_vals = []
    while feature:
        field_vals.append(feature.GetFieldAsString('klasa'))
        feature = layer.GetNextFeature()
    vals = np.unique(field_vals)
    return vals


def create_index_fld(input_shp, class_names, output_name='training_indexed.shp'):
    """
    Creates an extra field in shp with unique index value for each polygon
    input_shp - input shapefile
    class_names - list of unique class names to classify
    output name - output shapefile name
    """
    data_shp = input_shp
    driver_shp = ogr.GetDriverByName('ESRI Shapefile')
    vector = driver_shp.Open(data_shp, 1)
    lyr = vector.GetLayer()
    directory_out = os.getcwd()
    # if file with given name exists delete
    if output_name + '.shp' in os.listdir(directory_out):
        driver_shp.DeleteDataSource(output_name + '.shp')
    print('created file', output_name)
    out_ds = driver_shp.CreateDataSource(directory_out)

    lyr_copy = out_ds.CopyLayer(lyr, output_name)
    fieldDefn = ogr.FieldDefn('indeks', ogr.OFTInteger)
    fieldDefn.SetWidth(1)
    lyr_copy.CreateField(fieldDefn)

    for nb, f in enumerate(lyr_copy):
        f.SetField('indeks', nb)
        lyr_copy.SetFeature(f)

    fieldDefn = ogr.FieldDefn('kod', ogr.OFTInteger)
    fieldDefn.SetWidth(10)
    lyr_copy.CreateField(fieldDefn)

    code = 1
    for a in class_names:
        print(class_names[code - 1])
        lyr_copy.SetAttributeFilter("klasa = '{0}'".format(class_names[code - 1]))
        for f in lyr_copy:
            f.SetField('kod', code)
            lyr_copy.SetFeature(f)
        code += 1
    print('created')
    return output_name + '.shp'


def rasterize(in_raster, out_raster_name, shp_in):
    """
    Funkcja do rasteryzacji shapefila.
    Raster wejscie - raster, na podstawie którego utworzony zostanie raster wyjsciowy (ten sam układ wsp., wymiary, rodzielczosć)
    Raster wyjscie - scieżka do pliku wyjsciowego tiff
    shp_wejscie - dane, na podstawie ktorych tworzona bedzie maska. Scieżka dostępu do pliku shp.
    atrybut - wartosc opcjonalna, ktora przypisana zostanie zrasteryzowanym polom. Dobierana na podstawie atrybutu w pliku shp.
    """

    driver_raster = gdal.GetDriverByName('ENVI')
    driver_raster.Register()
    raster_in = gdal.Open(in_raster, gdalconst.GA_ReadOnly)

    driver_shp = ogr.GetDriverByName('ESRI Shapefile')
    shp_in = driver_shp.Open(shp_in, 1)
    shp_lyr = shp_in.GetLayer()

    ncol = raster_in.RasterXSize
    nrow = raster_in.RasterYSize

    proj = raster_in.GetProjectionRef()
    ext = raster_in.GetGeoTransform()

    # Create the raster dataset
    memory_driver = gdal.GetDriverByName('GTiff')
    out_raster_ds = memory_driver.Create(out_raster_name, ncol, nrow, 2, gdal.GDT_UInt16)

    out_raster_ds.SetProjection(proj)
    out_raster_ds.SetGeoTransform(ext)

    out_raster = out_raster_ds.ReadAsArray()

    for i in range(2):
        out_raster[i].fill(-999)

    status0 = gdal.RasterizeLayer(out_raster_ds,
                                  [2],
                                  shp_lyr,
                                  None, None,
                                  options=['ALL_TOUCHED=TRUE',
                                           "ATTRIBUTE={0}".format('kod')]
                                  )
    status = gdal.RasterizeLayer(out_raster_ds,
                                 [1],
                                 shp_lyr,
                                 None, None,
                                 options=['ALL_TOUCHED=TRUE',
                                          "ATTRIBUTE={0}".format('indeks')]
                                 )

    out_raster_ds = None

    if status != 0:
        print("I don't think it worked...")
    else:
        print("Success")


def extract(raster_in, mask, class_nb, class_names, out_file_pickle, out_file_csv):
    """
    Extract values from raster file. Saves into csv and pickle file.

    raster_in: in raster to extract values
    mask: mask with rasterised training data
    class_nb: number of classes to extract values
    class_names: class names to extract values
    out_file_csv: output csv
    out_file_pickle: output pickle
    """
    drivers_raster = gdal.GetDriverByName('ENVI')
    drivers_raster.Register()

    raster = gdal.Open(raster_in, gdalconst.GA_ReadOnly)

    inmask = gdal.Open(mask, gdalconst.GA_ReadOnly)
    band_mask = inmask.GetRasterBand(2)
    data_mask = band_mask.ReadAsArray(0, 0)

    coords = np.nonzero(data_mask)
    new_coords = np.array([0, 0])
    for i in range(len(coords[0])):  # reads coordinates from input raster
        m = np.array([coords[0][i], coords[1][i]])
        new_coords = np.vstack((new_coords, m))

    np.delete(new_coords, 0, 0)  # removers first empty row

    pixel_class = ([data_mask[x, y] for x, y in new_coords])
    px_vals = [[] for x in range(class_nb)]
    for nb, x in enumerate(pixel_class):
        px_vals[x - 1].append(new_coords[nb])

    data = []
    band_mask_index = inmask.GetRasterBand(1)

    for nb, class_nb in enumerate(px_vals):
        coord_list_class = px_vals[nb]
        class_id = nb + 1
        for counter, i in enumerate(coord_list_class):
            x, y = int(i[0]), int(i[1])
            bands = [raster.GetRasterBand(i) for i in range(1, raster.RasterCount + 1)]
            pix_val = np.squeeze(
                np.array([gdalnumeric.BandReadAsArray(band, y, x, 1, 1) for band in bands]).astype('int64'))
            pixel_extract = [x] + [y] + [pix_val] + ['{0}'.format(class_names[class_id])] + \
                            [int(band_mask_index.ReadAsArray(y, x, 1, 1))]
            data.append(pixel_extract)
            print('extracted', round((counter + 1) / len(coord_list_class), 2),
                  '% form class {0}'.format(class_names[class_id]))

    # cleaning data
    x = [x[0] for x in data]
    y = [x[1] for x in data]
    values = [x[2] for x in data]
    class_name = [x[3] for x in data]
    index = [x[4] for x in data]

    df = pd.DataFrame(list(zip(x, y, values, class_name, index)),
                      columns=['x', 'y', 'coordinates', 'values', 'class', 'index'])

    print(df.loc[:, 'class'].value_counts())
    df.to_csv(out_file_csv)
    df.to_pickle(out_file_pickle)
    print('done!')
