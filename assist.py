# -*- coding: utf-8 -*-
"""
Created on Wed Aug  8 11:11:53 2018

@author: Marcysia
"""
from gdal import ogr
import os
from osgeo import gdal, gdalnumeric, gdalconst
import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
import random
import copy

def open_envi_array(img):
    s=gdal.GetDriverByName('ENVI')
    s.Register()
    img_open=gdal.Open(img, gdalconst.GA_ReadOnly)
    img_arr=img_open.ReadAsArray()
    return img_arr

def get_class_names(shp):
    """
    Function gets unique class names from shp file
    shp - input shapefile
    """
    s_shp=ogr.GetDriverByName('ESRI Shapefile')
    data=s_shp.Open(shp, 1)
    layer= data.GetLayer()
    feature = layer.GetNextFeature()
    field_vals = []
    while feature:
      field_vals.append(feature.GetFieldAsString('klasa'))
      feature = layer.GetNextFeature()
    vals = np.unique(field_vals)

    return vals

def create_index_fld(input_shp, class_names, output_name):
    """
    Creates extra field in shp with unique index value for each polygon
    input_shp - input shapefile
    class_names - list of unique class names to classify
    output name - output shapefile name
    """
    data_shp=input_shp
    s_shp=ogr.GetDriverByName('ESRI Shapefile')
    vector=s_shp.Open(data_shp, 1)
    lyr=vector.GetLayer()
    print(lyr)
    directory_out=os.getcwd()
    #if file with given name exists delete
    if output_name + '.shp' in os.listdir(directory_out):
        s_shp.DeleteDataSource(output_name + '.shp')
    print(output_name)
    out_ds = s_shp.CreateDataSource(directory_out)
    print(out_ds)

    warstwa_kopia=out_ds.CopyLayer(lyr, output_name)
    print(warstwa_kopia)

    fieldDefn = ogr.FieldDefn('indeks', ogr.OFTInteger)
    fieldDefn.SetWidth(1)
    #fieldDefn.SetPrecision(1)
    warstwa_kopia.CreateField(fieldDefn)
    
    counter=0
    for f in warstwa_kopia:
        f.SetField('indeks', counter)
        warstwa_kopia.SetFeature(f)
        counter+=1
    
    fieldDefn = ogr.FieldDefn('kod', ogr.OFTInteger)
    fieldDefn.SetWidth(10)
    warstwa_kopia.CreateField(fieldDefn)
    
    code=1
    for a in class_names:
        print(class_names[code-1])
        warstwa_kopia.SetAttributeFilter("klasa = '{0}'".format(class_names[code-1]))
        for f in warstwa_kopia:
            f.SetField('kod', code)
            warstwa_kopia.SetFeature(f)
        code+=1
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

    s_raster=gdal.GetDriverByName('ENVI')
    s_raster.Register()
    raster_in=gdal.Open(in_raster, gdalconst.GA_ReadOnly)
    
    sterowniki_shp=ogr.GetDriverByName('ESRI Shapefile')
    shp_in=sterowniki_shp.Open(shp_in, 1)
    shp_warstwa=shp_in.GetLayer()

    # Fetch number of rows and columns
    ncol = raster_in.RasterXSize
    nrow = raster_in.RasterYSize
    
    # Fetch projection and extent
    proj = raster_in.GetProjectionRef()
    ext = raster_in.GetGeoTransform()

    # Create the raster dataset
    memory_driver = gdal.GetDriverByName('GTiff')
    out_raster_ds = memory_driver.Create(out_raster_name, ncol, nrow, 2, gdal.GDT_UInt16)
    
    # Set the ROI image's projection and extent to our input raster's projection and extent
    out_raster_ds.SetProjection(proj)
    out_raster_ds.SetGeoTransform(ext)
    
    out_raster=out_raster_ds.ReadAsArray()

    # Fill our output band with the -999 blank, no class label, value
    for i in range(2):
        out_raster[i].fill(-999)
  
    
    # Rasterize the shapefile layer to our new dataset
    status0 = gdal.RasterizeLayer(out_raster_ds,  # output to our new dataset
                                 [2],  # output to our new dataset's first band
                                 shp_warstwa,  # rasterize this layer
                                 None, None,  # don't worry about transformations since we're in same projection
                                # burn_values=[1],  # burn value 0
                                options= ['ALL_TOUCHED=TRUE',
                                  "ATTRIBUTE={0}".format('kod')] # put raster values according to the 'kod' field values
                                 )   # rasterize all pixels touched by polygons
    status = gdal.RasterizeLayer(out_raster_ds,  # output to our new dataset
                                 [1],  # output to our new dataset's first band
                                 shp_warstwa,  # rasterize this layer
                                 None, None,  # don't worry about transformations since we're in same projection
                                # burn_values=[1],  # burn value 0
                                options= ['ALL_TOUCHED=TRUE',
                                  "ATTRIBUTE={0}".format('indeks')] # put raster values according to the 'index' field values
                                 )   # rasterize all pixels touched by polygons

    # Close dataset
    out_raster_ds = None
    
    if status != 0:
        print("I don't think it worked...")
    else:
        print("Success")


def extract(raster_in, mask, class_nb, class_names, out_file_pickle, out_file_csv):
    """
    Extract vals from raster file. Saves into csv and pickle file.

    :param raster_in: in raster to extract values
    :param mask: mask with rasterised training data
    :param class_nb: number of classes to extract vals
    :param class_names: class names to extract vals
    :param out_file_csv: output csv
    :param out_file_pickle: output pickle
    :return: output csv, pickle
    """

    s_raster = gdal.GetDriverByName('ENVI')
    s_raster.Register()

    # wczytanie danych - rastra, z którego odczytywane są wartosci i maski z danymi testowymi
    raster = gdal.Open(raster_in, gdalconst.GA_ReadOnly)

    inmask = gdal.Open(mask, gdalconst.GA_ReadOnly)
    band_mask = inmask.GetRasterBand(2)
    data_mask = band_mask.ReadAsArray(0, 0)

    coords = np.nonzero(data_mask)
    new_coords = []

    for i in range(len(coords[0])):
        m = []
        m.append(coords[0][i])
        m.append(coords[1][i])
        new_coords.append(m)

    px_vals = []
    for i in range(class_nb):
        px_vals.append([])
    for x, y in new_coords:
        for i in range(1, class_nb + 1):
            if data_mask[x, y] == i:
                px_vals[i - 1].append([x, y])
    for nb, i in enumerate(px_vals):
        print(nb, i)

    class_id = -1
    data = []
    band_mask_index = inmask.GetRasterBand(1)


    # iteracja po wspolrzednych zebranych z maski
    for i in range(len(px_vals)):
        coord  = px_vals[i]
        class_id = class_id + 1
        # print(wsp)
        for counter, i in enumerate(coord):
            # utworzenie listy list - rekordu dla każdego piksela, który znajduje się na liscie współrzędnych
            pixel_extract = []
            x = int(i[0])
            y = int(i[1])
            print(x,y)
            pixel_extract.append([x, y])
            # odczyt wartosci dla piksela w kazdym kanale
            px_value = []
            for b in range(raster.RasterCount):
                band = raster.GetRasterBand(b + 1)
                band_val = (float(band.ReadAsArray(y, x, 1, 1)))
                px_value.append(band_val)
                # dodanie odczytanej wartosci do rekordu dla piksela
            pixel_extract.append(px_value)
            # dodanie informacji o klasie piksela do rekordu
            pixel_extract.append('{0}'.format(class_names[class_id]))
            # dodanie indeksu do rekordu
            pixel_extract.append(int(band_mask_index.ReadAsArray(y, x, 1, 1)))
            # dodanie rekordu dla piksela do zbioru odczytanych danych
            data.append(pixel_extract)
            print((class_names[class_id]), (counter + 1) / len(coord))
    print(data[0])

    #cleaning data
    coordinates = []
    values = []
    class_name = []
    index = []

    for element in data:
        coordinates.append(element[0])
        values.append(element[1])
        class_name.append(element[2])
        index.append(element[3])

    # data as pd Data Frame
    df = pd.DataFrame(list(zip(coordinates, values, class_name, index)),
                        columns=['coordinates', 'values', 'class', 'index'])

    # dodanie nowych, oddzielnych kolumn dla współrzędnej X i Y, usunięcie kolumny z obydwoma współrzędnymi
    df['X'] = df['coordinates'].str.get(0).astype(int, errors='ignore')
    df['Y'] = df['coordinates'].str.get(1).astype(int, errors='ignore')
    del df['coordinates']

    print(df.loc[:, 'class'].value_counts())

    #save file
    df.to_csv(out_file_csv)
    df.to_pickle(out_file_pickle)
    plot=(df['class'].value_counts()).plot(kind='bar', rot=0, fontsize=7)
    fig = plot.get_figure()
    fig.savefig('class_plot.png')

    print('done!')

