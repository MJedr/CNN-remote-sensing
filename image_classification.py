from osgeo import gdal, osr
import numpy as np
import pickle
import gdal, gdalnumeric
from gdalconst import *

def classify_tile(array, model):
    temp = np.zeros([array.shape[0], array.shape[1]])
    for nb, i in enumerate(array):
        temp[nb] = model.predict(i.reshape(1, -1))
    temp = temp.reshape(array.shape[0], array.shape[1])
    return temp

def classify_raster(raster_path, model_path, out_raster_name, x_block_size=256, y_block_size=160):
    ds = gdal.Open(raster_path, GA_ReadOnly)
    model = pickle.load(open(model_path, 'rb'))
    band = ds.GetRasterBand(1)
    b_array = band.ReadAsArray()
    x_im_size = band.XSize
    y_im_size = band.YSize
    x_block_size = int(x_block_size)
    y_block_size = int(y_block_size)
    xsize = b_array.shape[0]
    ysize = b_array.shape[1]
    xstride = np.floor(xsize / x_block_size).astype('int64')
    ystride = np.floor(ysize / y_block_size).astype('int64')
    max_xstride_range = int(x_block_size * xstride)
    max_ystride_range = int(y_block_size * ystride)
    nb_px_not_ystride = int(ysize % y_block_size)
    nb_px_not_xstride = int(xsize % x_block_size)
    out_raster = np.zeros([xsize, ysize])

    memory_driver = gdal.GetDriverByName('GTiff')
    proj = ds.GetProjectionRef()
    ext = ds.GetGeoTransform()
    out_raster_ds = memory_driver.Create(out_raster_name, x_im_size, y_im_size, 1, gdal.GDT_UInt16)
    out_raster_ds.SetProjection(proj)
    out_raster_ds.SetGeoTransform(ext)

    pixels = 0
    for yy in range(0, ystride):
        for xx in range(0, xstride):
            y = xx * x_block_size
            x = yy * y_block_size
            array = ds.ReadAsArray(y, x, y_block_size, x_block_size).reshape(x_block_size * y_block_size, 30)
            out_raster[x:x + x_block_size, y:y + y_block_size] = classify_tile(array, model)
            print('classified {0} pixels out of {1}'.format(pixels, xsize * ysize))
            print('-------------------------------------')
        if yy == (ystride - 1):
            if xsize % x_block_size != 0 and ysize % y_block_size != 0:
                array = ds.ReadAsArray(y, 0, int(ysize % y_block_size), int(xsize)). \
                    reshape((ysize % y_block_size) * xsize, 30)
                out_raster[y_block_size * ystride: y_block_size * ystride + ysize % y_block_size, 0:xsize] = classify_tile(array, model)
                new_x_stride = int(xsize - (xstride * x_block_size))
                new_x = int(xstride * x_block_size)
                new_y = 0
                for yyy in range(0, ystride):
                    array = ds.ReadAsArray(new_x, new_y, new_x_stride, y_block_size). \
                        reshape(new_x_stride * y_block_size, 30)
                    out_raster[new_y:new_y + y_block_size, new_x:xsize] = classify_tile(array)
                    new_y += y_block_size
            elif xsize % x_block_size != 0 and ysize % y_block_size == 0:
                array = ds.ReadAsArray(int(xstride * x_block_size), 0, int(xsize % x_block_size), int(ysize)). \
                    reshape((xsize % x_block_size) * ysize, 30)
                out_raster[0:ysize, x_block_size * xstride: x_block_size * xstride + xsize % x_block_size] = \
                    classify_tile(array, model)
            elif xsize % x_block_size == 0 and ysize % y_block_size != 0:
                array = ds.ReadAsArray(y, 0, int(ysize % y_block_size), int(xsize)). \
                    reshape(nb_px_not_ystride * xsize, 30)
                out_raster[y_block_size * ystride: y_block_size * ystride + ysize % y_block_size, 0:xsize] = \
                    classify_tile(array, model)
            print('classified {0} pixels out of {1}'.format(pixels, xsize * ysize))

    outband = out_raster_ds.GetRasterBand(1)
    outband.WriteArray(out_raster)
    outband.FlushCache()

    ds = None

    return out_raster
