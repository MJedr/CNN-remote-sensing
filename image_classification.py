from osgeo import gdal, osr
import pickle
import gdal, gdalnumeric
from gdalconst import *

def classify_raster(raster_path, model_path, out_raster_name):
    ds = gdal.Open(raster_path, GA_ReadOnly)
    model = pickle.load(open(model_path, 'rb'))
    band = ds.GetRasterBand(1)
    xsize = band.XSize
    ysize = band.YSize

    memory_driver = gdal.GetDriverByName('GTiff')
    proj = ds.GetProjectionRef()
    ext = ds.GetGeoTransform()
    out_raster_ds = memory_driver.Create(out_raster_name, xsize, ysize, 1, gdal.GDT_UInt16)
    out_raster_ds.SetProjection(proj)
    out_raster_ds.SetGeoTransform(ext)
    out_raster = out_raster_ds.ReadAsArray()

    block_sizes = ds.GetRasterBand(1).GetBlockSize()
    x_block_size = block_sizes[0] * 10
    y_block_size = block_sizes[1]
    xsize = ds.GetRasterBand(1).XSize
    ysize = ds.GetRasterBand(1).YSize

    blocks = 0
    pixels = 0
    for y in range(0, ysize, y_block_size):
        if y + y_block_size < ysize:
            rows = y_block_size
        else:
            rows = ysize - y
        for x in range(0, xsize, x_block_size):
            if x + x_block_size < xsize:
                cols = x_block_size
            else:
                cols = xsize - x
            array = ds.ReadAsArray(x, y, cols, rows).reshape(cols * rows, 430)
            print(array.shape)
            for i in array:
                pred = model.predict(i.reshape(1, -1))
                out_raster[x][y] = pred
                pixels += 1
            del array
            blocks += 1
            print('classified {0} pixels out of {1}'.format(pixels, xsize * ysize))
    ds = None
    out_raster = None

    return