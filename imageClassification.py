import numpy as np
from osgeo import gdal, ogr, gdalnumeric
import os
import pickle
import gdal, gdalnumeric
from gdalconst import *
import matplotlib.pyplot as plt
from osgeo import gdal_array
import ogr, os, osr

Image  = gdal.Open('data\\NI1_K4_HS_MNF.dat', GA_ReadOnly)

filename = 'C:\\cysia\\CNN-to-lc-classification\\wyniki\\SVM_MNF\\15\\svm_model.sav'
model = pickle.load(open(filename, 'rb'))


def array2raster(newRasterfn,rasterOrigin,pixelWidth,pixelHeight,array):

    cols = array.shape[1]
    rows = array.shape[0]
    originX = rasterOrigin[0]
    originY = rasterOrigin[1]

    driver = gdal.GetDriverByName('GTiff')
    outRaster = driver.Create(newRasterfn, cols, rows, 1, gdal.GDT_Byte)
    outRaster.SetGeoTransform((originX, pixelWidth, 0, originY, 0, pixelHeight))
    outband = outRaster.GetRasterBand(1)
    outband.WriteArray(array)
    outRasterSRS = osr.SpatialReference()
    outRasterSRS.ImportFromEPSG(4326)
    outRaster.SetProjection(outRasterSRS.ExportToWkt())
    outband.FlushCache()


def main(newRasterfn,rasterOrigin,pixelWidth,pixelHeight,array):
    reversed_arr = array[::-1] # reverse array so the tif looks like the array
    array2raster(newRasterfn,rasterOrigin,pixelWidth,pixelHeight,reversed_arr) # convert array to raster



coords=(Image.GetGeoTransform()[0], Image.GetGeoTransform()[3])

# Fetch number of rows and columns
ncol = Image.RasterXSize
nrow = Image.RasterYSize

# Fetch projection and extent
proj = Image.GetProjectionRef()
ext = Image.GetGeoTransform()

# Create the raster dataset
out_raster_name='klasyfikacja_test6.tiff'
memory_driver = gdal.GetDriverByName('GTiff')
out_raster_ds = memory_driver.Create(out_raster_name, 200, 200, 1, gdal.GDT_UInt16)

# Set the ROI image's projection and extent to our input raster's projection and extent
out_raster_ds.SetProjection(proj)
out_raster_ds.SetGeoTransform(ext)

out_raster=out_raster_ds.ReadAsArray()

Image.ReadAsArray(0, 1, 1, 1)
nBands = Image.RasterCount  # how many bands, to help you loop
nRows = Image.RasterYSize
nCols = Image.RasterXSize
newShape = (430, 400 * 400)
print(nRows, nCols)
nR = 200
nC = 200

outputImage = np.zeros((nR, nC))
outputImage.fill(-1)
outband = out_raster_ds.GetRasterBand(1)

print(Image.ReadAsArray(1, 1, 1, 1))
for x in range(nR):
    for y in range(nC):
        i = Image.ReadAsArray(x, y, 1, 1)
        pred = model.predict(i.reshape(1, -1))
        outputImage[x][y] = pred

plt.imshow(outputImage, cmap='tab20')
plt.colorbar()
plt.savefig('klasyfikacjaRFMNF_1.tif')
out_raster_ds = None

main('SVM_MNF_400.tiff', coords, nR, nC, outputImage)