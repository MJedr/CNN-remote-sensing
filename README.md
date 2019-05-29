# CNN-remote-sensing
  
A tool which enables perform a full per-pixel remote sensing image classification, 
which includes:
- reading sample data from *.shp* file
- rasterizing *.shp* file and extracting extracting signal vectors for samples from remote sensing image
- splitting data into training, test and validation set in a given proportion with a respect to the spatial correlation
- training CNN, RF, SVM models
- evaluating trained models
- classyfing a full remote sensing image.

## Requirements
To run the program it is neccesary to have Python 3.6. and **installed Python bindings** 
(more on a <a href="https://pypi.org/project/GDAL/"> PyPi website </a>).  
Then just use Python Pip to install other requirements:

```
pip install -r requirements.txt
```

## Data 
To run the classification, following files are neccessary:
- file in *.shp* format with trainig samples, which must include a field with class names
- raster file for classification in ENVI .hdr Labelled Raster.

## Getting started
To start a classification enter the paths to your files in main.py and then run the program. Trained model automatically will be saved.  
To perform image classification, please enter path for the model and the raster image in a image_classification.py file. 
**Please, note that the image classification can take a really long time.**

## Author
Marcjanna Jędrych 
