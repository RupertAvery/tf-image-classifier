# Python Image Classifier using Tensorflow

## Disclaimer

This is not my code. Most of it was taken from the very excellent article here:

* http://cv-tricks.com/tensorflow-tutorial/training-convolutional-neural-network-for-image-classification/ 

* https://github.com/sankit1/cv-tricks.com/tree/master/Tensorflow-tutorials/tutorial-2-image-classifier


# Prerequisites

* Python 3.x
* OpenCV for python 
    * `pip install opencv-python`
* Tensorflow 
    * `pip install tensorflow`

# Setup

This script assumes that the training data (images) is separated into folders using the label names in a folder called `training_data` in this format:

```
/training_data
    /label1
    /label2
```

Validation data will be taken from 20% of the training data

Minimum number of images for having an effective training will be 2000 images per class.

# Training

To train a model, run `train.py`

## Syntax

```
python train.py -m <model-name> -l <label1,label2[,label3]> -i <iterations>
```

## Parameters

```
-m, --model         The name of the model you used when training
-l, --labels        The labels in the training_data folder, comma separated
-i, --iterations    The number of iterations to perform. e.g. 3000
```

Once your model is trained, it will save the model files in the current folder. 

# Sorting

To sort images based on your model, run `sort.py` with the following parameters

## Syntax

```
python sort.py -m <model-name> -l <label1,label2[,label3]> -p <path/to/images>
```

# Parameters

```
-m, --model         The name of the model you used when training
-l, --labels        The labels to sort the images by comma separated, and in the order that they were used in training
-p, --path          The path containing the images to be sorted (jpg/jpeg)
```

The script will create a folder for each label in the same path, and begin executing the model against each image 