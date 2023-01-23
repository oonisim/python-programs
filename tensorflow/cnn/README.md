# Image Search Engine

## Objective

Implement a system to search similar images using the embedded image vector consine similarity.

Used Tensorflow Keras **ResNet50** ```avg_pool``` layer output before the fully connected layer as the model 
to transform the images in shape ```(height=224, width=224, channels=3)``` into the embedded image vectors 
of ```2048``` dimensions.

```
img2vec: Model = keras.Model(
    inputs=ResNet50.input,
    outputs=ResNet50.get_layer("avg_pool").output
)
```

## Function

| Input                                           | Output                     |
|-------------------------------------------------|----------------------------|
| An image with RGB channels that OpenCV can load | N number of similar images |

<img src="./image/search_engine_output.png" align="left"/>

---
# Terminologies
* FE: Feature Engineering
* npy: numpy serialised file


| Constant            | Description                                                                                                                                                                                                                                     |
|---------------------|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| NPY_IMAGE_VECTORS   | Embedded image vectors each of which represents a resized RGB image in the multidimensional latent space. The dimension size depends onthe vectorizer model, e.g. ResNet50 avg_pool layer output has 2048. Serialized with numpy.save() method. |
| NPY_RESIZED_RGB     | Images resized and transformed to have RGB channel order in memory. Serialized with numpy.save() method.  Each row in the array matches with the image name in NPY_IMAGE_NAMES.                                                                 |
| NPY_IMAGE_NAMES     | Names of the resized RGB images. Each row in the array matches with the image in NPY_RESIZED_RGB. Serialized with numpy.save().                                                                                                                 |
| TF_VECTORIZER_MODEL | Vectorizer Keras Model instance used at modelling to vectorize the images into embedded image vectors.  Serialized with the Keras Model.save() method with the default options.                                                                 |

---
# Note

## Image Data Format
Note the saved images have the channel order as RGB. The in-memory BGR order by OpenCV default
has been converted to RGB and saved with the order in disk.

image in memory passed to a most_similar() as argument MUST be BGR order as with OpenCV imread result, 
because the same transformation (resize, BGR to RGB, Keras/ResNet preprocess) is applied.

## Training/Serving Skew
To prevent training/serving skew (drifts), need to use th same artifacts
fitted to data for transformations (e.g. scaling, mean-centering, PCA), and the consistent 
serialisation and de-serialisation methods.

The numeric data has been serialised using numpy ```save()``` and the Tensorflow Keras model artifacts have been 
serialised using Keras ```Model.save()``` with the default format and options.



---
# System Requirements

1. Python 3.9.x
2. pip 22.3.x
3. OpenCV 4.7
4. Tensorflow 2.10.0
5. Tensorflow Keras Resnet50

### Development Environment
The environment used to develop and test.

```
$python --version
Python 3.9.13

$ pip --version
pip 22.3.1

$ pylint --version
pylint 2.15.6

$ pytest --version
pytest 7.2.0

>>> import cv2
>>> cv2.__version__
'4.7.0'

>>>import tensorflow as tf
>>>print(tf.__version__)
2.10.0
```

# Setup

1. Place original images in ${BASE}/data/master directory.
2. Create virtual environment.<br>
    ```
    python -m venv ${VENV_NAME}
    ```
3. Install python packages.
    ```
    pip install -r requirements.txt
    ```
4. Set the PYTHONPATH to include the ```lib/``` directory.

---

# Execution
1. To generate the embedding image vectors for image search, run the command:
```
run_modelling_pipeline.sh
```

# Standards
1. Not to use magic words - define constants or enumerations.
2. 