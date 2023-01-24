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

| Input                                                                                                                  | Output                                                                                                  |
|------------------------------------------------------------------------------------------------------------------------|---------------------------------------------------------------------------------------------------------|
| An query image to search similar ones. The image needs to have RGB channels and is in the format that OpenCV can load. | N number of similar images with their names and similarity scores, where the query image is on the left |

**Example**

![](./image/search_engine_output.png "search engine output")

---
# Terminologies
* FE: Feature Engineering
* npy: numpy serialised file
* Artifacts: intermediate and final outputs from the modelling processes e.g. transformed image data, model for prediction, etc. 


| Artifacts              | Description                                                                                                                                                                                                                                          |
|------------------------|------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| NPY_RESIZED_RGB        | Images data resized and transformed to have RGB channel order in memory. Serialized with numpy.save() method.  Each row in the array matches with the image name in NPY_IMAGE_NAMES.                                                                 |
| NPY_IMAGE_NAMES        | Name data of the resized RGB images. Each row in the array matches with the image in NPY_RESIZED_RGB. Serialized with numpy.save().                                                                                                                  |
| NPY_FEATURE_ENGINEERED | Images data preprocessed for the ResNet input layer.                                                                                                                                                                                                 |
| NPY_IMAGE_VECTORS      | Embedded image vectors data each of which represents a resized RGB image in the multidimensional latent space. The dimension size depends onthe vectorizer model, e.g. ResNet50 avg_pool layer output has 2048. Serialized with numpy.save() method. |
| TF_VECTORIZER_MODEL    | Vectorizer Keras Model instance used at modelling to vectorize the images into embedded image vectors.  Serialized with the Keras Model.save() method with the default options.                                                                      |

---
# Note

## Image Data Format
Note the saved images have the channel order as RGB. The in-memory BGR order by OpenCV default
has been converted to RGB and saved with the order in disk.

The image in memory passed to a ```most_similar()``` method as argument MUST be BGR order as with OpenCV imread result, 
so that the same transformation (resize, BGR to RGB, Keras/ResNet preprocess) is applied.

## Training/Serving Skew
To prevent training/serving skew (drifts), need to use th same artifacts
fitted to data for transformations (e.g. scaling, mean-centering, PCA), and the consistent 
serialisation and de-serialisation methods.

The numeric data has been serialised using numpy ```save()``` and the Tensorflow Keras model artifacts have been 
serialised using Keras ```Model.save()``` with the default format and options.

## Image Order 

In NPY_RESIZED_RGB and NPY_IMAGE_NAMES, the order is crucial to be able to correctly identify the 
(image data, image name) matching. The i-th row in NPY_IMAGE_NAMES needs to be the name of the image at the
i-th row in NPY_RESIZED_RGB.

---

# Standards

Follow the Python standards.

1. [PEP 8 – Style Guide for Python Code](https://peps.python.org/pep-0008/)
2. [Google Python Style Guide](https://google.github.io/styleguide/pyguide.html)
3. [Pylint](https://github.com/PyCQA/pylint)

---
# System Requirements

1. Python 3.9.x
2. pip 22.3.x
3. OpenCV 4.7
4. Numpy 1.24.x
5. Tensorflow 2.10.0
6. Tensorflow Keras Resnet50

### Development Environment
The environment used to develop and test.

```
MacOSX 13.0.1

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

>>> import numpy as np
>>> np.__version__
'1.24.1'

>>>import tensorflow as tf
>>>print(tf.__version__)
2.10.0
```

---
# Structure

## Directory
Code is organized in the directory:

```
BASE
├── README.md
├── data
│   ├── master                            <--- Original image data                   
│   ├── landing                           <--- ETL output
│   │   ├── image_names.npy               <--- NPY_IMAGE_NAMES (to be created by ETL)
│   │   └── resized_rgb.npy               <--- NPY_RESIZED_RGB (to be created by ETL)
│   ├── feature_store                     <--- Feature engineering output
│   │   └── feature_engineered.npy        <--- NPY_FEATURE_ENGINEERED (to be created by FE)
│   └── model                             <--- Modelling output
│       ├── embedded_image_vectors.npy    <--- NPY_IMAGE_VECTORS   (to be created by modelling)
│       └── vectorizer_model              <--- TF_VECTORIZER_MODEL (to be created by modelling)
├── lib                                   <--- Set PYTHONPATH to here
│   ├── util_constant.py                  <--- Common constant
│   ├── util_file.py                      <--- Python file utility
│   ├── util_numpy.py                     <--- Numpy utility
│   ├── util_opencv
│   │   └── image.py                      <--- OpenCV image utility
│   └── util_tf
│       └── resnet50.py                   <--- TF/Keras Resnet utility
└── src
    ├── requiments.txt
    ├── pylintrc
    ├── etl.py                            <--- Resize image and convert to RGB
    ├── feature_engineering.py            <--- Feature engineering e.g. ResNet preprocessing
    ├── model.py                          <--- Vectorizer model and image vector generation
    ├── serve.py                          <--- Image search
    ├── function.py                       <--- Utility
    ├── _common.sh
    ├── run_modelling_pipeline.sh         <--- Run modeling pipeline
    └── run_serving_pipeline.sh           <--- Run image serach

```

## Code

Commonly used functions are placed under ```lib``` directory for reusability not to to repeat the same efforts.

### src/serving.py

#### ```ImageSearchEngine.most_similar```

The method in the class implements the image search based on the cosine similarities.

```
    def most_similar(
            self, query: np.ndarray, n: int = 5  # pylint: disable=invalid-name
    ) -> List[Tuple[float, str]]:
        """
        Return top n most similar images.
```

### src/model.py

#### ```Vectorizer.transform()``` 

The method in the class implements the image vectorization using the ResNet50 ```avg_pool``` layer output
to embed the images into a vector of the latent space of 2048 dimensions.

```
def transform(self, images: Sequence[np.ndarray]) -> Optional[np.ndarray]:
        """Transform list of images into numpy vectors of image features.
        Images should be preprocessed first (padding, resize, normalize,..).

        The results are embedded vectors each of which represents an image
        in a multidimensional space where proximity represents the similarity
        of the images.
```

### lib/util_numpy.py

#### ```get_cosine_similarity()```

The method implements the cosine similarity in the vectorized manner.

```
def get_cosine_similarity(x: numpy.ndarray, y: np.ndarray) -> np.ndarray:
    """Calculate cosine similarity
```

### lib/util_tf/resnet.py

#### ```ResNet50Helper```

The class implements the TF/Keras ResNet50 utility functions.

```
class ResNet50Helper:
    """TF Keras ResNet50 helper function implementations"""
```

### lib/util_opencv/image.py

The file implements the OpenCV utility functions.


---


# Execution

## Overview

There are two pipelines to: 
1. run modelling and by going through ```Data Engineering (ETL)->Feature Engineering->Modelling``` phases.
2. run model serving pipeline to run image search by going through ```Data Engineering->Feature Engineering->Serving```.

### Modelling Pipeline

```Data Engineering/ETL``` phase resizes the images and converts them to RGB format in memory. It generates the artifacts
and save them in the landing zone.
* NPY_RESIZED_RGB
* NPY_IMAGE_NAMES

```Feature Engineering``` phase processes the NPY_RESIZED_RGB and runs the feature engineering to generate 
the features NPY_FEATURE_ENGINEERED and save it to feature store. Then it can be used as the input to the 
ResNet model input layer. ResNet50 preprocessing tool is provided by TF/Keras.

```Modelling``` phase processes the NPY_FEATURE_ENGINEERED and generates NPY_IMAGE_VECTORS as the embedded
image vectors. The KF/Keras ResNet model artifacts up to the ```avg_pool``` layer is used to embed the images 
into vectors. The model is serialized to TF_VECTORIZER_MODEL to be consistently re-used in the serving phase.

### Serving Pipeline

The same transformations in ```Data Engineering``` and ```Feature Engineering``` will be applied to the query
image. ```Serving``` phases takes the embedded vector of the query image and searches the images based on the
vector cosine similarity.


## Setup

1. Place original images in ```data/master``` directory`.
2. Create a Pyton virtual environment.
    ```
    python -m venv ${VENV_NAME}
    ```
3. Activate the viertual environment and install python packages.
    ```
    pip install -r src/requirements.txt
    ```
4. Set the PYTHONPATH environment variable to include the ```lib/``` directory.

## Commands


1. To run the modelling pipeline to generate the embedding image vectors for image search:
   ```
   $ cd src/
   $ ./run_modelling_pipeline.sh
   ```

2. To run the image search:
   ```
   $ cd src/
   $ ./run_serving_pipeline.sh
   ```


---

# TODO

## PyTest

## Code Performance Analysis

* [line_profiler and kernprof](https://github.com/pyutils/line_profiler)


## Memory  Profiling
* [Memory Profiler](https://github.com/pythonprofilers/memory_profiler)

---

# References

* [Arxiv - Deep Residual Learning for Image Recognition](https://arxiv.org/abs/1512.03385v1)
* [Kerass ResNet and ResNetV2](https://keras.io/api/applications/resnet/#resnet50-function)
* [tf.keras.applications.resnet50.preprocess_input](https://www.tensorflow.org/api_docs/python/tf/keras/applications/resnet50/preprocess_input)
* [How should I standardize input when fine-tuning a CNN?](https://stats.stackexchange.com/questions/384484/how-should-i-standardize-input-when-fine-tuning-a-cnn/388461#388461)