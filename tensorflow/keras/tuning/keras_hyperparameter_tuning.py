import numpy as np
import tensorflow as tf
from keras.layers import (
    Dense,
    Flatten,
    Normalization,
    Conv2D,
    MaxPooling2D,
)
from keras.models import (
    Sequential
)
from scikeras.wrappers import (
    KerasClassifier,
)
from sklearn.model_selection import (
    GridSearchCV
)

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
max_value = float(np.max(x_train))
x_train, x_test = x_train/max_value, x_test/max_value

input_shape = x_train[0].shape
number_of_classes = 10

# Data Normalization
normalization = Normalization(
    name="norm",
    input_shape=input_shape,  # (32, 32, 3)
    axis=-1  # Regard each pixel as a feature
)
normalization.adapt(x_train)


def create_model():
    model = Sequential([
        # normalization,
        Conv2D(
            name="conv",
            filters=32,
            kernel_size=(3, 3),
            strides=(1, 1),
            padding="same",
            activation='relu',
            input_shape=input_shape
        ),
        MaxPooling2D(
            name="maxpool",
            pool_size=(2, 2)
        ),
        Flatten(),
        Dense(
            name="full",
            units=100,
            activation="relu"
        ),
        Dense(
            name="label",
            units=number_of_classes,
            activation="softmax"
        )
    ])
    model.compile(
        loss=tf.keras.losses.sparse_categorical_crossentropy,
        optimizer='adam',
        metrics=['accuracy']
    )
    return model


model = KerasClassifier(model=create_model, verbose=2)
batch_size = [32]
epochs = [2, 3]
param_grid = dict(batch_size=batch_size, epochs=epochs)

grid = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=-1, cv=3)
grid_result = grid.fit(x_train, y_train)
