#!/usr/bin/env python
# coding: utf-8
import tensorflow as tf
from tensorflow import keras
import pandas as pd
import matplotlib.pyplot as plt

print(tf.__version__)
print(keras.__version__)

# Load fashion MNIST
fashion_mnist = keras.datasets.fashion_mnist
(X_train_full, y_train_full), (X_test, y_test) = fashion_mnist.load_data()

X_valid, X_train = X_train_full[:5000] / 255.0, X_train_full[5000:] / 255.0
y_valid, y_train = y_train_full[:5000], y_train_full[5000:]
X_test = X_test / 255.0

print(X_train.dtype)
print(X_train.shape)

class_names = [
    "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat", "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"
]
print(class_names[y_train[0]])

# Model to train
model = keras.models.Sequential()
model.add(keras.layers.Flatten(input_shape=[28, 28]))
model.add(keras.layers.Dense(300, activation="relu"))
model.add(keras.layers.Dense(100, activation="relu"))
model.add(keras.layers.Dense(len(class_names), activation="softmax"))

# First layter 784 × 300 connection weights, plus 300 bias terms, which adds up to 235,500 parameters
model.summary()

# Get the weidhts and biases in a layer
hidden1 = model.layers[1]
print(hidden1.name)

weights, biases = hidden1.get_weights()

# Attach computation logic to the model
model.compile(
    loss="sparse_categorical_crossentropy",  # Cross entropy loss for sparse categorical labels
    # optimizer="sgd",
    optimizer=keras.optimizers.SGD(lr=0.01),  # Use stochastic gradient descent
    metrics=["accuracy"]  # Evaluate the model performance with accuracy.
)

# Plot training history
history = model.fit(X_train, y_train, epochs=30, validation_data=(X_valid, y_valid))
pd.DataFrame(history.history).plot(figsize=(8, 5))
plt.grid(True)
plt.gca().set_ylim(0, 1)  # set the vertical range to [0-1]
plt.show()

# Evaluate the model with test data.
model.evaluate(X_test, y_test)
