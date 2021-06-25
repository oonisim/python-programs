import re
import os
import sys
from pathlib import Path

import numpy as np
import tensorflow as tf

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_curve, auc

import matplotlib.pyplot as plt
import pandas as pd

from transformers import (
    PreTrainedModel,
    DistilBertTokenizerFast,
    TFDistilBertForSequenceClassification,
    TFTrainer,
    TFTrainingArguments
)

pd.options.display.max_colwidth = 1000


"""## Google Colab"""

try:
    import google.colab

    IN_GOOGLE_COLAB = True
    DATA_PATH = "/content/drive/MyDrive/data/jigsaw-toxic-comment-classification-challenge.zip"
    google.colab.drive.mount('/content/drive')
except:
    IN_GOOGLE_COLAB = False
    DATA_PATH = ("Enter the data archive path")

"""---
# Data

First, upload data to 
"""

raw_train = pd.read_csv("./train.csv")
raw_test_data = pd.read_csv("./test.csv")
raw_test_label = pd.read_csv("./test_labels.csv")
raw_test = pd.merge(raw_test_data, raw_test_label, left_on='id', right_on='id', how='inner')

"""## Training (Raw)"""

raw_train.head()
raw_train.describe()

raw_train[raw_train['toxic'] > 0].head(5)

"""## Test (Raw)
The label value -1 is not clear. Remove the rows where value is -1.

> test_labels.csv - labels for the test data; value of -1 indicates it was not used for scoring
"""

raw_test = raw_test[(raw_test['toxic'] > 0)]  # Removing rows where 'toxic' label > 0 is sufficicent

"""## Trainig (Toxic) 

"""


class SavePretrainedCallback(tf.keras.callbacks.Callback):
    # Hugging Face models have a save_pretrained() method that saves both the weights and the necessary
    # metadata to allow them to be loaded as a pretrained model in future. This is a simple Keras callback
    # that saves the model with this method after each epoch.
    def __init__(self, output_dir, **kwargs):
        super().__init__()
        self.output_dir = output_dir

    def on_epoch_end(self, epoch, logs=None):
        self.model.save_pretrained(self.output_dir)


class Runner:
    # ================================================================================
    # Class
    # ================================================================================
    USE_HF_TRAINER = False
    _model_name = 'distilbert-base-cased'
    _tokenizer = DistilBertTokenizerFast.from_pretrained(_model_name)

    # ================================================================================
    # Instance
    # ================================================================================
    # --------------------------------------------------------------------------------
    # Instance properties
    # --------------------------------------------------------------------------------
    @property
    def batch_size(self):
        assert self._batch_size > 0
        return self._batch_size

    @property
    def X(self):
        """Training DataSet"""
        return self._X

    @property
    def V(self):
        """Validation DataSet"""
        return self._V

    @property
    def model_name(self):
        """HuggingFace pretrained model name"""
        return self._model_name

    @property
    def model(self):
        """Model"""
        return self._model

    @property
    def learning_rate(self):
        return self._learning_rate

    @property
    def num_epochs(self):
        return self._num_epochs

    @property
    def tokenizer(self):
        """"""
        return self._tokenizer

    @property
    def trainer(self):
        """"""
        return self._trainer

    @property
    def output_directory(self):
        """Directory to save models, etc"""
        return self._output_directory

    # --------------------------------------------------------------------------------
    # Instance initialization
    # --------------------------------------------------------------------------------
    def __init__(
            self,
            training_data,
            training_label,
            validation_data,
            validation_label,
            batch_size=16,
            learning_rate=5e-5,
            num_epochs=3,
            output_directory="./output"
    ):
        # --------------------------------------------------------------------------------
        # TensorFlow DataSet
        # --------------------------------------------------------------------------------
        assert np.all(np.isin(training_label, [0, 1]))
        assert np.all(np.isin(validation_label, [0, 1]))
        self._X = tf.data.Dataset.from_tensor_slices((
            dict(self.tokenizer(training_data, truncation=True, padding=True)),
            training_label
        ))
        self._V = tf.data.Dataset.from_tensor_slices((
            dict(self.tokenizer(validation_data, truncation=True, padding=True)),
            validation_label
        ))
        assert batch_size > 0
        self._batch_size = batch_size

        # --------------------------------------------------------------------------------
        # Keras Model
        # --------------------------------------------------------------------------------
        assert learning_rate > 0.0
        self._learning_rate = learning_rate
        self._model = None

        assert num_epochs > 0
        self._num_epochs = num_epochs

        assert os.path.isdir(output_directory) and os.access(output_directory, os.W_OK)
        self._output_directory = output_directory
        Path(self.output_directory).mkdir(parents=True, exist_ok=True)

        # --------------------------------------------------------------------------------
        # HuggingFace
        # --------------------------------------------------------------------------------
        self._trainer = None

    # --------------------------------------------------------------------------------
    # Instance methods
    # --------------------------------------------------------------------------------
    def _hf_train(self):
        self._training_args = TFTrainingArguments(
            output_dir='./results',             # output directory
            num_train_epochs=3,                 # total number of training epochs
            per_device_train_batch_size=self.batch_size,     # batch size per device during training
            per_device_eval_batch_size=self.batch_size,      # batch size for evaluation
            warmup_steps=500,                   # number of warmup steps for learning rate scheduler
            weight_decay=0.01,                  # strength of weight decay
            logging_dir='./logs',               # directory for storing logs
            logging_steps=10,
        )

        with self._training_args.strategy.scope():
            self._model = TFDistilBertForSequenceClassification.from_pretrained(self.model_name)

        self._trainer = TFTrainer(
            model=self._model,
            args=self._training_args,   # training arguments
            train_dataset=self.X,       # training dataset
            eval_dataset=self.V         # evaluation dataset
        )
        self.trainer.train()

    def _keras_train(self):
        self._model = TFDistilBertForSequenceClassification.from_pretrained(self.model_name)
        optimizer = tf.keras.optimizers.Adam(learning_rate=self.learning_rate)
        self.model.compile(optimizer=optimizer, loss=self.model.compute_loss)
        self.model.summary()
        self.model.fit(
            self.X.shuffle(1000).batch(self.batch_size).prefetch(1),
            epochs=self.num_epochs,
            batch_size=self.batch_size,
            validation_data=self.V.shuffle(1000).batch(self.batch_size).prefetch(1),
            callbacks=[SavePretrainedCallback(output_dir=self.output_directory)]
        )

    def train(self):
        if self.USE_HF_TRAINER:
            self._hf_train()
        else:
            self._keras_train()

    def evaluate(self, data, label):
        assert np.all(np.isin(label, [0, 1]))
        test_dataset = tf.data.Dataset.from_tensor_slices((
            dict(self.tokenizer(data, truncation=True, padding=True)),
            label
        ))
        evaluation = self.model.evaluate(
            test_dataset.shuffle(1000).batch(self.batch_size).prefetch(1)
        )
        print(f"Evaluation: (loss, accuracy):{evaluation}")

    def save(self, path_to_dir):
        if os.path.isdir(path_to_dir) and os.access(path_to_dir, os.W_OK):
            Path(path_to_dir).mkdir(parents=True, exist_ok=True)
            self.trainer.save_model(path_to_dir) if self.USE_HF_TRAINER else self.model.save(path_to_dir)
        else:
            raise RuntimeError(f"Cannot write to {path_to_dir} directory.")

    def load(self, path_to_dir):
        if os.path.isdir(path_to_dir) and os.access(path_to_dir, os.R_OK):
            self._model = PreTrainedModel.from_pretrained(path_to_dir)


toxic_data = raw_train['comment_text']
toxic_label = raw_train['toxic']
toxic_train_data, toxic_validation_data, toxic_train_label, toxic_validation_label = train_test_split(
    toxic_data,
    toxic_label,
    test_size=.2,
    shuffle=True
)


def test():
    runner = Runner(
        training_data=toxic_train_data,
        training_label=toxic_train_label,
        validation_data=toxic_validation_data,
        validation_label=toxic_validation_label,
    )
    runner.train()