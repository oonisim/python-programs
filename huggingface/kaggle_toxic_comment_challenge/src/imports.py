import os
import sys
from pathlib import Path
import transformers
import numpy as np
import tensorflow as tf
from sklearn.metrics import roc_auc_score
from tensorflow.python.keras import backend
from transformers import (
    DistilBertTokenizerFast,
    TFDistilBertModel,
    AdamWeightDecay,
    TFDistilBertForSequenceClassification,
    TFTrainingArguments,
    TFTrainer,
)
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_curve, auc

# Project ID
PROJECT = "toxic_comment_classification"

# To reduce the data volumen to run through the training in short timeframe.
TEST_MODE = False
NUM_TEST_SAMPLES = 256

## Execution recording (e.g. 2021JUL012322)
#TIMESTAMP = datetime.datetime.now().strftime("%Y%b%d%H%M").upper()
TIMESTAMP = input("Enter TIMESTAMP")
print(f"Execution timestamp {TIMESTAMP}")

# Directory to manage the data.
# Place jigsaw-toxic-comment-classification-challenge.zip in DATA_DIR
DATA_DIR = "."
OUTPUT_DIR = "."

# Flag to clear data or not
CLEANING_FOR_ANALYSIS = True
CLEANING_FOR_TRAINING = False

# Flag to overwrite the cleaned data
FORCE_OVERWRITE = False

# Labbels that classifies the type of the comment.
# CATEGORIES = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]
CATEGORIES = ["toxic"]
RESULT_DIRECTORY = ""

PICKLE_DIR = "."

def get_ipython():
    pass