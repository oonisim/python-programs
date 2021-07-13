import os
import sys

import numpy as np
import tensorflow as tf
from transformers import (
    PreTrainedModel,
    DistilBertTokenizerFast,
    TFDistilBertModel,
    TFDistilBertForSequenceClassification,
    TFTrainer,
    TFTrainingArguments
)
from tensorflow.keras.models import (
    Sequential
)
from tensorflow.keras.layers import (
    Dense
)


class Runner:
    """Fine tuning implementation class
    See:
    - https://www.tensorflow.org/guide/keras/train_and_evaluate
    - https://stackoverflow.com/questions/68172891/
    - https://stackoverflow.com/a/68172992/4281353

    The TF/Keras model has the base model, e.g distilbert for DistiBERT which is
    from the base model TFDistilBertModel.
    https://huggingface.co/transformers/model_doc/distilbert.html#tfdistilbertmodel

    TFDistilBertForSequenceClassification has classification layers added on top
    of TFDistilBertModel, hence not required to add fine-tuning layers by users.
    _________________________________________________________________
    Layer (type)                 Output Shape              Param #
    =================================================================
    distilbert (TFDistilBertMain multiple                  66362880
    _________________________________________________________________
    pre_classifier (Dense)       multiple                  590592
    _________________________________________________________________
    classifier (Dense)           multiple                  1538
    _________________________________________________________________
    dropout_59 (Dropout)         multiple                  0
    =================================================================
    """
    # ================================================================================
    # Class
    # ================================================================================
    USE_HF_TRAINER = False
    USE_CUSTOM_MODEL = True
    USE_METRIC_AUC = True
    TOKENIZER_LOWER_CASE = True
    # _model_name = 'distilbert-base-cased'
    _model_name = 'distilbert-base-uncased'
    _model_base_name = 'distilbert'
    _tokenizer = DistilBertTokenizerFast.from_pretrained(
        _model_name,
        do_lower_case=TOKENIZER_LOWER_CASE
    )

    # ================================================================================
    # Instance
    # ================================================================================
    # --------------------------------------------------------------------------------
    # Instance properties
    # --------------------------------------------------------------------------------
    @property
    def category(self):
        """Category of the text comment classification, e.g. toxic"""
        return self._category

    @property
    def num_labels(self):
        """Number of labels to classify"""
        assert self._num_labels > 0
        return self._num_labels

    @property
    def tokenizer(self):
        """BERT tokenizer. The Tokenzer must match the pretrained model"""
        return self._tokenizer

    @property
    def max_sequence_length(self):
        """Maximum token length for the BERT tokenizer can accept. Max 512
        """
        assert 128 <= self._max_sequence_length <= 512
        return self._max_sequence_length

    @property
    def X(self):
        """Training TensorFlow DataSet"""
        return self._X

    @property
    def V(self):
        """Validation TensorFlow DataSet"""
        return self._V

    @property
    def model_name(self):
        """HuggingFace pretrained model name"""
        return self._model_name

    @property
    def model_base_name(self):
        """HuggingFace pretrained base model name"""
        return self._model_base_name

    @property
    def model(self):
        """TensorFlow/Keras Model instance"""
        return self._model

    @property
    def freeze_pretrained_base_model(self):
        """Boolean to freeze the base model"""
        return self._freeze_pretrained_base_model

    @property
    def batch_size(self):
        """Mini batch size during the training"""
        assert self._batch_size > 0
        return self._batch_size

    @property
    def learning_rate(self):
        """Training learning rate"""
        return self._learning_rate

    @property
    def l2(self):
        """Regularizer decay rate"""
        return self._l2

    @property
    def reduce_lr_patience(self):
        """Training patience for reducing learinig rate"""
        return self._reduce_lr_patience

    @property
    def reduce_lr_factor(self):
        """Factor to reduce the learinig rate"""
        return self._reduce_lr_factor

    @property
    def early_stop_patience(self):
        """Training patience for early stopping"""
        return self._early_stop_patience

    @property
    def num_epochs(self):
        """Number of maximum epochs to run for the training"""
        return self._num_epochs

    @property
    def output_directory(self):
        """Parent directory to manage training artefacts"""
        return self._output_directory

    @property
    def model_directory(self):
        """Directory to save the trained models"""
        return self._model_directory

    @property
    def log_directory(self):
        """Directory to save logs, e.g. TensorBoard logs"""
        return self._log_directory

    @property
    def model_metric_names(self):
        """Model mtrics
        The attribute model.metrics_names gives labels for the scalar metrics
        to be returned from model.evaluate().
        """
        return self.model.metrics_names

    @property
    def history(self):
        """The history object returned from model.fit().
        The object holds a record of the loss and metric during training
        """
        assert self._history is not None
        return self._history

    @property
    def trainer(self):
        """HuggingFace trainer instance
        HuggingFace offers an optimized Trainer because PyTorch does not have
        the training loop as Keras/Model has. It is available for TensorFlow
        as well, hence to be able to hold the instance in case using it.
        """
        return self._trainer

    # --------------------------------------------------------------------------------
    # Instance initialization
    # --------------------------------------------------------------------------------
    def build_custom_model_acc_callbacks(self):
        """Callbacks for accuracy"""
        return [
            EarlyStoppingCallback(
                patience=self.early_stop_patience,
                monitor=self._monitor_metric,
                mode=self._monitor_mode
            ),
            ReduceLRCallback(
                patience=self.reduce_lr_patience,
                monitor=self._monitor_metric,
                mode=self._monitor_mode
            ),
            ModelCheckpointCallback(
                self.model_directory + os.path.sep + 'model.h5',
                monitor=self._monitor_metric,
                mode=self._monitor_mode
            ),
            TensorBoardCallback(self.log_directory),
        ]

    def build_custom_model_auc_callbacks(self, validation_data, validation_label):
        """Callbacks for ROC AUC"""
        return [
            ROCCallback(
                validation_data=dict(self.tokenize(validation_data)),
                validation_label=validation_label,
                output_path=self.model_directory + os.path.sep + 'model.h5',
                reduce_lr_patience=self.reduce_lr_patience,
                reduce_lr_factor=self.reduce_lr_factor,
                early_stop_patience=self.early_stop_patience,
                verbose=True
            ),
            TensorBoardCallback(self.log_directory),
        ]

    def build_huggingface_model(self):
        """Build model based on TFDistilBertForSequenceClassification which has
        classification heads added on top of the base BERT model.
        """
        # --------------------------------------------------------------------------------
        # Base model
        # --------------------------------------------------------------------------------
        config_file = self.model_directory + os.path.sep + "config.json"
        if os.path.isfile(config_file) and os.access(config_file, os.R_OK):
            # Load the saved model
            print(f"\nloading the saved huggingface model from {self.model_directory}...\n")
            self._pretrained_model = TFDistilBertForSequenceClassification.from_pretrained(
                self.model_directory,
                num_labels=self.num_labels
            )
        else:
            # Download the model from Huggingface
            self._pretrained_model = TFDistilBertForSequenceClassification.from_pretrained(
                self.model_name,
                num_labels=self.num_labels,
            )

        # Freeze base model if required
        if self.freeze_pretrained_base_model:
            for _layer in self._pretrained_model.layers:
                if _layer.name == self.model_base_name:
                    _layer.trainable = False

        self._model = self._pretrained_model

        # The number of classes in the output must match the num_labels
        _output = self._pretrained_model(self.tokenize(["i say hello"]))
        assert _output['logits'].shape[-1] == self.num_labels, "Number of labels mismatch"

        # --------------------------------------------------------------------------------
        # Model Metrics
        # --------------------------------------------------------------------------------
        self._metric_name = "accuracy"
        self._monitor_metric = "val_loss"
        self._monitor_mode = 'min'
        self._callbacks = self.build_custom_model_acc_callbacks()

        # --------------------------------------------------------------------------------
        # Build the model
        #     from_logits in SparseCategoricalCrossentropy(from_logits=[True|False])
        #     True  when the input is logits not  normalized by softmax.
        #     False when the input is probability normalized by softmax
        # --------------------------------------------------------------------------------
        optimizer = tf.keras.optimizers.Adam(learning_rate=self.learning_rate)
        self.model.compile(
            optimizer=optimizer,
            # loss=self.model.compute_loss,
            loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
            # loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
            # ["accuracy", "AUC"] causes an error:
            # ValueError: Shapes (None, 1) and (None, 2) are incompatible
            metrics=["accuracy"]
        )

    def build_custom_model(self, validation_data, validation_label):
        # --------------------------------------------------------------------------------
        # Input layer (token indices and attention masks)
        # --------------------------------------------------------------------------------
        input_ids = tf.keras.layers.Input(shape=(self.max_sequence_length,), dtype=tf.int32, name='input_ids')
        attention_mask = tf.keras.layers.Input((self.max_sequence_length,), dtype=tf.int32, name='attention_mask')

        # --------------------------------------------------------------------------------
        # Base layer
        # --------------------------------------------------------------------------------
        # TFBaseModelOutput.last_hidden_state has shape (batch_size, max_sequence_length, 768)
        # Each sequence has [CLS]...[SEP] structure of shape (max_sequence_length, 768)
        # Extract [CLS] embeddings of shape (batch_size, 768) as last_hidden_state[:, 0, :]
        # --------------------------------------------------------------------------------
        base = TFDistilBertModel.from_pretrained(
            self.model_name,
        )
        # Freeze the base model weights.
        if self.freeze_pretrained_base_model:
            for layer in base.layers:
                layer.trainable = False

        base.summary()
        output = base([input_ids, attention_mask]).last_hidden_state[:, 0, :]

        # --------------------------------------------------------------------------------
        # TODO:
        #    Need to verify the effect of regularizers.
        #
        #    [bias regularizer]
        #    It looks bias_regularizer adjusts the ROC threshold towards 0.5.
        #    Without it, the threshold of the ROC with BinaryCrossEntropy loss was approx 0.02.
        #    With    it, the threshold of the ROC with BinaryCrossEntropy loss was approx 0.6.
        # --------------------------------------------------------------------------------
        activation = "sigmoid" if self.num_labels == 1 else "softmax"
        output = tf.keras.layers.Dense(
            units=self.num_labels,
            kernel_initializer='glorot_uniform',
            # https://huggingface.co/transformers/v4.3.3/main_classes/optimizer_schedules.html#adamweightdecay-tensorflow
            # kernel_regularizer=tf.keras.regularizers.l2(l2=self.l2),
            # bias_regularizer=tf.keras.regularizers.l2(l2=self.l2),
            # activity_regularizer=tf.keras.regularizers.l2(l2=self.l2/10.0),
            activation=activation,
            name=activation
        )(output)

        # --------------------------------------------------------------------------------
        # Loss layer
        # --------------------------------------------------------------------------------
        if self.num_labels == 1:  # Binary classification
            loss_fn = tf.keras.losses.BinaryCrossentropy(from_logits=False)
        else:  # Categorical classification
            loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False)

        # --------------------------------------------------------------------------------
        # Model Metrics
        # --------------------------------------------------------------------------------
        if self.num_labels == 1:
            if self.USE_METRIC_AUC:  # ROC/AUC
                # AUC is for Binary Classification. Error if used for categorical"
                # "alueError: Shapes (None, <num_classes>) and (None, 1) are incompatible"
                # Because AUC is expecting shape(None, 1) as binary input into the loss fn.
                self._metric_name = "auc"
                self._monitor_metric = f"val_{self._metric_name}"
                self._monitor_mode = 'max'
                self._metrics = [
                    tf.keras.metrics.AUC(from_logits=False, name=self._metric_name),
                    tf.keras.metrics.Recall(name="recall"),
                    "accuracy"
                ]
                self._callbacks = self.build_custom_model_auc_callbacks(
                    validation_data,
                    validation_label
                )
            else:
                self._metric_name = "recall"  # Recall
                self._monitor_metric = f"val_{self._metric_name}"
                self._monitor_mode = 'max'
                self._metrics = [tf.keras.metrics.Recall(name=self._metric_name), "accuracy"]
                self._callbacks = self.build_custom_model_acc_callbacks()

        else:  # Validation loss
            self._metric_name = "accuracy"
            self._monitor_metric = "val_loss"
            self._monitor_mode = 'min'
            # metrics=[tf.keras.metrics.Accuracy(name=metric_name)]
            self._metrics = [self._metric_name]
            self._callbacks = self.build_custom_model_acc_callbacks()

        # --------------------------------------------------------------------------------
        # Build model
        # --------------------------------------------------------------------------------
        # TODO: Replace TIMESTAMP with instance variable
        name = f"{TIMESTAMP}_{self.model_name.upper()}"
        self._model = tf.keras.models.Model(inputs=[input_ids, attention_mask], outputs=output, name=name)
        self.model.compile(
            # https://huggingface.co/transformers/v4.3.3/main_classes/optimizer_schedules.html#adamweightdecay-tensorflow
            # optimizer=tf.keras.optimizers.Adam(learning_rate=self.learning_rate),
            optimizer=transformers.AdamWeightDecay(learning_rate=self.learning_rate),
            loss=loss_fn,
            metrics=self._metrics
        )

        # --------------------------------------------------------------------------------
        # Load model parameters if the saved weight file exits
        # --------------------------------------------------------------------------------
        path_to_h5 = self.model_directory + os.path.sep + "model.h5"
        if os.path.isfile(path_to_h5) and os.access(path_to_h5, os.R_OK):
            print(f"\nloading the saved model parameters from {path_to_h5}...\n")
            self.model.load_weights(path_to_h5)

    def __init__(
            self,
            category,
            training_data,
            training_label,
            validation_data,
            validation_label,
            num_labels=2,
            max_sequence_length=256,
            freeze_pretrained_base_model=False,
            batch_size=16,
            learning_rate=5e-5,
            l2=1e-4,
            metric_name="accuracy",
            monitor_metric='val_loss',
            monitor_mode='min',
            early_stop_patience=5,
            reduce_lr_patience=2,
            reduce_lr_factor=0.2,
            num_epochs=3,
            output_directory="./output"
    ):
        """
        Args:
            category:
            training_data:
            training_label:
            validation_data:
            validation_label:
            num_labels: Number of labels
            max_sequence_length=256: maximum tokens for tokenizer
            freeze_pretrained_base_model: flag to freeze pretrained model base layer
            batch_size:
            learning_rate:
            l2: L2 regularizer decay rate
            metric_name: metric for the model
            monitor_metric: metric to monitor for callbacks
            monitor_mode: auto|min|max
            early_stop_patience:
            reduce_lr_patience:
            reduce_lr_factor:
            num_epochs:
            output_directory: Directory to save the outputs
        """
        self._category = category
        self._trainer = None

        # --------------------------------------------------------------------------------
        # Model training configurations
        # --------------------------------------------------------------------------------
        assert 128 <= max_sequence_length <= 512, "Current max sequenth length is 512"
        self._max_sequence_length = max_sequence_length

        assert num_labels > 0
        self._num_labels = num_labels

        assert isinstance(freeze_pretrained_base_model, bool)
        self._freeze_pretrained_base_model = freeze_pretrained_base_model

        assert (0.0 < learning_rate) and (0 <= l2 < 1.0)
        self._learning_rate = learning_rate
        self._l2 = l2
        self._model = None

        assert num_epochs > 0
        self._num_epochs = num_epochs

        assert batch_size > 0
        self._batch_size = batch_size

        assert early_stop_patience > 0
        self._metric_name = metric_name
        self._monitor_metric = monitor_metric
        self._monitor_mode = monitor_mode
        self._early_stop_patience = early_stop_patience
        self._reduce_lr_patience = reduce_lr_patience
        self._reduce_lr_factor = reduce_lr_factor

        # model.fit() result holder
        self._history = None

        # --------------------------------------------------------------------------------
        # Output directories
        # --------------------------------------------------------------------------------
        # Parent directory
        self._output_directory = output_directory
        Path(self.output_directory).mkdir(parents=True, exist_ok=True)

        # Model directory
        self._model_directory = "{parent}/model_C{category}_B{size}_L{length}".format(
            parent=self.output_directory,
            category=self.category,
            size=self.batch_size,
            length=self.max_sequence_length
        )
        Path(self.model_directory).mkdir(parents=True, exist_ok=True)

        # Log directory
        self._log_directory = "{parent}/log_C{category}_B{size}_L{length}".format(
            parent=self.output_directory,
            category=self.category,
            size=self.batch_size,
            length=self.max_sequence_length
        )
        Path(self.log_directory).mkdir(parents=True, exist_ok=True)

        # --------------------------------------------------------------------------------
        # TensorFlow DataSet
        # --------------------------------------------------------------------------------
        if self.num_labels == 1:
            assert np.all(np.isin(training_label, [0, 1]))
            assert np.all(np.isin(validation_label, [0, 1]))
        else:
            assert np.all(np.isin(training_label, np.arange(self.num_labels)))
            assert np.all(np.isin(validation_label, np.arange(self.num_labels)))

        self._X = tf.data.Dataset.from_tensor_slices((
            dict(self.tokenize(training_data)),
            training_label
        ))
        self._V = tf.data.Dataset.from_tensor_slices((
            dict(self.tokenize(validation_data)),
            validation_label
        ))
        del training_data, training_label

        # --------------------------------------------------------------------------------
        # Model
        # --------------------------------------------------------------------------------
        if self.USE_CUSTOM_MODEL:
            self.build_custom_model(validation_data, validation_label)
            self._train_fn = self._keras_train_custom_model
            self._save_fn = self._save_custom_model
            self._load_fn = self._load_custom_model
            if self.num_labels == 1:
                self._predict_fn = self._keras_predict_custom_model_binary
            else:
                self._predict_fn = self._keras_predict_custom_model_categorical
        else:
            self.build_huggingface_model()
            self._train_fn = self._keras_train_huggingface_model
            self._predict_fn = self._keras_predict_huggnigface_model
            self._save_fn = self._save_huggingface_model
            self._load_fn = self._load_huggingface_model

        del validation_data, validation_label
        self.model.summary()

        # --------------------------------------------------------------------------------
        # Model validations
        # --------------------------------------------------------------------------------
        test_sentences = [
            "i am a cat who has no name.",
            "to be or not to be."
        ]
        test_tokens = self.tokenize(test_sentences, padding='max_length')
        TEST_BATCH_SIZE = len(test_tokens)

        # Model generates predictions for all the classes/labels.
        # including the binary classification where num_labels == 1
        test_model_output = self.model(test_tokens)
        assert test_model_output.shape == (TEST_BATCH_SIZE, self.num_labels), \
            "test_model_output type[%s] data [%s]" % \
            (type(test_model_output), test_model_output)

        # predict returns probabilities for the target class/label only
        # in a np array of shape (batch_size, 1). The probability value
        # is between 0 and 1.
        test_predictions = self.predict(test_sentences)
        assert test_predictions.shape == (TEST_BATCH_SIZE, 1), \
            "test_predictions shape[%s] data [%s]" % \
            (test_predictions.shape, test_predictions)
        assert np.all(0 < test_predictions) and np.all(test_predictions < 1)

        # --------------------------------------------------------------------------------

    # Instance methods
    # --------------------------------------------------------------------------------
    def tokenize(self, sentences, truncation=True, padding='longest'):
        """Tokenize using the Huggingface tokenizer
        Args:
            sentences: String or list of string to tokenize
            padding: Padding method ['do_not_pad'|'longest'|'max_length']
        """
        return self.tokenizer(
            sentences,
            truncation=truncation,
            padding=padding,
            max_length=self.max_sequence_length,
            return_tensors="tf"
        )

    def decode(self, tokens):
        return tokenizer.decode(tokens)

    def _hf_train(self):
        """Train the model using HuggingFace Trainer"""
        self._training_args = TFTrainingArguments(
            output_dir='./results',  # output directory
            num_train_epochs=3,  # total number of training epochs
            per_device_train_batch_size=self.batch_size,  # batch size per device during training
            per_device_eval_batch_size=self.batch_size,  # batch size for evaluation
            warmup_steps=500,  # number of warmup steps for learning rate scheduler
            weight_decay=0.01,  # strength of weight decay
            logging_dir='./logs',  # directory for storing logs
            logging_steps=10,
        )

        # with self._training_args.strategy.scope():
        #     self._model = TFDistilBertForSequenceClassification.from_pretrained(self.model_name)

        self._trainer = TFTrainer(
            model=self.model,
            args=self._training_args,  # training arguments
            train_dataset=self.X,  # training dataset
            eval_dataset=self.V  # evaluation dataset
        )
        self.trainer.train()

    def _keras_train_huggingface_model(self):
        """Train the model using Keras
        """
        # --------------------------------------------------------------------------------
        # Train the model
        # --------------------------------------------------------------------------------
        self._history = self.model.fit(
            self.X.shuffle(1000).batch(self.batch_size).prefetch(1),
            epochs=self.num_epochs,
            batch_size=self.batch_size,
            validation_data=self.V.shuffle(1000).batch(self.batch_size).prefetch(1),
            callbacks=[
                SavePretrainedCallback(output_dir=self.model_directory, verbose=True),
                ReduceLRCallback(patience=self.reduce_lr_patience),
                EarlyStoppingCallback(patience=self.early_stop_patience),
                # TensorBoardCallback(self.log_directory),
            ]
        )

    def _keras_train_custom_model(self):
        """Train the model using Keras
        """
        # --------------------------------------------------------------------------------
        # Train the model
        # --------------------------------------------------------------------------------
        self._history = self.model.fit(
            self.X.shuffle(1000).batch(self.batch_size).prefetch(1),
            epochs=self.num_epochs,
            batch_size=self.batch_size,
            validation_data=self.V.shuffle(1000).batch(self.batch_size).prefetch(1),
            callbacks=self._callbacks
        )

    def train(self):
        """Run the model trainig"""
        self._train_fn()

    def evaluate(self, data, label):
        """Evaluate the model on the given data and label.
        https://www.tensorflow.org/api_docs/python/tf/keras/Model#evaluate
        The attribute model.metrics_names gives labels for the scalar metrics
        to be returned from model.evaluate().

        Args:
            data: data to run the prediction
            label: label for the data
        Returns:
            scalar loss if the model has a single output and no metrics, OR
            list of scalars (if the model has multiple outputs and/or metrics).
        """
        if self.num_labels == 1:
            assert np.all(np.isin(label, [0, 1]))
        else:
            assert np.all(np.isin(label, np.arange(self.num_labels)))

        test_dataset = tf.data.Dataset.from_tensor_slices((
            dict(self.tokenize(data)),
            label
        ))
        evaluation = self.model.evaluate(
            test_dataset.shuffle(1000).batch(self.batch_size).prefetch(1)
        )
        return evaluation

    def _keras_predict_custom_model_binary(self, data):
        """Calcuate the prediction for the data
        Args:
            data: sentences to tokenize of type List[str]
        Returns: Probabilities for label 1 as numpy array of shape (batch_size, 1)
        """
        tokens = dict(self.tokenize(data, padding='max_length'))
        probabilities = self.model.predict(tokens)
        assert isinstance(probabilities, np.ndarray)
        return probabilities

    def _keras_predict_custom_model_categorical(self, data):
        """Calcuate the prediction for the data
        Args:
            data: sentences to tokenize of type List[str]
        Returns: Probabilities for label 1 as numpy array of shape (batch_size, 1)
        """
        tokens = dict(self.tokenize(data, padding='max_length'))
        probabilities = self.model.predict(tokens)
        assert isinstance(probabilities, np.ndarray) and probabilities.ndim == 2
        return probabilities[:, 1:2]

    def _keras_predict_huggnigface_model(self, data):
        """Calcuate the probability for each label
        Args:
            data: sentences to tokenize of type List[str]
        Returns: Probabilities for label 1 as numpy array of shape (batch_size, 1)
        """
        tokens = dict(self.tokenize(data))
        logits = self.model.predict(tokens)["logits"]
        # [:, 1:2] -> TensorFlow Tensor indices to select column 1 for all rows
        return tf.nn.softmax(logits)[:, 1:2].numpy()

    def predict(self, sentences):
        """Generate prediction (probabilities) for the target label
        Args:
            sentences: text sentences of type str or List[str]
        Return:
            normalized probabilities in numpy array via sigmoid or softmax
        """
        result = self._predict_fn(sentences)
        assert isinstance(result, np.ndarray) and result.shape[-1] == 1, \
            f"Expected np.ndarray but {type(result)} {result}"
        return result

    def _save_huggingface_model(self, path_to_dir):
        """Save Keras model in huggingface format
        """
        self.model.save_pretrained(path_to_dir)

    def _save_custom_model(self, path_to_dir):
        """Save Keras model in "tf" format for explicit save.
        Use h5 for auto-save model during the trainig to avoid overwriting
        the best model saved during the training.
        """
        self.model.save_weights(
            path_to_dir, overwrite=True, save_format='tf'
        )

    def save(self, path_to_dir):
        """Save the model from the HuggingFace.
        - config.json
        - tf_model.h5

        TODO:
            Save the best model metrics when saving the model as Keras config file.
            Reload the best metrics of the model when loading the model itself.

            If the saved best model is re-loaded, the best metric values that the
            model achieved need to be re-loaded as well. Otherwise the first epoch
            result, even if the metrics are worse than the best metrics achieved,
            will become the best results and the best model will be overwritten with
            the inferior model.

        Args:
            path_to_dir: directory path to save the model artefacts
        """
        # path_to_dir is mandatory to avoid overwriting the best model saved
        # during the training
        Path(path_to_dir).mkdir(parents=True, exist_ok=True)
        self._save_fn(path_to_dir)

    def _load_huggingface_model(self, path_to_dir):
        self._model = TFDistilBertForSequenceClassification.from_pretrained(path_to_dir)

    def _load_custom_model(self, path_to_dir):
        path_to_h5 = path_to_dir + os.path.sep + 'model.h5'
        self.model.load_weights(path_to_h5)

    def load(self, path_to_dir):
        """Load the model as the HuggingFace format.
        TODO:
            Reload the best metrics of the model when loading the model itself.
            If the saved best model is re-loaded, the best metric values that the
            model achieved need to be re-loaded as well. Otherwise the first epoch
            result, even if the metrics are worse than the best metrics achieved,
            will become the best results and the best model will be overwritten with
            the inferior model.

        Args:
            path_to_dir: Directory path from where to load config.json and .h5.
        """
        if os.path.isdir(path_to_dir) and os.access(path_to_dir, os.R_OK):
            self._load_fn(path_to_dir)
        else:
            raise RuntimeError(f"{path_to_dir} does not exit")