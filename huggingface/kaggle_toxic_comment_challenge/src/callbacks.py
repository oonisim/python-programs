import os
import sys
import numpy as np
import tensorflow as tf
from sklearn.metrics import roc_auc_score
from keras import backend


class ROCCallback(tf.keras.callbacks.Callback):
    """Take actions on the model training based on ROC/AUC
    """

    def __init__(
            self,
            validation_data,
            validation_label,
            output_path,
            output_format='h5',
            criterion=1,
            reduce_lr_patience=2,
            reduce_lr_factor=0.2,
            early_stop_patience=sys.maxsize,
            verbose=True
    ):
        """
        Args:
            validation_data: data to generate prediction to calculate ROC/AUC
            validation_label: label to calculate ROC/AUC
            output_path: path to save the model upon improvement
            output_format: model save format 'tf' or 'h5'
            reduce_lr_patience: number of consecutive no-improvement upon which to reduce LR.
            reduce_lr_factor: new learning rate = reduce_lr_factor * old learning rate
            early_stop_patience: total number of no-improvements upon which to stop the training
            verbose: [True|False] to output extra information
        """
        super().__init__()
        assert 0.0 < reduce_lr_factor < 1.0
        assert 0 < reduce_lr_patience
        assert 0 < early_stop_patience

        # --------------------------------------------------------------------------------
        # ROC/AUC calculation data
        # TODO:
        #    When recreating the data e.g. in the on_epoch_end in tf.keras.utils.Sequence
        #    then need to update the x, y accordingly here.
        # --------------------------------------------------------------------------------
        self.x = validation_data
        self.y = validation_label

        # --------------------------------------------------------------------------------
        # Training control parameters
        # --------------------------------------------------------------------------------
        self.criterion = criterion
        self.reduce_lr_patience = reduce_lr_patience
        self.reduce_lr_factor = reduce_lr_factor
        self.early_stop_patience = early_stop_patience
        self.output_path = output_path
        self.output_format = output_format
        self.verbose = verbose

        # --------------------------------------------------------------------------------
        # Statistics
        # --------------------------------------------------------------------------------
        self.max_roc_auc = -1
        self.min_val_loss = np.inf
        self.best_epoch = -1
        self.successive_no_improvement = 0
        self.total_no_improvement = 0

    def on_train_begin(self, logs={}):
        """Reset the statistics.
        The class instance can be re-used throughout multiple training runs
        """
        self.successive_no_improvement = 0
        self.total_no_improvement = 0

        # --------------------------------------------------------------------------------
        # DO NOT reset the best metric values that the model have achieved.
        # If restart the training on the same model, improvements needs to be measured
        # with the last best metrics of the model, not the initial values e.g -1 or np.inf.
        #
        # TODO:
        #    Save the best model metrics when saving the model as Keras config file.
        #    Reload the best metrics of the model when loading the model itself.
        #
        #    If the saved best model is re-loaded, the best metric values that the
        #    model achieved need to be re-loaded as well. Otherwise the first epoch
        #    result, even if the metrics are worse than the best metrics achieved,
        #    will become the best results and the best model will be overwritten with
        #    the inferior model.
        # --------------------------------------------------------------------------------
        # self.max_roc_auc = -1
        # self.min_val_loss = np.inf
        # self.best_epoch = -1

        return

    def on_train_end(self, logs={}):
        return

    def on_epoch_begin(self, epoch, logs={}):
        return

    def _reduce_learning_rate(self):
        old_lr = backend.get_value(self.model.optimizer.lr)
        new_lr = old_lr * self.reduce_lr_factor
        backend.set_value(self.model.optimizer.lr, new_lr)
        self.successive_no_improvement = 0
        if self.verbose:
            print(f"Reducing learning rate to {new_lr}.")

    def _stop_early(self):
        if self.verbose:
            print(
                "Early stopping: no improvement [%s] times. best epoch [%s] AUC [%5f] val_loss [%5f]" %
                (self.total_no_improvement, self.best_epoch + 1, self.max_roc_auc, self.min_val_loss)
            )
        self.model.stop_training = True
        self.total_no_improvement = 0
        self.successive_no_improvement = 0

    def _handle_improvement(self, epoch, roc_auc, roc_auc_prev, val_loss, val_loss_prev):
        if self.verbose:
            print(
                "Model improved auc [%5f > %5f] val_loss [%5f < %5f]. Saving to %s" %
                (roc_auc, roc_auc_prev, val_loss, val_loss_prev, self.output_path)
            )

        # --------------------------------------------------------------------------------
        # Update statistics
        # --------------------------------------------------------------------------------
        self.best_epoch = epoch
        self.successive_no_improvement = 0

        # --------------------------------------------------------------------------------
        # Save the model upon improvement
        # --------------------------------------------------------------------------------
        self.model.save_weights(
            self.output_path, overwrite=True, save_format=self.output_format
        )

        # --------------------------------------------------------------------------------
        # Stop when no more AUC improvement expected better than 1.0
        # --------------------------------------------------------------------------------
        if roc_auc_prev > (1.0 - 1e-10):
            self._stop_early()
            if self.verbose:
                print("Stopped as no AUC improvement can be made beyond 1.0")

    def _handle_no_improvement(self, epoch, roc_auc, roc_auc_prev, val_loss, val_loss_prev):
        if self.verbose:
            if roc_auc <= roc_auc_prev:
                print(f"AUC [%5f] did not improve from [%5f]." % (roc_auc, roc_auc_prev))
            if val_loss >= val_loss_prev:
                print(f"val_loss [%5f] did not improve from [%5f]." % (val_loss, val_loss_prev))

        # --------------------------------------------------------------------------------
        # Reduce LR
        # --------------------------------------------------------------------------------
        self.successive_no_improvement += 1
        if self.successive_no_improvement >= self.reduce_lr_patience:
            self._reduce_learning_rate()

        # --------------------------------------------------------------------------------
        # Early Stop
        # --------------------------------------------------------------------------------
        self.total_no_improvement += 1
        if self.total_no_improvement >= self.early_stop_patience:
            self._stop_early()

    def _has_improved(self, roc_auc, roc_auc_prev, val_loss, val_loss_prev):
        """Decide if an improvement has been achieved
        Criteria:
            1: Both AUC and val_loss improved
            2: AUC improved
            3: val_loss improved
        """
        if self.criterion == 1:
            return (roc_auc > roc_auc_prev) and (val_loss < val_loss_prev)
        if self.criterion == 2:
            return roc_auc > roc_auc_prev
        if self.criterion == 3:
            return val_loss < val_loss_prev

    def on_epoch_end(self, epoch, logs={}):
        """Verify the performance improvement metrics and make decisions on:
        - Reduce learning rate if count of no consecutive improvement >= reduce_lr_patience
        - Early stopping if total count of no improvement >= early_stop_patience

        TODO:
            If validation data is recreated e.g. at on_epoch_end tf.keras.utils.Sequence,
            NEED to update the self.x, self.y accordingly.
        """
        # [print(f"{k}:{v}") for k, v in logs.items()]
        predictions = self.model.predict(self.x)
        roc_auc = roc_auc_score(self.y, predictions)
        val_loss = logs.get('val_loss')

        roc_auc_prev = self.max_roc_auc
        val_loss_prev = self.min_val_loss
        self.max_roc_auc = np.maximum(roc_auc, self.max_roc_auc)
        self.min_val_loss = np.minimum(val_loss, self.min_val_loss)

        if self._has_improved(roc_auc, roc_auc_prev, val_loss, val_loss_prev):
            self._handle_improvement(epoch, roc_auc, roc_auc_prev, val_loss, val_loss_prev)
        else:
            self._handle_no_improvement(epoch, roc_auc, roc_auc_prev, val_loss, val_loss_prev)

        def on_batch_begin(self, batch, logs={}):
            return

    def on_batch_end(self, batch, logs={}):
        return


class SavePretrainedCallback(tf.keras.callbacks.Callback):
    """
    This is only for directly working on the Huggingface models.

    Hugging Face models have a save_pretrained() method that saves both
    the weights and the necessary metadata to allow them to be loaded as
    a pretrained model in future. This is a simple Keras callback that
    saves the model with this method after each epoch.

    """

    def __init__(self, output_dir, monitor='val_loss', mode='auto', verbose=True):
        super().__init__()
        self.output_dir = output_dir
        self.monitor = monitor
        self.mode = mode
        self.verbose = verbose

        self.lowest_val_loss = np.inf
        self.best_epoch = -1

    def on_epoch_end(self, epoch, logs={}):
        """
        Save only the best model
        - https://stackoverflow.com/a/68042600/4281353
        - https://www.tensorflow.org/guide/keras/custom_callback

        TODO:
        save_pretrained() method is in the HuggingFace model only.
        Need to implement an logic to update for Keras model saving.
        """
        assert self.monitor in logs, \
            f"monitor metric {self.monitor} not in valid metrics {logs.keys()}"

        val_loss = logs.get(self.monitor)
        if (self.best_epoch < 0) or (val_loss < self.lowest_val_loss):
            if self.verbose:
                print(f"Model val_loss improved: [{val_loss} < {self.lowest_val_loss}]")
                print(f"Saving to {self.output_dir}")
            self.lowest_val_loss = val_loss
            self.best_epoch = epoch
            self.model.save_pretrained(self.output_dir)


class TensorBoardCallback(tf.keras.callbacks.TensorBoard):
    """TensorBoard visualization of the model training
    See https://keras.io/api/callbacks/tensorboard/
    """

    def __init__(self, output_directory):
        super().__init__(
            log_dir=output_directory,
            write_graph=True,
            write_images=True,
            histogram_freq=1,  # log histogram visualizations every 1 epoch
            embeddings_freq=1,  # log embedding visualizations every 1 epoch
            update_freq="epoch",  # every epoch
        )


class EarlyStoppingCallback(tf.keras.callbacks.EarlyStopping):
    """Stop training when no progress on the metric to monitor
    https://keras.io/api/callbacks/early_stopping/
    https://machinelearningmastery.com/how-to-stop-training-deep-neural-networks-at-the-right-time-using-early-stopping/

    Using val_loss to monitor.
    https://datascience.stackexchange.com/a/49594/68313
    Prefer the loss to the accuracy. Why? The loss quantify how certain
    the model is about a prediction. The accuracy merely account for
    the number of correct predictions. Similarly, any metrics using hard
    predictions rather than probabilities have the same problem.
    """

    def __init__(self, patience=3, monitor='val_loss', mode='auto'):
        assert patience > 0
        super().__init__(
            monitor=monitor,
            mode=mode,
            verbose=1,
            patience=patience,
            restore_best_weights=True
        )

    def on_epoch_end(self, epoch, logs={}):
        assert self.monitor in logs, \
            f"monitor metric {self.monitor} not in valid metrics {logs.keys()}"
        super().on_epoch_end(epoch, logs)


class ModelCheckpointCallback(tf.keras.callbacks.ModelCheckpoint):
    """Check point to save the model
    See https://keras.io/api/callbacks/model_checkpoint/

    NOTE:
        Did not work with the HuggingFace native model with the error.
        NotImplementedError: Saving the model to HDF5 format requires the model
        to be a Functional model or a Sequential model.
        It does not work for subclassed models, because such models are defined
        via the body of a Python method, which isn't safely serializable.

        Did not work with the tf.keras.models.save_model nor model.save()
        as causing out-of-index errors or load_model() failures. Hence use
        save_weights_only=True.
    """

    def __init__(self, path_to_file, monitor='val_loss', mode='auto'):
        """
        Args:
            path_to_file: path to the model file to save at check points
        """
        super().__init__(
            filepath=path_to_file,
            monitor=monitor,
            mode=mode,
            save_best_only=True,
            save_weights_only=True,  # Cannot save entire model.
            save_freq="epoch",
            verbose=1
        )

    def on_epoch_end(self, epoch, logs={}):
        assert self.monitor in logs, \
            f"monitor metric {self.monitor} not in valid metrics {logs.keys()}"
        super().on_epoch_end(epoch, logs)


class ReduceLRCallback(tf.keras.callbacks.ReduceLROnPlateau):
    """Reduce learning rate when a metric has stopped improving.
    See https://keras.io/api/callbacks/reduce_lr_on_plateau/
    """

    def __init__(self, patience=3, monitor='val_loss', mode='auto'):
        assert patience > 0
        super().__init__(
            monitor=monitor,
            mode=mode,
            factor=0.2,
            patience=patience,
            verbose=1
        )

    def on_epoch_end(self, epoch, logs={}):
        assert self.monitor in logs, \
            f"monitor metric {self.monitor} not in valid metrics {logs.keys()}"
        super().on_epoch_end(epoch, logs)
