import os
import sys
from pathlib import Path
import logging

import numpy as np
import tensorflow as tf
from tensorflow.python.keras import backend


Logger = logging.getLogger(__name__)


class KerasCallbackBase(tf.keras.callbacks.Callback):
    """Base class for Keras custom callback"""
    def __init__(
            self,
            output_path,
            output_format='h5',
            output_overwrite=True,
            reduce_lr_patience=2,
            reduce_lr_factor=0.2,
            monitor_metric='val_loss',
            monitor_mode='min',
            monitor_best_value=np.inf,
            early_stop_patience=sys.maxsize,
            verbose=True
    ):
        """
        Args:
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
        assert monitor_mode in ['min', 'max']   # Do not handle 'auto'

        # --------------------------------------------------------------------------------
        # Training control parameters
        # --------------------------------------------------------------------------------
        self.monitor_metric = monitor_metric
        self.monitor_mode = monitor_mode
        self.monitor_best_value = monitor_best_value
        self.reduce_lr_patience = reduce_lr_patience
        self.reduce_lr_factor = reduce_lr_factor
        self.early_stop_patience = early_stop_patience
        self.output_path = output_path
        self.output_format = output_format
        self.output_overwrite = output_overwrite
        self.verbose = verbose

        # --------------------------------------------------------------------------------
        # Statistics
        # --------------------------------------------------------------------------------
        self.max_roc_auc = -1
        self.min_val_loss = np.inf
        self.best_epoch = -1
        self.successive_no_improvement = 0
        self.total_no_improvement = 0

    def on_batch_begin(self, batch, logs={}):
        return

    def on_batch_end(self, batch, logs={}):
        return

    def on_train_begin(self, logs={}):
        """Action at the start of training.
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
        """Ramp down the LR
        See the github of tf.keras.callbacks.ReduceLROnPlateau
        """
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

    def _handle_improvement(self, previous_metric_value):
        if self.verbose:
            Logger.info(
                "Model %s improved from %5f to %5f. Saving to %s" %
                (self.metric_name, previous_metric_value, self.monitor_best_value, self.output_path)
            )

        # --------------------------------------------------------------------------------
        # Update statistics
        # --------------------------------------------------------------------------------
        self.successive_no_improvement = 0

        # --------------------------------------------------------------------------------
        # Save the model upon improvement
        # --------------------------------------------------------------------------------
        self.model.save_weights(
            self.output_path, overwrite=self.output_overwrite, save_format=self.output_format
        )

    def _handle_no_improvement(self, epoch, metric_value):
        if self.verbose:
            Logger.info(
                "Model %s did not improve from %5f." % (self.metric_name, metric_value)
            )

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

    def on_epoch_end(self, epoch, logs={}):
        """Verify the performance improvement metrics and make decisions on:
        - Reduce learning rate if count of no consecutive improvement >= reduce_lr_patience
        - Early stopping if total count of no improvement >= early_stop_patience
        """
        # [print(f"{k}:{v}") for k, v in logs.items()]
        assert self.monitor_metric in logs, \
            "Specified monitor metric %s does not exist in available metrics %s." % \
            (self.monitor_metric, [metric for metric in logs.keys()])

        previous_best = self.monitor_best_value
        metric_value = logs.get(self.monitor_metric)
        if self.monitor_mode.lower() == 'min':
            self.monitor_best_value = np.minimum(metric_value, self.monitor_best_value)
        elif self.monitor_mode.lower() == 'max':
            self.monitor_best_value = np.maximum(metric_value, self.monitor_best_value)

        if metric_value == self.monitor_best_value:
            self.best_epoch = epoch
            self._handle_improvement(previous_metric_value=previous_best)
        else:
            self._handle_no_improvement(epoch=epoch, metric_value=metric_value)
