import os

import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_transform as tft

from transform import (
    BUCKET_FEATURES,
    LABEL_KEY,
    ONE_HOT_FEATURES,
    TEXT_FEATURES,
    transformed_name,
)


def get_model(show_summary=True):
    """
    This function defines a Keras model and returns the model as a
    Keras object.
    """

    # one-hot categorical features
    input_features = []
    for key, dim in ONE_HOT_FEATURES.items():
        input_features.append(
            tf.keras.Input(shape=(dim + 1,), name=transformed_name(key))
        )

    # adding bucketized features
    for key, dim in BUCKET_FEATURES.items():
        input_features.append(
            tf.keras.Input(shape=(dim + 1,), name=transformed_name(key))
        )

    # adding text input features
    input_texts = []
    for key in TEXT_FEATURES.keys():
        input_texts.append(
            tf.keras.Input(
                shape=(1,), name=transformed_name(key), dtype=tf.string
            )
        )

    # embed text features
    MODULE_URL = "https://tfhub.dev/google/universal-sentence-encoder/4"
    embed = hub.KerasLayer(MODULE_URL)
    reshaped_narrative = tf.reshape(input_texts[0], [-1])
    embed_narrative = embed(reshaped_narrative)
    deep_ff = tf.keras.layers.Reshape((512,), input_shape=(1, 512))(
        embed_narrative
    )

    deep = tf.keras.layers.Dense(256, activation="relu")(deep_ff)
    deep = tf.keras.layers.Dense(64, activation="relu")(deep)
    deep = tf.keras.layers.Dense(16, activation="relu")(deep)

    wide_ff = tf.keras.layers.concatenate(input_features)
    wide = tf.keras.layers.Dense(16, activation="relu")(wide_ff)

    both = tf.keras.layers.concatenate([deep, wide])

    output = tf.keras.layers.Dense(1, activation="sigmoid")(both)

    inputs = input_features + input_texts

    keras_model = tf.keras.models.Model(inputs, output)
    keras_model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss="binary_crossentropy",
        metrics=[
            tf.keras.metrics.BinaryAccuracy(),
            tf.keras.metrics.TruePositives(),
        ],
    )
    if show_summary:
        keras_model.summary()

    return keras_model


def _gzip_reader_fn(filenames):
    """Small utility returning a record reader that can read gzip'ed files."""
    return tf.data.TFRecordDataset(filenames, compression_type="GZIP")


def _get_serve_tf_examples_fn(model, tf_transform_output):
    """Returns a function that parses a serialized tf.Example."""

    model.tft_layer = tf_transform_output.transform_features_layer()

    @tf.function
    def serve_tf_examples_fn(serialized_tf_examples):
        """Returns the output to be used in the serving signature."""
        feature_spec = tf_transform_output.raw_feature_spec()
        feature_spec.pop(LABEL_KEY)
        parsed_features = tf.io.parse_example(
            serialized_tf_examples, feature_spec
        )

        transformed_features = model.tft_layer(parsed_features)

        outputs = model(transformed_features)
        return {"outputs": outputs}

    return serve_tf_examples_fn


def _input_fn(file_pattern, tf_transform_output, batch_size=64):
    """Generates features and label for tuning/training.

    Args:
        file_pattern: input tfrecord file pattern.
        tf_transform_output: A TFTransformOutput.
        batch_size: representing the number of consecutive elements of
          returned dataset to combine in a single batch

    Returns:
        A dataset that contains (features, indices) tuple where features
        is a dictionary of Tensors, and indices is a single Tensor of
        label indices.
    """
    transformed_feature_spec = (
        tf_transform_output.transformed_feature_spec().copy()
    )

    dataset = tf.data.experimental.make_batched_features_dataset(
        file_pattern=file_pattern,
        batch_size=batch_size,
        features=transformed_feature_spec,
        reader=_gzip_reader_fn,
        label_key=transformed_name(LABEL_KEY),
    )

    return dataset


# TFX Trainer will call this function.
def run_fn(fn_args):
    """Train the model based on given args.

    Args:
    fn_args: Holds args used to train the model as name/value pairs.
    """
    tf_transform_output = tft.TFTransformOutput(fn_args.transform_output)

    train_dataset = _input_fn(fn_args.train_files, tf_transform_output, 64)
    eval_dataset = _input_fn(fn_args.eval_files, tf_transform_output, 64)

    model = get_model()

    log_dir = os.path.join(os.path.dirname(fn_args.serving_model_dir), "logs")
    tensorboard_callback = tf.keras.callbacks.TensorBoard(
        log_dir=log_dir, update_freq="batch"
    )

    model.fit(
        train_dataset,
        steps_per_epoch=fn_args.train_steps,
        validation_data=eval_dataset,
        validation_steps=fn_args.eval_steps,
        callbacks=[tensorboard_callback],
    )

    signatures = {
        "serving_default": _get_serve_tf_examples_fn(
            model, tf_transform_output
        ).get_concrete_function(
            tf.TensorSpec(shape=[None], dtype=tf.string, name="examples")
        ),
    }
    model.save(
        fn_args.serving_model_dir, save_format="tf", signatures=signatures
    )
