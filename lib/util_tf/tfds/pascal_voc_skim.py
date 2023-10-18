from typing import (
    Dict,
)

import tensorflow as tf
import tensorflow_datasets as tfds


def generate_yolo_v1_labels_from_pascal_voc(record: Dict) -> tf.Tensor:
    """
    Generate YOLO v1 Class labels of shape (C,) per image where C is 20 classes

    Args:
        record: Row in VOC Dataset
    Returns: tf.Tensor of shape (C,)
    """
    label: tf.Tensor = record['objects']['label'] - 1
    tf.print("-" * 80)
    tf.print("label", label)
    indices = tf.reshape(
        tensor=label,
        shape=(-1, 1)
    )
    num_objects = tf.shape(indices)[0]
    positions: tf.Tensor = tf.concat(
        values=[
            tf.reshape(tf.range(num_objects, dtype=indices.dtype), (-1, 1)),  # row i
            indices                                                           # column j=labels[i]
        ],
        axis=-1
    )
    tf.print("positions:\n", positions)
    classes = tf.tensor_scatter_nd_update(
        tensor=tf.zeros(shape=(num_objects, 20), dtype=tf.float32),
        indices=positions,
        updates=tf.ones(shape=(num_objects,))
    )
    return classes


def main():
    voc, info = tfds.load(
        name='voc',
        data_dir="/Volumes/SSD/data/tfds/",
        with_info=True,
    )
    tf.config.run_functions_eagerly(True)
    labels = voc['train'].take(10).map(
        generate_yolo_v1_labels_from_pascal_voc,
        num_parallel_calls=1,
        deterministic=True
    )
    for index, label in enumerate(labels):
        print('*' * 80)
        print(index, label.shape)


if __name__ == "__main__":
    main()