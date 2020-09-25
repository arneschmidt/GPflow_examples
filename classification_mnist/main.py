import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt
import gpflow
from gpflow.ci_utils import ci_niter
from scipy.cluster.vq import kmeans2

from typing import Dict, Optional, Tuple
import tensorflow as tf
import tensorflow_datasets as tfds
import gpflow
from gpflow.utilities import to_default_float
from cnn import Cnn

def load_data(batch_size: int):
    (ds_train, ds_test), ds_info = tfds.load(
        'mnist',
        split=['train', 'test'],
        shuffle_files=True,
        as_supervised=True,
        with_info=True,
    )
    image_shape = ds_info.features["image"].shape
    image_size = tf.reduce_prod(image_shape)

    def map_fn(image, label):
        image = to_default_float(image) / 255.0
        label = to_default_float(label)
        return tf.reshape(image, [-1, image_size]), label

    ds_train = ds_train.cache()
    ds_train = ds_train.shuffle(ds_info.splits['train'].num_examples)
    ds_train = ds_train.batch(batch_size)
    ds_train = ds_train.map(
        map_fn, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    ds_train = ds_train.prefetch(tf.data.experimental.AUTOTUNE)

    ds_test = ds_test.cache()
    ds_test = ds_test.batch(batch_size)
    ds_test = ds_test.map(
        map_fn, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    ds_test = ds_test.prefetch(tf.data.experimental.AUTOTUNE)

    return ds_train, ds_test, ds_info


def main():
    batch_size = 128
    ds_train, ds_test, info = load_data(batch_size)

    image_shape = info.features["image"].shape

    cnn = Cnn()
    cnn.create_model(batch_size=batch_size, image_shape=image_shape, feature_outputs=int(5))
    cnn.train(ds_train, ds_test)
    cnn.save()

    cnn_loaded = Cnn()
    cnn_loaded.load_combined_model()
    cnn_loaded.test(ds_test)


if __name__ == '__main__':
    main()