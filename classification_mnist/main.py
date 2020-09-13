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

def main():
    (ds_train, ds_test), ds_info = tfds.load(
        'mnist',
        split=['train', 'test'],
        shuffle_files=True,
        as_supervised=True,
        with_info=True,
    )

    def normalize_img(image, label):
        """Normalizes images: `uint8` -> `float32`."""
        return tf.cast(image, tf.float32) / 255., label

    ds_train = ds_train.map(
        normalize_img, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    ds_train = ds_train.cache()
    ds_train = ds_train.shuffle(ds_info.splits['train'].num_examples)
    ds_train = ds_train.batch(128)
    ds_train = ds_train.prefetch(tf.data.experimental.AUTOTUNE)

    ds_test = ds_test.map(
        normalize_img, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    ds_test = ds_test.batch(128)
    ds_test = ds_test.cache()
    ds_test = ds_test.prefetch(tf.data.experimental.AUTOTUNE)

    cnn = Cnn()
    cnn.train(ds_train, ds_test)
    cnn.save()

if __name__ == '__main__':
    main()