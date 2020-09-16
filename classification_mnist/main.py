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

def load_data():
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

    return ds_train, ds_test



def main():
    ds_train, ds_test = load_data()

    cnn = Cnn()
    cnn.create_model()
    cnn.train(ds_train, ds_test)
    cnn.save()

    num_mnist_classes = 10
    output_dim = 5
    num_inducing_points = 100
    images_subset, labels_subset = next(iter(dataset.batch(32)))
    images_subset = tf.reshape(images_subset, [-1, image_size])
    labels_subset = tf.reshape(labels_subset, [-1, 1])

    kernel = gpflow.kernels.SquaredExponential()
    likelihood = gpflow.likelihoods.MultiClass(num_mnist_classes)

    inducing_variable_kmeans = kmeans2(images_subset.numpy(), num_inducing_points, minit="points")[0]
    inducing_variable_cnn = kernel.cnn(inducing_variable_kmeans)
    inducing_variable = KernelSpaceInducingPoints(inducing_variable_cnn)

    model = gpflow.models.SVGP(
        kernel,
        likelihood,
        inducing_variable=inducing_variable,
        num_data=total_num_data,
        num_latent_gps=num_mnist_classes,
    )

    cnn_loaded = Cnn()
    cnn_loaded.load_combined_model()
    cnn_loaded.model.evaluate(ds_test)

if __name__ == '__main__':
    main()