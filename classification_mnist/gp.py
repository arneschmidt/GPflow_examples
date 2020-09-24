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

class DeepKernelGP():
    def __init__(self, dataset, total_num_data: int, batch_size: int, image_size: int, num_mnist_classes: int, feature_outputs: int,
                 num_inducing_points: int):
        self.dataset = dataset
        self.model = self.initialize_model(total_num_data, batch_size, image_size, num_mnist_classes, feature_outputs,
                 num_inducing_points)



    def initialize_model(self, total_num_data: int, batch_size: int, image_size: int, num_mnist_classes: int, feature_outputs: int,
                 num_inducing_points: int):
        images_subset, labels_subset = next(iter(self.dataset.batch(32)))
        images_subset = tf.reshape(images_subset, [-1, image_size])
        labels_subset = tf.reshape(labels_subset, [-1, 1])

        kernel = KernelWithConvNN(
            feature_outputs, gpflow.kernels.SquaredExponential(), batch_size=batch_size
        )

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
        return model

class KernelWithConvNN(gpflow.kernels.Kernel):
    def __init__(
        self,
        cnn: tf.keras.models.Model,
        base_kernel: gpflow.kernels.Kernel,
        batch_size: Optional[int] = None,
    ):
        super().__init__()
        with self.name_scope:
            self.base_kernel = base_kernel
            self.cnn = cnn
            self.cnn.build()

    def K(self, a_input: tf.Tensor, b_input: Optional[tf.Tensor] = None) -> tf.Tensor:
        transformed_a = self.cnn(a_input)
        transformed_b = self.cnn(b_input) if b_input is not None else b_input
        return self.base_kernel.K(transformed_a, transformed_b)

    def K_diag(self, a_input: tf.Tensor) -> tf.Tensor:
        transformed_a = self.cnn(a_input)
        return self.base_kernel.K_diag(transformed_a)

class KernelSpaceInducingPoints(gpflow.inducing_variables.InducingPoints):
    pass


@gpflow.covariances.Kuu.register(KernelSpaceInducingPoints, KernelWithConvNN)
def Kuu(inducing_variable, kernel, jitter=None):
    func = gpflow.covariances.Kuu.dispatch(
        gpflow.inducing_variables.InducingPoints, gpflow.kernels.Kernel
    )
    return func(inducing_variable, kernel.base_kernel, jitter=jitter)


@gpflow.covariances.Kuf.register(KernelSpaceInducingPoints, KernelWithConvNN, object)
def Kuf(inducing_variable, kernel, a_input):
    return kernel.base_kernel(inducing_variable.Z, kernel.cnn(a_input))

