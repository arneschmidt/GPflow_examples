import gpflow
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from gpflow.utilities import print_summary

def f(x):
    """ Objective function f(x) for data generation"""
    return 1/(1+np.exp(-x)) + 0.001*x**2

def regression_with_gp():
    print("Hola muchachos!")
    # define training points with noise (random but fixed for reproduction)
    x_train = np.array([-15,  -4,  4, 9, 13, 18], dtype=np.float)
    np.random.seed(1)
    noise = np.random.normal(scale=0.1, size=x_train.size)
    y_train = f(x_train) + noise

    x_train = x_train.reshape(-1, 1)
    y_train = y_train.reshape(-1, 1)

    x_function_plot = np.arange(-25, 25, 0.01).reshape(-1, 1)

    k = gpflow.kernels.SquaredExponential()
    print_summary(k)

    gp = gpflow.models.GPR(data=(x_train, y_train), kernel=k, mean_function=None)
    print_summary(gp)

    opt = gpflow.optimizers.Scipy()
    opt_logs = opt.minimize(gp.training_loss, gp.trainable_variables, options=dict(maxiter=100))

    mean, var = gp.predict_f(x_function_plot)
    ## plot
    plt.figure(figsize=(12, 6))
    plt.plot(x_train, y_train, "kx", mew=2)
    plt.plot(x_function_plot, mean, "C0", lw=2)
    plt.fill_between(
        x_function_plot[:, 0],
        mean[:, 0] - 1.96 * np.sqrt(var[:, 0]),
        mean[:, 0] + 1.96 * np.sqrt(var[:, 0]),
        color="C0",
        alpha=0.2,
    )
    plt.savefig("regression_with_GP.jpg")


if __name__ == '__main__':
    regression_with_gp()

