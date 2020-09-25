import argparse
import tensorflow as tf
import tensorflow_datasets as tfds
from gpflow.utilities import to_default_float
from cnn import Cnn
from gp import DeepKernelGP

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
    ds_train = ds_train.batch(batch_size, drop_remainder=True)
    ds_train = ds_train.map(
        map_fn, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    ds_train = ds_train.prefetch(tf.data.experimental.AUTOTUNE)
    ds_train = ds_train.repeat()

    ds_test = ds_test.cache()
    ds_test = ds_test.batch(batch_size, drop_remainder=True)
    ds_test = ds_test.map(
        map_fn, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    ds_test = ds_test.prefetch(tf.data.experimental.AUTOTUNE)

    return ds_train, ds_test, ds_info


def main(args):
    batch_size = 128
    ds_train, ds_test, info = load_data(batch_size)

    image_shape = info.features["image"].shape
    image_size = tf.reduce_prod(image_shape)
    num_train_samples = info.splits["train"].num_examples
    num_test_samples = info.splits["test"].num_examples

    cnn = Cnn()
    if "init_model" in args.cnn_mode:
        cnn.create_model(batch_size=batch_size, image_shape=image_shape, feature_outputs=int(100))
    if "train" in args.cnn_mode:
        cnn.train(ds_train, ds_test, epochs=10, steps_per_epoch=int(num_train_samples/batch_size))
    if "save" in args.cnn_mode:
        cnn.save()
    if "load" in args.cnn_mode:
        cnn.load_combined_model()
    if "test" in args.cnn_mode:
        cnn.test(ds_test)

    gp = DeepKernelGP(ds_train, num_train_samples, image_size, 10, cnn.feature_extractor, num_inducing_points=100)
    if "train" in args.gp_mode:
        gp.train(10000)
    if "test" in args.gp_mode:
        gp.test(ds_test, image_size, batch_size, num_test_samples)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cnn_mode', nargs='+', choices=["init_model", "train", "save", "load", "test"],
                        default=["init_model", "train", "test"],
                        help='Choose the actions of the Cnn, multiple choices possible.')
    parser.add_argument('--gp_mode', nargs='+', choices=["train", "test"], default=["train", "test"],
                        help="Choose mode of the deep kernel gp. Optional.")
    args = parser.parse_args()
    main(args)