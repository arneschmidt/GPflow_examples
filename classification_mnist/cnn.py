import tensorflow as tf
from gpflow.utilities import to_default_float
import os

class Cnn():
    def __init__(self, batch_size: int, image_shape: int, feature_outputs: int):
        self.batch_size = batch_size
        self.image_shape = image_shape
        self.feature_outputs = feature_outputs,
        self.model = None
        self.feature_extractor = None
        self.head = None

    def create_model(self):
        input_size = int(tf.reduce_prod(self.image_shape))
        input_shape = (input_size,)
        self.feature_extractor = tf.keras.Sequential(
                [
                    tf.keras.layers.InputLayer(input_shape=input_shape, batch_size=self.batch_size),
                    tf.keras.layers.Reshape(self.image_shape),
                    tf.keras.layers.Conv2D(
                        filters=32, kernel_size=self.image_shape[:-1], padding="same", activation="relu"
                    ),
                    tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=2),
                    tf.keras.layers.Conv2D(
                        filters=64, kernel_size=(5, 5), padding="same", activation="relu"
                    ),
                    tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=2),
                    tf.keras.layers.Flatten(),
                    tf.keras.layers.Dense(self.feature_outputs[0], activation="relu"),
                    #tf.keras.layers.Lambda(to_default_float),
                ]
            )
        self.head = tf.keras.models.Sequential([
            tf.keras.layers.Dense(10, activation='softmax')
        ])

        self.model = tf.keras.models.Sequential([self.feature_extractor, self.head])

        self.model.compile(
            loss='sparse_categorical_crossentropy',
            optimizer=tf.keras.optimizers.Adam(0.001),
            metrics=['accuracy'],
        )

    def train(self, train_data, test_data):
        self.model.fit(
            train_data,
            epochs=2,
            validation_data=test_data,
        )

    def save(self, path: str = "./models/",  name: str = "cnn"):
        self.feature_extractor.save(os.path.join(path, name + "_feature_extractor.h5"))
        self.head.save(os.path.join(path, name + "_head.h5"))

    def load_combined_model(self, path: str = "./models/",  name: str = "cnn"):
        self.feature_extractor = tf.keras.models.load_model(os.path.join(path, name + "_feature_extractor.h5"))
        self.head = tf.keras.models.load_model(os.path.join(path, name + "_head.h5"))
        self.model = tf.keras.models.Sequential([self.feature_extractor, self.head])
        self.model.compile(
            loss='sparse_categorical_crossentropy',
            optimizer=tf.keras.optimizers.Adam(0.001),
            metrics=['accuracy'],
        )
        self.model.summary()


