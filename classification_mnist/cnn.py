import tensorflow as tf
from gpflow.utilities import to_default_float
import os

class Cnn():
    def __init__(self):
        self.batch_size = None
        self.image_shape = None
        self.feature_outputs = None
        self.model = None
        self.feature_extractor = None
        self.head = None

    def create_model(self, batch_size: int, image_shape: int, feature_outputs: int):
        self.batch_size = batch_size
        self.image_shape = image_shape
        self.feature_outputs = feature_outputs
        input_size = int(tf.reduce_prod(image_shape))
        input_shape = (input_size,)

        self.feature_extractor = tf.keras.models.Sequential([
            tf.keras.layers.Flatten(input_shape=input_shape),
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Lambda(to_default_float)])
        self.head = tf.keras.models.Sequential([
            tf.keras.layers.Dense(10, activation='softmax')
        ])

        self.model = tf.keras.models.Sequential([self.feature_extractor, self.head])

        self.model.compile(
            loss='sparse_categorical_crossentropy',
            optimizer=tf.keras.optimizers.Adam(0.001),
            metrics=['accuracy'],
        )
        self.model.summary()

    def train(self, train_data, test_data, epochs: int):
        self.model.fit(
            train_data,
            epochs=2,
            steps_per_epoch=1000,
            validation_data=test_data,
        )
    def test(self, test_data):
        self.model.evaluate(test_data)

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

    def load_feature_extractor(self, path: str = "./models/",  name: str = "cnn"):
        self.feature_extractor = tf.keras.models.load_model(os.path.join(path, name + "_feature_extractor.h5"))