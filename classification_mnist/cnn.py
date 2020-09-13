import tensorflow as tf
import os

class Cnn():
    def __init__(self):
        self.model = None
        self.feature_extractor = None
        self.head = None

    def create_model(self):
        self.feature_extractor = tf.keras.models.Sequential([
            tf.keras.layers.Flatten(input_shape=(28, 28, 1)),
            tf.keras.layers.Dense(128, activation='relu')])
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

    def train(self, train_data, test_data):
        self.model.fit(
            train_data,
            epochs=2,
            validation_data=test_data,
        )

    def save(self, path: str = "./models/",  name: str = "cnn"):
        # feature_extractor = tf.keras.models.Model(inputs=self.model.layers[0].input,
        #                                           outputs=self.model.layers[1].output)
        # head = tf.keras.models.Model(inputs=self.model.layers[2].input,
        #                              outputs=self.model.layers[2].output)
        self.feature_extractor.save(os.path.join(path, name + "_feature_extractor.h5"))
        self.head.save(os.path.join(path, name + "_head.h5"))

    def load_combined_model(self, path: str = "./models/",  name: str = "cnn"):
        self.feature_extractor = tf.saved_model.load(os.path.join(path, name + "_feature_extractor.h5"))
        self.head = tf.saved_model.load(os.path.join(path, name + "_head.h5"))
        self.model = tf.keras.models.Sequential([self.feature_extractor, self.head])


