import tensorflow as tf
import os

class Cnn():
    def __init__(self):
        self.model, self.feature_extractor, self.head = self._create_model()

    def _create_model(self):
        feature_extractor = tf.keras.models.Sequential([
            tf.keras.layers.Flatten(input_shape=(28, 28, 1)),
            tf.keras.layers.Dense(128, activation='relu')])
        head = tf.keras.models.Sequential([
            tf.keras.layers.Dense(10, activation='softmax')
        ])

        model = tf.keras.models.Sequential([feature_extractor, head])

        model.compile(
            loss='sparse_categorical_crossentropy',
            optimizer=tf.keras.optimizers.Adam(0.001),
            metrics=['accuracy'],
        )
        return model, feature_extractor, head

    def train(self, train_data, test_data):
        self.model.fit(
            train_data,
            epochs=6,
            validation_data=test_data,
        )

    def save(self, path: str = "./models/",  name: str = "cnn"):
        # feature_extractor = tf.keras.models.Model(inputs=self.model.layers[0].input,
        #                                           outputs=self.model.layers[1].output)
        # head = tf.keras.models.Model(inputs=self.model.layers[2].input,
        #                              outputs=self.model.layers[2].output)
        self.feature_extractor.save(os.path.join(path, name + "_feature_ext"))
        self.head.save(os.path.join(path, name + "_head"))
