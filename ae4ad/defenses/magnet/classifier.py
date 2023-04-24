import os
from abc import abstractmethod
import tensorflow as tf
from tensorflow.keras import models
from tensorflow.keras import layers
from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.keras.optimizers import SGD


class Classifier:
    """
    Abstract classifier.
    """

    def __init__(self, data, optimizer=SGD(0.01), with_softmax=False, save_path='./models/', pretrained=False, batch_size=32):
        """
        Constructor of classifier.

        :param data: data to train, validate, test.
        :param optimizer: optimizer to train.
        :param with_softmax: train classifier with softmax output.
        :param save_path: folder path to save classifier.
        :param pretrained: set True to use load() to load pretrained models
        """
        self.history = None
        self.classifier = None
        self.data = data
        self.optimizer = optimizer
        self.with_softmax = with_softmax
        self.save_path = save_path
        self.pretrained = pretrained
        self.batch_size = batch_size
        if not pretrained:
            self.build()
            self.compile()

    @abstractmethod
    def build(self):
        """
        Build model.

        :return: None.
        """
        pass

    def compile(self):
        """
        Compile model.

        :return: None.
        """
        from_logits = True
        if self.with_softmax:
            from_logits = False

        self.classifier.compile(
            optimizer=self.optimizer, loss=CategoricalCrossentropy(from_logits=from_logits), metrics=['accuracy']
        )

    def train(self, epochs=50, batch_size=128, data_generator=None, save_name='classifier.h5'):
        """
        Train model.

        :param data_generator:
        :param batch_size:
        :param epochs: epochs to train classifier.
        :param save_name: save classifier with name.
        :return: training history.
        """
        if data_generator is not None:
            data_generator.fit(self.data.x_train)

            self.history = self.classifier.fit(
                data_generator.flow(self.data.x_train, self.data.y_train, batch_size=batch_size),
                validation_data=data_generator.flow(self.data.x_val, self.data.y_val, batch_size=batch_size),
                steps_per_epoch=len(self.data.x_train) / batch_size, epochs=epochs
            )
        else:
            self.history = self.classifier.fit(
                self.data.x_train, self.data.y_train, batch_size=batch_size, epochs=epochs,
                validation_data=(self.data.x_val, self.data.y_val)
            )

        if save_name is not None:
            self.classifier.save(os.path.join(self.save_path, save_name))
            print(os.path.join(self.save_path, save_name))

        return self.history

    def load(self, save_name='classifier.h5', save_path=None):
        """
        Load pretrained classifier.

        :param save_name: name of pretrained classifier.
        :param save_path: folder contains pretrained classifier.
        :return: None.
        """
        if save_path is None:
            save_path = self.save_path

        path = os.path.join(save_path, save_name)

        if self.pretrained:
            self.classifier = models.load_model(path)
        else:
            self.classifier.load_weights(path)

    def evaluate(self, data=None, label=None):
        """
        Evaluate data.

        :param data: input data.
        :param label: input label.
        :return: None.
        """
        if data is None or label is None:
            return self.classifier.evaluate(self.data.x_test, self.data.y_test)
        else:
            return self.classifier.evaluate(data, label)

    def predict(self, x, t=1):
        """
        Classifier predicts test examples.

        :param x: an array of test examples.
        :param t: t > 1 if the largest element in logits output is much larger than its second-largest element.
        :return: prediction of input examples.
        """
        if self.with_softmax:
            return self.classifier.predict(x, batch_size=self.batch_size)
        else:
            logits = self.classifier.predict(x, batch_size=self.batch_size) / t
            return tf.nn.softmax(logits)
