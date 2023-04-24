from abc import abstractmethod
import numpy as np
import tensorflow as tf
from tensorflow.keras.losses import kl_divergence
from classifier import Classifier


np.random.seed(7)


class Detector:
    def __init__(self):
        pass

    @abstractmethod
    def score(self, x):
        """
        Calculate score on a test example.

        :param x: an array of test examples.
        :return: score of test examples array.
        """
        pass


class LpNormDetector(Detector):
    """
    Detector bases on reconstruction error.
    """

    def __init__(self, autoencoder, p):
        """
        Constructor of reconstruction-error-based detector.

        :param autoencoder: path to reformer autoencoder.
        :param p: order of the Lp norm.
        """
        super().__init__()
        self.ae = autoencoder
        self.p = p
    def score(self, x):
        """
        Calculate reconstruction error on a test example is E(x) = LpNorm(x - ae(x)).

        :param x: an array of test examples.
        :return: Lp norm score of test examples array.
        """
        x_ = self.ae.predict(x)
        diff = np.abs(x.reshape((len(x), -1)) - x_.reshape((len(x), -1)))
        score_x = None
        if self.p == 1:
            score_x = np.mean(diff, axis=1)
        elif self.p == 2:
            score_x = np.mean(diff ** 2, axis=1)
        else:
            raise Exception('Lp not implemented')
        return score_x


class DivergenceDetector(Detector):
    """
    Detector bases on probability divergence.
    """
    def __init__(self, autoencoder, classifier: Classifier, t=1):
        """
        Constructor of probability-divergence-based detector

        :param reformer: reformer autoencoder.
        :param classifier: target classifier.
        :param t: temperature t > 1 to calculate softmax output of classifier.
        """
        super().__init__()
        self.ae = autoencoder
        self.classifier = classifier
        self.t = t

    @staticmethod
    def js_divergence(p, q):
        """
        Jensen-Shannon divergence.

        :param p: softmax output 1.
        :param q: softmax output 2.
        :return:
        """
        m = 0.5 * (p + q)
        return 0.5 * kl_divergence(p, m) + 0.5 * kl_divergence(q, m)

    def score(self, x):
        """
        Calculate JS divergence score of text examples array
        :param x: an array of test examples.
        :return: JS divergence score.
        """
        x_predict = self.classifier.predict(x, t=self.t)  # softmax predict of x with t.

        x_ae = self.ae.predict(x)  # reform x using autoencoder.
        x_ae_predict = self.classifier.predict(x_ae, t=self.t)  # softmax predict of x_reform.

        # calculate JS divergence distance between 2 probability x and x_reform.
        return self.js_divergence(x_predict, x_ae_predict)
