from abc import abstractmethod
from sklearn.model_selection import train_test_split
from tensorflow.keras.datasets import mnist, cifar10
from tensorflow.keras.utils import to_categorical
import numpy as np
import random
import sklearn as sk


class NormalData:
    def __init__(self, val_ratio=0.2, random_state=7):
        self.val_ratio = val_ratio
        self.y_test = None
        self.x_test = None
        self.y_val = None
        self.x_val = None
        self.y_train = None
        self.x_train = None
        self.random_sate = random_state
        self.load_data()

    @abstractmethod
    def load_data(self):
        pass

    def summary(self):
        print('x_train shape:\t', self.x_train.shape)
        print('y_train shape:\t', self.y_train.shape)
        print('x_val shape:\t', self.x_val.shape)
        print('y_val shape:\t', self.y_val.shape)
        print('x_test shape:\t', self.x_test.shape)
        print('y_test shape:\t', self.y_test.shape)


class MNISTData(NormalData):
    def __init__(self, val_ratio=0.2):
        super().__init__(val_ratio)

    def load_data(self):
        (x_train_all, y_train_all), (x_test, y_test) = mnist.load_data()

        x_train_all = x_train_all.astype('float32') / 255.0
        x_test = x_test.astype('float32') / 255.0

        x_train_all = x_train_all.reshape((x_train_all.shape[0], 28, 28, 1))
        self.x_test = x_test.reshape((x_test.shape[0], 28, 28, 1))

        y_train_all = to_categorical(y_train_all, 10)
        self.y_test = to_categorical(y_test, 10)

        self.x_train, self.x_val, self.y_train, self.y_val = \
            train_test_split(x_train_all, y_train_all, test_size=self.val_ratio, random_state=self.random_sate)

        print('MNISTData')
        self.summary()


class Cifar10Data(NormalData):
    def __init__(self, val_ratio=0.2):
        super().__init__(val_ratio)

    def load_data(self):
        (x_train_all, y_train_all), (x_test, y_test) = cifar10.load_data()

        x_train_all = x_train_all.astype('float32') / 255.0
        x_test = x_test.astype('float32') / 255.0

        x_train_all = x_train_all.reshape((x_train_all.shape[0], 32, 32, 3))
        self.x_test = x_test.reshape((x_test.shape[0], 32, 32, 3))

        y_train_all = to_categorical(y_train_all, 10)
        self.y_test = to_categorical(y_test, 10)

        self.x_train, self.x_val, self.y_train, self.y_val = \
            train_test_split(x_train_all, y_train_all, test_size=self.val_ratio, random_state=self.random_sate)

        print('Cifar10Data')
        self.summary()


class UntrustedData:
    def __init__(self, origin_data, origin_label, adv_data, adv_label):
        self.origin_data = origin_data
        self.origin_label = origin_label
        self.adv_data = adv_data
        self.adv_label = adv_label

    def get_untrusted_data(self):
        random_idx = random.sample(range(len(self.origin_data)), len(self.adv_data))
        data = np.concatenate([self.adv_data, self.origin_data[random_idx]], axis=0)
        label = np.concatenate([self.adv_label, self.origin_label[random_idx]], axis=0)
        mark = np.concatenate([np.full(len(self.adv_data), 1), np.full(len(self.adv_data), 0)], axis=0)
        data, label, mark = sk.utils.shuffle(data, label, mark)

        return data, label, mark
