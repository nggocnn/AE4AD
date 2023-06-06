import tensorflow as tf
import numpy as np


class CheckClassifierAccuracy(tf.keras.callbacks.Callback):
    def __init__(self, classifier, x_test, y_test, data_name='val_data', every_epoch=10):
        super().__init__()
        self.classifier = classifier
        self.x_test = x_test
        self.y_test = np.argmax(y_test, axis=1).reshape(-1)
        self.data_name = data_name
        self.every_epoch = every_epoch

    def on_epoch_end(self, epoch, logs=None):
        if epoch % self.every_epoch == 0:
            reformed_data = self.model.predict(self.x_test)
            reformed_pred = np.argmax(self.classifier.predict(reformed_data), axis=1).reshape(-1)
            logs[f'classifier_acc_on_{self.data_name}'] = (reformed_pred == self.y_test).sum() / len(self.y_test)