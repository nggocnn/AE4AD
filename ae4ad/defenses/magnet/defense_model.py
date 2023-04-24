import os
from abc import abstractmethod

import numpy as np
from tensorflow.keras import layers
from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.keras import models
from tensorflow.keras.regularizers import L2

from data import NormalData


class AbstractDefenseModel:
    def __init__(self, input_shape, optimizer, noise_volume: float, model_path: str, loss_function='mse',
                 regularizer=L2(1e-9)):
        """
        Constructor of defense model.

        :param input_shape: input shape.
        :param optimizer:
        :param noise_volume: volume of Gaussian noise.
        :param model_path: path of folder to save model.
        :param regularizer: regularizer.
        """
        self.defense_model = None
        self.input_shape = input_shape
        self.optimizer = optimizer
        self.noise_volume = noise_volume
        self.model_path = model_path
        self.loss_function = loss_function
        self.regularizer = regularizer
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
        self.defense_model.compile(loss=self.loss_function, optimizer=self.optimizer)

    def train(self, train_data, val_data, save_name='defense_mode.h5',
              epochs=100, batch_size=256, callbacks=None):
        """
        Train defense model.

        :param callbacks:
        :param train_data: train data.
        :param val_data: validation data.
        :param save_name: save model with name.
        :param epochs: epochs to train model.
        :param batch_size: batch size to train model.
        :return:
        """
        train_noise = self.noise_volume * np.random.normal(size=train_data.shape)
        val_noise = self.noise_volume * np.random.normal(size=val_data.shape)

        noisy_train_data = np.clip(train_data + train_noise, 0.0, 1.0)
        noisy_val_data = np.clip(val_data + val_noise, 0.0, 1.0)

        self.defense_model.fit(
            x=noisy_train_data, y=train_data,
            validation_data=(noisy_val_data, val_data),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks
        )

        if save_name is not None:
            self.defense_model.save(os.path.join(self.model_path, save_name))
            print(os.path.join(self.model_path, save_name))

    def load(self, save_name='defense_model.h5', model_path=None):
        """
        Load pretrained model.

        :param save_name: name of pretrained model.
        :param model_path:  folder contains pretrained model.
        :return:
        """
        if model_path is None:
            model_path = self.model_path
        self.defense_model.load_weights(os.path.join(model_path, save_name))


class DefenseModelI(AbstractDefenseModel):
    """
    Defense model bases on autoencoder architecture 1.
    """

    def build(self):
        input_layer = layers.Input(shape=self.input_shape)
        x = layers.Conv2D(filters=3, kernel_size=(3, 3),
                          padding='same', activation='sigmoid',
                          activity_regularizer=self.regularizer)(input_layer)
        x = layers.AveragePooling2D(pool_size=(2, 2))(x)

        x = layers.Conv2D(filters=3, kernel_size=(3, 3),
                          padding='same', activation='sigmoid',
                          activity_regularizer=self.regularizer)(x)

        x = layers.Conv2D(filters=3, kernel_size=(3, 3),
                          padding='same', activation='sigmoid',
                          activity_regularizer=self.regularizer)(x)
        x = layers.UpSampling2D(size=(2, 2))(x)

        x = layers.Conv2D(filters=3, kernel_size=(3, 3),
                          padding='same', activation='sigmoid',
                          activity_regularizer=self.regularizer)(x)

        output_layer = layers.Conv2D(filters=self.input_shape[-1], kernel_size=(3, 3),
                                     padding='same', activation='sigmoid',
                                     activity_regularizer=self.regularizer)(x)

        self.defense_model = models.Model(inputs=[input_layer], outputs=[output_layer])


class DefenseModelII(AbstractDefenseModel):
    """
    Defense model bases on autoencoder architecture 2.
    """

    def build(self):
        input_layer = layers.Input(shape=self.input_shape)
        x = layers.Conv2D(filters=3, kernel_size=(3, 3),
                          padding='same', activation='sigmoid',
                          activity_regularizer=self.regularizer)(input_layer)

        x = layers.Conv2D(filters=3, kernel_size=(3, 3),
                          padding='same', activation='sigmoid',
                          activity_regularizer=self.regularizer)(x)

        output_layer = layers.Conv2D(filters=self.input_shape[-1], kernel_size=(3, 3),
                                     padding='same', activation='sigmoid',
                                     activity_regularizer=self.regularizer)(x)

        self.defense_model = models.Model(inputs=[input_layer], outputs=[output_layer], name='autoencoder')


class DefenseModelIII(AbstractDefenseModel):
    """
    Defense model bases on autoencoder architecture 2.
    """

    def build(self):
        input_layer = layers.Input(shape=self.input_shape, name="input_layer")
        x = layers.Conv2D(filters=32, kernel_size=(3, 3),
                          padding='same', activation="relu")(input_layer)
        x = layers.Conv2D(filters=32, kernel_size=(3, 3),
                          padding='same', activation="relu")(x)
        x = layers.MaxPooling2D(pool_size=(2, 2))(x)

        x = layers.Conv2D(filters=64, kernel_size=(3, 3),
                          padding='same', activation="relu")(x)
        x = layers.Conv2D(filters=64, kernel_size=(3, 3),
                          padding='same', activation="relu")(x)
        x = layers.MaxPooling2D(pool_size=(2, 2))(x)

        x = layers.Conv2D(filters=128, kernel_size=(3, 3),
                          padding='same', activation="relu")(x)
        x = layers.Conv2D(filters=128, kernel_size=(3, 3),
                          padding='same', activation="relu")(x)
        x = layers.MaxPooling2D(pool_size=(2, 2))(x)

        x = layers.Flatten()(x)
        x = layers.Dense(units=1024, activation="relu")(x)

        x = layers.Dense(units=512)(x)

        x = layers.Dense(1024, activation="relu")(x)
        x = layers.Dense(2048, activation="relu")(x)
        x = layers.Reshape((4, 4, 128))(x)

        x = layers.UpSampling2D()(x)
        x = layers.Conv2DTranspose(filters=128, kernel_size=(3, 3),
                                   padding='same', activation="relu")(x)
        x = layers.Conv2DTranspose(filters=128, kernel_size=(3, 3),
                                   padding='same', activation="relu")(x)
        x = layers.UpSampling2D()(x)

        x = layers.Conv2DTranspose(filters=64, kernel_size=(3, 3),
                                   padding='same', activation="relu")(x)
        x = layers.Conv2DTranspose(filters=64, kernel_size=(3, 3),
                                   padding='same', activation="relu")(x)

        x = layers.UpSampling2D()(x)
        x = layers.Conv2DTranspose(filters=32, kernel_size=(3, 3),
                                   padding='same', activation="relu")(x)
        x = layers.Conv2DTranspose(filters=32, kernel_size=(3, 3),
                                   padding='same', activation="relu")(x)

        output_layer = layers.Conv2DTranspose(3, 3, activation="sigmoid", padding="same")(x)

        self.defense_model = models.Model(input_layer, output_layer, name="autoencoder_III")


class DefenseModelIV:
    def __init__(self, input_shape, classifier, optimizer, noise_volume: float, model_path: str, regularizer=L2(1e-9)):
        """
        Constructor of defense model.

        :param input_shape: input shape.
        :param optimizer:
        :param noise_volume: volume of Gaussian noise.
        :param model_path: path of folder to save model.
        :param regularizer: regularizer.
        """
        self.defense_model = None
        self.input_shape = input_shape
        self.classifier = classifier
        self.classifier.trainable = False
        self.optimizer = optimizer
        self.noise_volume = noise_volume
        self.model_path = model_path
        self.regularizer = regularizer
        self.custom_model = None
        self.build()
        self.compile()

    def build(self):
        """
        Build model.

        :return: None.
        """
        input_layer_ = layers.Input(shape=self.input_shape)
        x = layers.Conv2D(filters=3, kernel_size=(3, 3),
                   padding='same', activation='sigmoid',
                   activity_regularizer=self.regularizer)(input_layer_)
        x = layers.AveragePooling2D(pool_size=(2, 2))(x)

        x = layers.Conv2D(filters=3, kernel_size=(3, 3),
                   padding='same', activation='sigmoid',
                   activity_regularizer=self.regularizer)(x)

        x = layers.Conv2D(filters=3, kernel_size=(3, 3),
                   padding='same', activation='sigmoid',
                   activity_regularizer=self.regularizer)(x)
        x = layers.UpSampling2D(size=(2, 2))(x)

        x = layers.Conv2D(filters=3, kernel_size=(3, 3),
                   padding='same', activation='sigmoid',
                   activity_regularizer=self.regularizer)(x)

        output_layer = layers.Conv2D(filters=self.input_shape[-1], kernel_size=(3, 3),
                              padding='same', activation='sigmoid',
                              activity_regularizer=self.regularizer)(x)

        self.defense_model = models.Model(inputs=[input_layer_], outputs=[output_layer], name='autoencoder')

        input_layer = layers.Input(shape=self.input_shape)
        reform_layer = self.defense_model(input_layer)

        logits_layer = self.classifier(reform_layer)
        predict_layer = layers.Softmax(name="predict_layer")(logits_layer)

        self.custom_model = models.Model(inputs=[input_layer], outputs=[reform_layer, predict_layer])

    def load(self, save_name='defense_model.h5', model_path=None):
        """
        Load pretrained model.

        :param save_name: name of pretrained model.
        :param model_path:  folder contains pretrained model.
        :return:
        """
        if model_path is None:
            model_path = self.model_path
        self.defense_model.load_weights(os.path.join(model_path, save_name))

    def compile(self):
        self.defense_model.compile(loss="mse", optimizer=self.optimizer, metrics="mse")
        self.custom_model.compile(
            loss={
                "autoencoder": "mse",
                "predict_layer": CategoricalCrossentropy(from_logits=True)
            },
            optimizer=self.optimizer,
            metrics={
                "autoencoder": "mse",
                "predict_layer": "acc"
            }
        )

    def train(self, data: NormalData, save_name=None,
              custom_name=None, epochs=100, batch_size=256):
        """
        Train defense model
        :param custom_name:
        :param data: data
        :param save_name: save model with name
        :param epochs: epochs to train model
        :param batch_size: batch size to train model
        :return:
        """

        train_noise = self.noise_volume * np.random.normal(size=data.x_train.shape)
        val_noise = self.noise_volume * np.random.normal(size=data.x_val.shape)

        noisy_train_data = np.clip(data.x_train + train_noise, 0.0, 1.0)
        noisy_val_data = np.clip(data.x_val + val_noise, 0.0, 1.0)

        self.custom_model.fit(
            x=noisy_train_data, y=[data.x_train, data.y_train],
            validation_data=(noisy_val_data, [data.x_val, data.y_val]),
            epochs=epochs,
            batch_size=batch_size
        )
        self.custom_model.evaluate(noisy_val_data, [data.x_val, data.y_val])
        self.defense_model.evaluate(noisy_val_data, data.x_val)
        self.classifier.evaluate(self.defense_model.predict(noisy_val_data), data.y_val)
        self.classifier.evaluate(self.defense_model.predict(data.x_val), data.y_val)
        if save_name is not None:
            self.defense_model.save(os.path.join(self.model_path, save_name))
            print(os.path.join(self.model_path, save_name))

        if custom_name is not None:
            self.custom_model.save(os.path.join(self.model_path, custom_name))
