import numpy as np

from tensorflow.keras import models, layers, optimizers, callbacks

from sklearn.model_selection import train_test_split

from src.defenses.config_parser import AE4AD_Config
from src.utils.logger import AE4AD_Logger
from src.defenses.callbacks import CheckClassifierAccuracy


logger = AE4AD_Logger.get_logger()


class AE4AD_Autoencoder:
    def __init__(self, config: AE4AD_Config):
        self.model = None
        self.config = config
        self.build()
        self.compile()
        self.checkpoint_path = f'{self.config.autoencoder_path}/autoencoder_' \
                               f'{self.config.classifier_name}'

        self.callbacks = []

        self.callbacks.append(callbacks.ModelCheckpoint(
            filepath=str(self.checkpoint_path) + "_{epoch:04d}",
            monitor="val_loss",
            save_best_only=True
        ))

        if self.config.early_stopping:
            self.callbacks.append(callbacks.EarlyStopping(
                monitor="val_loss",
                min_delta=self.config.min_delta,
                patience=10
            ))

    def build(self):
        model = models.Sequential([
            layers.Input(shape=self.config.image_shape),
            layers.Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.Conv2D(filters=32, kernel_size=(3, 3), strides=(2, 2), activation='relu', padding='same'),
            layers.Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.UpSampling2D(),
            layers.Conv2DTranspose(filters=32, kernel_size=(3, 3), strides=(1, 1), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.Conv2D(filters=self.config.image_shape[-1],
                          kernel_size=(1, 1), strides=(1, 1), activation='sigmoid', padding='same')
        ])

        if self.config.image_shape != model.outputs[0].shape[1:]:
            raise ValueError(f'Autoencoder input shape {model.inputs[0].shape} '
                             f'is not identical to output shape {model.outputs[0].shape[1:]}')

        self.model = model

    def compile(self):
        self.model.compile(
            loss=self.config.loss,
            optimizer=optimizers.Adam(learning_rate=self.config.learning_rate),
            metrics=['mse']
        )

    def train(self, save_final_epoch):
        logger.info(f'Training autoencoder from adversarial data')
        x_train_data = []
        x_val_data = []
        y_train_data = []
        y_val_data = []
        z_train_data = []
        z_val_data = []

        for i in range(len(self.config.adversarial_data)):
            x_train, x_val, y_train, y_val, z_train, z_val = train_test_split(
                self.config.adversarial_data[i],
                self.config.gt_original_data[i],
                self.config.gt_labels_data[i],
                test_size=self.config.valid_ratio
            )

            x_train_data.append(x_train)
            x_val_data.append(x_val)
            y_train_data.append(y_train)
            y_val_data.append(y_val)
            z_train_data.append(z_train)
            z_val_data.append(z_val)

        x_train_data = np.concatenate(x_train_data)
        x_val_data = np.concatenate(x_val_data)
        y_train_data = np.concatenate(y_train_data)
        y_val_data = np.concatenate(y_val_data)
        z_train_data = np.concatenate(z_train_data)
        z_val_data = np.concatenate(z_val_data)

        logger.info(f'Length of training data {len(x_train_data)}, length of validation data {len(x_val_data)}')

        self.callbacks.append(CheckClassifierAccuracy(
            self.config.target_classifier,
            x_val_data, z_val_data, every_epoch=5
        ))

        self.model.fit(
            x_train_data, y_train_data,
            validation_data=(x_val_data, y_val_data),
            epochs=self.config.epochs,
            batch_size=self.config.batch_size,
            callbacks=self.callbacks,
        )

        reformed_train = self.model.predict(x_train_data)
        reformed_train_pred = np.argmax(self.config.target_classifier.predict(reformed_train), axis=1)
        train_accuracy = (reformed_train_pred == np.argmax(z_train_data, axis=1)).sum() / len(z_train_data)
        logger.info(f'Final epoch: Accuracy of target classifier on reformed train data: {train_accuracy}')

        reformed_val = self.model.predict(x_val_data)
        reformed_val_pred = np.argmax(self.config.target_classifier.predict(reformed_val), axis=1)
        val_accuracy = (reformed_val_pred == np.argmax(z_val_data, axis=1)).sum() / len(z_val_data)
        logger.info(f'Final epoch: Accuracy of target classifier on reformed validation data: {val_accuracy}')

        if save_final_epoch:
            self.model.save(f'{self.checkpoint_path}_final_epoch')

    def evaluate_on_target_classifier(self, x, y):
        model = self.model

        reformed_x = model.predict(x)
        reformed_x_pred = np.argmax(self.config.target_classifier.predict(reformed_x), axis=1)
        accuracy = (reformed_x_pred == np.argmax(y, axis=1)).sum() / len(y)
        logger.info(f'Accuracy of target classifier on reformed data: {accuracy}')
