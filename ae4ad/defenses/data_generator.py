import tensorflow as tf
import numpy as np


class GaussianDataGenerator(tf.keras.utils.Sequence):
    def __init__(self, data, noise_volume=0.01, batch_size=128, shuffle=True):
        """Initialization"""
        self.indexes = None
        self.data = data
        self.noise_volume = noise_volume
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.data_idx = np.arange(len(self.data))
        self.on_epoch_end()

    def on_epoch_end(self):
        """Updates indexes after each epoch"""
        self.indexes = np.arange(len(self.data))
        if self.shuffle:
            np.random.shuffle(self.indexes)

    def __len__(self):
        """Denotes the number of batches per epoch"""
        return int(np.floor(len(self.data) / self.batch_size))

    def __getitem__(self, index):
        """Generate one batch of data"""

        # Generate indexes of the batch
        indexes = self.indexes[index * self.batch_size: (index + 1) * self.batch_size]

        # Generate data
        x, y = self.__data_generation(indexes)

        return x, y

    def __data_generation(self, id_list):
        batch_data = self.data[id_list]
        train_noise = [self.noise_volume * np.random.normal(size=batch_data.shape[1:]) for i in range(self.batch_size)]
        x = np.clip(batch_data + train_noise, 0.0, 1.0)
        y = batch_data

        return x, y
