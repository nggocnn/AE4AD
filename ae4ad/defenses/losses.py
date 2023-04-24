import tensorflow as tf


def mse_ce_loss(substitute_classifier):
    def loss(y_true, y_pred):
        squared_difference = tf.square(y_true - y_pred)
        ce = tf.keras.losses.CategoricalCrossentropy()(
            substitute_classifier(y_true), substitute_classifier(y_pred)
        )
        return tf.reduce_mean(squared_difference, axis=-1) + ce

    return loss
