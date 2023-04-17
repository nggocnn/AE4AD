import numpy as np
import tensorflow as tf


@tf.function
def compute_gradient(model_fn, loss_fn, x, y, targeted):
    with tf.GradientTape() as tape:
        tape.watch(x)
        loss = loss_fn(y, model_fn(x))
        if targeted:
            loss = -loss

    grad = tape.gradient(loss, x)
    return grad


def get_or_guess_labels(model_fn, x, y=None, targeted=False):
    if targeted is True and y is None:
        raise ValueError("Must provide y for a targeted attack!")

    preds = model_fn(x)
    nb_classes = preds.shape[-1]

    # labels set by the user
    if y is not None:
        # inefficient when y is a tensor, but this function only get called once
        y = np.asarray(y)

        if len(y.shape) == 1:
            # the user provided categorical encoding
            y = tf.one_hot(y, nb_classes)

        y = tf.cast(y, x.dtype)
        return y, nb_classes

    # must be an untargeted attack
    labels = tf.cast(
        tf.equal(tf.reduce_max(preds, axis=1, keepdims=True), preds), x.dtype
    )

    return labels, nb_classes


def set_with_mask(x, x_other, mask):
    mask = tf.cast(mask, x.dtype)
    ones = tf.ones_like(mask, dtype=x.dtype)
    return x_other * mask + x * (ones - mask)
