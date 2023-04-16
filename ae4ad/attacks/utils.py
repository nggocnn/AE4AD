import numpy as np
import tensorflow as tf


@tf.function
def compute_gradient(model_fn, loss_fn, x, y, targeted):
    with tf.GradientTape() as tape:
        tape.watch(x)
        loss = loss_fn(labels=y, logits=model_fn)
        if targeted:
            loss = -loss

    grad = tape.gradient(loss, x)
    return grad


def optimize_linear(grad, eps, norm=np.inf):
    # Convert the iterator returned by `range` into a list.
    axis = list(range(1, len(grad.get_shape())))
    avoid_zero_div = 1e-12
    if norm == np.inf:
        optimal_perturbation = tf.sign(grad)
        optimal_perturbation = tf.stop_gradient(optimal_perturbation)

    elif norm == 1:
        abs_grad = tf.abs(grad)
        sign = tf.sign(grad)
        max_abs_grad = tf.reduce_max(abs_grad, axis, keepdims=True)
        tied_for_max = tf.dtypes.cast(
            tf.equal(abs_grad, max_abs_grad), dtype=tf.float32
        )
        num_ties = tf.reduce_sum(tied_for_max, axis, keepdims=True)
        optimal_perturbation = sign * tied_for_max / num_ties

    elif norm == 2:
        square = tf.maximum(
            avoid_zero_div, tf.reduce_sum(tf.square(grad), axis, keepdims=True)
        )
        optimal_perturbation = grad / tf.sqrt(square)

    else:
        raise NotImplementedError(
            "Only L-inf, L1 and L2 norms are currently implemented."
        )

    scaled_perturbation = tf.multiply(eps, optimal_perturbation)
    return scaled_perturbation


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


def clip_eta(eta, norm, eps):
    # Clipping perturbation eta to self.norm norm ball
    if norm not in [np.inf, 1, 2]:
        raise ValueError("norm must be np.inf, 1, or 2.")
    axis = list(range(1, len(eta.get_shape())))
    avoid_zero_div = 1e-12
    if norm == np.inf:
        eta = tf.clip_by_value(eta, -eps, eps)
    else:
        if norm == 1:
            raise NotImplementedError("")

        elif norm == 2:
            # avoid_zero_div must go inside sqrt to avoid a divide by zero in the gradient through this operation
            norm = tf.sqrt(
                tf.maximum(
                    avoid_zero_div, tf.reduce_sum(tf.square(eta), axis, keepdims=True)
                )
            )
        # We must *clip* to within the norm ball, not *normalize* onto the surface of the ball
        factor = tf.minimum(1.0, tf.math.divide(eps, norm))
        eta = eta * factor
    return eta

