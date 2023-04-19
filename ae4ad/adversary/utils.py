import os.path

import numpy as np
import tensorflow as tf

from matplotlib import pyplot as plt, gridspec

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


def export_adversarial_npy(folder, name, prefix, suffix, data, separate_folder):

    assert isinstance(data, list)

    for i in range(len(data)):
        if isinstance(prefix, list):
            assert len(prefix) == len(data)
            if separate_folder:
                save_folder = os.path.join(folder, prefix[i])
                save_name = f'{name}'
            else:
                save_folder = folder
                save_name = f'{prefix[i]}_{name}'
        else:
            if separate_folder:
                save_folder = os.path.join(folder, prefix)
                save_name = f'{name}'
            else:
                save_folder = folder
                save_name = f'{prefix}_{name}'

        if isinstance(suffix, list):
            assert len(suffix) == len(data)
            save_name = f'{save_name}{("_" + str(suffix[i])) if suffix[i] is not None else ""}.npy'
        else:
            save_name = f'{save_name}{("_" + str(suffix)) if suffix is not None else ""}.npy'

        if not os.path.exists(save_folder):
            os.makedirs(save_folder)

        save_path = os.path.join(save_folder, save_name)

        np.save(save_path, data[i])


def save_adversarial_samples(figname, x_adv, x_origin, y_adv, y_origin, shape, n=10, dpi=600):
    fig, axs = plt.subplots(nrows=n, ncols=2, figsize=(10, 15))
    indices = np.random.randint(low=0, high=len(x_adv), size=n)
    for i in range(n):
        idx = indices[i]
        axs[i, 0].imshow(x_origin[idx].reshape(shape))
        axs[i, 0].set_title(f'Original image - label {y_origin[idx]}')
        axs[i, 0].axis('off')

        axs[i, 1].imshow(x_adv[idx].reshape(shape))
        axs[i, 1].set_title(f'Original image - label {y_adv[idx]} - L2 {np.linalg.norm(x_adv - x_origin):.4f}')
        axs[i, 1].axis('off')

    fig.savefig(figname, bbox_inches='tight', dpi=dpi)

