import numpy as np
import tensorflow as tf
from tqdm import tqdm

from ae4ad.adversary.utils import compute_gradient
from ae4ad.utils.utils import data_filter


class FGSM:
    def __init__(
            self,
            model_fn,
            x,
            y,
            eps=0.06,
            loss_fn=None,
            batch_size=128,
            clip_min=0.0,
            clip_max=1.0,
            targeted=False
    ):
        if clip_min is not None:
            if not np.all(tf.math.greater_equal(x, clip_min)):
                raise ValueError(
                    f"The input is smaller than the minimum value of {clip_min}!"
                )

        if clip_max is not None:
            if not np.all(tf.math.less_equal(x, clip_max)):
                raise ValueError(
                    f"The input is greater than the maximum value of {clip_max}!"
                )

        self.model_fn = model_fn
        self.x = x
        self.y = y
        self.eps = eps
        self.loss_fn = tf.nn.softmax_cross_entropy_with_logits if loss_fn is None else loss_fn
        self.batch_size = batch_size
        self.clip_min = clip_min
        self.clip_max = clip_max
        self.targeted = targeted

    def attack(self):
        x_adv = np.zeros_like(self.x)

        for i in tqdm(
                range(0, len(self.x), self.batch_size),
                desc=f'Attacking batch of {self.batch_size} images: '
        ):
            x_batch = self.x[i: i + self.batch_size]
            y_batch = self.y[i: i + self.batch_size]
            x_adv[i: i + self.batch_size] = self._attack(x_batch, y_batch).numpy()

        indexes = data_filter(self.model_fn, x_adv, self.y, self.batch_size, equal=False)

        return x_adv[indexes], self.x[indexes], self.y[indexes]

    def _attack(self, x, y):

        x = tf.cast(x, tf.float32)

        if y is None:
            y = self.model_fn(x)
            self.targeted = False

        grad = compute_gradient(self.model_fn, self.loss_fn, x, y, self.targeted)
        eta = tf.multiply(self.eps, tf.stop_gradient(tf.sign(grad)))

        x_adv = x + eta

        if self.clip_min is not None and self.clip_max is not None:
            x_adv = tf.clip_by_value(x_adv, self.clip_min, self.clip_max)

        return x_adv
