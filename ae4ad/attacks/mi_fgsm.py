import numpy as np
import tensorflow as tf

from ae4ad.attacks.fgsm import FGSM
from ae4ad.attacks.utils import compute_gradient, optimize_linear, clip_eta


class MI_FGSM(FGSM):
    def __init__(
            self,
            model_fn,
            x,
            y,
            eps=0.3,
            eps_iter=0.06,
            n_iters=10,
            loss_fn=None,
            batch_size=128,
            clip_min=0.0,
            clip_max=1.0,
            decay_factor=1.0,
            targeted=False,
    ):
        assert eps_iter <= eps

        super().__init__(
            model_fn,
            x,
            y,
            eps,
            loss_fn,
            batch_size,
            clip_min,
            clip_max,
            targeted
        )

        self.eps_iter = eps_iter
        self.n_iters = n_iters
        self.decay_factor = decay_factor,
        self.targeted = targeted

    def _attack(self, x, y):
        x = tf.cast(x, tf.float32)

        if y is None:
            y = self.model_fn(x)
            self.targeted = False

        momentum = tf.zeros_like(x)
        x_adv = x

        for i in range(self.n_iters):
            grad = compute_gradient(self.model_fn, self.loss_fn, x_adv, y, self.targeted)

            reduce_axis = list(range(1, len(grad.shape)))
            avoid_zero_div = tf.cast(1e-12, grad.dtype)
            grad = grad / tf.math.maximum(
                avoid_zero_div,
                tf.math.reduce_mean(tf.math.abs(grad), reduce_axis, keepdims=True),
            )

            momentum = self.decay_factor * momentum + grad

            optimal_perturbation = optimize_linear(momentum, self.eps_iter, norm=np.inf)

            x_adv = x_adv + optimal_perturbation

            x_adv = x + clip_eta(x_adv - x, np.inf, self.eps)

            if self.clip_min is not None and self.clip_max is not None:
                x_adv = tf.clip_by_value(x_adv, self.clip_min, self.clip_max)

        return x_adv
