import numpy as np
import tensorflow as tf

from tqdm import tqdm


class AdversarialGauss:
    def __init__(
            self,
            model_fn,
            x,
            y,
            mu=0.0,
            sigma=1.0,
            eps=0.3,
            eps_iter=0.06,
            n_iters=10,
            clip_min=0.0,
            clip_max=1.0,
    ):
        assert eps_iter <= eps

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
        self.mu = mu
        self.sigma = sigma
        self.eps = eps
        self.eps_iter = eps_iter
        self.n_iters = n_iters
        self.clip_min = clip_min
        self.clip_max = clip_max

    def attack(self):
        x_adv = np.zeros_like(self.x)

        for i in tqdm(range(len(self.x)), desc=f'Attacking: '):
            x_adv[i] = self._attack(self.x[i])

        return x_adv

    def _attack(self, x):
        x_adv = x
        for i in range(self.n_iters):
            eta = self.eps * np.random.normal(
                loc=self.mu,
                scale=self.sigma,
                size=x.shape
            )

            x_adv = np.clip(x_adv + eta, self.clip_min, self.clip_max)

        return x_adv
