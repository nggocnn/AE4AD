import tensorflow as tf

from ae4ad.adversary.attacks.fgsm import FGSM


class BIM(FGSM):
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

    def _attack(self, x, y):
        x_adv = x
        for i in range(self.n_iters):
            x_adv = super()._attack(x_adv, y)
            x_adv = x + tf.clip_by_value(x_adv - x, -self.eps, self.eps)

        return x_adv
