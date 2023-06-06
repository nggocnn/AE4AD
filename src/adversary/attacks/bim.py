import tensorflow as tf

from src.adversary.attacks.fgsm import FGSM


class BIM(FGSM):
    def __init__(
            self,
            model_fn,
            x,
            y,
            eps=0.039,
            n_iters=10,
            loss_fn=None,
            batch_size=128,
            clip_min=0.0,
            clip_max=1.0,
            targeted=False,
    ):

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

        self.n_iters = n_iters

    def _attack(self, x, y):
        x_adv = x
        for i in range(self.n_iters):
            x_adv = super()._attack(x_adv, y)

        return x_adv
