import numpy as np

from ae4ad.attacks.fgsm import FGSM
from ae4ad.attacks.utils import clip_eta


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
            x_adv = x + clip_eta(x_adv - x, np.inf, self.eps)

        return x_adv
