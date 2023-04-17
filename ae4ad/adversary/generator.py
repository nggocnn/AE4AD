import numpy as np

from ae4ad.adversary.attacks import fgsm, bim, mi_fgsm, cw_l2
from ae4ad.adversary.config_parser import AdversarialConfig
from ae4ad.adversary.const import *
from ae4ad.utils.logger import AE4AD_Logger

logger = AE4AD_Logger.get_logger()


class AdversarialGenerator:
    def __init__(self, config: AdversarialConfig):
        self.config = config

        self.output_folder = self.config.output_path
        self.target_classifier = self.config.target_classifier
        self.classifier_name = self.config.classifier_name

        self.images = self.config.images
        self.labels = self.config.labels

        pred_labels = np.zeros_like(self.labels)

        for i in range(0, len(self.images), 32):
            pred_labels[i: i + 32] = self.target_classifier(self.images[i: i + 32])

        true_pred_indexes = np.where(np.argmax(self.labels, axis=1) == np.argmax(pred_labels, axis=1))

        self.images = self.images[true_pred_indexes]
        self.labels = self.labels[true_pred_indexes]

        self.limit = self.config.limit
        if self.limit > 0:
            _, counts = zip(np.unique(self.labels, axis=0, return_counts=True))

            self.limit = np.min(counts)
            logger.warning(f'Smallest numbers of images belongs to label {np.argmin(counts)} ({self.limit}).')
            logger.warning(f'Set limitation of each label to {self.limit}.')

            labels_indices = [np.where(self.labels[: i] == 1)[0] for i in range(self.config.n_classes)]

            balanced_images = []
            balanced_labels = []
            for indices in labels_indices:
                np.random.shuffle(indices)
                balanced_images.append(self.images[indices[:self.limit]])
                balanced_labels.append(self.labels[indices[:self.limit]])

            self.images = np.concatenate(balanced_images)
            self.labels = np.concatenate(balanced_labels)

        logger.info(dict(zip(*np.unique(np.argmax(self.labels, axis=1), return_counts=True))))

        self.adversarial_config = self.config.adversarial_config

    def adversary_generate(self):
        for key in self.adversarial_config.keys():
            try:
                if key == FGSM:
                    fgsm_config = self.adversarial_config[key]
                    if fgsm_config[ENABLE] == TRUE:
                        epsilon_list = fgsm_config[EPSILON]
                        if not isinstance(epsilon_list, list):
                            epsilon_list = [epsilon_list]

                        for epsilon in epsilon_list:
                            logger.info(f'Generating adversarial examples with {key}: {epsilon}.')
                            fgsm_generator = fgsm.FGSM(
                                model_fn=self.target_classifier,
                                x=self.images,
                                y=self.labels,
                                eps=float(epsilon),
                                batch_size=int(fgsm_config[BATCH_SIZE]),
                                clip_min=self.config.input_range[0],
                                clip_max=self.config.input_range[1],
                                targeted=bool(fgsm_config[TARGETED])
                            )

                            x_adv = fgsm_generator.attack()

                elif key == BIM:
                    bim_config = self.adversarial_config[key]
                    if bim_config[ENABLE] == TRUE:
                        epsilon_list = bim_config[EPSILON]
                        if not isinstance(epsilon_list, list):
                            epsilon_list = [epsilon_list]

                        for epsilon in epsilon_list:
                            logger.info(f'Generating adversarial examples with {key}: {epsilon}.')
                            bim_generator = bim.BIM(
                                model_fn=self.target_classifier,
                                x=self.images,
                                y=self.labels,
                                eps=float(bim_config[MAX_BALL]),
                                eps_iter=float(epsilon),
                                n_iters=int(bim_config[N_ITERATIONS]),
                                batch_size=int(bim_config[BATCH_SIZE]),
                                clip_min=self.config.input_range[0],
                                clip_max=self.config.input_range[1],
                                targeted=bool(bim_config[TARGETED])
                            )

                            x_adv = bim_generator.attack()

                elif key == MI_FGSM:
                    mi_fgsm_config = self.adversarial_config[key]
                    if mi_fgsm_config[ENABLE] == TRUE:
                        epsilon_list = mi_fgsm_config[EPSILON]
                        if not isinstance(epsilon_list, list):
                            epsilon_list = [epsilon_list]

                        for epsilon in epsilon_list:
                            logger.info(f'Generating adversarial examples with {key}: {epsilon}.')
                            mi_fgsm_generator = mi_fgsm.MI_FGSM(
                                model_fn=self.target_classifier,
                                x=self.images,
                                y=self.labels,
                                eps=float(mi_fgsm_config[MAX_BALL]),
                                eps_iter=float(epsilon),
                                n_iters=int(mi_fgsm_config[N_ITERATIONS]),
                                batch_size=int(mi_fgsm_config[BATCH_SIZE]),
                                clip_min=self.config.input_range[0],
                                clip_max=self.config.input_range[1],
                                decay_factor=float(mi_fgsm_config[DECAY_FACTOR]),
                                targeted=bool(mi_fgsm_config[TARGETED])
                            )

                            x_adv = mi_fgsm_generator.attack()

                elif key == CW_L2:
                    cw_l2_config = self.adversarial_config[key]
                    if cw_l2_config[ENABLE] == TRUE:
                        conf_list = cw_l2_config[CONFIDENCE]
                        if not isinstance(conf_list, list):
                            conf_list = [conf_list]

                        for conf in conf_list:
                            logger.info(f'Generating adversarial examples with {key}: {conf}.')
                            cw_generator = cw_l2.CarliniWagnerL2(
                                model_fn=self.target_classifier,
                                x=self.images,
                                batch_size=int(cw_l2_config[BATCH_SIZE]),
                                clip_min=self.config.input_range[0],
                                clip_max=self.config.input_range[1],
                                confidence=float(conf)
                            )

                            x_adv = cw_generator.attack()
            except Exception as e:
                logger.debug(e)
