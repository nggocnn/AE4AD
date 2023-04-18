from configobj import ConfigObj

from ae4ad.adversary.const import *
from ae4ad.utils.logger import AE4AD_Logger
from ae4ad.utils.utils import *

logger = AE4AD_Logger().get_logger()


class AdversarialConfig:
    def __init__(self, filepath):
        self.config = ConfigObj(filepath)

        self.general_config = None
        self.output_path = None

        self.target_classifier = None
        self.classifier_name = None
        self.input_shape = None
        self.output_shape = None
        self.n_classes = None

        self.images = None
        self.labels = None
        self.limit = 0

        self.input_range = [0.0, 1.0]

        self.adversarial_config = {}

        self.config_parse()

    def config_parse(self):
        if self.config.__contains__(GENERAL_CONFIG):
            # load general configuration keys
            self.general_config = self.config[GENERAL_CONFIG]

            # load output folder path
            if self.general_config.__contains__(OUTPUT_PATH):
                self.output_path = self.general_config[OUTPUT_PATH]

            if self.output_path is None or self.output_path == "":
                self.output_path = DEFAULT_OUTPUT_PATH

            if not os.path.exists(self.output_path):
                os.makedirs(self.output_path)
                logger.info(f'Created output folder {self.output_path}.')
            elif any(os.scandir(self.output_path)):
                logger.warning(f'Output folder {self.output_path} is not empty.')

            # load targeted classifier
            if self.general_config.__contains__(TARGET_CLASSIFIER_PATH):
                target_classifier_path = self.general_config[TARGET_CLASSIFIER_PATH]

                logger.info(f'Loading target classifier at {target_classifier_path}.')
                classifier, classifier_name, success, message = load_model(target_classifier_path)

                if not success:
                    raise IOError(message)

                logger.info(message)

                self.target_classifier = classifier
                self.classifier_name = classifier_name
                self.input_shape = classifier.inputs[0].shape[1:]
                self.output_shape = classifier.outputs[0].shape[1:]
                self.n_classes = self.output_shape[0]

            # load range of input value
            if self.general_config.__contains__(PIXEL_RANGE):
                input_range = np.array(self.general_config[PIXEL_RANGE], dtype='float32')
                if len(input_range) == 2:
                    self.input_range = input_range
                else:
                    logger.warning('Please check input range configuration!')

            logger.info(f'Range of input value is set to {str(self.input_range)}.')

            # load images set and labels set
            if self.general_config.__contains__(IMAGES_FILE_PATH) and \
                    self.general_config.__contains__(LABELS_FILE_PATH):

                self.images = load_data_from_npy(self.general_config[IMAGES_FILE_PATH])
                self.labels = load_data_from_npy(self.general_config[LABELS_FILE_PATH])

                # validate value range of images
                if np.min(self.images) < self.input_range[0] or np.max(self.images) > self.input_range[1]:
                    raise ValueError(f'Value range of images data is not valid '
                                     f'(min={np.min(self.images)}, max={np.max(self.images)}).')

                # validate size of images set and labels set
                if len(self.images) != len(self.labels):
                    raise ValueError(f'Size of images set ({len(self.images)}) must be '
                                     f'identical to  size of labels set ({len(self.labels)}).')

                # validate shape of images
                if self.images.shape[1:] != self.input_shape:
                    raise ValueError(f'Images\' shape {self.images.shape[1:]} must be '
                                     f'identical to classifier\'s input shape {self.input_shape}.')

                # validate shape of labels
                if self.labels.shape[1:] == (1, ) or len(self.labels.shape) == 1:
                    logger.warning(f'Labels set may not be in one-hot vector shape ({self.labels.shape[1:]}).')
                    logger.debug(f'Converting labels set to one-hot vectors.')

                    # check whether sparse labels data value is in valid range or not
                    if np.min(self.labels) < 0 or np.max(self.labels) > self.output_shape[0]:
                        raise ValueError(f'Error in sparse labels data '
                                         f'(min={np.min(self.labels)}, max={np.max(self.labels)}).')

                    # convert sparse labels data to one hot vectors
                    # some labels may not appear but still convert to match output of classifier
                    self.labels = tf.keras.utils.to_categorical(self.labels, self.n_classes)
                    logger.debug(f'Labels\' shape after converted to one-hot vectors: {self.labels.shape[1:]}.')

                indexes = data_filter(self.target_classifier, self.images, self.labels, 32, verbose=False)

                self.images = self.images[indexes]
                self.labels = self.labels[indexes]

                logger.info(f'Number of correct-predicted images: {len(self.images)}.')
            else:
                raise ValueError('Not found configurations for input data')

            # load length's limitation
            if self.general_config.__contains__(LIMIT_PER_LABEL):
                self.limit = self.general_config[LIMIT_PER_LABEL]

            if self.limit == 0:
                logger.info('Adversarial examples are being generated from full dataset.')
            else:
                logger.info(f'Adversarial examples are being generated from maximum {self.limit} data points each label.')

            logger.info('Loaded general configurations completely!')
        else:
            raise ValueError('Not found general configurations!')

        self.adversarial_config[FGSM] = self.config[FGSM] if self.config.__contains__(FGSM) else None

        self.adversarial_config[BIM] = self.config[BIM] if self.config.__contains__(BIM) else None

        self.adversarial_config[MI_FGSM] = self.config[MI_FGSM] if self.config.__contains__(MI_FGSM) else None

        self.adversarial_config[CW_L2] = self.config[CW_L2] if self.config.__contains__(CW_L2) else None

        self.adversarial_config[GAUSS] = self.config[GAUSS] if self.config.__contains__(GAUSS) else None
