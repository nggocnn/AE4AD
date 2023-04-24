import os

from configobj import ConfigObj

from ae4ad.defenses.ae4ad.const import *
from ae4ad.defenses.losses import mse_ce_loss
from ae4ad.utils.logger import AE4AD_Logger
from ae4ad.utils.utils import *

logger = AE4AD_Logger.get_logger()


class AE4AD_Config:
    def __init__(self, filepath):
        self.config = ConfigObj(filepath)

        self.training_config = None
        self.autoencoder_path = None
        self.epochs = 100
        self.valid_ratio = 0.1
        self.batch_size = 128
        self.learning_rate = 0.001
        self.early_stopping = False
        self.min_delta = 0.0005
        self.loss = 'mse'

        self.general_config = None
        self.output_path = None
        self.target_classifier = None
        self.classifier_name = None
        self.image_shape = None
        self.n_classes = None

        self.input_range = [0.0, 1.0]

        self.data_config = None

        self.adversarial_folder = None
        self.gt_original_folder = None
        self.gt_labels_folder = None

        self.original_images_file = None
        self.original_labels_file = None

        self.adversarial_data = []
        self.gt_original_data = []
        self.gt_labels_data = []

        self.config_parse()

    def config_parse(self):
        if self.config.__contains__(GENERAL_CONFIG):
            # load general configuration keys
            self.general_config = self.config[GENERAL_CONFIG]

            # load output folder path
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

                classifier.summary()

                self.target_classifier = classifier
                self.classifier_name = classifier_name
                self.image_shape = classifier.inputs[0].shape[1:]
                self.n_classes = classifier.outputs[0].shape[1:][0]

                # load range of input value
                if self.general_config.__contains__(PIXEL_RANGE):
                    input_range = np.array(self.general_config[PIXEL_RANGE], dtype='float32')
                    if len(input_range) == 2:
                        self.input_range = input_range
                    else:
                        logger.warning('Please check input range configuration!')

                logger.info(f'Range of input value is set to {str(self.input_range)}.')
        else:
            raise ValueError('Not found general configurations!')

        if self.config.__contains__(TRAINING_CONFIG):
            self.training_config = self.config[TRAINING_CONFIG]

            if self.training_config.__contains__(AUTOENCODER_PATH):
                self.autoencoder_path = self.training_config[AUTOENCODER_PATH]
                if self.autoencoder_path == '':
                    logger.debug(
                        f'Autoencoder path has not been defined, set to default path {DEFAULT_AUTOENCODER_PATH}')
                    self.autoencoder_path = DEFAULT_AUTOENCODER_PATH

            self.autoencoder_path = os.path.abspath(self.autoencoder_path)
            if not os.path.exists(self.autoencoder_path):
                os.makedirs(self.autoencoder_path)

            logger.debug(f'Autoencoder will be saved to {self.autoencoder_path}')

            if self.training_config.__contains__(TRAINING_CONFIG):
                self.epochs = int(self.training_config[EPOCHS])

            logger.info(f'Training Autoencoder for {self.epochs}')

            if self.training_config.__contains__(LOSS):
                loss_type = self.training_config[LOSS]
                if loss_type == MSE:
                    self.loss = 'mse'
                elif loss_type == MSE_CE or loss_type == MSE_CE_O:
                    self.loss = mse_ce_loss(self.target_classifier)
                else:
                    logger.error(f'Loss function type has not been defined {loss_type}, set to default {self.loss}')
            logger.info(f'Loss functions: {str(self.loss)}')


            if self.training_config.__contains__(VALID_RATIO):
                self.valid_ratio = float(self.training_config[VALID_RATIO])

            if self.training_config.__contains__(BATCH_SIZE):
                self.batch_size = int(self.training_config[BATCH_SIZE])

            if self.training_config.__contains__(LEARNING_RATE):
                self.learning_rate = float(self.training_config[LEARNING_RATE])

            if self.training_config.__contains__(EARLY_STOPPING):
                self.early_stopping = True if self.training_config[EARLY_STOPPING] == TRUE else False

            if self.training_config.__contains__(MIN_DELTA):
                self.min_delta = float(self.training_config[MIN_DELTA]) if self.early_stopping else None
        else:
            raise ValueError(f'Not found training configuration!')

        if self.config.__contains__(DATA_CONFIG):
            self.data_config = self.config[DATA_CONFIG]

            if self.data_config.__contains__(ADVERSARIAL_FOLDER):
                self.adversarial_folder = self.data_config[ADVERSARIAL_FOLDER]
            else:
                raise ValueError('Not found adversarial folder!')

            if self.data_config.__contains__(GT_ORIGINAL_FOLDER):
                self.gt_original_folder = self.data_config[GT_ORIGINAL_FOLDER]
            else:
                raise ValueError('Not found ground truth images folder!')

            if self.data_config.__contains__(GT_LABELS_FOLDER):
                self.gt_labels_folder = self.data_config[GT_LABELS_FOLDER]
            else:
                raise ValueError('Not found ground truth labels folder!')

            self.adversarial_data, self.gt_original_data, self.gt_labels_data = \
                self.load_adversarial_data(self.adversarial_folder, self.gt_original_folder, self.gt_labels_folder)

            if self.data_config.__contains__(ORIGINAL_IMAGES_FILE):
                self.original_images_file = self.data_config[ORIGINAL_IMAGES_FILE]
            else:
                raise ValueError('Not found ground original images file!')

            if self.data_config.__contains__(ORIGINAL_LABELS_FILE):
                self.original_labels_file = self.data_config[ORIGINAL_LABELS_FILE]
            else:
                raise ValueError('Not found ground original labels file!')

            original_images_data, original_labels_data = \
                self.load_original_data(self.original_images_file, self.original_labels_file)

            self.adversarial_data.append(original_images_data)
            self.gt_original_data.append(original_images_data)
            self.gt_labels_data.append(original_labels_data)
        else:
            raise ValueError(f'Not found data configurations!')

    def load_adversarial_data(self, adversarial_folder, gt_original_folder, gt_labels_folder):
        adversarial_filenames_list = sorted(os.listdir(adversarial_folder))
        original_filenames_list = sorted(os.listdir(gt_original_folder))
        labels_filenames_list = sorted(os.listdir(gt_labels_folder))

        adversarial_data = []
        gt_original_data = []
        gt_labels_data = []

        for adversarial_filename, gt_original_filename, gt_labels_filename \
                in zip(adversarial_filenames_list, original_filenames_list, labels_filenames_list):
            adversarial_filepath = os.path.join(adversarial_folder, adversarial_filename)
            gt_original_filepath = os.path.join(gt_original_folder, gt_original_filename)
            gt_labels_filepath = os.path.join(gt_labels_folder, gt_labels_filename)

            logger.info(f'Loading adversarial data from {adversarial_filepath}')

            is_valid = True

            temp_adversarial = load_data_from_npy(adversarial_filepath)
            temp_gt_original = load_data_from_npy(gt_original_filepath)
            temp_gt_labels = load_data_from_npy(gt_labels_filepath)

            if np.min(temp_adversarial) < self.input_range[0] or np.max(temp_adversarial) > self.input_range[1] \
                    or np.min(temp_gt_original) < self.input_range[0] or np.max(temp_gt_original) > self.input_range[1]:
                raise ValueError(f'Value range of input data is not in range {self.input_range}')

            if len(temp_adversarial) != len(temp_gt_original) \
                    or len(temp_adversarial) != len(temp_gt_labels) \
                    or len(temp_gt_original) != len(temp_gt_labels):
                raise ValueError(f'Length of input data is invalid: length of adversaries {len(temp_adversarial)} '
                                 f'length of origins {len(temp_gt_original)} '
                                 f'length of labels {len(temp_gt_labels)}')

            if temp_adversarial.shape[1:] != self.image_shape \
                    or temp_gt_original.shape[1:] != self.image_shape:
                raise ValueError(f'Images shape (advs shape {temp_adversarial.shape[1:]}, '
                                 f'origins shape {temp_gt_original.shape[1:]}) must be '
                                 f'identical to classifier\'s input shape {self.image_shape}')

                # whether label is in one-hot vector shape or not
            if temp_gt_labels.shape[1:] == (1,) or len(temp_gt_labels.shape) == 1:
                logger.warning(f'Labels set may not be in one-hot vector shape: {temp_gt_labels.shape[1:]}')
                logger.debug(f'Converting labels set to one-hot vectors')

                # check whether sparse labels data value is in valid range or not
                if np.min(temp_gt_labels) < 0 or np.max(temp_gt_labels) > self.n_classes:
                    raise ValueError(f'Error in sparse labels data: '
                                     f'min={np.min(temp_gt_labels)}, max={np.max(temp_gt_labels)}')

                # convert sparse labels data to one hot vectors
                # some labels may not appear but still convert to match output of classifier
                temp_gt_labels = tf.keras.utils.to_categorical(temp_gt_labels, self.n_classes)
                logger.debug(f'Labels\' shape after converting: {temp_gt_labels.shape[1:]}')

            if is_valid:
                logger.info(f'Loaded adversarial data: {len(temp_gt_labels)}')
                adversarial_data.append(temp_adversarial)
                gt_original_data.append(temp_gt_original)
                gt_labels_data.append(temp_gt_labels)

        if len(adversarial_data) == 0 or len(gt_original_data) == 0 or len(gt_labels_data) == 0:
            raise ValueError(f'Error in loading data!')

        return adversarial_data, gt_original_data, gt_labels_data

    def load_original_data(self, original_images_file, original_labels_file):
        logger.info(f'Loading original data from {original_images_file}')
        temp_original_images = load_data_from_npy(original_images_file)
        temp_original_labels = load_data_from_npy(original_labels_file)

        if np.min(temp_original_images) < self.input_range[0] \
                or np.max(temp_original_images) > self.input_range[1]:
            raise ValueError(f'Value range of input data is not in range {self.input_range}')

        if len(temp_original_images) != len(temp_original_labels):
            raise ValueError(f'Length of input data is invalid:'
                             f'length of origins {len(temp_original_images)}, '
                             f'length of labels {len(temp_original_labels)}')

        if temp_original_images.shape[1:] != self.image_shape:
            raise ValueError(f'Images shape (origins shape {temp_original_images.shape[1:]}) must be '
                             f'identical to classifier\'s input shape {self.image_shape}')

        # whether label is in one-hot vector shape or not
        if temp_original_labels.shape[1:] == (1,) or len(temp_original_labels.shape) == 1:
            logger.warning(f'Labels set may not be in one-hot vector shape: {temp_original_labels.shape[1:]}')
            logger.debug(f'Converting labels set to one-hot vectors')

            # check whether sparse labels data value is in valid range or not
            if np.min(temp_original_labels) < 0 or np.max(temp_original_labels) > self.n_classes:
                raise ValueError(
                    f'Error in sparse labels data: min={np.min(temp_original_labels)}, max={np.max(temp_original_labels)}')

            # convert sparse labels data to one hot vectors
            # some labels may not appear but still convert to match output of classifier
            temp_original_labels = tf.keras.utils.to_categorical(temp_original_labels, self.n_classes)
            logger.debug(f'Labels\' shape after converting: {temp_original_labels.shape[1:]}')

        if len(temp_original_images) == 0 or len(temp_original_labels) == 0:
            raise ValueError(f'Error in loading data!')

        if self.training_config[LOSS] == MSE_CE_O:
            target_pred = np.argmax(self.target_classifier.predict(temp_original_images), axis=1).reshape(-1)
            temp_label = np.argmax(temp_original_labels, axis=1).reshape(-1)

            true_pred_idx = np.where(target_pred == temp_label)

            temp_original_images = temp_original_images[true_pred_idx]
            temp_original_labels = temp_original_labels[true_pred_idx]
        logger.info(f'Loaded original data: {len(temp_original_labels)}')
        return temp_original_images, temp_original_labels
