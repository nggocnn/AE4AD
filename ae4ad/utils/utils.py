import os
import argparse
import sys

import cv2
import numpy as np

from datetime import datetime
from tqdm import tqdm

import tensorflow as tf

H5_EXTENSION = 'h5'
NP_EXTENSION = '.npy'


def str2bool(s):
    if s.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif s.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Unsupported value encountered.')


def recover_image_cv2(image):
    # Recover normalization
    image = image * 255.0

    # Process image_copy and do not destroy the data of image
    image_copy = image.clone().data.permute(1, 2, 0).cpu().numpy()
    image_copy = np.clip(image_copy, 0, 255.0).astype(np.uint8)

    return image_copy


def load_model(model_file_path: str):
    if not os.path.exists(model_file_path):
        return None, None, False, f'Path not found: {model_file_path}.'

    model, model_name, success, message = None, None, False, ""
    if os.path.isdir(model_file_path):
        model, model_name, success, message = load_model_from_folder(model_file_path)
    elif os.path.isfile(model_file_path):
        model, model_name, success, message = load_model_from_h5(model_file_path)

    if success:
        message = f'Successfully loaded model {message}.'
    else:
        message = f'Failed to load model from {message}.'

    return model, model_name, success, message


def load_model_from_h5(model_file_path: str):
    basename = os.path.basename(model_file_path)
    spiltext = os.path.splitext(basename)
    if not spiltext[1] == H5_EXTENSION:
        return None, None, False, f'File format does not match to {H5_EXTENSION}.'
    try:
        model = tf.keras.models.load_model(model_file_path, compile=False)
        model_name = spiltext[0]
    except (ImportError, IOError) as e:
        return None, None, False, e

    return model, model_name, True, f'Loaded model {model_name} with format {H5_EXTENSION}.'


def load_model_from_folder(model_folder_path: str):
    basename = os.path.basename(model_folder_path)
    try:
        model = tf.keras.models.load_model(model_folder_path, compile=False)
        model_name = basename
    except (ImportError, IOError) as e:
        return None, None, False, e

    return model, model_name, True, f'Loaded model {model_name} saved in folder.'


def load_data_from_npy(npy_file_path: str):
    if not os.path.isfile(npy_file_path):
        raise FileNotFoundError(f'Not found numpy binary file {npy_file_path}')

    splitext = os.path.splitext(npy_file_path)

    if not splitext[1] == NP_EXTENSION:
        raise ValueError(f'File type {splitext[1]} is not valid. '
                         f'Please choose file with {NP_EXTENSION} extension.')

    return np.load(npy_file_path)

# def load_data_from_folder(folder_path):
#     if not os.path.isdir(folder_path):
#         raise ValueError(f'Folder path {folder_path} is not valid')
#
#     for class_folder in os.listdir(folder_path):
#         if not class_folder.isnumeric():
#             raise ValueError(f'Sub folder {class_folder} is not valid. '
#                              f'It needs to be numeric as class labels.')


def data_filter(model_fn, x, y, batch_size, equal=True):
    y_ = np.zeros_like(y)
    for i in tqdm(range(0, len(x), batch_size), desc=f'Filtering images: '):
        y_[i: i + batch_size] = model_fn(x[i: i + batch_size])
    if equal:
        return np.where(np.argmax(y_, axis=1) == np.argmax(y, axis=1))
    else:
        return np.where(np.argmax(y_, axis=1) != np.argmax(y, axis=1))
