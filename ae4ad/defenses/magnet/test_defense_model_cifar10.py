import os

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras.datasets import mnist, fashion_mnist, cifar10
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import to_categorical
from tqdm import tqdm

from defender import Defender
from detector import DivergenceDetector
from classifier import Classifier
from data import Cifar10Data


if __name__ == '__main__':
    ae_reformer = load_model("D:\\Things\\PyProject\\AE4AD\\Data\\MagNet\\best_trained_magnet_reformer\\cifar_magnet_0.01")
    # ae_reformer = load_model("D:\\Things\\PyProject\\AE4AD\\Data\\MagNet\\best_trained_magnet_reformer\\cifar_magnet_0.025")
    # ae_reformer = load_model("D:\\Things\\PyProject\\AE4AD\\Data\\MagNet\\best_trained_magnet_reformer\\cifar_magnet_0.05")

    ae_detector = load_model("D:\\Things\\PyProject\\AE4AD\\Data\\MagNet\\best_trained_magnet_detector\\detector_magnet_cifar10")

    classifier = Classifier(data=Cifar10Data, with_softmax=False, pretrained=True)
    classifier.load("D:\\Things\\PyProject\\AE4AD\\Data\\CNN\\cifar10_classifier\\cifar10_cnn_model.h5")

    cifar10_classifier = load_model("D:\\Things\\PyProject\\AE4AD\\Data\\CNN\\cifar10_classifier\\cifar10_cnn_model.h5")



    detector_I = DivergenceDetector(ae_detector, classifier, t=40)

    detectors = {
        "detector": detector_I,
    }

    drop_rates = {
        "detector": 0.01,
    }

    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    x_train = np.expand_dims(x_train, axis=-1) / 255.0
    y_train = to_categorical(y_train)
    x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=float(5/50), random_state=7)

    defender = Defender(
        x_val, y_val, classifier, detectors, drop_rates
    )

    print(defender.thresholds)

    summary_first_time = True
    images_path = "D:\\Things\\PyProject\\AE4AD\\Data\\ADV\\cifar10_test_adv\\adv"
    labels_path = "D:\\Things\\PyProject\\AE4AD\\Data\\ADV\\cifar10_test_adv\\label"

    result_path = "D:\\Things\\PyProject\\AE4AD\\Data\\MagNet\\"
    summary_result_path = result_path + "cifar10_summary_0.01.csv"
    # summary_result_path = result_path + "cifar10_summary_0.025.csv"
    # summary_result_path = result_path + "cifar10_summary_0.05.csv"

    image_files_list = sorted(os.listdir(images_path))
    label_files_list = sorted(os.listdir(labels_path))

    for idx, image_file in tqdm(enumerate(image_files_list), desc="Testing..."):
        images = np.load(os.path.join(images_path, image_file))
        labels = np.load(os.path.join(labels_path, label_files_list[idx]))
        if len(labels.shape) == 2:
            labels = np.argmax(labels, axis=1)

        idx_passed, idx_not_passed = defender.filter(images)

        acc_on_passed_images = 0
        acc_on_reformed_passed_images = 0
        n_pass_pred_true = 0

        if len(idx_passed) > 0:
            passed_images = images[idx_passed]
            passed_labels = labels[idx_passed]

            pred_reformed_passed = np.argmax(
                classifier.predict(ae_reformer.predict(passed_images)), axis=1)
            n_pass_pred_true = (pred_reformed_passed == passed_labels).sum()
            acc_on_reformed_passed_images = n_pass_pred_true / len(idx_passed)

        pred_reformed_all = np.argmax(classifier.predict(ae_reformer.predict(images)), axis=1)
        acc_on_reformed_all = (pred_reformed_all == labels).sum() / len(images)

        summary_results = pd.DataFrame({
            'config': image_file,
            'n_images': len(images),
            'n_not_passed': len(idx_not_passed),
            'magnet_all_acc': (n_pass_pred_true + len(idx_not_passed)) / len(images),
            'acc_on_reformed_all': acc_on_reformed_all,
        }, index=[idx])


        if summary_first_time:
            summary_results.to_csv(summary_result_path, mode='w', header=True, index=True, float_format='%.6f')
            summary_first_time = False
        else:
            summary_results.to_csv(summary_result_path, mode='a', header=False, index=True, float_format='%.6f')

    images_test = np.load("D:\\Things\\PyProject\\AE4AD\\Data\\CNN\\cifar10_classifier\\cifar10_test_data.npy")
    labels_test = np.load("D:\\Things\\PyProject\\AE4AD\\Data\\CNN\\cifar10_classifier\\cifar10_test_label.npy")
    if len(labels_test.shape) == 2:
        labels = np.argmax(labels_test, axis=1)

    idx_passed, idx_not_passed = defender.filter(images_test)

    acc_on_passed_images = 0
    acc_on_reformed_passed_images = 0
    n_pass_pred_true = 0

    if len(idx_passed) > 0:
        passed_images = images_test[idx_passed]
        passed_labels = labels_test[idx_passed]

        pred_reformed_passed = np.argmax(
            classifier.predict(ae_reformer.predict(passed_images)), axis=1)
        n_pass_pred_true = (pred_reformed_passed == passed_labels).sum()
        acc_on_reformed_passed_images = n_pass_pred_true / len(idx_passed)

    pred_reformed_all = np.argmax(classifier.predict(ae_reformer.predict(images_test)), axis=1)
    acc_on_reformed_all = (pred_reformed_all == labels_test).sum() / len(images_test)

    summary_results = pd.DataFrame({
        'config': 'test',
        'n_images': len(images_test),
        'n_not_passed': len(idx_not_passed),
        'magnet_all_acc': n_pass_pred_true / len(images_test),
        'acc_on_reformed_all': acc_on_reformed_all,
    }, index=[6])

    summary_results.to_csv(summary_result_path, mode='a', header=False, index=True, float_format='%.6f')
