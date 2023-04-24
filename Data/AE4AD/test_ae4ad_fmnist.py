import os

import numpy as np
import pandas as pd
from tensorflow.keras.datasets import mnist, fashion_mnist, cifar10
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import to_categorical
from tqdm import tqdm

if __name__ == '__main__':
    classifier = load_model("D:\\Things\\PyProject\\AE4AD\\Data\\CNN\\fmnist_classifier\\fmnist_cnn_model.h5")
    ae4ad_reformer = load_model("D:\\Things\\PyProject\\AE4AD\\Data\AE4AD\\autoencoder_fmnist_cnn_model_softmax_0439", compile=False)

    summary_first_time = True
    images_path = "D:\\Things\\PyProject\\AE4AD\\Data\ADV\\fmnist_test_adv\\adv"
    labels_path = "D:\\Things\\PyProject\\AE4AD\\Data\ADV\\fmnist_test_adv\\label"

    result_path = "D:\\Things\\PyProject\\AE4AD\\Data\\AE4AD\\"

    summary_result_path = result_path + "ae4ad_fmnist_summary.csv"

    image_files_list = sorted(os.listdir(images_path))
    label_files_list = sorted(os.listdir(labels_path))

    for idx, image_file in tqdm(enumerate(image_files_list), desc="Testing..."):
        images = np.load(os.path.join(images_path, image_file))
        labels = np.load(os.path.join(labels_path, label_files_list[idx]))
        if len(labels.shape) == 2:
            labels = np.argmax(labels, axis=1)

        reformed_image = ae4ad_reformer.predict(images)
        pred = np.argmax(classifier.predict(reformed_image), axis=1)
        acc = (pred == labels).sum() / len(images)

        summary_results = pd.DataFrame({
            'config': image_file,
            'n_images': len(images),
            'acc': acc
        }, index=[idx])

        if summary_first_time:
            summary_results.to_csv(summary_result_path, mode='w', header=True, index=True, float_format='%.4f')
            summary_first_time = False
        else:
            summary_results.to_csv(summary_result_path, mode='a', header=False, index=True, float_format='%.4f')

    test_images = np.load("D:\\Things\\PyProject\\AE4AD\\Data\\CNN\\fmnist_classifier\\fmnist_test_data.npy")
    test_labels = np.load("D:\\Things\\PyProject\\AE4AD\\Data\\CNN\\fmnist_classifier\\fmnist_test_label.npy")

    reformed_image = ae4ad_reformer.predict(test_images)
    pred = np.argmax(classifier.predict(reformed_image), axis=1)
    acc = (pred == test_labels).sum() / len(test_images)

    summary_results = pd.DataFrame({
        'config': 'test',
        'n_images': len(test_images),
        'acc': acc
    }, index=[6])
    summary_results.to_csv(summary_result_path, mode='a', header=False, index=True, float_format='%.4f')
