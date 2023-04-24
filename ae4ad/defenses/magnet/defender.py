from typing import Dict
import matplotlib
import numpy as np
from detector import Classifier, Detector

matplotlib.use('agg')


class Defender:
    def __init__(
            self, x_val, y_val,
            classifier: Classifier,
            detectors: Dict[str, Detector],
            drop_rates: Dict[str, float]
    ):
        self.val_data = x_val
        self.val_label = y_val
        self.classifier = classifier
        self.detectors = detectors
        self.drop_rates = drop_rates
        self.thresholds = dict()
        self.set_threshold()

    def set_threshold(self):
        """
        Use detectors with specific drop rates to score validation set and set defender's threshold
        Set the threshold of detectors' score such that the false positive rate
        of the detector on the validation is at most drop rates

        :return: None
        """
        for name, detector in self.detectors.items():
            n_drops = int(len(self.val_data) * self.drop_rates[name])
            scores = np.sort(detector.score(self.val_data))
            self.thresholds[name] = scores[-n_drops - 1]

    def filter(self, test_data):
        """
        Use detector's threshold to filter test data
        :param test_data: input to test
        :return: index of passed data, index of not-passed data
        """
        passed_idx = np.array(range(len(test_data)))
        for name, detector in self.detectors.items():
            scores = detector.score(test_data)
            passed_idx = np.intersect1d(passed_idx, np.argwhere(scores < self.thresholds[name]))
        not_passed_idx = np.setdiff1d(np.array(range(len(test_data))), passed_idx)
        return passed_idx, not_passed_idx

    def test_detector(self, test_data, test_mark):
        """
        Test detector
        :param test_data:
        :param test_mark:
        :return: detector's accuracy, index of passed data, index of not-passed data
        """
        passed_idx, not_passed_idx = self.filter(test_data)
        n_data = len(test_data)

        detector_predict = np.full(n_data, 1)
        detector_predict[passed_idx] = 0
        true_positive = sum(1 for a, b in zip(detector_predict, test_mark) if a == 1 and b == 1)
        false_positive = sum(1 for a, b in zip(detector_predict, test_mark) if a == 1 and b == 0)
        true_negative = sum(1 for a, b in zip(detector_predict, test_mark) if a == 0 and b == 0)
        false_negative = sum(1 for a, b in zip(detector_predict, test_mark) if a == 0 and b == 1)

        return true_positive, false_positive, true_negative, false_negative, passed_idx, not_passed_idx
