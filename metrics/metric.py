import numpy as np
import operator
import constants as c
from . import metric_utils


class Metric():
    def __init__(self, name, minimize=True):
        self.name = name
        self.minimize = minimize

    def get_best_epoch(self, values):
        if self.minimize:
            idx, value = min(enumerate(values),
                key=operator.itemgetter(1))
        else:
            idx, value = max(enumerate(values),
                key=operator.itemgetter(1))
        epoch = idx + 1 # epochs start at 1
        return epoch, value

    def evaluate(self, loss, preds, probs, targets):
        pass

    def format(self, value):
        pass


class AuxiliaryMetric():
    def __init__(self, name, units):
        self.name = name
        self.units = units


class Accuracy(Metric):
    def __init__(self):
        super().__init__(c.ACCURACY, minimize=False)

    def evaluate(self, loss, preds, probs, targets):
        return metric_utils.get_accuracy(preds, targets)

    def format(self, value):
        return value


class Loss(Metric):
    def __init__(self):
        super().__init__(c.LOSS, minimize=True)

    def evaluate(self, loss, preds, probs, targets):
        return loss

    def format(self, value):
        return value


class F2Score(Metric):
    def __init__(self, target_threshold=None):
        super().__init__(c.F2_SCORE, minimize=False)
        self.target_threshold = target_threshold  # pseudo soft targets

    def evaluate(self, loss, preds, probs, targets):
        average = 'samples' if targets.shape[1] > 1 else 'binary'
        if self.target_threshold is not None:
            targets = targets > self.target_threshold

        return metric_utils.get_f2_score(preds, targets, average)

    def format(self, value):
        return value


class DiceScore(Metric):
    def __init__(self):
        super().__init__(c.DICE_SCORE, minimize=False)

    def evaluate(self, loss, preds, probs, targets):
        return metric_utils.get_dice_score(preds, targets)

    def format(self, value):
        return value


class EnsembleF2(Metric):
    def __init__(self, ens_probs, threshold):
        super().__init__('EnsembleF2', minimize=False)
        self.ens_probs = ens_probs
        self.threshold = threshold

    def evaluate(self, loss, preds, probs, targets):
        if probs.shape[0] != self.ens_probs.shape[1]:
            return .950
        average = 'samples' if targets.shape[1] > 1 else 'binary'
        probs = np.expand_dims(probs, 0)
        joined_probs = np.concatenate([self.ens_probs, probs])
        joined_probs = np.mean(joined_probs, axis=0)
        preds = joined_probs > self.threshold
        return metric_utils.get_f2_score(preds, targets, average)

    def format(self, value):
        return value