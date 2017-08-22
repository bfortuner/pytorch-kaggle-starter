import constants as c
from . import metric


SUPPORTED_METRICS = {
    c.ACCURACY: metric.Accuracy(),
    c.LOSS: metric.Loss(),
    c.F2_SCORE: metric.F2Score(),
    c.ENSEMBLE_F2: metric.EnsembleF2(None,None),
    c.DICE_SCORE: metric.DiceScore(),
}
SUPPORTED_AUX_METRICS = {}


def get_metric_by_name(name):
    return SUPPORTED_METRICS[name]


def get_metrics_from_config(config):
    primary_metrics = []
    for m in config.metrics:
        new_metric = get_metric_by_name(m)
        primary_metrics.append(new_metric)
    return primary_metrics


def get_aux_metrics_from_config(config):
    aux_metrics = []
    for m in config.aux_metrics:
        new_metric = metric.AuxiliaryMetric(m['name'], m['units'])
        aux_metrics.append(new_metric)
    return aux_metrics
