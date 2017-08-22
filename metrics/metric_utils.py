import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from sklearn.metrics import fbeta_score
from sklearn import metrics as scipy_metrics
import torch
import torch.nn.functional as F
from torch.autograd import Variable
import warnings
import constants as c
import predictions


def get_default_loss(probs, targets, **kwargs):
    return get_cross_entropy_loss(probs, targets)


def get_default_score(preds, targets, avg='samples', **kwargs):
    return get_f2_score(preds, targets, avg)


def get_metric_in_blocks(outputs, targets, block_size, metric):
    sum_ = 0
    n = 0
    i = 0
    while i < len(outputs):
        out_block = outputs[i:i+block_size]
        tar_block = targets[i:i+block_size]
        score = metric(out_block, tar_block)
        sum_ += len(out_block) * score
        n += len(out_block)
        i += block_size
    return sum_ / n


def get_metrics_in_batches(model, loader, thresholds, metrics):
    model.eval()
    n_batches = len(loader)
    metric_totals = [0 for m in metrics]

    for data in loader:
        if len(data[1].size()) == 1:
            targets = data[1].float().view(-1, 1)
        inputs = Variable(data[0].cuda(async=True))
        targets = Variable(data[1].cuda(async=True))

        output = model(inputs)

        labels = targets.data.cpu().numpy()
        probs = output.data.cpu().numpy()
        preds = predictions.get_predictions(probs, thresholds)

        for i,m in enumerate(metrics):
            score = m(preds, labels)
            metric_totals[i] += score

    metric_totals = [m / n_batches for m in metric_totals]
    return metric_totals


def get_accuracy(preds, targets):
    preds = preds.flatten() 
    targets = targets.flatten()
    correct = np.sum(preds==targets)
    return correct / len(targets)


def get_cross_entropy_loss(probs, targets):
    return F.binary_cross_entropy(
              Variable(torch.from_numpy(probs)),
              Variable(torch.from_numpy(targets).float())).data[0]


def get_recall(preds, targets):
    return scipy_metrics.recall_score(targets.flatten(), preds.flatten())


def get_precision(preds, targets):
    return scipy_metrics.precision_score(targets.flatten(), preds.flatten())


def get_roc_score(probs, targets):
    return scipy_metrics.roc_auc_score(targets.flatten(), probs.flatten())


def get_dice_score(preds, targets):
    eps = 1e-7
    batch_size = preds.shape[0]
    preds = preds.reshape(batch_size, -1)
    targets = targets.reshape(batch_size, -1)

    total = preds.sum(1) + targets.sum(1) + eps
    intersection = (preds * targets).astype(float)
    score = 2. * intersection.sum(1) / total
    return np.mean(score)


def get_f2_score(y_pred, y_true, average='samples'):
    y_pred, y_true, = np.array(y_pred), np.array(y_true)
    return fbeta_score(y_true, y_pred, beta=2, average=average) 


def find_f2score_threshold(probs, targets, average='samples',
                           try_all=True, verbose=False, step=.01):
    best = 0
    best_score = -1
    totry = np.arange(0.1, 0.9, step)
    for t in totry:
        score = get_f2_score(probs, targets, t)
        if score > best_score:
            best_score = score
            best = t
    if verbose is True:
        print('Best score: ', round(best_score, 5),
              ' @ threshold =', round(best,4))
    return round(best,6)

