import os
import re
from glob import glob
from config import PATHS
from experiments import experiment
from experiments import exp_utils
import training.utils as train_utils
import constants as c


LATEST = 'latest'

def load_single_weights(exp_name, epoch):
    return epoch


def load_multiple_weights(exp_name, epoch):
    return list(epoch)


def get_weights_path(exp_name):
    return os.path.join(PATHS['experiments']['root'], exp_name, 'weights')


def load_experiment(name):
    exp = experiment.Experiment(name, PATHS['experiments']['root'])
    exp.review()
    exp.history.load_history_from_file(c.VAL)
    return exp


def get_f2_scores_by_epoch(exp_name, sort_by_score):
    exp = load_experiment(exp_name)
    weight_fpaths = exp_utils.get_weights_fpaths(exp.weights_dir)
    epochs = exp_utils.get_weight_epochs_from_fpaths(weight_fpaths)
    epochs.insert(0,'latest')
    f2_scores = exp.history.metrics_history[c.F2_SCORE][c.VAL]
    score_by_epoch = {}
    for epoch in epochs[1:]:
        score_by_epoch[epoch] = float('{:4g}'.format(f2_scores[epoch-1]))
    score_by_epoch[LATEST] = float('{:4g}'.format(f2_scores[-1]))
    if sort_by_score:
        sorted_epochs_by_score = {}
        sorted_epochs = sorted(score_by_epoch, key=score_by_epoch.get,
                                reverse=sort_by_score)
        for epoch in sorted_epochs:
            if epoch == LATEST:
                sorted_epochs_by_score[epoch] = '{:4g}'.format(f2_scores[-1])
            else:
                score = '{:4g}'.format(f2_scores[epoch-1])
                sorted_epochs_by_score[epoch] = score
        return append_score_wpaths(exp_name, sorted_epochs_by_score)
    return append_score_wpaths(exp_name, score_by_epoch)

def append_score_wpaths(exp_name, epoch_dict):
    new_dict = {}
    for epoch in epoch_dict.keys():
        new_key = '{:} ({:4g})'.format(
            epoch, float(epoch_dict[epoch]))
        wpath = get_weights_fpath(epoch, exp_name)
        new_dict[new_key] = wpath
    return new_dict


def get_weights_fpath(epoch, exp_name):
    weights_path = get_weights_path(exp_name)
    if epoch == LATEST:
        return os.path.join(weights_path, c.LATEST_WEIGHTS_FNAME)
    return weights_path+'/weights-'+str(epoch)+'.pth'


def get_weights_epoch_path_dict(exp_name):
    wtdict = {}
    epochs = get_weights_epochs(exp_name)
    for epoch in epochs:
        wtdict[epoch] = get_weights_fpath(epoch, exp_name)
    return wtdict
