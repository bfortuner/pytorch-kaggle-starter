import os
import math
import shutil
from glob import glob

import config as cfg
import constants as c
import numpy as np
import utils.general as gen_utils
import utils.files
import models.utils
from clients import s3_client, es_client



def cleanup_experiments(exp_dir):
    exp_paths = glob(exp_dir+'/*/')
    for path in exp_paths:
        config_path = os.path.join(path, c.EXPERIMENT_CONFIG_FNAME)
        if not os.path.isfile(config_path):
            shutil.rmtree(path)


def delete_experiment(exp_name, exp_dir, local=True, s3=False, es=False):
    if local:
        pattern = os.path.join(exp_dir, exp_name)
        exp_path_list = glob(pattern)
        if len(exp_path_list) > 0:
            for p in exp_path_list:
                print("Deleting local exp")
                shutil.rmtree(p)
        else:
            print("Local copy of exp not found!")
    if s3:
        print("Deleting S3 document")
        s3_client.delete_experiment(exp_name)
    if es:
        print("ES delete not implemented")
        es_client.delete_experiment_by_field(field='exp_name', value=exp_name)


def prune(weights_dir, keep_epochs):
    prune_weights_and_optims(weights_dir, keep_epochs)


def auto_prune(exp, n_bins=5, metric_name=c.LOSS, func=min):
    best_epochs = get_best_epochs(exp, metric_name, n_bins, func)
    print(best_epochs)
    prune(exp.weights_dir, best_epochs)


def get_best_epochs(exp, metric_name, n_bins=5, func=max, end_epoch=10000):
    metric_arr = exp.history.metrics_history[metric_name][c.VAL][:end_epoch+1]
    idx, _ = get_best_values_in_bins(metric_arr, n_bins, func)
    return [i+1 for i in idx] #epoch starts at 1


def get_best_values_in_bins(arr, n_bins, func):
    bucket_size = math.ceil(len(arr)/n_bins)
    if isinstance(arr, list):
        arr = np.array(arr)
    best_idxfunc = np.argmax if func is max else np.argmin
    best_valfunc = np.amax if func is max else np.amin
    best_idx, best_vals = [], []
    for i in range(0, len(arr), bucket_size):
        best_idx.append(i+best_idxfunc(arr[i:i+bucket_size]))
        best_vals.append(best_valfunc(arr[i:i+bucket_size]))
    return best_idx, best_vals


def prune_weights_and_optims(weights_dir, keep_epochs):
    matches, fpaths = utils.files.get_matching_files_in_dir(
        weights_dir, c.WEIGHTS_OPTIM_FNAME_REGEX)
    print(matches)
    for i in range(len(matches)):
        epoch = int(matches[i].group(2))
        if epoch not in keep_epochs:
            os.remove(fpaths[i])


def get_weights_fpaths(weights_dir):
    return utils.files.get_matching_files_in_dir(
        weights_dir, c.WEIGHTS_FNAME_REGEX)[1]


def get_weight_epochs_from_fpaths(fpaths):
    epochs = []
    found_latest = False
    for path_ in fpaths:
        ## FIX THIS override
        if 'latest' not in path_:
            epochs.append(int(path_.strip(c.WEIGHTS_EXT).split('-')[-1]))
        else:
            found_latest = True
    epochs.sort()
    if found_latest:
        epochs.insert(0,'latest')
    return epochs


def get_weight_fpaths_by_epoch(weights_dir, epochs):
    matches, fpaths = utils.files.get_matching_files_in_dir(
        weights_dir, c.WEIGHTS_FNAME_REGEX)
    weight_fpaths = []
    for i in range(len(matches)):
        epoch = int(matches[i].group(1))
        if epoch in epochs:
            weight_fpaths.append(fpaths[i])
    return weight_fpaths


def get_optim_fpaths_by_epoch(optims_dir, keep_epochs):
    matches, fpaths = utils.files.get_matching_files_in_dir(
        optims_dir, c.OPTIM_FNAME_REGEX)
    weight_fpaths = []
    for i in range(len(matches)):
        epoch = int(matches[i].group(1))
        if epoch in keep_epochs:
            weight_fpaths.append(fpaths[i])
    return weight_fpaths


def get_weights_fname(epoch):
    if epoch is None:
        return c.LATEST_WEIGHTS_FNAME
    return 'weights-%d%s' % (epoch, c.WEIGHTS_EXT)


def get_optim_fname(epoch):
    if epoch is None:
        return c.LATEST_OPTIM_FNAME
    return 'optim-%d%s' % (epoch, c.OPTIM_EXT)


def load_weights_by_exp_and_epoch(model, exp_name, epoch='latest'):
    if epoch is None or epoch == 'latest':
        weights_fname = c.LATEST_WEIGHTS_FNAME
    else:
        weights_fname = 'weights-{:d}.th'.format(epoch)
    fpath = os.path.join(cfg.PATHS['experiments'], exp_name, 'weights', weights_fname)
    models.utils.load_weights(model, fpath)


def download_experiment(dest_dir, exp_name):
    fpath = os.path.join(dest_dir, exp_name + c.EXP_FILE_EXT)
    s3_client.download_experiment(fpath, exp_name)
    utils.files.unzipdir(fpath, dest_dir)


def upload_experiment(parent_dir, exp_name):
    print(('Uploading experiment {:s}. '
           'This may take a while..').format(exp_name))
    exp_path = os.path.join(parent_dir, exp_name)
    exp_copy_path = exp_path+'_copy'
    exp_copy_archive_path = os.path.join(exp_copy_path, exp_name)
    archive_path = exp_path + c.EXP_FILE_EXT
    shutil.copytree(exp_path, exp_copy_archive_path)
    print('Archiving..')
    utils.files.zipdir(exp_copy_path, archive_path)
    shutil.rmtree(exp_copy_path)
    print('Uploading..')
    s3_client.upload_experiment(archive_path, exp_name)
    os.remove(archive_path)
    print('Upload complete!')


def generate_display_name(base_name, *args):
    unique_id = gen_utils.gen_unique_id()
    return base_name+'-'.join(args[0])+'-id'+unique_id


def get_id_from_name(exp_name):
    return exp_name.split('-id')[-1]


def get_transforms_config(transforms):
    data_aug = []
    for r in transforms.transforms:
        data_aug.append((str(r.__class__.__name__),
                         r.__dict__))
    return data_aug