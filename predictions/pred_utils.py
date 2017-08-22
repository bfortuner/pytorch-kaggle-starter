import os
import json
import time
import scipy
import numpy as np
import pandas as pd
import bcolz
import random
from io import StringIO
from torch.autograd import Variable
import cv2
import h5py

import config as cfg
import constants as c
from .pred_constants import *
from . import pred_builder
import utils.general
import utils.files
import models.utils
import clients.s3_client as s3
from datasets import data_loaders
from datasets import metadata
from metrics import metric_utils
from experiments import exp_utils
from datasets.datasets import FileDataset



def predict_batch(net, inputs):
    v = Variable(inputs.cuda(), volatile=True)
    return net(v).data.cpu().numpy()


def get_probabilities(model, loader):
    model.eval()
    return np.vstack(predict_batch(model, data[0]) for data in loader)


def get_predictions(probs, thresholds):
    preds = np.copy(probs)
    preds[preds >= thresholds] = 1
    preds[preds < thresholds] = 0
    return preds.astype('uint8')


def get_mask_predictions(model, loader, thresholds, W=None, H=None):
    probs = get_probabilities(model, loader)
    preds = get_predictions(probs, thresholds)

    if W is not None and H is not None:
        preds = resize_batch(preds, W, H)
    return preds


def get_mask_probabilities(model, loader, W=None, H=None):
    model.eval()
    probs = get_probabilities(model, loader)  
    if W is not None and H is not None:
        probs = resize_batch(probs, W, H)
    return probs


def resize_batch(pred_batch, W=None, H=None):
    preds = []
    for i in range(len(pred_batch)):
        arr = resize_arr(pred_batch[i], W, H)
        preds.append(arr)
    return np.stack(preds)


def resize_arr(arr, W, H, mode=cv2.INTER_LINEAR):
    """
    We assume shape is (C, H, W) like tensor
    # arr = scipy.misc.imresize(arr.squeeze(), shape, interp='bilinear', mode=None)
    To shrink: 
        - INTER_AREA
    To enlarge:
        - INTER_CUBIC (slow, best quality)
        - INTER_LINEAR (faster, good quality).
    """
    arr = arr.transpose(1,2,0)
    arr = cv2.resize(arr, (W, H), mode)
    if len(arr.shape) < 3:
        arr = np.expand_dims(arr, 2)
    arr = arr.transpose(2,0,1)
    return arr


def get_targets(loader):
    targets = None
    for data in loader:
        if targets is None:
            shape = list(data[1].size())
            shape[0] = 0
            targets = np.empty(shape)
        target = data[1]
        if len(target.size()) == 1:
            target = target.view(-1,1)
        target = target.numpy()
        targets = np.vstack([targets, target])
    return targets


def save_pred(fpath, pred_arr, meta_dict=None):
    bc = bcolz.carray(pred_arr, mode='w', rootdir=fpath, 
            cparams=bcolz.cparams(clevel=9, cname='lz4'))
    if meta_dict is not None:
        bc.attrs['meta'] = meta_dict
    bc.flush()
    return bc


def append_to_pred(bc_arr, pred_arr, meta_dict=None):
    bc_arr.append(pred_arr)
    if meta_dict is not None:
        bc_arr.attrs['meta'] = meta_dict
    bc_arr.flush()
    return bc_arr


def append_pred_to_file(fpath, pred_arr, meta_dict=None):
    bc_arr = bcolz.open(rootdir=fpath)
    bc_arr.append(pred_arr)
    if meta_dict is not None:
        bc_arr.attrs['meta'] = meta_dict
    bc_arr.flush()
    return bc_arr


def save_or_append_pred_to_file(fpath, pred_arr, meta_dict=None):
    if os.path.exists(fpath):
        return append_pred_to_file(fpath, pred_arr, meta_dict)
    else:
        return save_pred(fpath, pred_arr, meta_dict)


def load_pred(fpath, numpy=False):
    bc = bcolz.open(rootdir=fpath)
    if numpy:
        return np.array(bc)
    return bc


def get_local_pred_fpath(name):
    return os.path.join(cfg.PATHS['predictions'], name+c.PRED_FILE_EXT)


def list_local_preds(dset=c.TEST, fnames_only=False):
    pattern = '_' + dset + c.PRED_FILE_EXT
    _, fpaths = utils.files.get_matching_files_in_dir(
        cfg.PATHS['predictions'], pattern)
    if fnames_only:
        return [utils.files.get_fname_from_fpath(f) for f in fpaths]
    return fpaths


def ensemble_with_method(arr, method):
    if method == c.MEAN:
        return np.mean(arr, axis=0)
    elif method == c.GMEAN:
        return scipy.stats.mstats.gmean(arr, axis=0)
    elif method == c.VOTE:
        return scipy.stats.mode(arr, axis=0)[0][0]
    raise Exception("Operation not found")


def get_prediction_fpath(basename, dset):
    fname = '{:s}_{:s}'.format(basename, dset + c.PRED_FILE_EXT)
    return os.path.join(cfg.PATHS['predictions'], fname)



# refactor notebook helpers

def build_pred_df_from_dir(dir_path):
    fpaths, _ = utils.files.get_paths_to_files(dir_path)
    summary = []
    for f in fpaths:
        if c.PRED_FILE_EXT in f:
            pred = load_pred(f)
            summary_dict = build_pred_summary_dict(pred)
            summary.append(summary_dict)
    return pd.DataFrame(summary)


def get_pred_summary_from_dicts(dicts):
    summary = []
    for d in dicts:
        summary.append(build_pred_summary_dict(d))
    return pd.DataFrame(summary)


def build_pred_summary_dict(pred):
    meta = pred['meta']
    return {
        'id': pred.get_id(),
        'name': pred.name,
        'pred_type': pred.pred_type,
        'dset': meta['dset'],
        c.LOSS : meta['scores'][c.LOSS],
        c.SCORE : meta['scores'][c.SCORE],
        'threshold' : meta['thresholds'],
        'created': meta['created'],
        'fpath': get_local_pred_fpath(pred.name)
    }


def get_clean_tta_str(tta):
    STRIP = [
    'torchvision.transforms.',
    'torchsample.tensor_transforms.',
    'torchsample.affine_transforms.',
    'torchsample.transforms.tensor_transforms.',
    'torchsample.transforms.affine_transforms.',
    'object at ',
    '<', '>',
    ]
    str_ = str(tta.transforms)
    for s in STRIP:
        str_ = str_.replace(s,'')
    return str_