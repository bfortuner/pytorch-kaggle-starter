import os
import time
import random
import numpy as np
import pandas as pd
import shutil

import config as cfg
import constants as c
import utils
import predictions



def get_ensemble_fpath(basename, dset):
    fname = '{:s}_{:s}_{:s}'.format(basename, 'ens', dset + c.PRED_FILE_EXT)
    return os.path.join(cfg.PATHS['predictions'], fname)


def get_ensemble_meta(name, fpaths):
    preds = [predictions.load_pred(f) for f in fpaths]
    meta = preds[0].attrs['meta'].copy()
    meta['name'] = name
    meta['members'] = {p.attrs['meta']['name']:p.attrs['meta'] for p in preds}
    print("members", list(meta['members'].keys()))
    return meta


def ens_prediction_files(ens_fpath, pred_fpaths, block_size=1, 
                         method=c.MEAN, meta=None):
    preds = [predictions.load_pred(f) for f in pred_fpaths]
    n_inputs = preds[0].shape[0]
    if os.path.exists(ens_fpath):
        print('Ens file exists. Overwriting')
        time.sleep(2)
        shutil.rmtree(ens_fpath)
    
    i = 0
    start = time.time()
    while i < n_inputs:
        pred_block = np.array([p[i:i+block_size] for p in preds])
        ens_block = predictions.ensemble_with_method(pred_block, method)
        if i == 0:
            ens_pred = predictions.save_pred(ens_fpath, ens_block, meta)
        else:
            ens_pred = predictions.append_to_pred(ens_pred, ens_block)
        i += block_size
    print(utils.logger.get_time_msg(start))
    return ens_fpath


def build_scores(loss, score):
    return {
        c.LOSS: loss,
        c.SCORE: score
    }


def build_metadata(labels, scores, thresholds, pred_type, dset):
    return {
        'label_names': labels,
        'scores': scores,
        'thresholds': thresholds,
        'pred_type': pred_type,
        'dset': dset,
        'created': time.strftime("%m/%d/%Y %H:%M:%S", time.localtime())
    }