import time

import utils.files
import constants as c
from .pred_constants import *
from .prediction import Prediction


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


def build_pred(name, preds, probs, val_preds, val_probs, labels, loss,
               score, thresholds, w_fpath, exp_name, tta, dset):
    name = PRED_TYPE + '-' + name
    scores = build_scores(loss, score)
    metadata = build_metadata(labels, scores, thresholds, PRED_TYPE, dset)
    metadata['w_fpath'] = w_fpath
    metadata['exp_name'] = exp_name
    metadata['tta'] = get_tta_doc(tta)
    return Prediction(name, metadata, preds=preds, probs=probs,
                     val_preds=val_preds, val_probs=val_probs)


def get_tta_doc(transforms):
    data_aug = []
    for r in transforms.transforms:
        data_aug.append((str(r.__class__.__name__),
                         r.__dict__))
    return str(data_aug)
