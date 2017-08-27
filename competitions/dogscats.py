import os
import numpy as np
import pandas as pd

import config as cfg
import constants as c

import datasets.metadata as meta
import utils


LABEL_NAMES = ['cat', 'dog']
LABEL_TO_IDX = meta.get_labels_to_idxs(LABEL_NAMES)
IDX_TO_LABEL = meta.get_idxs_to_labels(LABEL_NAMES)
SUB_HEADER = 'id,label'


def make_metadata_file():
    '''
    First move the cats/dogs data in train.zip 
    to `catsdogs/datasets/inputs/trn_jpg` 
    '''
    train_path = cfg.PATHS['datasets']['inputs']['trn_jpg']
    _, fnames = utils.files.get_paths_to_files(
        train_path, strip_ext=True)
    lines = []
    for name in fnames:
        label = name.split('.')[0]
        lines.append('{:s},{:s}\n'.format(name, label))
    with open(cfg.METADATA_PATH, 'w') as f:
        f.writelines(lines)