import os
import random
import numpy as np

import utils.files
import constants as c


def make_bag(fpaths, targets):
    bag_fpaths = []
    bag_targets = []
    for i in range(len(fpaths)):
        idx = random.randint(1,len(fpaths)-1)
        bag_fpaths.append(fpaths[idx])
        bag_targets.append(targets[idx])
    return bag_fpaths, np.array(bag_targets)


def verify_bag(trn_fpaths_bag):
    trn_dict = {}
    for f in trn_fpaths_bag:
        if f in trn_dict:
            trn_dict[f] += 1
        else:
            trn_dict[f] = 1
    return max(trn_dict.values())


def make_fold(name, trn_path, tst_path, folds_dir, 
                val_size, shuffle=True):
    _, trn_fnames = utils.files.get_paths_to_files(
        trn_path, file_ext=c.JPG, sort=True, strip_ext=True)
    _, tst_fnames = utils.files.get_paths_to_files(
        tst_path, file_ext=c.JPG, sort=True, strip_ext=True)

    if shuffle:
        random.shuffle(trn_fnames)

    fold = {
        c.TRAIN: trn_fnames[:-val_size],
        c.VAL: trn_fnames[-val_size:],
        c.TEST: tst_fnames
    }

    fpath = os.path.join(folds_dir, name + c.DSET_FOLD_FILE_EXT)
    utils.files.save_json(fpath, fold)
    return fold


def load_data_fold(folds_dir, name):
    fpath = os.path.join(folds_dir, name + c.DSET_FOLD_FILE_EXT)
    return utils.files.load_json(fpath)


def get_fpaths_from_fold(fold, dset, dset_path, postfix=''):
    fnames = fold[dset]
    fpaths = [os.path.join(dset_path, f+postfix) for f in fnames]
    return fpaths


def get_targets_from_fold(fold, dset, lookup):
    img_names = [f.split('.')[0] for f in fold[dset]]
    targets = []
    for img in img_names:
        targets.append(lookup[img])
    return np.array(targets)


def get_fpaths_targets_from_fold(fold, dset, dset_path, lookup):
    fpaths = get_fpaths_from_fold(fold, dset, dset_path)
    targets = get_fpaths_from_fold(fold, dset, lookup)
    return fpaths, targets
