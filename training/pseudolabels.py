import numpy as np
import torch
import random
from predictions import pred_utils
import utils.files as file_utils
import datasets.data_aug as data_aug
import pickle
from datasets.datasets import FileDataset


def get_pseudo_label_targets(fpaths, model, img_scale, n_labels, thresholds):
    dataset = FileDataset(fpaths, targets=None,
                          transform=data_aug.get_basic_transform(img_scale))
    dataloader = torch.utils.data.DataLoader(dataset, 64, shuffle=False,
                            pin_memory=False, num_workers=1)
    probs = pred_utils.get_probabilities(model, dataloader)
    preds = pred_utils.get_predictions(probs, thresholds)
    return preds, probs


def get_pseudo_labeled_fpaths_targets(dir_path, model, n_samples,
                                      img_scale, n_labels, thresholds):
    fpaths, _ = file_utils.get_paths_to_files(dir_path)
    random.shuffle(fpaths)
    fpaths = fpaths[:n_samples]
    targets, _ = get_pseudo_label_targets(fpaths, model, img_scale,
                                          n_labels, thresholds)
    return fpaths, targets


def combined_train_and_pseudo_fpaths_targets(trn_fpaths, trn_targets,
                                             pseudo_fpaths, pseudo_targets):
    combined_fpaths = trn_fpaths + pseudo_fpaths
    combined_targets = np.vstack([trn_targets, pseudo_targets])
    return combined_fpaths, combined_targets


def save_pseudo_labels(pseudo_preds, img_paths, out_fpath):
    obj = {'preds':pseudo_preds, 'img_paths':img_paths}
    with open(out_fpath, 'wb') as f:
        pickle.dump(obj, f)


def load_pseudo_labels(fpath):
    obj = pickle.load(open(fpath, 'rb'))
    return obj['img_paths'], obj['preds']
