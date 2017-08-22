import os
import numpy as np
import time
import predictions
import datasets.metadata as meta
import training
import constants as c
import config as cfg
import utils.files



def write_preds_to_file(fpath, ids, preds, header):
    ids = np.array(ids).T
    preds = np.array(preds).T
    submission = np.stack([ids, preds], axis=1)
    np.savetxt(fpath, submission, fmt='%s', delimiter=',',
               header=header, comments='')


def make_tags_submission(sub_fpath, ids, preds, label_names, header):
    tags = meta.get_tags_from_preds(preds, label_names)
    write_preds_to_file(sub_fpath, ids, tags, header)


def make_preds_submission(sub_fpath, ids, preds, header):
    preds = [' '.join(map(str, p.tolist())) for p in preds]
    write_preds_to_file(sub_fpath, ids, preds, header)


def get_sub_path_from_pred_path(pred_fpath):
    sub_fname = os.path.basename(pred_fpath).rstrip(
        c.PRED_FILE_EXT) + c.SUBMISSION_FILE_EXT
    sub_fpath = os.path.join(cfg.PATHS['submissions'], sub_fname)
    return sub_fpath


def run_length_encode(mask_image):
    pixels = mask_image.flatten()
    # We avoid issues with '1' at the start or end (at the corners of 
    # the original image) by setting those pixels to '0' explicitly.
    # We do not expect these to be non-zero for an accurate mask, 
    # so this should not harm the score.
    pixels[0] = 0
    pixels[-1] = 0
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 2
    runs[1::2] = runs[1::2] - runs[:-1:2]
    return rle_to_string(runs)


def rle_to_string(runs):
    return ' '.join(str(x) for x in runs)


def run_length_decode(rel, H, W, fill_value=1):
    mask = np.zeros((H*W),np.uint8)
    rel  = np.array([int(s) for s in rel.split(' ')]).reshape(-1,2)
    for r in rel:
        start = r[0]
        end   = start +r[1]
        mask[start:end]=fill_value
    mask = mask.reshape(H,W)
    return mask





def submit_to_kaggle(fpath, competition, username, password):
    pass
    


# Refactor classification stuff from amazon..

def make_multi_label_submission(preds, img_paths, label_names, out_path,
                    name, file_ext='.csv.gz'):
    pred_tags = convert_preds_to_tags(preds, label_names)
    fnames = utils.files.get_fnames_from_fpaths(img_paths)
    fnames = np.array(fnames)
    fnames = np.expand_dims(fnames, 1)
    submission_fpath = os.path.join(out_path, name+'-submission'+file_ext)
    write_preds_to_file(fnames, pred_tags, submission_fpath)


def convert_preds_to_tags(preds, tags_list):
    tag_list = []
    for pred in preds:
        tags = ' '.join(meta.convert_one_hot_to_tags(pred, tags_list))
        tag_list.append(tags)
    tag_arr = np.array(tag_list)
    return np.expand_dims(tag_arr,1)