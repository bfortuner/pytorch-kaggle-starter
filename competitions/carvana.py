import os
import time
import shutil
import gzip

import utils.files
import config as cfg
import constants as c
from datasets import datasets
from datasets import data_loaders
import predictions
import training
import submissions



def get_submission_lines(pred_arr, fnames):
    lines = []
    for i in range(len(pred_arr)):
        rle = submissions.run_length_encode(pred_arr[i])
        lines.append(fnames[i]+','+rle)
    return lines


def make_submission(pred, block_size=10000, header=None, compress=True):
    meta = pred.attrs['meta']
    print("Preds", pred.shape, meta['name'], meta['dset'])
    input_fnames = meta['input_fnames']
    sub_fpath = os.path.join(cfg.PATHS['submissions'], meta['name']+c.SUBMISSION_FILE_EXT)
    lines = [] if header is None else [header]

    i = 0
    while i < len(pred):
        start = time.time()
        pred_block = pred[i:i+block_size].squeeze().astype('uint8')
        newlines = get_submission_lines(pred_block, input_fnames[i:i+block_size])
        lines.extend(newlines)
        i += block_size
        print(training.get_time_msg(start))

    sub_fpath = utils.files.write_lines(sub_fpath, lines, compress)
    return sub_fpath


def get_block_predict_dataloaders(dataset, block_size, batch_size):
    loaders = []
    i = 0
    while i < len(dataset):
        inp_fpaths = dataset.input_fpaths[i:i+block_size]
        tar_fpaths = (None if dataset.target_fpaths is None 
                      else dataset.target_fpaths[i:i+block_size])
        block_dset = datasets.ImageTargetDataset(inp_fpaths, tar_fpaths, 
                          'pil', 'pil', input_transform=dataset.input_transform, 
                            target_transform=dataset.target_transform, 
                            joint_transform=dataset.joint_transform)
        block_loader = data_loaders.get_data_loader(block_dset, batch_size,
                              shuffle=False, n_workers=2, pin_memory=False)
        loaders.append(block_loader)
        i += block_size
    return loaders


def predict_binary_mask_blocks(name, dset, model, dataset, block_size,
                                batch_size, threshold, W=None, H=None):
    pred_fpath = os.path.join(cfg.PATHS['predictions'], name + '_' 
                              + dset + c.PRED_FILE_EXT)
    if os.path.exists(pred_fpath):
        print('Pred file exists. Overwriting')
        time.sleep(2)
        shutil.rmtree(pred_fpath)
    
    loaders = get_block_predict_dataloaders(dataset, block_size, batch_size)
    input_fnames = utils.files.get_fnames_from_fpaths(dataset.input_fpaths)
    target_fnames = (None if dataset.target_fpaths is None else 
        utils.files.get_fnames_from_fpaths(dataset.target_fpaths))
    meta = {
            'name': name,
            'dset': dset,
            'input_fnames': input_fnames,
            'target_fnames': target_fnames
    }
    
    i = 0
    for loader in loaders:
        print('Predicting block_{:d}, inputs: {:d}'.format(i, len(loader.dataset)))
        start = time.time()
        pred_block = predictions.get_mask_predictions(
            model, loader, threshold, W, H).astype('uint8')
        if i == 0:
            preds = predictions.save_pred(pred_fpath, pred_block, meta)
        else:
            preds = predictions.append_to_pred(preds, pred_block)
        i += 1
        print(training.get_time_msg(start))
    return pred_fpath


def upsample_preds(preds, block_size, W, H):
    meta = preds[0].attrs['meta'].copy()
    n_inputs = preds[0].shape[0]
    up_fpath = os.path.join(cfg.PATHS['predictions'], 
        meta['name']+'_up'+c.PRED_FILE_EXT)
    print("inputs", n_inputs, "preds",len(preds), up_fpath)

    if os.path.exists(up_fpath):
        print('Ens file exists. Overwriting')
        time.sleep(2)
        shutil.rmtree(up_fpath)
    
    i = 0
    start = time.time()
    while i < n_inputs:
        up_block = predictions.resize_batch(
            preds[i:i+block_size], W, H).astype('uint8')
        if i == 0:
            up_pred = predictions.save_pred(up_fpath, up_block, meta)
        else:
            up_pred = predictions.append_to_pred(up_pred, up_block)
        i += block_size
    
    print(utils.logger.get_time_msg(start))
    return up_pred



def get_and_write_probabilities_to_bcolz():
    """If I need extra speed"""
    pass