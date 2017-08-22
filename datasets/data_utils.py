import os
import shutil
import numpy as np 
import utils
from glob import glob
from PIL import Image
from skimage import io
import torch

import config as cfg
import constants as c
from datasets import metadata



def pil_loader(path):
    return Image.open(path).convert('RGB')


def tensor_loader(path):
    return torch.load(path)


def numpy_loader(path):
    return np.load(path)


def io_loader(path):
    return io.imread(path)


def tif_loader(path):
    return calibrate_image(io.imread(path)[:,:,(2,1,0,3)])


def calibrate_image(rgb_image, ref_stds, ref_means):
    res = rgb_image.astype('float32')
    return np.clip((res - np.mean(res,axis=(0,1))) / np.std(res,axis=(0,1))
           * ref_stds + ref_means,0,255).astype('uint8')


def get_inputs_targets(fpaths, dframe):
    ## REFACTOR
    inputs = []
    targets = []
    for fpath in fpaths:
        # Refactor
        name, tags = metadata.get_img_name_and_tags(METADATA_DF, fpath)
        inputs.append(img_utils.load_img_as_arr(fpath))
        targets.append(meta.get_one_hots_by_name(name, dframe))
    return np.array(inputs), np.array(targets)