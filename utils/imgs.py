import os
import random
import numpy as np
from skimage import io
from PIL import Image, ImageFilter
from  scipy import ndimage
import cv2
import scipy.misc
import matplotlib.image as mpimg
import matplotlib as mpl
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
import torchsample
from torch.utils.data import DataLoader, TensorDataset

import config as cfg
import constants as c
from datasets import metadata
from . import files


CLASS_COLORS = {
    'green': (0, 128, 0),
    'red': (128, 0, 0),
    'blue': (0, 0, 128),
    'black': (0, 0, 0),
    'white': (255, 255, 255),
    'grey':(128, 128, 128),
}


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in c.IMG_EXTS)


def load_rgb_pil(img_path):
    return Image.open(img_path).convert('RGB')


def load_tif_as_arr(img_path):
    return io.imread(img_path)


def load_img_as_arr(img_path):
    return plt.imread(img_path)


def load_img_as_tensor(img_path):
    img_arr = load_img_as_arr(img_path)
    return transforms.ToTensor()(img_arr)


def load_img_as_pil(img_path):
    return Image.open(img_path).convert('RGB')


def save_pil_img(pil_img, fpath):
    pil_img.save(fpath)


def save_arr(arr, fpath):
    scipy.misc.imsave(fpath, arr)


def norm_meanstd(arr, mean, std):
    return (arr - mean) / std


def denorm_meanstd(arr, mean, std):
    return (arr * std) + mean


def norm255_tensor(arr):
    """Given a color image/where max pixel value in each channel is 255
    returns normalized tensor or array with all values between 0 and 1"""
    return arr / 255.


def denorm255_tensor(arr):
    return arr * 255.


def plot_img_arr(arr, fs=(6,6), title=None):
    plt.figure(figsize=fs)
    plt.imshow(arr.astype('uint8'))
    plt.title(title)
    plt.show()


def plot_img_tensor(tns, fs=(6,6), title=None):
    tns = denorm255_tensor(tns)
    arr = tns.numpy().transpose((1,2,0))
    plot_img_arr(arr, fs, title)


def tensor_to_arr(tns):
    tns = denorm255_tensor(tns)
    return tns.numpy().transpose((1,2,0))


def plot_img_from_fpath(img_path, fs=(8,8), title=None):
    plt.figure(figsize=fs)
    plt.imshow(plt.imread(img_path))
    plt.title(title)
    plt.show()


def plot_meanstd_normed_tensor(tns, mean, std, fs=(6,6), title=None):
    """If normalized with mean/std"""
    tns = denorm255_tensor(tns)
    arr = tns.numpy().transpose((1, 2, 0))
    arr = denorm_meanstd(arr, mean, std)
    plt.figure(figsize=fs)
    plt.imshow(arr)
    if title:
        plt.title(title)
    plt.show()


def get_mean_std_of_dataset(dir_path, sample_size=5):
    fpaths, fnames = files.get_paths_to_files(dir_path)
    random.shuffle(fpaths)
    total_mean = np.array([0.,0.,0.])
    total_std = np.array([0.,0.,0.])
    for f in fpaths[:sample_size]:
        if 'tif' in f:
            img_arr = io.imread(f)
        else:
            img_arr = load_img_as_arr(f)
        mean = np.mean(img_arr, axis=(0,1))
        std = np.std(img_arr, axis=(0,1))
        total_mean += mean
        total_std += std
    avg_mean = total_mean / sample_size
    avg_std = total_std / sample_size
    print("mean: {}".format(avg_mean), "stdev: {}".format(avg_std))
    return avg_mean, avg_std


def plot_binary_mask(arr, threshold=0.5, title=None, color=(255,255,255)):
    arr = format_1D_binary_mask(arr.copy())
    print(arr.shape)
    for i in range(3):
        arr[:,:,i][arr[:,:,i] >= threshold] = color[i]
    arr[arr < threshold] = 0
    plot_img_arr(arr, title=title)


def format_1D_binary_mask(mask):
    if len(mask.shape) == 2:
        mask = np.expand_dims(mask, 0)
    mask = np.stack([mask,mask,mask],axis=1).squeeze().transpose(1,2,0)
    return mask.astype('float32')


def plot_binary_mask_overlay(mask, img_arr, fs=(18,18), title=None):
    mask = format_1D_binary_mask(mask.copy())
    fig = plt.figure(figsize=fs)
    a = fig.add_subplot(1,2,1)
    a.set_title(title)
    plt.imshow(img_arr.astype('uint8'))
    plt.imshow(mask, cmap='jet', alpha=0.5) # interpolation='none'
    plt.show()


def plot_binary_mask_overlay(mask, img_arr, fs=(18,18), title=None):
    mask = format_1D_binary_mask(mask.copy())
    fig = plt.figure(figsize=fs)
    a = fig.add_subplot(1,2,1)
    a.set_title(title)
    plt.imshow(img_arr.astype('uint8'))
    plt.imshow(mask, cmap='jet', alpha=0.5) # interpolation='none'
    plt.show()


def plot_samples_from_dir(dir_path, shuffle=False):
    fpaths, fnames = files.get_paths_to_files(dir_path)
    plt.figure(figsize=(16,12))
    start = random.randint(0,len(fpaths)-1) if shuffle else 0
    j = 1
    for idx in range(start, start+6):
        plt.subplot(2,3,j)
        plt.imshow(plt.imread(fpaths[idx]))
        plt.title(fnames[idx])
        plt.axis('off')
        j += 1


def plot_sample_preds(fpaths, preds, targs, label_names, shuffle=False):
    fnames = files.get_fnames_from_fpaths(fpaths)
    plt.figure(figsize=(16,12))
    start = random.randint(0,len(preds)-1) if shuffle else 0
    j = 1
    for idx in range(start, start+6):
        plt.subplot(2,3,j)
        pred_tags = 'P: ' + ','.join(metadata.convert_one_hot_to_tags(preds[idx], label_names))
        if targs is not None:
            targ_tags = 'T: ' + ','.join(metadata.convert_one_hot_to_tags(
                targs[idx], label_names))
        else:
            targ_tags = ''
        title = '\n'.join([fnames[idx], pred_tags, targ_tags])
        plt.imshow(plt.imread(fpaths[idx]))
        plt.title(title)
        j += 1


def plot_sample_preds_masks(fnames, inputs, preds, fs=(9,9), 
        n_samples=8, shuffle=False):
    start = random.randint(0,len(inputs)-1) if shuffle else 0
    for idx in range(start, start+n_samples):
        print(fnames[idx])
        img = tensor_to_arr(inputs[idx])
        plot_binary_mask_overlay(preds[idx], img, fs, fnames[idx])
