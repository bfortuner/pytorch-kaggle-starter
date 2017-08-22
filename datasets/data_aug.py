import math
import random
from PIL import Image, ImageFilter
import cv2
import numpy as np

import torch
import torchsample
import torchvision
from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset


#http://docs.opencv.org/3.0-beta/doc/py_tutorials/py_imgproc/py_geometric_transformations/py_geometric_transformations.html

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

IMAGENET_NORMALIZE = torchvision.transforms.Normalize(
            mean=IMAGENET_MEAN,
            std=IMAGENET_STD
)

def get_data_aug_summary(transforms):
    data_aug = []
    for r in transforms.transforms:
        data_aug.append((str(r.__class__.__name__), r.__dict__))
    return data_aug


def get_basic_transform(scale, normalize=None):
    data_aug = [
        torchvision.transforms.Scale(scale),
        torchvision.transforms.ToTensor()
    ]
    if normalize is not None:
        data_aug.append(normalize)
    return torchvision.transforms.Compose(data_aug)


def get_single_pil_transform(scale, augmentation, normalize=None):
    data_aug = [
        torchvision.transforms.Scale(scale),
        augmentation,
        torchvision.transforms.ToTensor()
    ]
    if normalize is not None:
        data_aug.append(normalize)
    return torchvision.transforms.Compose(data_aug)


def get_single_tensor_transform(scale, augmentation, normalize=None):
    data_aug = [
        torchvision.transforms.Scale(scale),
        torchvision.transforms.ToTensor(),
        augmentation
    ]
    if normalize is not None:
        data_aug.append(normalize)
    return torchvision.transforms.Compose(data_aug)


class RandomRotate90(object):
    def __init__(self, p=0.75):
        self.p = p

    def __call__(self, *inputs):
        outputs = []
        for idx, input_ in enumerate(inputs):
            input_ = random_rotate_90(input_, self.p)
            outputs.append(input_)
        return outputs if idx > 1 else outputs[0]


class BinaryMask(object):
    def __init__(self, thresholds):
        self.thresholds = thresholds

    def __call__(self, *inputs):
        outputs = []
        for idx, input_ in enumerate(inputs):
            input_[input_ >= self.thresholds] = 1.0
            input_[input_ < self.thresholds] = 0.0
            outputs.append(input_)
        return outputs if idx > 1 else outputs[0]


class Slice1D(object):
    def __init__(self, dim=0, slice_idx=0):
        self.dim = dim
        self.slice_idx = slice_idx

    def __call__(self, *inputs):
        outputs = []
        for idx, input_ in enumerate(inputs):
            input_ = torch.unsqueeze(input_[self.slice_idx,:,:], dim=self.dim)
            outputs.append(input_)
        return outputs if idx > 1 else outputs[0]


class RandomHueSaturation(object):
    def __init__(self, hue_shift=(-180, 180), sat_shift=(-255, 255),
                    val_shift=(-255, 255), u=0.5):
        self.hue_shift = hue_shift
        self.sat_shift = sat_shift
        self.val_shift = val_shift
        self.u = u

    def __call__(self, *inputs):
        outputs = []
        for idx, input_ in enumerate(inputs):
            input_ = random_hue_saturation(input_, self.hue_shift,
                self.sat_shift, self.val_shift, self.u)
            outputs.append(input_)
        return outputs if idx > 1 else outputs[0]


class RandomShiftScaleRotate(object):
    def __init__(self, shift=(-0.0625,0.0625), scale=(-0.1,0.1),
                    rotate=(-45,45), aspect=(0,0), u=0.5):
        self.shift = shift
        self.scale = scale
        self.rotate = rotate
        self.aspect = aspect
        self.border_mode = cv2.BORDER_CONSTANT
        self.u = u

    def __call__(self, input_, target):
        input_, target = random_shift_scale_rot(input_, target, self.shift, 
        self.scale, self.rotate, self.aspect, self.border_mode, self.u)
        return [input_, target]


def random_rotate_90(pil_img, p=1.0):
    if random.random() < p:
        angle=random.randint(1,3)*90
        if angle == 90:
            pil_img = pil_img.rotate(90)
        elif angle == 180:
            pil_img = pil_img.rotate(180)
        elif angle == 270:
            pil_img = pil_img.rotate(270)
    return pil_img


def random_hue_saturation(image, hue_shift=(-180, 180), sat_shift=(-255, 255),
                            val_shift=(-255, 255), u=0.5):
    image = np.array(image)
    if np.random.random() < u:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(image)
        hue_shift = np.random.uniform(hue_shift[0], hue_shift[1])
        h = cv2.add(h, hue_shift)
        sat_shift = np.random.uniform(sat_shift[0], sat_shift[1])
        s = cv2.add(s, sat_shift)
        val_shift = np.random.uniform(val_shift[0], val_shift[1])
        v = cv2.add(v, val_shift)
        image = cv2.merge((h, s, v))
        image = cv2.cvtColor(image, cv2.COLOR_HSV2BGR)
    
    return Image.fromarray(image)


def random_shift_scale_rot(image, label, shift_limit=(-0.0625,0.0625), 
        scale_limit=(-0.1,0.1), rotate_limit=(-45,45), aspect_limit = (0,0),  
        borderMode=cv2.BORDER_CONSTANT, u=0.5):
    image = image.numpy().transpose(1,2,0)
    label = label.numpy().squeeze()
    if random.random() < u:
        height,width,channel = image.shape

        angle  = random.uniform(rotate_limit[0],rotate_limit[1])  #degree
        scale  = random.uniform(1+scale_limit[0],1+scale_limit[1])
        aspect = random.uniform(1+aspect_limit[0],1+aspect_limit[1])
        sx    = scale*aspect/(aspect**0.5)
        sy    = scale       /(aspect**0.5)
        dx    = round(random.uniform(shift_limit[0],shift_limit[1])*width )
        dy    = round(random.uniform(shift_limit[0],shift_limit[1])*height)

        cc = math.cos(angle/180*math.pi)*(sx)
        ss = math.sin(angle/180*math.pi)*(sy)
        rotate_matrix = np.array([ [cc,-ss], [ss,cc] ])

        box0 = np.array([ [0,0], [width,0],  [width,height], [0,height], ])
        box1 = box0 - np.array([width/2,height/2])
        box1 = np.dot(box1,rotate_matrix.T) + np.array([width/2+dx,height/2+dy])
        box0 = box0.astype(np.float32)
        box1 = box1.astype(np.float32)
        mat = cv2.getPerspectiveTransform(box0,box1)
        image = cv2.warpPerspective(image, mat, (width,height),
        flags=cv2.INTER_LINEAR,borderMode=borderMode,borderValue=(0,0,0,))  
        #cv2.BORDER_CONSTANT, borderValue = (0, 0, 0))  #cv2.BORDER_REFLECT_101

        box0 = np.array([ [0,0], [width,0],  [width,height], [0,height], ])
        box1 = box0 - np.array([width/2,height/2])
        box1 = np.dot(box1,rotate_matrix.T) + np.array([width/2+dx,height/2+dy])
        box0 = box0.astype(np.float32)
        box1 = box1.astype(np.float32)
        mat = cv2.getPerspectiveTransform(box0,box1)
        label = cv2.warpPerspective(label, mat, (width,height),
        flags=cv2.INTER_LINEAR,borderMode=borderMode,borderValue=(0,0,0,)) 
        #cv2.BORDER_CONSTANT, borderValue = (0, 0, 0))  #cv2.BORDER_REFLECT_101
    image = torch.from_numpy(image.transpose(2,0,1))
    label = np.expand_dims(label, 0)
    label = torch.from_numpy(label)#.transpose(2,0,1)) 
    return image,label


blurTransform = torchvision.transforms.Lambda(
    lambda img: img.filter(ImageFilter.GaussianBlur(1.5)))