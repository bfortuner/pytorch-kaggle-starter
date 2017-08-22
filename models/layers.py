import torch.nn as nn


def conv_relu(in_channels, out_channels, kernel_size=3, stride=1,
              padding=1, bias=True):
    return [
        nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, 
            stride=stride, padding=padding, bias=bias),
        nn.ReLU(inplace=True),
    ]

def conv_bn_relu(in_channels, out_channels, kernel_size=3, stride=1, 
                 padding=1, bias=False):
    return [
        nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size,
        stride=stride, padding=padding, bias=bias),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
    ]

def linear_bn_relu_drop(in_channels, out_channels, dropout=0.5, bias=False):
    layers = [
        nn.Linear(in_channels, out_channels, bias=bias),
        nn.BatchNorm1d(out_channels),
        nn.ReLU(inplace=True)
    ]
    if dropout > 0:
        layers.append(nn.Dropout(dropout))
    return layers


