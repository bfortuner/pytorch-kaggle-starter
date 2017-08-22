import torch
import torch.nn as nn
import torchvision.models
import models.utils


class SimpleResnet(nn.Module):
    def __init__(self, resnet, classifier):
        super().__init__()
        self.__class__.__name__ = "SimpleResnet"
        self.resnet = resnet
        self.classifier = classifier

    def forward(self, x):
        x = self.resnet(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


class ConcatResnet(nn.Module):
    def __init__(self, resnet, classifier):
        super().__init__()
        self.__class__.__name__ = 'ConcatResnet'
        self.resnet = resnet
        self.ap = nn.AdaptiveAvgPool2d((1,1))
        self.mp = nn.AdaptiveMaxPool2d((1,1))
        self.classifier = classifier

    def forward(self, x):
        x = self.resnet(x)
        x = torch.cat([self.mp(x), self.ap(x)], 1)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

    
def get_resnet18(pretrained, n_freeze):
    resnet = torchvision.models.resnet18(pretrained)
    if n_freeze > 0:
        models.utils.freeze_layers(resnet, n_freeze)
    return resnet


def get_resnet34(pretrained, n_freeze):
    resnet = torchvision.models.resnet34(pretrained)
    if n_freeze > 0:
        models.utils.freeze_layers(resnet, n_freeze)
    return resnet


def get_resnet50(pretrained, n_freeze):
    resnet = torchvision.models.resnet50(pretrained)
    if n_freeze > 0:
        models.utils.freeze_layers(resnet, n_freeze)
    return resnet
