import torch
import torch.nn as nn
import torchvision.models
import models.utils


def get_fc(in_feat, n_classes, activation=None):
    layers = [
        nn.Linear(in_features=in_feat, out_features=n_classes)
    ]
    if activation is not None:
        layers.append(activation)
    return nn.Sequential(*layers)


def get_classifier(in_feat, n_classes, activation, p=0.5):
    layers = [
        nn.BatchNorm1d(num_features=in_feat),
        nn.Dropout(p),
        nn.Linear(in_features=in_feat, out_features=n_classes),
        activation
    ]
    return nn.Sequential(*layers)


def get_mlp_classifier(in_feat, out_feat, n_classes, activation, p=0.01, p2=0.5):
    layers = [
        nn.BatchNorm1d(num_features=in_feat),
        nn.Dropout(p),
        nn.Linear(in_features=in_feat, out_features=out_feat),
        nn.ReLU(),
        nn.BatchNorm1d(num_features=out_feat),
        nn.Dropout(p2),
        nn.Linear(in_features=out_feat, out_features=n_classes),
        activation
    ]
    return nn.Sequential(*layers)


def cut_model(model, cut):
    return nn.Sequential(*list(model.children())[:cut])