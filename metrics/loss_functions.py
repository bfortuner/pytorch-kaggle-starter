import torch
import torch.nn.functional as F
from . import metric_utils


class DiceLoss():
    '''
    http://campar.in.tum.de/pub/milletari2016Vnet/milletari2016Vnet.pdf
    https://github.com/faustomilletari/VNet/blob/master/pyLayer.py
    https://github.com/pytorch/pytorch/issues/1249
    '''
    def __init__(self):
        self.__class__.__name__ = 'Dice'

    def __call__(self, output, target):
        return 1.0 - get_torch_dice_score(output, target)


class DiceBCELoss():
    def __init__(self, dice_weight=1.0):
        self.__class__.__name__ = 'DiceBCE'
        self.dice_weight = dice_weight
        self.bce_weight = 1.0 - dice_weight

    def __call__(self, output, target):
        bce = F.binary_cross_entropy(output, target)
        dice = 1 - get_torch_dice_score(output, target)
        return (dice * self.dice_weight) + (bce * self.bce_weight)


class WeightedBCELoss():
    def __init__(self, weights):
        self.weights = weights
        self.__class__.__name__ = 'WeightedBCE'

    def __call__(self, output, target):
        return F.binary_cross_entropy(output, target, self.weights)


class KnowledgeDistillLoss():
    def __init__(self, target_weight=0.25):
        self.__class__.__name__ = 'KnowledgeDistill'
        self.target_weight = target_weight

    def __call__(self, output, target, soft_target):
        target_loss = F.binary_cross_entropy(output, target) * self.target_weight
        soft_target_loss = F.binary_cross_entropy(output, soft_target)
        return target_loss + soft_target_loss


class HuberLoss():
    def __init__(self, c=0.5):
        self.c = c
        self.__class__.__name__ = 'Huber'

    def __call__(self, output, target):
        bce = F.binary_cross_entropy(output, target)
        return self.c**2 * (torch.sqrt(1 + (bce/self.c)**2) - 1)


class SmoothF2Loss():
    def __init__(self, c=10.0, f2_weight=0.2, bce_weight=1.0):
        self.__class__.__name__ = 'SmoothF2'
        self.c = c
        self.f2_weight = f2_weight
        self.bce_weight = bce_weight

    def __call__(self, output, target, thresholds):
        f2 = get_smooth_f2_score(output, target, thresholds, self.c) * self.f2_weight
        bce = F.binary_cross_entropy(output, target) * self.bce_weight
        return f2 + bce



# Helpers / Shared Methods

def get_torch_dice_score(outputs, targets):
    eps = 1e-7
    batch_size = outputs.size()[0]
    outputs = outputs.view(batch_size, -1)
    targets = targets.view(batch_size, -1)

    total = torch.sum(outputs, dim=1) + torch.sum(targets, dim=1)
    intersection = torch.sum(outputs * targets, dim=1).float()

    dice_score = (2.0 * intersection) / (total + eps)
    return torch.mean(dice_score)


def sigmoid(z, c=1.0):
    return 1.0 / (1.0 + torch.exp(-c*z))


def get_smooth_f2_score(outputs, targets, thresholds, c=10.0):
    eps = 1e-9
    outputs = sigmoid(thresholds - outputs, c).float()
    tot_out_pos = torch.sum(outputs, dim=1)
    tot_tar_pos = torch.sum(targets, dim=1)
    TP = torch.sum(outputs * targets, dim=1)

    P = TP / (tot_out_pos + eps)
    R = TP / tot_tar_pos + eps
    F2 = 5.0 * (P*R / (4*P + R))
    return torch.mean(F2)