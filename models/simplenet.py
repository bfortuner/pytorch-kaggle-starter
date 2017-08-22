import torch.nn as nn
import models.layers as layers


class SimpleNet(nn.Module):
    def __init__(self, in_feat, n_classes):
        super().__init__()
        self.features = nn.Sequential(
            *layers.conv_bn_relu(in_feat, 8, kernel_size=1, stride=1, padding=0, bias=False),
            *layers.conv_bn_relu(8, 32, kernel_size=3, stride=1, padding=1, bias=False),
            nn.MaxPool2d(kernel_size=2, stride=2), #size/2
            *layers.conv_bn_relu(32, 32, kernel_size=3, stride=1, padding=1, bias=False),
            nn.MaxPool2d(kernel_size=2, stride=2), #size/2     
            *layers.conv_bn_relu(32, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.MaxPool2d(kernel_size=2, stride=2), #size/2     
        )
        self.classifier = nn.Sequential(
            *layers.linear_bn_relu_drop(64, 512, dropout=0.0, bias=False),
            nn.Linear(512, n_classes, bias=False),
            nn.Sigmoid()   
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x