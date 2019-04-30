import torch.nn as nn
from torchvision import models


def build_model(input_channels, num_classes):
    model = models.alexnet(pretrained=True, num_classes=1000)
    model.features[0] = nn.Conv2d(input_channels, 64, kernel_size=11, stride=4, padding=2)
    modules = list(model.features.children())
    modules.insert(3, nn.BatchNorm2d(64))
    modules.insert(7, nn.BatchNorm2d(192))
    model.features = nn.Sequential(*modules)

    model.classifier[-1] = nn.Linear(4096, num_classes)

    return model