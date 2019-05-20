import torch
import torch.nn as nn
from torchvision import models
from .inception import inception_v3

__all__ = ['three_stream_net']


class ThreeStreamNet(nn.Module):
    def __init__(self, num_classes, input_length=15):
        super(ThreeStreamNet, self).__init__()
        self.rgb = inception_v3(pretrained=True, input_length=input_length)

        self.flow = inception_v3(pretrained=True, input_length=input_length)
        self.skeleton = inception_v3(pretrained=True, input_length=input_length)
        self.classifier = nn.Sequential(
            nn.Linear(2048*3, 2048),
            nn.ReLU(inplace=True),
            nn.Linear(2048, num_classes)
        )

    def forward(self, x):
        x1 = self.rgb(x['rgb'])
        x2 = self.flow(x['flow'])
        x3 = self.skeleton(x['skeleton'])
        x = torch.stack((x1, x2, x3), dim=2)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


def my_alexnet(pretrained=True):

    alexnet = models.alexnet(pretrained=pretrained)
    # freeze features layers
    for p in alexnet.features.parameters():
        p.requires_grad = False

    # remove final layer
    modules = list(alexnet.classifier.children())[:-4]
    alexnet.classifier = nn.Sequential(*modules)

    return alexnet


def three_stream_net(**kwargs):

    model = ThreeStreamNet(**kwargs)

    return model

