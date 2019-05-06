import torch.nn as nn
import torch.utils.model_zoo as model_zoo
import math


__all__ = ['my_simple_convnet']


class BBNN(nn.Module):

    def __init__(self, features, num_classes=16):
        super(BBNN, self).__init__()
        self.features = features
        self.classifier = nn.Sequential(
            nn.Linear(128*9*9, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(),
        )
        # Change the dropout value to 0.9 and 0.9 for rgb model
        self.fc_action = nn.Linear(4096, num_classes)
        # self._initialize_weights()

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        x = self.fc_action(x)
        return x


def make_layers(cfg, batch_norm=False):
    layers = []
    in_channels = 3
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=(4, 11), stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=(1, 11))
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)


def my_simple_convnet(**kwargs):
    """
    """
    return SimpleConvnet(make_layers(cfg['F'], batch_norm=True), **kwargs)