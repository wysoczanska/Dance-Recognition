import torch.nn as nn
import torch
import torch.utils.model_zoo as model_zoo
import math


__all__ = ['my_simple_convnet']


class SimpleConvnet(nn.Module):

    def __init__(self, num_classes=16):
        super(SimpleConvnet, self).__init__()
        self.features_11 = make_layers(cfg['F'], 11, batch_norm=True)
        self.features_7 = make_layers(cfg['F'], 7, batch_norm=True)
        self.features_4 = make_layers(cfg['F'], 4, batch_norm=True)
        self.features_1 = make_layers(cfg['F'], 1, batch_norm=True)

        self.classifier = nn.Sequential(
            nn.Linear(256, 50),
            nn.ReLU(True),
            nn.Dropout(),
            # nn.Linear(4096, 4096),
            # nn.ReLU(True),
            # nn.Dropout(),
        )
        # Change the dropout value to 0.9 and 0.9 for rgb model
        self.fc_action = nn.Linear(50, num_classes)
        # self._initialize_weights()

    def forward(self, x):
        x1 = self.features_11(x)
        x2 = self.features_7(x)
        x3 = self.features_4(x)
        x4 = self.features_1(x)
        x = torch.cat((x1, x2, x3, x4), 1)

        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        x = self.fc_action(x)
        return x


def make_layers(cfg, kernel, batch_norm=False):
    layers = []
    in_channels = 3
    for v in cfg:
        if v == 'M':
            # layers += [nn.MaxPool2d(kernel_size=(4, kernel))]
            layers += [nn.AdaptiveAvgPool2d((1))]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=(1, kernel))
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)


cfg = {
    'A': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'B': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'D': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'E': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
    'F': [64, 'M']
}


def my_simple_convnet(**kwargs):
    """
    """
    return SimpleConvnet(**kwargs)