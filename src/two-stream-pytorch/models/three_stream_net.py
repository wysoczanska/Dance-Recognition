import torch
import torch.nn as nn
from torchvision import models

__all__ = ['three_stream_net']


class ThreeStreamNet(nn.Module):
    def __init__(self, num_classes):
        super(ThreeStreamNet, self).__init__()
        self.rgb = my_alexnet(pretrained=True)
        self.optical_flow = my_alexnet(pretrained=True)
        self.skeleton = my_alexnet(pretrained=True)
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(4096*3, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes)
        )

    def forward(self, x):
        x1 = self.rgb(x['rgb'])
        x2 = self.optical_flow(x['flow'])
        x3 = self.skeleton(x['skeleton'])
        x = torch.cat((x1, x2, x3), 1)
        x = self.classifier(x)
        return x


def my_alexnet(pretrained=True):

    alexnet = models.alexnet(pretrained=pretrained)
    modules = list(alexnet.classifier.children())[:-4]
    alexnet.classifier = nn.Sequential(*modules)
    for p in alexnet.features.parameters():
        p.requires_grad = False
    return alexnet





def three_stream_net(**kwargs):
    r"""AlexNet model architecture from the
    `"One weird trick..." <https://arxiv.org/abs/1404.5997>`_ paper.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ThreeStreamNet(**kwargs)

    return model