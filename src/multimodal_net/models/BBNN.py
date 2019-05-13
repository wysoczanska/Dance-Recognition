import torch.nn as nn
import torch
import torch.nn.functional as F
import torch.utils.model_zoo as model_zoo
import math


__all__ = ['BBNN']


class BBNN(nn.Module):

    def __init__(self, num_channels, num_classes=16):
        super(BBNN, self).__init__()
        self.sl = nn.Sequential(nn.Conv2d(3, 32, kernel_size=3), nn.BatchNorm2d(32, eps=0.001), nn.MaxPool2d(kernel_size=(1, 4)))
        self.inception_a = InceptionA(32, 32)
        self.inception_b = InceptionA(160, 32)
        self.inception_c = InceptionA(288, 32)
        self.tl = nn.Sequential(BasicConv2d(416, 32, kernel_size=1), nn.AvgPool2d(2, 2))
        self.dl = nn.AdaptiveAvgPool2d((1,1))
        self.fc = nn.Linear(32, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                import scipy.stats as stats
                stddev = m.stddev if hasattr(m, 'stddev') else 0.1
                X = stats.truncnorm(-2, 2, scale=stddev)
                values = torch.as_tensor(X.rvs(m.weight.numel()), dtype=m.weight.dtype)
                values = values.view(m.weight.size())
                with torch.no_grad():
                    m.weight.copy_(values)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x_sl = self.sl(x)
        x_a = self.inception_a(x_sl)
        x_b = self.inception_b(torch.cat((x_sl, x_a), 1))
        x_c = self.inception_c(torch.cat((x_sl, x_a, x_b), 1))
        x = self.tl(torch.cat((x_sl, x_a, x_b, x_c), 1))
        x = self.dl(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


class InceptionA(nn.Module):

    def __init__(self, in_channels, pool_features):
        super(InceptionA, self).__init__()
        self.branch1x1 = BasicConv2d(in_channels, 32, kernel_size=1)

        self.branch5x5_1 = BasicConv2d(in_channels, 32, kernel_size=1)
        self.branch5x5_2 = BasicConv2d(32, 32, kernel_size=5, padding=2)

        self.branch3x3dbl_1 = BasicConv2d(in_channels, 32, kernel_size=1)
        self.branch3x3dbl_2 = BasicConv2d(32, 32, kernel_size=3, padding=1)
        # self.branch3x3dbl_3 = BasicConv2d(96, 96, kernel_size=3, padding=1)

        self.branch_pool = BasicConv2d(in_channels, pool_features, kernel_size=1)

    def forward(self, x):
        branch1x1 = self.branch1x1(x)

        branch5x5 = self.branch5x5_1(x)
        branch5x5 = self.branch5x5_2(branch5x5)

        branch3x3dbl = self.branch3x3dbl_1(x)
        branch3x3dbl = self.branch3x3dbl_2(branch3x3dbl)
        # branch3x3dbl = self.branch3x3dbl_3(branch3x3dbl)

        branch_pool = F.max_pool2d(x, kernel_size=3, stride=1, padding=1)
        branch_pool = self.branch_pool(branch_pool)

        outputs = [branch1x1, branch5x5, branch3x3dbl, branch_pool]
        return torch.cat(outputs, 1)


class BasicConv2d(nn.Module):

    def __init__(self, in_channels, out_channels, **kwargs):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, bias=False, **kwargs)
        self.bn = nn.BatchNorm2d(in_channels, eps=0.001)

    def forward(self, x):
        x = self.bn(x)
        x = self.conv(x)
        return F.relu(x, inplace=True)
