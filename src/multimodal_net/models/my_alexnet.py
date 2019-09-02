import torch.nn as nn
from torchvision import models
import collections
import torch.utils.model_zoo as model_zoo

__all__ = ['alexnet']

model_urls = {
    'alexnet': 'https://download.pytorch.org/models/alexnet-owt-4df8aa71.pth',
}


def alexnet(input_length, num_classes):

    model = models.alexnet(pretrained=True, num_classes=1000)
    model.features[0] = nn.Conv2d(3*input_length, 64, kernel_size=11, stride=4, padding=2)
    model.classifier[-1] = nn.Linear(4096, num_classes)

    pretrained_dict = model_zoo.load_url(model_urls['alexnet'])

    model_dict = model.state_dict()

    new_pretrained_dict = change_key_names(pretrained_dict, input_length)
    # 1. filter out unnecessary keys
    new_pretrained_dict = {k: v for k, v in new_pretrained_dict.items() if k in model_dict}
    # 2. overwrite entries in the existing state dict
    model_dict.update(new_pretrained_dict)
    # 3. load the new state dict
    model.load_state_dict(model_dict)

    for p in model.features.parameters():
        p.requires_grad = False

    # remove final layer
    modules = list(model.classifier.children())[:-4]
    model.classifier = nn.Sequential(*modules)

    return model


def change_key_names(old_params, input_length):
    new_params = collections.OrderedDict()
    layer_count = 0
    allKeyList = old_params.keys()
    for layer_key in allKeyList:
        if layer_count >= len(allKeyList) - 2:
            # exclude fc layers
            continue
        else:
            if layer_count == 0:
                rgb_weight = old_params[layer_key]
                # rgb_weight_mean = torch.mean(rgb_weight, dim=1)
                # TODO: ugly fix here, why torch.mean() turn tensor to Variable
                # print(type(rgb_weight_mean))
                flow_weight = rgb_weight.repeat(1, input_length, 1, 1)
                new_params[layer_key] = flow_weight
                layer_count += 1
                # print(layer_key, new_params[layer_key].size(), type(new_params[layer_key]))
            else:
                new_params[layer_key] = old_params[layer_key]
                layer_count += 1
                # print(layer_key, new_params[layer_key].size(), type(new_params[layer_key]))

    return new_params