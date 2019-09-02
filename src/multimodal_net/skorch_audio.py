import argparse
import time
import models
import datasets
from tensorboardX import SummaryWriter
import os
import shutil
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.parallel
import torch.optim
import torch.utils.data
import audio_transforms
from torch.optim.lr_scheduler import ReduceLROnPlateau
import visualization_utils as vis
from torchvision import transforms, utils
import eval
import numpy as np
from sklearn.model_selection import GridSearchCV
from torch.utils.data.dataloader import default_collate
import skorch
from skorch.callbacks import Checkpoint
from skorch.net import NeuralNet
from skorch.helper import predefined_split
from sklearn.metrics import accuracy_score, make_scorer

from skorch.callbacks import LRScheduler
# from skorch.callbacks.lr_scheduler import CyclicLR

model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))

dataset_names = sorted(name for name in datasets.__all__)

parser = argparse.ArgumentParser(description='PyTorch Lets dance audio data training')
parser.add_argument('data', metavar='DIR',
                    help='path to dataset')
parser.add_argument('--logdir',
                    help='tensorboard log directory')
parser.add_argument('--train_split_file', metavar='DIR',
                    help='path to train files list')
parser.add_argument('--test_split_file', metavar='DIR',
                    help='path to test files list')
parser.add_argument('--dataset', '-d', default='ucf101',
                    help='dataset: ucf101 | hmdb51 | letsdance')
parser.add_argument('--model_path')
parser.add_argument('--arch', '-a', metavar='ARCH', default='vgg16',
                    help='model architecture (default: alexnet)')
parser.add_argument('-s', '--split', default=1, type=int, metavar='S',
                    help='which split of data to work on (default: 1)')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=250, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=25, type=int,
                    metavar='N', help='mini-batch size (default: 50)')
parser.add_argument('--iter-size', default=5, type=int,
                    metavar='I', help='iter size as in Caffe to reduce memory usage (default: 5)')
parser.add_argument('--new_length', default=1, type=int,
                    metavar='N', help='length of sampled video frames (default: 1)')
parser.add_argument('--new_width', default=224, type=int,
                    metavar='N', help='resize width (default: 340)')
parser.add_argument('--new_height', default=224, type=int,
                    metavar='N', help='resize height (default: 256)')
parser.add_argument('--lr', '--learning-rate', default=0.001, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--lr_steps', default=[50, 100, 150, 200], type=float, nargs="+",
                    metavar='LRSteps', help='epochs to decay learning rate by 10')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight-decay', '--wd', default=5e-4, type=float,
                    metavar='W', help='weight decay (default: 5e-4)')
parser.add_argument('--print-freq', default=50, type=int,
                    metavar='N', help='print frequency (default: 50)')
parser.add_argument('--save-freq', default=25, type=int,
                    metavar='N', help='save frequency (default: 25)')
parser.add_argument('--resume', default='./checkpoints', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')

best_prec1 = 0
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class SliceDatasetX(skorch.dataset.Dataset):
    """Helper class that wraps a torch dataset to make it work with sklearn"""

    def __init__(self, dataset, collate_fn=default_collate):
        self.dataset = dataset
        self.collate_fn = collate_fn

        self._indices = list(range(len(self.dataset)))

    def __len__(self):
        return len(self.dataset)

    @property
    def shape(self):
        return len(self),

    def __getitem__(self, i):
        if isinstance(i, (int, np.integer)):
            Xb = self.transform(*self.dataset[i])[0]
            return Xb

        if isinstance(i, slice):
            i = self._indices[i]

        Xb = self.collate_fn([self.transform(*self.dataset[j])[0] for j in i])
        return Xb


def load_checkpoint(model, optimizer, model_path):
    print("=> loading checkpoint '{}'".format(model_path))
    checkpoint = torch.load(model_path)
    start_epoch = checkpoint['epoch']
    model.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    print("=> loaded checkpoint '{}' (epoch {})".format(model_path, checkpoint['epoch']))
    return model, optimizer, start_epoch, 0

def ds_accuracy(y_true, y_pred):
    proba = np.argmax(y_pred, axis=1)
    return accuracy_score(y_true, proba)

def build_model(num_classes, model_name):
    model = models.__dict__[model_name](3)
    # model.fc_action = nn.Linear(4096, num_classes)
    if args.arch.endswith('alexnet') or args.arch.startswith('vgg'):
        model.features = torch.nn.DataParallel(model.features)
    else:
        model = torch.nn.DataParallel(model)
    return model


def main():
    global args, best_prec1
    args = parser.parse_args()
    writer = SummaryWriter(args.logdir)

    # create model
    print("Building model ... ")

    model = build_model(num_classes=16, model_name=args.arch)
    model = model.to(device)
    print(model)
    print("Model %s is loaded. " % (args.arch))

    if not os.path.exists(args.resume):
        os.makedirs(args.resume)
    print("Saving everything to directory %s." % (args.resume))

    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss().cuda()

    # optimizer = torch.optim.SGD(model.parameters(), args.lr,
    #                             momentum=args.momentum,
    #                             weight_decay=args.weight_decay)

    # optimizer =torch.optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    # scheduler = ReduceLROnPlateau(optimizer, 'max', patience=3, factor=0.5, verbose=True)

    if not os.path.exists(args.resume):
        os.makedirs(args.resume)
    print("Saving everything to directory %s." % (args.resume))

    cudnn.benchmark = True

    train_transform = transforms.Compose([
            # video_transforms.Scale((256)),
            audio_transforms.Resize((args.new_height, args.new_width)),
            audio_transforms.ToTensor(),
            # normalize,
        ])

    val_transform = transforms.Compose([
            # video_transforms.Scale((256)),
            audio_transforms.Resize((args.new_height, args.new_width)),
            audio_transforms.ToTensor(),
            # normalize,
        ])

    train_dataset = datasets.__dict__[args.dataset](root=args.data,
                                                    source=args.train_split_file,
                                                    phase="train",
                                                    is_color=True,
                                                    new_length=args.new_length,
                                                    transform=train_transform)
    val_dataset = datasets.__dict__[args.dataset](root=args.data,
                                                  source=args.test_split_file,
                                                  phase="val",
                                                  is_color=True,
                                                  new_length=args.new_length,
                                                  transform=val_transform)

    print('{} samples found, {} train samples and {} test samples.'.format(len(val_dataset)+len(train_dataset),
                                                                           len(train_dataset),
                                                                           len(val_dataset)))
    #
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        num_workers=args.workers, pin_memory=True, shuffle=True)

    net = NeuralNet(
        model,
        batch_size=32,
        max_epochs=150,
        lr=0.005,
        criterion=nn.NLLLoss,
        optimizer=torch.optim.Adam,
        iterator_train__shuffle=True,
        iterator_train__num_workers=16,
        iterator_valid__shuffle=False,
        iterator_valid__num_workers=16,
        train_split=None,
        device='cuda',
        callbacks=[skorch.callbacks.EpochScoring(ds_accuracy, use_caching=False)]
    )
    # we have to extract the target data, otherwise sklearn will complain
    # y_from_ds = np.asarray([train_dataset[i][1] for i in range(len(train_dataset))]).reshape(-1)
    # print(y_from_ds.shape)
    print(len(train_dataset))
    inputs = []
    targets = []

    for i, (input, target) in enumerate(train_loader):
        inputs.append(input.cpu().numpy())
        targets.append(target.cpu().numpy())
    y_from_ds = np.concatenate(targets, axis=0)
    print(y_from_ds.shape)
        # np.asarray([train_loader[i][1] for i, (input, target) in enumerate(train_loader)])
    ds = np.concatenate(inputs, axis=0)
    from sklearn.model_selection import cross_val_score

    y_pred = cross_val_score(net, ds, y=y_from_ds, cv=3, scoring=make_scorer(ds_accuracy))
    print(y_pred)


if __name__ == '__main__':
    main()

