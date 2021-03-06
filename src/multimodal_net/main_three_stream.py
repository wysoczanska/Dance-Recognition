import argparse
import os
import shutil
import time

import datasets
import models
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import video_transforms
from tensorboardX import SummaryWriter
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torchvision import utils

model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))

dataset_names = sorted(name for name in datasets.__all__)

parser = argparse.ArgumentParser(description='PyTorch Three-Stream Action Recognition')
parser.add_argument('data', metavar='DIR',
                    help='path to dataset')
parser.add_argument('--train_split_file', metavar='DIR',
                    help='path to train files list')
parser.add_argument('--test_split_file', metavar='DIR',
                    help='path to test files list')
parser.add_argument('--dataset', '-d', default='Letsdance')
parser.add_argument('--logdir',
                    help='tensorboard log directory')
parser.add_argument('--out_dir', type=str, help='Directory with saved models', default='./checkpoints')
parser.add_argument('--model_path', type=str, help='Absolute path to the checkpoint. Only valid when resume or eval'
                                                   '', default='./checkpoints/model_best.pth.tar')
parser.add_argument('--resume', dest='resume', help='Best model evaluation mode', action='store_true')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=250, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=25, type=int,
                    metavar='N', help='mini-batch size (default: 50)')
parser.add_argument('--new_length', default=1, type=int,
                    metavar='N', help='length of sampled video frames (default: 1)')
parser.add_argument('--new_width', default=340, type=int,
                    metavar='N', help='resize width (default: 340)')
parser.add_argument('--new_height', default=256, type=int,
                    metavar='N', help='resize height (default: 256)')
parser.add_argument('--lr', '--learning-rate', default=0.001, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight-decay', '--wd', default=5e-4, type=float,
                    metavar='W', help='weight decay (default: 5e-4)')
parser.add_argument('--print-freq', default=50, type=int,
                    metavar='N', help='print frequency (default: 50)')
parser.add_argument('--save-freq', default=25, type=int,
                    metavar='N', help='save frequency (default: 25)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--num_classes', default=16,
                    help='Number of classes in dataset')


best_acc1 = 0
ngpus = torch.cuda.device_count()


def main():
    global args, best_acc1
    args = parser.parse_args()
    num_classes = args.num_classes
    start_epoch=0
    writer = SummaryWriter(args.logdir)

    model = build_model(num_classes=num_classes, input_length=args.new_length)

    print(model)

    # create model
    print("Building model ... ")

    model = torch.nn.DataParallel(model)
    model.cuda()

    if not os.path.exists(args.out_dir):
        os.makedirs(args.out_dir)
    print("Saving everything to directory %s." % (args.out_dir))

    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss().cuda()

    optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)
    scheduler = ReduceLROnPlateau(optimizer, verbose=True, patience=4)

    # if resume set to True, load the model and continue training
    if args.resume or args.evaluate:
        if os.path.isfile(args.model_path):
            model, optimizer, start_epoch = load_checkpoint(model, optimizer, args.model_path)

    cudnn.benchmark = True

    is_color = True
    # scale_ratios = [1.0, 0.875, 0.75, 0.66]
    clip_mean = {'rgb': [0.485, 0.456, 0.406] * args.new_length, 'flow': [0.9432, 0.9359, 0.9511] *args.new_length,
                 'skeleton': [0.0071, 0.0078, 0.0079]*args.new_length}
    clip_std = {'rgb': [0.229, 0.224, 0.225] * args.new_length, 'flow': [0.0788, 0.0753, 0.0683] * args.new_length,
                'skeleton': [0.0581, 0.0621, 0.0623] * args.new_length}

    normalize = video_transforms.Normalize(mean=clip_mean,
                                           std=clip_std)
    train_transform = video_transforms.Compose([
            video_transforms.Resize((args.new_width, args.new_height)),
            video_transforms.ToTensor(),
            normalize,
        ])

    val_transform = video_transforms.Compose([
            video_transforms.Resize((args.new_width, args.new_height)),
            video_transforms.ToTensor(),
            normalize,
        ])

    train_dataset = datasets.__dict__[args.dataset](root=args.data,
                                                    source=args.train_split_file,
                                                    phase="train",
                                                    is_color=is_color,
                                                    new_length=args.new_length,
                                                    video_transform=train_transform)
    val_dataset = datasets.__dict__[args.dataset](root=args.data,
                                                  source=args.test_split_file,
                                                  phase="val",
                                                  is_color=is_color,
                                                  new_length=args.new_length,
                                                  video_transform=val_transform,
                                                  return_id=True)

    print('{} samples found, {} train samples and {} test samples.'.format(len(val_dataset)+len(train_dataset),
                                                                           len(train_dataset),
                                                                           len(val_dataset)))

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=True)
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True)

    if args.evaluate:

        validate(val_loader, model, criterion, epoch=0, writer=writer, classes=val_dataset.classes)
        return

    for epoch in range(start_epoch, args.epochs):
        # train for one epoch
        train(train_loader, model, criterion, optimizer, epoch, args, writer)

        # evaluate on validation set
        acc1, loss = validate(val_loader, model, criterion, epoch, writer)
        scheduler.step(loss, epoch=epoch)

        # remember best acc@1 and save checkpoint
        is_best = acc1 > best_acc1
        best_acc1 = max(acc1, best_acc1)

        save_checkpoint({
                'epoch': epoch + 1,
                'arch': 'ThreeStreamTemporal',
                'state_dict': model.state_dict(),
                'best_acc1': best_acc1,
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(),
            }, is_best, 'last_checkpoint.pth.tar', args.out_dir)

    writer.close()


def build_model(input_length, num_classes):
    model = models.three_stream_net(num_classes=num_classes, input_length=input_length)
    return model


def train(train_loader, model, criterion, optimizer, epoch, args, writer):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top3 = AverageMeter('Acc@3', ':6.2f')
    progress = ProgressMeter(len(train_loader), batch_time, data_time, losses, top1,
                             top3, prefix="Epoch: [{}]".format(epoch))

    # switch to train mode
    model.train()

    end = time.time()
    for i, (input, target) in enumerate(train_loader):

        # measure data loading time
        data_time.update(time.time() - end)
        for modality, data in input.items():
            input[modality] = data.cuda(non_blocking=True)
        target = target.cuda(non_blocking=True)

        # compute output
        output, aux_outputs = model(input)
        loss = criterion(output, target)

        # measure accuracy and record loss
        acc1, acc3 = accuracy(output, target, topk=(1, 3))
        losses.update(loss.item(), input[modality].size(0))
        top1.update(acc1[0], input[modality].size(0))
        top3.update(acc3[0], input[modality].size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        writer.add_scalar('Train/Acc1', top1.avg, epoch*len(train_loader)+i)
        writer.add_scalar('Train/Acc3', top3.avg, epoch*len(train_loader)+i)
        writer.add_scalar('Train/Loss', losses.avg, epoch*len(train_loader)+i)
        if i % args.print_freq == 0:
            progress.print(i)
    if epoch < 1:
        writer.add_image('Input RGB', utils.make_grid(input['rgb'][0].view(args.new_length,3,args.new_width, args.new_height)).cpu(), epoch)
        writer.add_image('Input Flow', utils.make_grid(input['flow'][0].view(args.new_length,3,args.new_width, args.new_height)).cpu(), epoch)
        writer.add_image('Input Skeleton', utils.make_grid(input['skeleton'][0].view(args.new_length,3,args.new_width, args.new_height)).cpu(), epoch)


def validate(val_loader, model, criterion, epoch, writer=None, classes=None):
    batch_time = AverageMeter('Time', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top3 = AverageMeter('Acc@3', ':6.2f')
    progress = ProgressMeter(len(val_loader), batch_time, losses, top1, top3,
                             prefix='Test: ')
    decisions={}
    targets_per_clip={}
    outputs_clip ={}

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        end = time.time()
        for i, (input, target, clip_id) in enumerate(val_loader):
            for modality, data in input.items():
                input[modality] = data.cuda(non_blocking=True)
            target = target.cuda(non_blocking=True)

            # compute output
            output = model(input)
            loss = criterion(output, target)

            # measure accuracy and record loss
            acc1, acc3 = accuracy(output, target, topk=(1, 3))
            losses.update(loss.item(), input[modality].size(0))
            top1.update(acc1[0], input[modality].size(0))
            top3.update(acc3[0], input[modality].size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                progress.print(i)

        print(' * Acc@1 {top1.avg:.3f} Acc@3 {top3.avg:.3f}'
              .format(top1=top1, top3=top3))
        if writer:
            writer.add_scalar('Test/Acc1', top1.avg, epoch)
            writer.add_scalar('Test/Acc3', top3.avg, epoch)
            writer.add_scalar('Test/Loss', losses.avg, epoch)

    return top1.avg, loss.item()


def load_checkpoint(model, optimizer, model_path):
    print("=> loading checkpoint '{}'".format(model_path))
    checkpoint = torch.load(model_path)
    start_epoch = checkpoint['epoch']
    model.load_state_dict(checkpoint['state_dict'])
    if optimizer:
        optimizer.load_state_dict(checkpoint['optimizer'])
    print("=> loaded checkpoint '{}' (epoch {})".format(model_path, checkpoint['epoch']))
    return model, optimizer, start_epoch


def save_checkpoint(state, is_best, filename, resume_path):
    cur_path = os.path.join(resume_path, filename)
    best_path = os.path.join(resume_path, 'model_best.pth.tar')
    torch.save(state, cur_path)
    if is_best:
        shutil.copyfile(cur_path, best_path)

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, *meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def print(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


if __name__ == '__main__':
    main()
