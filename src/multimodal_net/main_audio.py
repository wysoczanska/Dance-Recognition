import argparse
import os
import shutil
import time

import audio_transforms
import datasets
import models
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.parallel
import torch.optim
import torch.utils.data
from tensorboardX import SummaryWriter
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torchvision import transforms, utils

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
parser.add_argument('--dataset', '-d', default='Letsdance_audio')
parser.add_argument('--model_path')
parser.add_argument('--arch', '-a', metavar='ARCH', default='BBNN')
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
parser.add_argument('--new_width', default=224, type=int,
                    metavar='N', help='resize width )')
parser.add_argument('--new_height', default=224, type=int,
                    metavar='N', help='resize height ')
parser.add_argument('--lr', '--learning-rate', default=0.001, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--weight-decay', '--wd', default=5e-4, type=float,
                    metavar='W', help='weight decay (default: 5e-4)')
parser.add_argument('--print-freq', default=50, type=int,
                    metavar='N', help='print frequency (default: 50)')
parser.add_argument('--save-freq', default=25, type=int,
                    metavar='N', help='save frequency (default: 25)')
parser.add_argument('--resume', default='./checkpoints_bbnn', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')

best_prec1 = 0.0
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_checkpoint(model, optimizer, model_path):
    print("=> loading checkpoint '{}'".format(model_path))
    checkpoint = torch.load(model_path)
    start_epoch = checkpoint['epoch']
    model.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    print("=> loaded checkpoint '{}' (epoch {})".format(model_path, checkpoint['epoch']))
    return model, optimizer, start_epoch, 0


def build_model(num_classes, model_name):
    model = models.__dict__[model_name](3)
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
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = ReduceLROnPlateau(optimizer, 'max', patience=3, factor=0.5, verbose=True)

    if not os.path.exists(args.resume):
        os.makedirs(args.resume)
    print("Saving everything to directory %s." % (args.resume))

    cudnn.benchmark = True

    train_transform = transforms.Compose([
            audio_transforms.Resize((args.new_height, args.new_width)),
            audio_transforms.ToTensor(),
        ])

    val_transform = transforms.Compose([
            audio_transforms.Resize((args.new_height, args.new_width)),
            audio_transforms.ToTensor(),
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
                                                  transform=val_transform,
                                                  return_id=True)

    print('{} samples found, {} train samples and {} test samples.'.format(len(val_dataset)+len(train_dataset),
                                                                           len(train_dataset),
                                                                           len(val_dataset)))

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        num_workers=args.workers, pin_memory=True, shuffle=True)
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=True)
    if args.evaluate:
        if os.path.isfile(args.model_path):
            model, optimizer, start_epoch, best_acc1 = load_checkpoint(model, optimizer, args.model_path)
        validate(val_loader, model, criterion, writer, epoch=0, classes=val_dataset.classes)
        return

    for epoch in range(args.start_epoch, args.epochs):
        # train for one epoch
        train(train_loader, model, criterion, optimizer, epoch, writer)

        # evaluate on validation set
        prec1 = 0.0
        if (epoch + 1) % args.save_freq == 0:
            prec1 = validate(val_loader, model, criterion, writer, epoch, val_dataset.classes)
            scheduler.step(prec1, epoch=epoch)

        # remember best prec@1 and save checkpoint
        is_best = prec1 > best_prec1
        best_prec1 = max(prec1, best_prec1)

        if (epoch + 1) % args.save_freq == 0:
            checkpoint_name = "last_checkpoint.pth.tar"
            save_checkpoint({
                'epoch': epoch + 1,
                'arch': args.arch,
                'state_dict': model.state_dict(),
                'best_prec1': best_prec1,
                'optimizer': optimizer.state_dict(),
            }, is_best, checkpoint_name, args.resume)
    writer.close()


def train(train_loader, model, criterion, optimizer, epoch, writer):
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
        input = input.cuda(async=True)
        target = target.cuda(async=True)

        # compute output
        output = model(input)
        loss = criterion(output, target)

        # measure accuracy and record loss
        acc1, acc3 = accuracy(output, target, topk=(1, 3))
        losses.update(loss.item(), input.size(0))
        top1.update(acc1[0], input.size(0))
        top3.update(acc3[0], input.size(0))

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
    if epoch < 2:
        writer.add_image('Input', utils.make_grid(input), epoch)


def validate(val_loader, model, criterion, writer, epoch, classes):
    batch_time = AverageMeter('Time', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top3 = AverageMeter('Acc@3', ':6.2f')
    progress = ProgressMeter(len(val_loader), batch_time, losses, top1, top3,
                             prefix='Test: ')

    # switch to evaluate mode

    with torch.no_grad():
        end = time.time()

        for i, (input, target, clip_id) in enumerate(val_loader):
            input = input.cuda(async=True)
            target = target.cuda(async=True)

            # compute output
            output = model(input)
            loss = criterion(output, target)

            # measure accuracy and record loss
            acc1, acc3 = accuracy(output, target, topk=(1, 3))
            losses.update(loss.item(), input.size(0))
            top1.update(acc1[0], input.size(0))
            top3.update(acc3[0], input.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            prediction = output.max(1, keepdim=True)[1].view(-1)
            target = target.view_as(prediction).cpu().numpy()

            if i % args.print_freq == 0:
                progress.print(i)

        print(' * Acc@1 {top1.avg:.3f} Acc@3 {top3.avg:.3f}'
              .format(top1=top1, top3=top3))
        if writer:
            writer.add_scalar('Test/Acc1', top1.avg, epoch)
            writer.add_scalar('Test/Acc3', top3.avg, epoch)
            writer.add_scalar('Test/Loss', losses.avg, epoch)

    return top1.avg


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


def adjust_learning_rate(optimizer, epoch, args):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = args.lr * (0.1 ** (epoch // 30))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


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
