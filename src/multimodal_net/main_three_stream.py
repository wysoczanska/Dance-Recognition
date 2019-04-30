import argparse
import time

import datasets
import models
import os
import shutil
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import video_transforms

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
parser.add_argument('--dataset', '-d', default='ucf101',
                    choices=["ucf101", "Letsdance"],
                    help='dataset: ucf101 | letsdance')
parser.add_argument('--out_dir', type=str, help='Directory with saved models', default='./checkpoints')
parser.add_argument('--model_path', type=str, help='Absolute path to the checkpoint. Only valid when resume or eval'
                                                   '', default='./checkpoints/model_best.pth.tar')
parser.add_argument('--resume', dest='resume', help='Best model evaluation mode', action='store_true')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=250, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='tmanual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=25, type=int,
                    metavar='N', help='mini-batch size (default: 50)')
parser.add_argument('--iter-size', default=5, type=int,
                    metavar='I', help='iter size as in Caffe to reduce memory usage (default: 5)')
parser.add_argument('--new_length', default=1, type=int,
                    metavar='N', help='length of sampled video frames (default: 1)')
parser.add_argument('--new_width', default=340, type=int,
                    metavar='N', help='resize width (default: 340)')
parser.add_argument('--new_height', default=256, type=int,
                    metavar='N', help='resize height (default: 256)')
parser.add_argument('--lr', '--learning-rate', default=0.001, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--lr_steps', default=[100, 200], type=float, nargs="+",
                    metavar='LRSteps', help='epochs to decay learning rate by 10')
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


best_acc1 = 0
ngpus = torch.cuda.device_count()


def main():
    global args, best_acc1
    args = parser.parse_args()
    num_classes = 16
    start_epoch=0

    model = build_model(num_classes=num_classes, input_length=args.new_length*3)

    print(model)

    # create model
    print("Building model ... ")

    model.rgb.features = torch.nn.DataParallel(model.rgb.features)
    model.skeleton.features = torch.nn.DataParallel(model.skeleton.features)
    model.flow.features = torch.nn.DataParallel(model.flow.features)
    model.cuda()

    if not os.path.exists(args.out_dir):
        os.makedirs(args.out_dir)
    print("Saving everything to directory %s." % (args.out_dir))

    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss().cuda()

    optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)

    # if resume set to True, load the model and continue training
    if args.resume:
        if os.path.isfile(args.model_path):
            model, optimizer, start_epoch, best_acc1 = load_checkpoint(model, optimizer, args.model_path)

    cudnn.benchmark = True

    is_color = True
    scale_ratios = [1.0, 0.875, 0.75, 0.66]
    clip_mean = [0.485, 0.456, 0.406] * args.new_length
    clip_std = [0.229, 0.224, 0.225] * args.new_length

    normalize = video_transforms.Normalize(mean=clip_mean,
                                           std=clip_std)
    train_transform = video_transforms.Compose([
            # video_transforms.Scale((256)),
            video_transforms.Resize((224, 224)),
            # video_transforms.RandomHorizontalFlip(),
            video_transforms.ToTensor(),
            normalize,
        ])

    val_transform = video_transforms.Compose([
            # video_transforms.Scale((256)),
            video_transforms.Resize((224, 224)),
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
                                                  video_transform=val_transform)

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
        validate(val_loader, model, criterion)
        return

    for epoch in range(start_epoch, args.epochs):
        adjust_learning_rate(optimizer, epoch, args)

        # train for one epoch
        train(train_loader, model, criterion, optimizer, epoch, args)

        # evaluate on validation set
        acc1 = validate(val_loader, model, criterion)

        # remember best acc@1 and save checkpoint
        is_best = acc1 > best_acc1
        best_acc1 = max(acc1, best_acc1)

        save_checkpoint({
                'epoch': epoch + 1,
                'arch': 'ThreeStreamTemporal',
                'state_dict': model.state_dict(),
                'best_acc1': best_acc1,
                'optimizer': optimizer.state_dict(),
            }, is_best, 'last_checkpoint.pth.tar', args.out_dir)


def build_model(input_length, num_classes):
    model = models.three_stream_net(num_classes=num_classes)
    model.rgb.features[0] = nn.Conv2d(input_length, 64, kernel_size=11, stride=4, padding=2)
    model.flow.features[0] = nn.Conv2d(input_length, 64, kernel_size=11, stride=4, padding=2)
    model.skeleton.features[0] = nn.Conv2d(input_length, 64, kernel_size=11, stride=4, padding=2)
    return model


def train(train_loader, model, criterion, optimizer, epoch, args):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(len(train_loader), batch_time, data_time, losses, top1,
                             top5, prefix="Epoch: [{}]".format(epoch))

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
        output = model(input)
        loss = criterion(output, target)

        # measure accuracy and record loss
        acc1, acc5 = accuracy(output, target, topk=(1, 5))
        losses.update(loss.item(), input[modality].size(0))
        top1.update(acc1[0], input[modality].size(0))
        top5.update(acc5[0], input[modality].size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            progress.print(i)


def validate(val_loader, model, criterion):
    batch_time = AverageMeter('Time', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(len(val_loader), batch_time, losses, top1, top5,
                             prefix='Test: ')

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        end = time.time()
        for i, (input, target) in enumerate(val_loader):
            for modality, data in input.items():
                input[modality] = data.cuda(non_blocking=True)
            target = target.cuda(non_blocking=True)

            # compute output
            output = model(input)
            loss = criterion(output, target)

            # measure accuracy and record loss
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            losses.update(loss.item(), input[modality].size(0))
            top1.update(acc1[0], input[modality].size(0))
            top5.update(acc5[0], input[modality].size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                progress.print(i)

        print(' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'
              .format(top1=top1, top5=top5))

    return top1.avg


def load_checkpoint(model, optimizer, model_path):
    print("=> loading checkpoint '{}'".format(model_path))
    checkpoint = torch.load(model_path)
    start_epoch = checkpoint['epoch']
    best_acc1 = checkpoint['best_acc1']
    model.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    print("=> loaded checkpoint '{}' (epoch {})".format(model_path, checkpoint['epoch']))
    return model, optimizer, start_epoch, best_acc1


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
