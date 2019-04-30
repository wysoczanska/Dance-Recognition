from __future__ import print_function

from sklearn.metrics import f1_score, recall_score, precision_score, accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.utils.multiclass import unique_labels
import argparse
import datasets
import models
from collections import Counter
import numpy as np
import torch
import matplotlib.pyplot as plt
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import video_transforms
from main_three_stream import build_model

model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))

dataset_names = sorted(name for name in datasets.__all__)


parser = argparse.ArgumentParser(description='Three stream eval ')
parser.add_argument('data', metavar='DIR',
                    help='path to dataset')
parser.add_argument('--new_length', default=1, type=int,
                    metavar='N', help='length of sampled video frames (default: 1)')
parser.add_argument('--test_split_file', metavar='DIR',
                    help='path to test files list')
parser.add_argument('--dataset', '-d', default='ucf101',
                    choices=["Letsdance"])
parser.add_argument('--out_dir', type=str, help='Directory with saved models', default='./checkpoints')
parser.add_argument('--model_path', type=str, help='Absolute path to the checkpoint. Only valid when resume or eval'
                                                   '', default='./checkpoints/model_best.pth.tar')
parser.add_argument('-b', '--batch-size', default=25, type=int,
                    metavar='N', help='mini-batch size (default: 50)')


def get_metrics(target, prediction):
    """
    Calculates basic evaluation metrics
    :param prediction: numpy array (1, nr samples) with predictions
    :param target: numpy array (1, nr samples) with predictions
    :return: dictionary with metrics: {f1, accuracy, precision, recall}
    """

    return {'Test/acc': accuracy_score(target, prediction),
            'Test/f1': f1_score(target, prediction, average='macro'),
            'Test/precision': precision_score(target, prediction, average='macro'),
            'Test/recall': recall_score(target, prediction, average='macro'),
            'Test/cm': confusion_matrix(target, prediction)}


def plot_confusion_matrix(y_true, y_pred, classes,
                          normalize=False,
                          title=None,
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if not title:
        if normalize:
            title = 'Normalized confusion matrix'
        else:
            title = 'Confusion matrix, without normalization'

    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    # Only use the labels that appear in the data
    classes = classes[unique_labels(y_true, y_pred)]
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    fig, ax = plt.subplots(figsize=(15,15))
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
#     fig.tight_layout()
    return fig


def main():
    num_classes = 16
    args = parser.parse_args()
    model = build_model(num_classes=num_classes, input_length=args.new_length * 3)

    print(model)

    # create model
    print("Building model ... ")

    model.rgb.features = torch.nn.DataParallel(model.rgb.features)
    model.skeleton.features = torch.nn.DataParallel(model.skeleton.features)
    model.flow.features = torch.nn.DataParallel(model.flow.features)
    model.cuda()

    print("=> loading checkpoint '{}'".format(args.model_path))
    checkpoint = torch.load(args.model_path)
    model.load_state_dict(checkpoint['state_dict'])
    print("=> loaded checkpoint '{}' (epoch {})".format(args.model_path, checkpoint['epoch']))

    clip_mean = [0.485, 0.456, 0.406] * args.new_length
    clip_std = [0.229, 0.224, 0.225] * args.new_length

    normalize = video_transforms.Normalize(mean=clip_mean,
                                           std=clip_std)

    val_transform = video_transforms.Compose([
        video_transforms.Resize((224, 224)),
        video_transforms.ToTensor(),
        normalize
    ])

    val_dataset = datasets.__dict__[args.dataset](root=args.data,
                                                  source=args.test_split_file,
                                                  phase="val",
                                                  is_color=False,
                                                  new_length=args.new_length,
                                                  video_transform=val_transform,
                                                  return_id=True)

    val_loader = torch.utils.data.DataLoader(val_dataset,
        batch_size=args.batch_size, shuffle=False,
        num_workers=4, pin_memory=True)
    model.eval()

    with torch.no_grad():
        targets = []
        predictions = []
        clip_ids = []
        for i, (input, target, clip_id) in enumerate(val_loader):
            for modality, data in input.items():
                input[modality] = data.cuda(non_blocking=True)
            target = target.cuda(non_blocking=True)

            # compute output
            output = model(input)

            # measure accuracy and record loss
            prediction = output.max(1, keepdim=True)[1].view(-1)

            targets.append(target.view_as(prediction).cpu().numpy())
            predictions.append(prediction.cpu().numpy())
            clip_ids += clip_id
        unique_clips = Counter(clip_ids)
        predictions = np.concatenate(predictions, axis=0)
        targets = np.concatenate(targets, axis=0)

        # max voting
        final_preds = {}
        decisions = {}
        for clip, _ in unique_clips.items():
            preds = []
            for idx, x in enumerate(clip_ids):
                if x == clip:
                    preds.append(predictions[idx])
                    target = targets[idx]
            preds = Counter(preds)
            decisions[clip] = np.array([preds.most_common(1)[0][0], target])
            print('pred:' + str(preds.most_common(1)))
            print('target '+ str(target))
            # final_targets[clip] = target
        decisions = np.array(list(decisions.values())).reshape(-1, 2)

        # final_preds = np.fromiter(final_preds.values(), dtype=int)
        # final_targets = np.fromiter(final_targets.values(), dtype=int)
        # print(decisions)
        # print(final_targets.reshape(1, -1))
        print(get_metrics(decisions[:, 1], decisions[:, 0]))
        plot_confusion_matrix(decisions[:, 1], decisions[:, 0], np.arange(0, 16), title='conf').savefig('conf.png')

if __name__ == '__main__':
    main()