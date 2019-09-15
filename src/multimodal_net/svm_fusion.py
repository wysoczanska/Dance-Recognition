import argparse
import os
import pickle

import audio_transforms
import datasets
import main_audio
import main_three_stream
import models
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import tqdm
import video_transforms
from scipy import stats
from scipy.special import softmax
from sklearn import svm
from sklearn.model_selection import RandomizedSearchCV
from torchvision import transforms

model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))

dataset_names = sorted(name for name in datasets.__all__)

parser = argparse.ArgumentParser(description='PyTorch Three-Stream Action Recognition')
parser.add_argument('data', metavar='DIR',
                    help='path to dataset')
parser.add_argument('--train_split_file', default='datasets/letsdance_splits/train.csv',
                    help='path to train files list')
parser.add_argument('--test_split_file', default ='datasets/letsdance_splits/val.csv',
                    help='path to val files list')
parser.add_argument('--device', default='cpu',
                    help='Whether to use gpu or cpu for models inference')
parser.add_argument('--classes_to_idx',
                    help='Path to classes to idx dictionary pickle file', default="datasets/classes_to_idx.pkl")
parser.add_argument('--vision_model_path', type=str, help='Absolute path to the vision-based model checkpoint.')
parser.add_argument('--audio_model_path', type=str, help='Absolute path to the audio checkpoint.')
parser.add_argument('--svm_model_path', type=str, help='Path to SVM model pickled file.')
parser.add_argument('--train', dest='train', action='store_true',
                    help='whether to train svm')
parser.add_argument('--extract', dest='extract', action='store_true',
                    help='whether to extarct audio and wisual representations')
parser.add_argument('--audio_representations_train', default='./extracted_representations/audio_predictions_train.pkl',
                    help='Path to extracted audio representations from train split')
parser.add_argument('--audio_representations_val', default='./extracted_representations/audio_predictions_val.pkl',
                    help='Path to extracted audio representations validation split')
parser.add_argument('--visual_representations_train', default='./extracted_representations/final_outs_train_vis.pkl',
                    help='Path to extracted visual representations (train)')
parser.add_argument('--visual_representations_val', default='./extracted_representations/final_outs_vis.pkl',
                    help='Path to extracted visual representations (validation split)')
parser.add_argument('--num_classes', default=16,
                    help='Number of classes in dataset')
parser.add_argument('--out_dir', type=str, help='Directory with saved extracted representations', default='./extracted_representations')
parser.add_argument('--new_length', type=str, help='Length of visual input in frames', default=15)
parser.add_argument('--new_width', default=340, type=int,
                    metavar='N', help='Visual input resize width (default: 340)')
parser.add_argument('--new_height', default=256, type=int,
                    metavar='N', help='Visual input resize height (default: 256)')
parser.add_argument('--new_audio_width', default=216, type=int,
                    metavar='N', help='Audio input resize width (default: 340)')
parser.add_argument('--new_audio_height', default=128, type=int,
                    metavar='N', help='Audio input resize height (default: 256)')
parser.add_argument('--dataset', default="Letsdance",
                    help='Dataset, for now only Lets dance is supported')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')


def infer_three_stream(val_loader, model, classes):
    decisions={}
    targets_per_clip={}
    outputs_clip ={}

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        for i, (input, target, clip_id) in tqdm.tqdm(enumerate(val_loader), total=len(val_loader)):
            for modality, data in input.items():
                input[modality] = data.cuda(non_blocking=True)
            target = target.cuda(non_blocking=True)

            # compute output
            output = model(input)
            prediction = output.max(1, keepdim=True)[1].view(-1)
            targets = target.view_as(prediction).cpu().numpy()

            if clip_id[0] not in decisions.keys():
                decisions[clip_id[0]] = prediction.cpu().numpy()
                targets_per_clip[clip_id[0]] = targets
                outputs_clip[clip_id[0]] = output.view(-1).cpu().numpy()
            else:
                decisions[clip_id[0]] = np.append(decisions[clip_id[0]], prediction.cpu().numpy())
                targets_per_clip[clip_id[0]] = np.append(targets_per_clip[clip_id[0]], targets)
                outputs_clip[clip_id[0]] = np.append(outputs_clip[clip_id[0]], output.view(-1).cpu().numpy())

        # eval.max_voting(decisions, targets_per_clip, classes)

        return outputs_clip


def extract_from_three_stream(args):
    model = main_three_stream.build_model(num_classes=args.num_classes, input_length=args.new_length)
    # create model
    print("Building model ... ")
    model = torch.nn.DataParallel(model)
    model = model.to(args.device)

    # define loss function (criterion) and optimizer

    if os.path.isfile(args.vision_model_path):
        model, _, start_epoch = main_three_stream.load_checkpoint(model, None, args.vision_model_path)
    is_color = True

    clip_mean = {'rgb': [0.485, 0.456, 0.406] * args.new_length, 'flow': [0.9432, 0.9359, 0.9511] * args.new_length,
                 'skeleton': [0.0071, 0.0078, 0.0079] * args.new_length}
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
                                                    video_transform=train_transform,
                                                    return_id=True)
    val_dataset = datasets.__dict__[args.dataset](root=args.data,
                                                  source=args.test_split_file,
                                                  phase="val",
                                                  is_color=is_color,
                                                  new_length=args.new_length,
                                                  video_transform=val_transform,
                                                  return_id=True)

    print('{} samples found, {} train samples and {} test samples.'.format(len(val_dataset) + len(train_dataset),
                                                                           len(train_dataset),
                                                                           len(val_dataset)))

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=1, shuffle=True,
        num_workers=args.workers, pin_memory=True)
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=1, shuffle=False,
        num_workers=args.workers, pin_memory=True)

    print("Extracting train visual representations")
    outputs_clip_train = infer_three_stream(train_loader, model,  classes=val_dataset.classes)
    pickle.dump(outputs_clip_train, open(args.visual_representations_train, 'wb'))

    print("Extracting validation visual representations")
    outputs_clip_val = infer_three_stream(val_loader, model, classes=val_dataset.classes)
    pickle.dump(outputs_clip_val, open(args.visual_representations_val, 'wb'))

    return outputs_clip_train, outputs_clip_val


def infer_bbnn(val_loader, model, classes):
    # switch to evaluate mode
    model.eval()
    predictions = []
    targets = []
    decisions_per_clip = {}

    with torch.no_grad():

        for i, (input, target, clip_id) in enumerate(val_loader):
            input = input.cuda(async=True)
            target = target.cuda(async=True)

            # compute output
            output = model(input)
            prediction = output.max(1, keepdim=True)[1].view(-1)
            target = target.view_as(prediction).cpu().numpy()
            predictions.append(prediction.cpu().numpy())
            targets.append(target)
            decisions_per_clip[clip_id[0]] = output.view(-1).cpu().numpy()


    return decisions_per_clip


def extract_from_bbnn(args):
    # create model
    print("Building model ... ")

    model = main_audio.build_model(num_classes=args.num_classes, model_name='BBNN')
    model = model.to(args.device)
    print("Model %s is loaded. " % ("BBNN"))

    # define loss function (criterion) and optimizer
    train_transform = transforms.Compose([
        audio_transforms.Resize((args.new_audio_height, args.new_audio_width)),
        audio_transforms.ToTensor(),
    ])
    val_transform = transforms.Compose([
        audio_transforms.Resize((args.new_audio_height, args.new_audio_width)),
        audio_transforms.ToTensor(),
    ])

    train_dataset = datasets.__dict__[args.dataset + '_audio'](root=args.data, source=args.train_split_file,
                                                               phase="train",
                                                               is_color=True,
                                                               new_length=1,
                                                               transform=train_transform,
                                                               return_id=True)
    val_dataset = datasets.__dict__[args.dataset + '_audio'](root=args.data,
                                                  source=args.test_split_file,
                                                  phase="val",
                                                  is_color=True,
                                                  new_length=1,
                                                  transform=val_transform,
                                                  return_id=True)

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=1,
        num_workers=args.workers, pin_memory=True, shuffle=True)
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=1, shuffle=True,
        num_workers=args.workers, pin_memory=True)
    model, _, start_epoch = main_three_stream.load_checkpoint(model, None, args.audio_model_path)
    decisions_per_clip_train = infer_bbnn(train_loader, model, classes=val_dataset.classes)
    pickle.dump(decisions_per_clip_train, open(args.audio_representations_train, 'wb'))
    decisions_per_clip_val = infer_bbnn(val_loader, model,  classes=val_dataset.classes)
    pickle.dump(decisions_per_clip_val, open(args.audio_representations_val, 'wb'))

    return decisions_per_clip_train, decisions_per_clip_val


def train_svm(vision_train, audio_train, train_csv, classes_to_idx):

    # data preparation
    audio_df = pd.DataFrame.from_dict(audio_train, orient='index')
    train_df = pd.read_csv(train_csv, header=None, sep='\t')
    train_df = train_df.set_index(1)

    vision_df = pd.DataFrame.from_dict(vision_train, orient='index').reset_index()
    vision_df = vision_df.set_index('index')
    vision_df = pd.DataFrame(np.nanmean(vision_df.values.reshape(len(train_df), -1, len(classes_to_idx)), axis=1),
                             index=vision_df.index)

    audio_df = audio_df.apply(softmax, axis=1)
    vision_df = vision_df.apply(softmax, axis=1)
    df = pd.concat([audio_df, vision_df], axis=1)
    y_train = train_df[0].replace(classes_to_idx)
    y_train = y_train.reindex(df.index)

    clf = svm.SVC()
    tuned_parameters = {'kernel': ['rbf', 'linear'], 'gamma': stats.uniform(1e-4, 10), 'C': stats.uniform(0.01, 10)}

    grid_search = RandomizedSearchCV(clf, tuned_parameters, scoring='accuracy',
                                     cv=10, return_train_score=True, n_jobs=20, n_iter=500)
    grid_search.fit(df, y_train)
    best_model = grid_search.best_estimator_
    print("Best model:")
    print(best_model)
    pickle.dump(best_model, open('svm.pkl', 'wb'))
    return best_model


def main(args):
    # if train infer both models and save pickles
    vision_train = None
    audio_train = None
    if args.extract:
        # vision_train, vision_val = extract_from_three_stream(args)
        audio_train, audio_val = extract_from_bbnn(args)
    if args.train:
        if vision_train is None:
            audio_train = pickle.load(open(args.audio_representations_train, "rb"))
            vision_train = pickle.load(open(args.visual_representations_train, "rb"))
        print("Training SVM...")
        svm_model = train_svm(vision_train, audio_train, args.train_split_file, pickle.load(open(args.classes_to_idx, "rb" )))

    assert os.path.exists("svm.pkl"), "No SVM model! :("
    assert os.path.exists(args.audio_representations_val), "No audio validation representations! :("
    assert os.path.exists(args.visual_representations_val), "No vision validation representations! :("
    audio_val = pickle.load(open(args.audio_representations_val, "rb"))
    vision_val = pickle.load(open(args.visual_representations_val, "rb"))
    svm_model = pickle.load(open("svm.pkl", "rb"))
    evaluate(audio_val, vision_val, svm_model, args)


def evaluate(audio_val, vision_val, svm_model, args):
    classes_to_idx = pickle.load(open(args.classes_to_idx, "rb"))
    idxes_to_classes = {val: key for (key, val) in classes_to_idx.items()}

    vision_df_val = pd.DataFrame.from_dict(vision_val, orient='index').reset_index()
    vision_df_val = vision_df_val.set_index('index')
    val_df = pd.read_csv(args.test_split_file, header=None, sep='\t',
                         names = ['true', 'index', 'duration']).set_index('index')

    vision_df_val = pd.DataFrame(np.nanmean(vision_df_val.values.reshape(len(val_df), -1, len(classes_to_idx)), axis=1),
                                 index=vision_df_val.index)
    audio_df_val = pd.DataFrame.from_dict(audio_val, orient='index')
    audio_df_val_preds = audio_df_val.apply(softmax, axis=1)
    vision_df_val_preds = vision_df_val.apply(softmax, axis=1)
    vision_df_val['vision_final'] = vision_df_val.idxmax(axis=1)
    audio_df_val['audio_final'] = audio_df_val.idxmax(axis=1)
    final_df = val_df.join(audio_df_val['audio_final']).drop(columns=['duration'])
    final_df = final_df.join(vision_df_val['vision_final'])
    df_val = pd.concat([audio_df_val_preds, vision_df_val_preds], axis=1)
    df_val = df_val.reindex(final_df.index)
    final_df['svm_final'] = svm_model.predict(df_val)
    final_df['svm_final'] = final_df['svm_final'].replace(idxes_to_classes)
    final_df['audio_final'] = final_df['audio_final'].replace(idxes_to_classes)
    final_df['vision_final'] = final_df['vision_final'].replace(idxes_to_classes)
    from sklearn.metrics import accuracy_score, precision_score, recall_score
    print(pd.DataFrame({'Accuracy': accuracy_score(final_df['true'], final_df['svm_final']),
                  'Precision': precision_score(final_df['true'], final_df['svm_final'], average='macro'),
                  "Recall": recall_score(final_df['true'], final_df['svm_final'], average='macro')}, index=[0]))

    final_df.to_csv("results.csv")


if __name__ == '__main__':
    args = parser.parse_args()
    if not os.path.exists('./extracted_representations'):
        os.mkdir("./extracted_representations")
    main(args)