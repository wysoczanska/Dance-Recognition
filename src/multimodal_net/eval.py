from __future__ import print_function

from sklearn.metrics import f1_score, recall_score, precision_score, accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.utils.multiclass import unique_labels
from collections import Counter
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams.update({'font.size': 22})


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
    classes_idx=np.arange(0,16)
    classes_idx = classes_idx[unique_labels(y_true, y_pred)]
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
           ylabel='True labels',
           xlabel='Predicted labels')

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
    fig.tight_layout()
    return fig


def max_voting(decisions_per_clip, targets_per_clip, classes):
    final_preds = []
    preds_per_clip = {}
    final_tars = []
    for clip_id, predictions in decisions_per_clip.items():
        print('Video metrics: ')
        print(get_metrics(targets_per_clip[clip_id], predictions))
        dec = Counter(predictions).most_common(1)[0][0]
        print('Decision: ' + str(dec) + ' Target: ' + str(targets_per_clip[clip_id][0]))
        preds_per_clip[clip_id] = dec
        final_preds.append(dec)
        final_tars.append(targets_per_clip[clip_id][0])
    # pickle.dump(final_tars, open('final_targets.pkl', 'wb'))
    # pickle.dump(preds_per_clip, open('final_preds.pkl', 'wb'))
    print('voting metrics')
    print(get_metrics(final_tars, final_preds))

    plot_confusion_matrix(final_tars, final_preds, classes, title='Confusion matrix').savefig('conf.png')

