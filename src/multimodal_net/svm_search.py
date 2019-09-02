import pickle
import pandas as pd
import numpy as np
import torch.nn.functional as F
import torch
import matplotlib.pyplot as plt
from scipy.special import softmax

vision_preds = pickle.load( open( "final_outs_train_vis.pkl", "rb" ) )
audio_preds = pickle.load( open( "audio_predictions_train.pkl", "rb" ) )
audio_df = pd.DataFrame.from_dict(audio_preds, orient='index')
audio_df
# audio_df.columns = ['audio_dec']
train_df = pd.read_csv("datasets/letsdance_splits/train.csv", header=None, sep='\t')
#                     )
classes_to_idx = pickle.load( open( "classes_to_idx.pkl", "rb" ) )
train_df = train_df.set_index(1)
vision_df = pd.DataFrame.from_dict(vision_preds, orient='index').reset_index()
# vision_df.columns = ['filename', 'vision_dec']
vision_df = vision_df.set_index('index')

vision_df = pd.DataFrame(np.nanmean(vision_df.values.reshape(1107, -1, 16), axis=1), index=vision_df.index)
audio_df = audio_df.apply(softmax, axis=1)
vision_df = vision_df.apply(softmax, axis=1)
df = pd.concat([audio_df, vision_df], axis=1)
y_train=train_df[0].replace(classes_to_idx)
y_train.reindex(df.index)
# from scipy import stats
from sklearn import svm
from sklearn.model_selection import GridSearchCV, train_test_split, RandomizedSearchCV, StratifiedShuffleSplit
from sklearn.preprocessing import normalize

X = normalize(df.values, 'l2')
clf = svm.SVC(probability=False)
tuned_parameters = {'kernel': ['rbf', 'linear'], 'gamma': [0.1], 'C': [1, 10]}


grid_search = GridSearchCV(clf, tuned_parameters, scoring='accuracy',
                           cv=3, return_train_score=False, n_jobs=-1)
grid_search.fit(X, np.array(y_train.values, dtype=np.int))
print(grid_search.best_estimator_)