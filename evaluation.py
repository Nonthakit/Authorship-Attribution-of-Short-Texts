from __future__ import print_function
from __future__ import division
import pandas as pd
import numpy as np
import predict_def
import sys
from keras import models
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report
np.random.seed(0)

subset = None

print('Loading predicted data...')

post_pred = pd.read_csv(sys.argv[1])
post_pred =  post_pred.dropna()

print('Loading corrected data...')

post_corr = pd.read_csv(sys.argv[2])
post_corr = post_corr.dropna()
post_corr = post_corr[post_corr.time_stamp!='created_at']

feature_cols = ['raw_text']

batch_size = 128

maxlen = 140

print('Evaluating...')

header = True;

print(classification_report(post_pred['username'], post_corr['username']))

print('Recording differences...')

post =  pd.concat([post_pred.rename(
    columns = {'username':'predicted_username'})['predicted_username']
    ,post_corr['username']], axis = 1)

post = post[post.predicted_username!=post.username]
post = post['predicted_username']
post.to_csv(sys.argv[3], header = True)

print('The differences have been recorded in ' + sys.argv[3])
