from __future__ import print_function
from __future__ import division
import pandas as pd
import numpy as np
import predict_def
import sys
from keras import models
from sklearn.preprocessing import LabelEncoder
np.random.seed(0)

subset = None

print('load model...')
model = models.load_model(sys.argv[2])

print('load classes encoder...')
le = LabelEncoder()
le.classes_ = np.load(sys.argv[2] + '/classes.npy', allow_pickle=True)

print('Loading data...')

post = pd.read_csv(sys.argv[3])
post =  post.dropna()
post = post[post.time_stamp!='created_at']

#le.fit(post['username'])
#np.save('classes.npy', le.classes_)
#le.classes_ = np.load('classes.npy')

feature_cols = ['raw_text']

batch_size = 128

vocab, reverse_vocab, vocab_size, check = predict_def.create_vocab_set()
maxlen = 140

ngram = int(sys.argv[1])

print('predicting...')

header = True;

with open(sys.argv[4], mode='w') as writer:
    for i in range(0, post.shape[0], batch_size):
        size = batch_size
        if (i + size > post.shape[0]):
            size = post.shape[0] - i
        if (ngram == 1):
            X_batch = predict_def.encode_data(post.raw_text[i:i+size], 
                    maxlen, vocab, vocab_size, check)
        
        else:
            X_batch = predict_def.encode_data2(post.raw_text[i:i+size], 
                    maxlen, vocab, vocab_size, check)
        
        np.set_printoptions(threshold=sys.maxsize)
        
        
        Y = model.predict(X_batch)
        Y = np.argmax(Y, axis=1)
        Y = le.inverse_transform(Y)

        if (header):
            df = pd.DataFrame({'username': np.asarray(Y)})
            df.to_csv(writer, index=True, header=header)
        else:
            df = pd.DataFrame({'index': range(i,i+size)
                ,'username': np.asarray(Y)})
            df.to_csv(writer, index=False, header=header)
        header = False
