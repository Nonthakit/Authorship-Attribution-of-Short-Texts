from __future__ import print_function
from __future__ import division
import datetime
import pandas as pd
import numpy as np
import predict_def
import sys
from sklearn.preprocessing import LabelBinarizer
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
np.random.seed(0)

if (len(sys.argv) < 3):
    print("invalid arguments")
    print("usage: python main.py model_type saved_model_name")
    exit()

ngram = int(sys.argv[1])
if (ngram < 1 or ngram > 2):
    print("no such model")
    exit()

subset = None

#Maximum length. Longer gets chopped. Shorter gets padded.
maxlen = 140

#Model params
#Filters for conv layers
nb_filter = 500
#Number of units in the dense layer
dense_outputs = 256
#Conv layer kernel size
filter_kernels = [3, 4, 5]
#Number of units in the final output layer. Number of classes.
cat_output = 22

#Compile/fit params
batch_size = 32
torelation_epoch = 3
nb_epoch = 40

print('Loading data...')

url = 'dataset.csv'
post = pd.read_csv(url)
post =  post.dropna()
post = post[post.time_stamp!='created_at']

feature_cols = ['raw_text']


X = (post[feature_cols])
le = LabelEncoder()
enc = LabelBinarizer()
print ('Converting ...')

post['username'] = le.fit_transform(post['username'])
# print (list(post['username']))
enc.fit(list(post['username']))
Y = enc.transform(list(post['username']))

vocab, reverse_vocab, vocab_size, check = predict_def.create_vocab_set()
X_train, X_test, y_train, y_test = train_test_split(X, Y, random_state=1)

print('Build model...')
if (ngram == 1):
    model = predict_def.model(filter_kernels, dense_outputs, maxlen, vocab_size,
                       nb_filter, cat_output)
else:
    model = predict_def.model2(filter_kernels, dense_outputs, maxlen, vocab_size,
                       nb_filter, cat_output)
print('Fit model...') 
initial = datetime.datetime.now()
print (len(X_test), len(y_test))

max_acc = 0

for e in range(nb_epoch):
    xi, yi = predict_def.shuffle_matrix(X_train, y_train)
    xi_test, yi_test = predict_def.shuffle_matrix(X_test, y_test)
    if subset:
        batches = predict_def.mini_batch_generator(xi[:subset], yi[:subset],
                                                    vocab, vocab_size, check,
                                                    maxlen,
                                                    batch_size=batch_size, ngram=ngram)
    else:
        batches = predict_def.mini_batch_generator(xi, yi, vocab, vocab_size,
                                                    check, maxlen,
                                                    batch_size=batch_size, ngram=ngram)

    test_batches = predict_def.mini_batch_generator(xi_test, yi_test, vocab,
                                                     vocab_size, check, maxlen,
                                                     batch_size=batch_size,ngram=ngram)

    accuracy = 0.0
    loss = 0.0
    step = 1
    start = datetime.datetime.now()
    print('Epoch: {}'.format(e))
    for x_train, y_train_ in batches:
        f = model.train_on_batch(np.asarray(x_train).astype(np.int), np.asarray(y_train_).astype(np.int))
        loss += f[0]
        loss_avg = loss / step
        accuracy += f[1]
        accuracy_avg = accuracy / step
        if step % 100 == 0:
            print('  Step: {}'.format(step))
            print('\tLoss: {}. Accuracy: {}'.format(loss_avg, accuracy_avg))
        step += 1

    test_accuracy = 0.0
    test_loss = 0.0
    test_step = 1
    
    for x_test_batch, y_test_batch in test_batches:
        f_ev = model.test_on_batch(np.asarray(x_test_batch).astype(np.int), np.asarray(y_test_batch).astype(np.int))
        test_loss += f_ev[0]
        test_loss_avg = test_loss / test_step
        test_accuracy += f_ev[1]
        test_accuracy_avg = test_accuracy / test_step
        test_step += 1
    stop = datetime.datetime.now()
    e_elap = stop - start
    t_elap = stop - initial
    print('Epoch {}. Loss: {}. Accuracy: {}\nEpoch time: {}. Total time: {}\n'.format(e, test_loss_avg, test_accuracy_avg, e_elap, t_elap))
    if (max_acc < test_accuracy_avg):
        max_acc = test_accuracy_avg
        model.save(sys.argv[2])
        best_epoch = e
    elif (e - best_epoch >= torelation_epoch):
        print('Accuracy doesn\'t increase for {} epoches. Training ends.\n'.format(torelation_epoch))
        pass
