import numpy as np
import datetime
import pandas
import theano
import theano.tensor as T
import math
import time
import itertools
import csv
from agebuckets import getbucket
from inputcreate import inputcreate
from lasagne.init import *
from lasagne import layers
from lasagne.updates import nesterov_momentum, rmsprop, adagrad
from lasagne.objectives import categorical_crossentropy
from lasagne.nonlinearities import *
from lasagne.layers import *
from sklearn.utils import shuffle
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import scale

from numpy import genfromtxt

def elu(x):
    return T.switch(x > 0, x, T.exp(x) - 1)

def myScore(yb, y_prob):
	scores = []
	for i in range(len(yb)):
		top5 = np.argsort(y_prob[i])[::-1][:5]
		dcg = 0
		for j in range(5):
			rel = 1 if top5[j] == yb[i] else 0
			dcg += (2**rel - 1) / math.log(j+1+1,2)
		scores.append(dcg)
	return np.mean(np.array(scores))

import cPickle
with open('keyswithsessions.pickle', 'rb') as f:
	keyswithsessions = cPickle.load(f)

users = []

# Read training file
with open('train_users_2.csv', 'rb') as csvfile:
	reader = csv.reader(csvfile, delimiter=',')
	reader.next()
	for row in reader:
		users.append(row)

users = np.array(users)

# Create input matrix
numusers = np.empty((users.shape[0], 969))
starti = 0

# Remove id column
trainids = users[:,0]
users = users[:,1:]

#import cPickle
with open('sessiontabs.pickle', 'rb') as f:
	sessiontabs = cPickle.load(f)

def float32(k):
    return np.cast['float32'](k)

class AdjustVariable(object):
    def __init__(self, name, start=0.03, stop=0.001):
        self.name = name
        self.start, self.stop = start, stop
        self.ls = None

    def __call__(self, nn, train_history):
        if self.ls is None:
            self.ls = np.linspace(self.start, self.stop, nn.max_epochs)

        epoch = train_history[-1]['epoch']
        new_value = float32(self.ls[epoch - 1])
        getattr(nn, self.name).set_value(new_value)

#y = np.empty(users.shape[0])
label = users[:,14]
countrylabels = sorted(list(set(users[:,14])))
y = np.array([countrylabels.index(x) for x in label])

#X = numusers
y = y.astype(np.int32)

# Only those that contain sessions
ntrainsids = []
ny = []
for trainid, yi in itertools.izip(trainids, y):
	if trainid in keyswithsessions:
		ntrainsids.append(trainid)
		ny.append(yi)
trainids = np.array(ntrainsids)
y = np.array(ny)

#trainids = trainids[-76430:]
#y = y[-76430:]

trainids, y = shuffle(trainids, y, random_state=42)

val_size = 10000
#X_train, X_val = X[:-val_size], X[-val_size:]
ids_train, ids_val = trainids[:-val_size], trainids[-val_size:]
y_train, y_val = y[:-val_size], y[-val_size:]
"""
#rfc = GradientBoostingClassifier(n_estimators=5, verbose=1, learning_rate=0.05, max_depth=5)
rfc.fit(X_train, y_train)

prd = rfc.predict_proba(X_val)
print myScore(y_val, prd)
"""
"""
# random few sessions
ids, targets = ids_train[0:2], y_train[0:2]

lengths_train = np.empty(0)
X = np.empty((0, 9))

for i, id in enumerate(ids):
	# Get session tab for id
	tab = np.load('sessiontabs/' + id + '.npy')
	X = np.append(X, tab, axis=0)
	lengths_train = np.append(lengths_train, len(tab))

targets = np.array(targets, dtype=np.str)
lengths_train = np.array(lengths_train, dtype=np.int32)

from seqlearn.hmm import MultinomialHMM
from seqlearn.perceptron import StructuredPerceptron

clf = StructuredPerceptron()
clf.fit(X, targets, lengths_train)
"""

def iterate_minibatches(inputs, targets, batchsize, shuffle=False):
	assert len(inputs) == len(targets)
	if shuffle:
		indices = np.arange(len(inputs))
		np.random.shuffle(indices)
	for start_idx in range(0, len(inputs) - batchsize + 1, batchsize):
		if shuffle:
			excerpt = indices[start_idx:start_idx + batchsize]
		else:
			excerpt = slice(start_idx, start_idx + batchsize)
		yield inputs[excerpt], targets[excerpt]

num_epochs = 2000
seq_len = 50

input_var = T.tensor3('inputs')
#mask_var = T.matrix('masks')
#mask_var = T.TensorType(dtype=theano.config.floatX, broadcastable=(False,True))('mask')
target_var = T.ivector('targets')

#l_maskin = InputLayer(shape=(None, 34496), input_var=mask_var)
l_in = InputLayer(shape=(None, seq_len, 9), input_var=input_var)
l_dropout = DropoutLayer(l_in)
l_hidden = RecurrentLayer(l_dropout, num_units=50, nonlinearity=sigmoid)
l_dropout2 = DropoutLayer(l_hidden)
l_hidden2 = DenseLayer(l_dropout2, num_units=50, nonlinearity=sigmoid)
l_dropout3 = DropoutLayer(l_hidden2)
#l_hidden3 = DenseLayer(l_hidden2, num_units=50, nonlinearity=tanh)
#l_shp = ReshapeLayer(l_hidden, (-1, 9))
l_out = DenseLayer(l_dropout3, num_units=12, nonlinearity=softmax)

prediction = get_output(l_out)
loss = categorical_crossentropy(prediction, target_var)
loss = loss.mean()

test_prediction = get_output(l_out, deterministic=True)
test_loss = categorical_crossentropy(test_prediction, target_var)
test_loss = test_loss.mean()

#test_acc = myScoreT(test_prediction, target_var)
test_acc = T.mean(T.eq(T.argmax(test_prediction, axis=1), target_var), dtype=theano.config.floatX)
#test_acc = theano.function(inputs=[test_prediction, target_var], outputs=myScore, allow_input_downcast=True)

params = get_all_params(l_out, trainable=True)
updates = nesterov_momentum(loss, params, learning_rate=0.01, momentum=0.9)

train_fn = theano.function([input_var, target_var], loss, updates=updates)
val_fn = theano.function([input_var, target_var], [test_loss, test_acc])
predict_fn = theano.function([input_var], test_prediction)

for epoch in range(num_epochs):
	start_time = time.time()
	train_err = 0
	train_batches = 0
	
	for batch in iterate_minibatches(ids_train, y_train, 100, shuffle=True):
		ids, targets = batch
		batchtab = np.empty((len(ids), seq_len, 9), dtype=np.float32)
		
		for i, id in enumerate(ids):
			# Get session tab for id
			#tab = np.load('sessiontabs/' + id + '.npy')
			tab = sessiontabs[id]
			batchtab[i,-tab.shape[0]:,:] = tab[-seq_len:]
			#print "1", tab[-seq_len:]
			#print "2", tab[:seq_len]
			#tabs.append(tab)
			#batchmask[i,:] = batchtab[i,:,0] != 0
		
		train_err += train_fn(batchtab, targets)
		#print "trained b"
		train_batches += 1
	
	val_err = 0
	val_acc = 0
	val_myscore = 0
	val_batches = 0
	
	for batch in iterate_minibatches(ids_val, y_val, 100, shuffle=False):
		ids, targets = batch
		batchtab = np.empty((len(ids), seq_len, 9), dtype=np.float32)
		
		for i, id in enumerate(ids):
			# Get session tab for id
			tab = sessiontabs[id]
			batchtab[i,-tab.shape[0]:,:] = tab[-seq_len:]
		
		err, acc = val_fn(batchtab, targets)
		prd = predict_fn(batchtab)
		#print prd
		val_myscore += myScore(targets, prd)
		val_err += err
		val_acc += acc
		val_batches += 1
	
	
	print("Epoch {} of {} took {:.3f}s".format(epoch + 1, num_epochs, time.time() - start_time))
	print("  training loss:\t\t{:.6f}".format(train_err / train_batches))
	print("  validation loss:\t\t{:.6f}".format(val_err / val_batches))
	print("  train/val:\t\t\t{:.6f}".format((train_err / train_batches) / (val_err / val_batches)))
	print("  validation accuracy:\t\t{:.2f} %".format(val_acc / val_batches * 100))
	print("  validation myscore:\t\t{:.2f} %".format(val_myscore / val_batches * 100))

testids, rawtest, testinput = inputcreate('test_users.csv')
#testoutput = net1.predict_proba(testinput)
testoutput = predict_fn(testinput)
#testoutput = rfc.predict_proba(testinput)
