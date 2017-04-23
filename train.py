import numpy as np
import datetime
import pandas
import theano
import theano.tensor as T
import math
import time
import itertools
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

trainids, users, numusers = inputcreate('train_users_2.csv')
#np.save('trainids', trainids)
#np.save('users', users)
#np.save('numusers', numusers)

#import cPickle
#with open('sessiontabs.pickle', 'rb') as f:
#	sessionstabs = cPickle.load(f)

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

"""
net1 = NeuralNet(
	layers=[
		('input', layers.InputLayer),
		#('mask', layers.InputLayer),
		('hidden', layers.DenseLayer),
		#('hidden', layers.RecurrentLayer),
		('hidden2', layers.DenseLayer),
		('output', layers.DenseLayer),
	],
	input_shape=(None, 158),
	#input_shape=(None, 158),
	#mask_shape=(None, None),
	hidden_num_units=80,
	hidden2_num_units=80,
	#hidden3_num_units=200,
	hidden_nonlinearity=tanh,
	hidden2_nonlinearity=tanh,
	#dropout_p=0.3,
	output_nonlinearity=softmax,
	output_num_units=12,
	train_split=TrainSplit(eval_size=0.1),
	custom_score=['myscore',myScore],
	
	update=nesterov_momentum,
	#update_learning_rate=theano.shared(float32(0.03)),
	#update_momentum=theano.shared(float32(0.9)),
	update_learning_rate=0.01,
	update_momentum=0.9,
	#batch_iterator_train=BatchIterator(batch_size=2048),
	#batch_iterator_test=BatchIterator(batch_size=2048),
	
	#on_epoch_finished=[
	#    AdjustVariable('update_learning_rate', start=0.03, stop=0.0001),
    #    AdjustVariable('update_momentum', start=0.9, stop=0.999),
	#],
	
	regression=False,
	max_epochs=5000,
	verbose=1,
)
net1.initialize()

X = numusers
y = y.astype(np.int32)

X, y = shuffle(X, y, random_state=42)
print X.shape
net1.fit(X, y)
"""


X = numusers
y = y.astype(np.int32)

X, y = shuffle(X, y, random_state=42)

val_size = 10000
X_train, X_val = X[:-val_size], X[-val_size:]
y_train, y_val = y[:-val_size], y[-val_size:]
"""
#rfc = GradientBoostingClassifier(n_estimators=5, verbose=1, learning_rate=0.05, max_depth=5)
rfc.fit(X_train, y_train)

prd = rfc.predict_proba(X_val)
print myScore(y_val, prd)
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

input_var = T.matrix('inputs')
target_var = T.ivector('targets')

l_in = InputLayer(shape=(None, 969), input_var=input_var)
l_hidden = DenseLayer(l_in, num_units=50, nonlinearity=tanh)
l_hidden2 = DenseLayer(l_hidden, num_units=50, nonlinearity=tanh)
#l_hidden3 = DenseLayer(l_hidden2, num_units=50, nonlinearity=tanh)
l_out = DenseLayer(l_hidden, num_units=12, nonlinearity=softmax)

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
	
	for batch in iterate_minibatches(X_train, y_train, 2000, shuffle=True):
		inputs, targets = batch
		train_err += train_fn(inputs, targets)
		train_batches += 1
	
	val_err = 0
	val_acc = 0
	val_myscore = 0
	val_batches = 0
	
	for batch in iterate_minibatches(X_val, y_val, 2000, shuffle=False):
		inputs, targets = batch
		err, acc = val_fn(inputs, targets)
		val_err += err
		val_acc += acc
		val_batches += 1
	
	prd = predict_fn(X_val)
	val_myscore += myScore(y_val, prd)
	
	print("Epoch {} of {} took {:.3f}s".format(epoch + 1, num_epochs, time.time() - start_time))
	print("  training loss:\t\t{:.6f}".format(train_err / train_batches))
	print("  validation loss:\t\t{:.6f}".format(val_err / val_batches))
	print("  train/val:\t\t\t{:.6f}".format((train_err / train_batches) / (val_err / val_batches)))
	print("  validation accuracy:\t\t{:.2f} %".format(val_acc / val_batches * 100))
	print("  validation myscore:\t\t{:.2f} %".format(val_myscore * 100))

testids, rawtest, testinput = inputcreate('test_users.csv')
#testoutput = net1.predict_proba(testinput)
testoutput = predict_fn(testinput)
#testoutput = rfc.predict_proba(testinput)

with open('output.csv', 'wb') as f:
	f.write('id,country\n')
	index = 0
	for row in testoutput:
		id = testids[index]
		top5 = np.argsort(row)[::-1][:5]
		for n in top5:
			f.write(id + ',' + countrylabels[n] + '\n')
		index += 1

#for row in testinput:
#	print row