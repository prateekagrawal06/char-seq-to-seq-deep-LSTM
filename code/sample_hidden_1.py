import tensorflow as tf
import numpy as np 
import pickle


with open("../data/processedData.pickle",'r') as pd:
	text = pickle.load(pd)

with open("../data/uniqueChar.pickle",'r') as uc:
	unique_char = pickle.load(uc)

with open("../data/uniqueCharToInt.pickle",'r') as uc1:
	uniqueCharToInt = pickle.load(uc1)

with open("../data/intToUniqueChar.pickle",'r') as uc2:
	intToUniqueChar = pickle.load(uc2)

print len(text)
print "No. of unique characters: ", len(unique_char)
print uniqueCharToInt
print intToUniqueChar
print unique_char

nSteps = 1
nInputs = len(unique_char)
nHiddenUnits = 256
nOutputs = len(unique_char)

x = tf.placeholder(tf.float32,[None,nInputs])

hPrev = tf.placeholder(tf.float32,[nHiddenUnits,1])
cPrev = tf.placeholder(tf.float32,[nHiddenUnits,1])

weights = {
    # (nInputs, nHiddenUnit1)
    'input': tf.Variable(tf.random_normal([nInputs, nHiddenUnits]), name = 'weightsIn') * 0.1,

    'i' : tf.Variable(tf.random_normal([nHiddenUnits,(2 * nHiddenUnits)]), name = 'weightsi') * 0.1,
    'f' : tf.Variable(tf.random_normal([nHiddenUnits,(2 * nHiddenUnits)]),name = 'weightsf') * 0.1,
    'o' : tf.Variable(tf.random_normal([nHiddenUnits,(2 * nHiddenUnits)]),name = 'weightso') * 0.1,
    'g' : tf.Variable(tf.random_normal([nHiddenUnits,(2 * nHiddenUnits)]),name = 'weightsg') * 0.1,
    # (nHiddenUnits1, nOutputs)
    'output': tf.Variable(tf.random_normal([nHiddenUnits, nOutputs]),name = 'weightsOut') * 0.1
}
biases = {
    # (nHiddenUnits1, )
    'input': tf.Variable(tf.constant(0.0, shape=[nHiddenUnits, ]),name = 'biasesIn'),

    'i' : tf.Variable(tf.constant(0.0,shape=[nHiddenUnits, ]), name = 'biasesi'),
    'f' : tf.Variable(tf.constant(0.0,shape=[nHiddenUnits, ]), name = 'biasesf'),
    'o' : tf.Variable(tf.constant(0.0,shape=[nHiddenUnits, ]), name = 'biaseso'),
    'g' : tf.Variable(tf.constant(0.0,shape=[nHiddenUnits, ]), name = 'biasesg'),

    # (nOutputs, )
    'output': tf.Variable(tf.constant(0.0, shape=[nOutputs, ]), name = 'biasesOut')
}



def cell(x,cPrev,hPrev):
	x = tf.reshape(x,[nHiddenUnits,1])
	hPrev = tf.reshape(hPrev,[nHiddenUnits,1])
	cPrev = tf.reshape(cPrev,[nHiddenUnits,-1])

	
	ic = tf.reshape(tf.concat([hPrev,x], axis = 0),[2*nHiddenUnits,-1])
	ib = tf.reshape(biases['i'],[nHiddenUnits,-1])
	i = tf.sigmoid(tf.matmul(weights['i'],ic) + ib)

	fc = tf.reshape(tf.concat([hPrev,x], axis = 0),[2*nHiddenUnits,-1])
	fb = tf.reshape(biases['f'],[nHiddenUnits,-1])
	f = tf.sigmoid(tf.matmul(weights['f'],fc) + fb)

	oc = tf.reshape(tf.concat([hPrev,x], axis = 0),[2*nHiddenUnits,-1])
	ob = tf.reshape(biases['o'],[nHiddenUnits,-1])
	o = tf.sigmoid(tf.matmul(weights['o'],oc) + ob)

	gc = tf.reshape(tf.concat([hPrev,x], axis = 0),[2*nHiddenUnits,-1])
	gb = tf.reshape(biases['g'],[nHiddenUnits,-1])
	g = tf.tanh(tf.matmul(weights['g'],gc) + gb)

	
	cCurrent = tf.add(tf.multiply(f,cPrev) , tf.multiply(i,g))
	hCurrent = tf.multiply(o,tf.tanh(cCurrent))

	return cCurrent,hCurrent


def unroll_LSTM(x_in,cPrev,hPrev):
	cStates = list()
	hStates = list()
	for i in range(nSteps):
		cCurrent,hCurrent = cell(x_in[i],cPrev,hPrev)
		cStates.append(cCurrent)
		hStates.append(hCurrent)

		cPrev = cCurrent
		hPrev = hCurrent
		
	return tf.reshape(hStates,[nSteps,nHiddenUnits]),tf.reshape(cPrev,[nHiddenUnits,1]),tf.reshape(hPrev,[nHiddenUnits,1])

x = tf.reshape(x,[-1,nInputs])
x_in = tf.add(tf.matmul(x,weights['input']),biases['input'])

hStates, cPrevBatch, hPrevBatch = unroll_LSTM(x_in,cPrev,hPrev)

results = tf.matmul(hStates, weights['output']) + biases['output']
results = tf.nn.softmax(tf.reshape(results,[nSteps,nOutputs]))


saver = tf.train.Saver()
with tf.Session() as sess:
	saver.restore(sess,"../hidden_1_haikus/model_checkpoint/save_net.ckpt")
	print "Model Restored"
	#with open("../hidden_1_lr_0.001_clip_100_steps_128/cPrev.pickle","r") as c:
		#cPrevSess = pickle.load(c)
	#with open("../hidden_1_lr_0.001_clip_100_steps_128/hPrev.pickle","r") as h:
		#hPrevSess = pickle.load(h)
	hPrevSess = np.zeros(shape = [nHiddenUnits,1])
	cPrevSess = np.zeros(shape = [nHiddenUnits,1])

	char = []
	startChar = np.zeros(shape = [1,nInputs])
	startChar[0,8] = 1

	for i in range(1000):
		nextCharProb, cPrevSess, hPrevSess = sess.run([results,cPrevBatch,hPrevBatch],{ x : startChar, cPrev : cPrevSess, hPrev : hPrevSess})
		nextCharIndex = np.random.choice(range(nOutputs), p = nextCharProb.ravel())
		nextChar = intToUniqueChar[nextCharIndex]
		char.append(nextChar)
		startChar = np.zeros(shape = [1,nInputs])
		startChar[0,nextCharIndex] = 1		
	print "text sampled"
	print "".join(char)

	
