import tensorflow as tf
import numpy as np 
import pickle


def getData(fileDir):
	text = "prateek Agrawal prateek Agrawal prateek Agrawal prateek Agrawal prateek Agrawal prateek Agrawal prateek Agrawal prateek Agrawal"
	unique_char = set(text)
	uniqueCharToInt = {s : i for i,s in enumerate(unique_char)}
	intToUniqueChar = {i : s for i,s in enumerate(unique_char)}
	data = []
	for s in text:
		a = np.zeros(shape=[len(unique_char)])
		a[uniqueCharToInt[s]] = 1
		data.append(a)
	data = np.array(data)
	#data = data[:12800]
	return data, uniqueCharToInt, intToUniqueChar,unique_char

data, uniqueCharToInt, intToUniqueChar,unique_char = getData("../data/input.txt")
print uniqueCharToInt

nSteps = 1
nInputs = len(unique_char)
nHiddenUnits = 512
nOutputs = len(unique_char)

x = tf.placeholder(tf.float32,[None,nInputs])
#hPrev = tf.Variable(initial_value = tf.zeros(shape = [nHiddenUnits,]), trainable = True, name = "hiddenPrevious")
#cPrev = tf.Variable(initial_value = tf.zeros(shape = [nHiddenUnits,]), trainable = True, name = "statePrevious")

hPrev = tf.placeholder(tf.float32,[nHiddenUnits,1])
cPrev = tf.placeholder(tf.float32,[nHiddenUnits,1])
weights = {
    # (nInputs, nHiddenUnit1)
    'input': tf.Variable(tf.random_normal([nInputs, nHiddenUnits]), name = 'weightsIn'),

    'i' : tf.Variable(tf.random_normal([nHiddenUnits,(2 * nHiddenUnits)]), name = 'weightsi'),
    'f' : tf.Variable(tf.random_normal([nHiddenUnits,(2 * nHiddenUnits)]),name = 'weightsf'),
    'o' : tf.Variable(tf.random_normal([nHiddenUnits,(2 * nHiddenUnits)]),name = 'weightso'),
    'g' : tf.Variable(tf.random_normal([nHiddenUnits,(2 * nHiddenUnits)]),name = 'weightsg'),
    # (nHiddenUnits1, nOutputs)
    'output': tf.Variable(tf.random_normal([nHiddenUnits, nOutputs]),name = 'weightsOut')
}
biases = {
    # (nHiddenUnits1, )
    'input': tf.Variable(tf.constant(0.1, shape=[nHiddenUnits, ]),name = 'biasesIn'),

    'i' : tf.Variable(tf.random_normal(shape=[nHiddenUnits, ]), name = 'biasesi'),
    'f' : tf.Variable(tf.random_normal(shape=[nHiddenUnits, ]), name = 'biasesf'),
    'o' : tf.Variable(tf.random_normal(shape=[nHiddenUnits, ]), name = 'biaseso'),
    'g' : tf.Variable(tf.random_normal(shape=[nHiddenUnits, ]), name = 'biasesg'),

    # (nOutputs, )
    'output': tf.Variable(tf.constant(0.1, shape=[nOutputs, ]), name = 'biasesOut')
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
	saver.restore(sess,"../hidden_1/model_checkpoint/save_net.ckpt")
	print "Model Restored"
	with open("../hidden_1/cPrev.pickle","r") as c:
		cPrevSess = pickle.load(c)
	with open("../hidden_1/hPrev.pickle","r") as h:
		hPrevSess = pickle.load(h)

	char = []
	startChar = np.zeros(shape = [1,nInputs])
	startChar[0,5] = 1

	for i in range(200):
		nextCharProb, cPrevSess, hPrevSess = sess.run([results,cPrevBatch,hPrevBatch],{ x : startChar, cPrev : cPrevSess, hPrev : hPrevSess})
		nextCharIndex = np.random.choice(range(nOutputs), p = nextCharProb.ravel())
		nextChar = intToUniqueChar[nextCharIndex]
		startChar = np.zeros(shape = [1,nInputs])
		startChar[0,nextCharIndex] = 1
		char.append(nextChar)
	print "text sampled"
	print "".join(char)

	
