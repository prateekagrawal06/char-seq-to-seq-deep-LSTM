import tensorflow as tf
import numpy as np
import pickle

with open("../data/processedDataTrain.pickle",'r') as pd:
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

nOutputs = len(unique_char)
nInputs = len(unique_char)
nHiddenUnits = 512
lr = .0001
nSteps = 128
clipValue = 500
path = "../hidden_1_limericks/"

print "learning rate : ", lr
print "no of sequesnce : " , nSteps
print "clipping value : " , clipValue
print "hidden units : " , nHiddenUnits

x = tf.placeholder(tf.float32,[None,nInputs])
y = tf.placeholder(tf.float32,[None,nOutputs])


hPrev1 = tf.placeholder(tf.float32,[nHiddenUnits,1])
cPrev1 = tf.placeholder(tf.float32,[nHiddenUnits,1])


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

	
def unroll_LSTM(x_in, cPrev, hPrev):
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

hStates,cPrevBatch,hPrevBatch = unroll_LSTM(x_in, cPrev1, hPrev1)

results = tf.matmul(hStates, weights['output']) + biases['output']
results = tf.reshape(results,[nSteps,nOutputs])
### checkout the loss function,,,might be wrong
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = results, labels = y))


optimizer = tf.train.AdamOptimizer(lr)
dVar = optimizer.compute_gradients(loss)
dVarClipped = [(tf.clip_by_value(grad, -clipValue,clipValue), var) for grad, var in dVar]
train = optimizer.apply_gradients(dVarClipped)


saver = tf.train.Saver()
with tf.Session() as sess:
	init = tf.global_variables_initializer()
	sess.run(init)
	cPrevSess = np.zeros(shape = [nHiddenUnits,1])
	hPrevSess = np.zeros(shape = [nHiddenUnits,1])
	i = 0
	j = 0
	epoch_loss = 0
	batchLossFile = open(path + "batchLossFile.txt","w")
	epochLossFile = open(path + "epochLossFile.txt","w")
	
	while True:
		print "Iteration : ", j
		if (nSteps*(1 + i) + 1) <= len(text):
			text_x = text[(i*nSteps) : (nSteps*(1 + i))]
			text_y = text[(i*nSteps + 1) : (nSteps*(1 + i) + 1)]
			batch_x = []
			for s in text_x:
				a = np.zeros(shape=[len(unique_char)])
				a[uniqueCharToInt[s]] = 1
				batch_x.append(a)
			batch_x = np.array(batch_x)

			batch_y = []
			for s in text_y:
				a = np.zeros(shape=[len(unique_char)])
				a[uniqueCharToInt[s]] = 1
				batch_y.append(a)
			batch_y = np.array(batch_y)

			_, batch_loss, cPrevSess, hPrevSess =  sess.run([train,loss,cPrevBatch,hPrevBatch],{x : batch_x, y : batch_y, cPrev1 : cPrevSess, hPrev1 : hPrevSess})			
			print "loss : ", batch_loss
			batchLossFile.write("%s\n" % batch_loss)
			epoch_loss += batch_loss
			j += 1
			i += 1

			if j % 100 == 0 :
				save_path = saver.save(sess, path + "model_checkpoint/save_net.ckpt")
				
				print "model saved"


		else:
			print "One epoch done"
			print "epoch loss : ", epoch_loss
			epochLossFile.write("%s\n" % epoch_loss)
			i = 0
			epoch_loss = 0




