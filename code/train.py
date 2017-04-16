import tensorflow as tf
import numpy as np
import pickle
#import matplotlib.pyplot as plt
def getData(fileDir):
	#with open (fileDir, "r") as myfile:
	    #text=myfile.read()
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
	#data = data[:1000]
	return data, uniqueCharToInt, intToUniqueChar,unique_char

data, uniqueCharToInt, intToUniqueChar,unique_char = getData("../data/input.txt")
print data.shape
print "No. of unique characters: ", len(unique_char)

nOutputs = len(unique_char)
nInputs = len(unique_char)
nHiddenUnits = 512
lr = .001
#n_epoch = 2
nSteps = 5


x = tf.placeholder(tf.float32,[None,nInputs])
y = tf.placeholder(tf.float32,[None,nOutputs])


#cStates = list()
#hStates = list()

#hPrev = tf.Variable(initial_value = tf.zeros(shape = [nHiddenUnits,]), trainable = False, name = "hiddenPrevious")
#cPrev = tf.Variable(initial_value = tf.zeros(shape = [nHiddenUnits,]), trainable = False, name = "statePrevious")

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

hStates,cPrevBatch,hPrevBatch = unroll_LSTM(x_in, cPrev, hPrev)

results = tf.matmul(hStates, weights['output']) + biases['output']
results = tf.reshape(results,[nSteps,nOutputs])

loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = results, labels = y))


optimizer = tf.train.AdamOptimizer(lr)
dVar = optimizer.compute_gradients(loss)
dVarClipped = [(tf.clip_by_value(grad, -5.,5.), var) for grad, var in dVar]
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
	batchLossFile = open("../hidden_1/batchLossFile.txt","w")
	epochLossFile = open("../hidden_1/epochLossFile.txt","w")
	cPrevFile = open("../hidden_1/cPrev.pickle",'w')
	hPrevFile = open("../hidden_1/hPrev.pickle",'w')
	while True:
		print "Iteration : ", j
		if (nSteps*(1 + i) + 1) <= data.shape[0]:
			batch_x = data[(i*nSteps) : (nSteps*(1 + i)), 0:nInputs]
			batch_y = data[(i*nSteps + 1) : (nSteps*(1 + i) + 1), 0:nInputs]
			_, batch_loss, cPrevSess, hPrevSess =  sess.run([train,loss,cPrevBatch,hPrevBatch],{x : batch_x, y : batch_y, cPrev : cPrevSess, hPrev : hPrevSess})			
			print "loss : ", batch_loss
			batchLossFile.write("%s\n" % batch_loss)
			epoch_loss += batch_loss
			j += 1
			i += 1

			if j % 10 == 0 :
				save_path = saver.save(sess, "../hidden_1/model_checkpoint/save_net.ckpt")
				pickle.dump(cPrevSess,cPrevFile)
				pickle.dump(hPrevSess,hPrevFile)
				print "model saved"


		else:
			print "One epoch done"
			print "epoch loss : ", epoch_loss
			epochLossFile.write("%s\n" % epoch_loss)
			i = 0
			epoch_loss = 0



'''
saver = tf.train.Saver()
lossFile = open('../hidden_1/lossList.txt','w')

with tf.Session() as sess:
	init = tf.global_variables_initializer()
	sess.run(init)
	cPrevSess = np.zeros(shape = [nHiddenUnits,1])
	hPrevSess = np.zeros(shape = [nHiddenUnits,1])
	for epoch in range(n_epoch):		
		epoch_loss = 0
		print "epoch : ",epoch
		for steps in range(int(data.shape[0]/nSteps)):			
			#if steps % 10 == 0 :
			print "step : ", steps
			if (nSteps*(1 + steps) + 1) <= data.shape[0]:

				batch_x = data[(steps*nSteps) : (nSteps*(1 + steps)), 0:nInputs]
				batch_y = data[(steps*nSteps + 1) : (nSteps*(1 + steps) + 1), 0:nInputs]
				print cPrevSess
				_, batch_loss, cPrevSess, hPrevSess =  sess.run([train,loss,cPrevBatch,hPrevBatch],{x : batch_x, y : batch_y, cPrev : cPrevSess, hPrev : hPrevSess})			
				print cPrevSess
				epoch_loss += batch_loss
			else:
				break


		print 'epoch_loss : ', epoch_loss
		lossFile.write("%s\n" % epoch_loss)
		#print sess.run(hPrev)


		if epoch % 10 == 0:
			save_path = saver.save(sess, "../hidden_1/model_checkpoint/save_net.ckpt")
			print "model saved"

'''



		
				
	
	






