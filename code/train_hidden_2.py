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

hPrev1 = tf.placeholder(tf.float32,[nHiddenUnits,1])
cPrev1 = tf.placeholder(tf.float32,[nHiddenUnits,1])

hPrev2 = tf.placeholder(tf.float32,[nHiddenUnits,1])
cPrev2 = tf.placeholder(tf.float32,[nHiddenUnits,1])


weights = {
    # (nInputs, nHiddenUnit1)
    'input': tf.Variable(tf.random_normal([nInputs, nHiddenUnits]), name = 'weightsIn'),

    'i1' : tf.Variable(tf.random_normal([nHiddenUnits,(2 * nHiddenUnits)]), name = 'weightsi1'),
    'f1' : tf.Variable(tf.random_normal([nHiddenUnits,(2 * nHiddenUnits)]),name = 'weightsf1'),
    'o1' : tf.Variable(tf.random_normal([nHiddenUnits,(2 * nHiddenUnits)]),name = 'weightso1'),
    'g1' : tf.Variable(tf.random_normal([nHiddenUnits,(2 * nHiddenUnits)]),name = 'weightsg1'),
    # (nHiddenUnits1, nOutputs)

    'hh' : tf.Variable(tf.random_normal([nHiddenUnits,nHiddenUnits]), name = 'weightshh'),

    'i2' : tf.Variable(tf.random_normal([nHiddenUnits,(2 * nHiddenUnits)]), name = 'weightsi2'),
    'f2' : tf.Variable(tf.random_normal([nHiddenUnits,(2 * nHiddenUnits)]),name = 'weightsf2'),
    'o2' : tf.Variable(tf.random_normal([nHiddenUnits,(2 * nHiddenUnits)]),name = 'weightso2'),
    'g2' : tf.Variable(tf.random_normal([nHiddenUnits,(2 * nHiddenUnits)]),name = 'weightsg2'),

    # (nHiddenUnits1, nOutputs)
    'output': tf.Variable(tf.random_normal([nHiddenUnits, nOutputs]),name = 'weightsOut')
}
biases = {
    # (nHiddenUnits1, )
    'input': tf.Variable(tf.constant(0.01, shape=[nHiddenUnits, ]),name = 'biasesIn'),

    'i1' : tf.Variable(tf.constant(0.01,shape=[nHiddenUnits, ]), name = 'biasesi1'),
    'f1' : tf.Variable(tf.constant(0.01,shape=[nHiddenUnits, ]), name = 'biasesf1'),
    'o1' : tf.Variable(tf.constant(0.01,shape=[nHiddenUnits, ]), name = 'biaseso1'),
    'g1' : tf.Variable(tf.constant(0.01,shape=[nHiddenUnits, ]), name = 'biasesg1'),

    'hh' : tf.Variable(tf.constant(0.01,shape=[nHiddenUnits, ]), name = 'biaseshh'),

    'i2' : tf.Variable(tf.constant(0.01,shape=[nHiddenUnits, ]), name = 'biasesi2'),
    'f2' : tf.Variable(tf.constant(0.01,shape=[nHiddenUnits, ]), name = 'biasesf2'),
    'o2' : tf.Variable(tf.constant(0.01,shape=[nHiddenUnits, ]), name = 'biaseso2'),
    'g2' : tf.Variable(tf.constant(0.01,shape=[nHiddenUnits, ]), name = 'biasesg2'),    


    # (nOutputs, )
    'output': tf.Variable(tf.constant(0.1, shape=[nOutputs, ]), name = 'biasesOut')
}



def cell(x,cPrev,hPrev, layer):
	
	x = tf.reshape(x,[nHiddenUnits,1])
	hPrev = tf.reshape(hPrev,[nHiddenUnits,1])
	cPrev = tf.reshape(cPrev,[nHiddenUnits,-1])

	
	ic = tf.reshape(tf.concat([hPrev,x], axis = 0),[2*nHiddenUnits,-1])
	ib = tf.reshape(biases['i'+str(layer)],[nHiddenUnits,-1])
	i = tf.sigmoid(tf.matmul(weights['i' + str(layer)],ic) + ib)

	fc = tf.reshape(tf.concat([hPrev,x], axis = 0),[2*nHiddenUnits,-1])
	fb = tf.reshape(biases['f'+str(layer)],[nHiddenUnits,-1])
	f = tf.sigmoid(tf.matmul(weights['f'+str(layer)],fc) + fb)

	oc = tf.reshape(tf.concat([hPrev,x], axis = 0),[2*nHiddenUnits,-1])
	ob = tf.reshape(biases['o'+str(layer)],[nHiddenUnits,-1])
	o = tf.sigmoid(tf.matmul(weights['o'+str(layer)],oc) + ob)

	gc = tf.reshape(tf.concat([hPrev,x], axis = 0),[2*nHiddenUnits,-1])
	gb = tf.reshape(biases['g'+str(layer)],[nHiddenUnits,-1])
	g = tf.tanh(tf.matmul(weights['g'+str(layer)],gc) + gb)

	
	cCurrent = tf.add(tf.multiply(f,cPrev) , tf.multiply(i,g))
	hCurrent = tf.multiply(o,tf.tanh(cCurrent))

	return cCurrent,hCurrent

	
def unroll_LSTM(x, cPrev, hPrev,layer):
	cStates = list()
	hStates = list()
	
	for i in range(nSteps):
		cCurrent,hCurrent = cell(x[i],cPrev,hPrev, layer)
		cStates.append(cCurrent)
		hStates.append(hCurrent)

		cPrev = cCurrent
		hPrev = hCurrent
		
		
	return tf.reshape(hStates,[nSteps,nHiddenUnits]),tf.reshape(cPrev,[nHiddenUnits,1]),tf.reshape(hPrev,[nHiddenUnits,1])

x = tf.reshape(x,[-1,nInputs])
inputHidden1 = tf.add(tf.matmul(x,weights['input']),biases['input'])

hStates1,cPrev1Batch,hPrev1Batch = unroll_LSTM(inputHidden1, cPrev1, hPrev1,1)

inputHidden2 = tf.matmul(hStates1, weights['hh']) + biases['hh']

hStates2,cPrev2Batch,hPrev2Batch = unroll_LSTM(inputHidden2, cPrev2, hPrev2,2)

results = tf.matmul(hStates2, weights['output']) + biases['output']
results = tf.reshape(results,[nSteps,nOutputs])

loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = results, labels = y))


optimizer = tf.train.AdamOptimizer(lr)
dVar = optimizer.compute_gradients(loss)
dVarClipped = [(tf.clip_by_value(grad, -10.,10.), var) for grad, var in dVar]
train = optimizer.apply_gradients(dVarClipped)


saver = tf.train.Saver()
with tf.Session() as sess:
	init = tf.global_variables_initializer()
	sess.run(init)
	cPrev1Sess = np.zeros(shape = [nHiddenUnits,1])
	hPrev1Sess = np.zeros(shape = [nHiddenUnits,1])
	cPrev2Sess = np.zeros(shape = [nHiddenUnits,1])
	hPrev2Sess = np.zeros(shape = [nHiddenUnits,1])
	i = 0
	j = 0
	epoch_loss = 0
	batchLossFile = open("../hidden_2/batchLossFile.txt","w")
	epochLossFile = open("../hidden_2/epochLossFile.txt","w")
	cPrev1File = open("../hidden_2/cPrev1.pickle",'w')
	hPrev1File = open("../hidden_2/hPrev1.pickle",'w')
	cPrev2File = open("../hidden_2/cPrev2.pickle",'w')
	hPrev2File = open("../hidden_2/hPrev2.pickle",'w')
	while True:
		print "Iteration : ", j
		if (nSteps*(1 + i) + 1) <= data.shape[0]:
			batch_x = data[(i*nSteps) : (nSteps*(1 + i)), 0:nInputs]
			batch_y = data[(i*nSteps + 1) : (nSteps*(1 + i) + 1), 0:nInputs]
			_, batch_loss, cPrev2Sess, hPrev2Sess, cPrev1Sess, hPrev1Sess =  sess.run([train,loss,cPrev2Batch,hPrev2Batch,cPrev1Batch,hPrev1Batch],{x : batch_x, y : batch_y, cPrev1 : cPrev1Sess, hPrev1 : hPrev1Sess, cPrev2 : cPrev2Sess, hPrev2 : hPrev2Sess})			
			print "loss : ", batch_loss
			batchLossFile.write("%s\n" % batch_loss)
			epoch_loss += batch_loss
			j += 1
			i += 1

			if j % 10 == 0 :
				save_path = saver.save(sess, "../hidden_2/model_checkpoint/save_net.ckpt")
				pickle.dump(cPrev1Sess,cPrev1File)
				pickle.dump(hPrev1Sess,hPrev1File)
				pickle.dump(cPrev2Sess,cPrev2File)
				pickle.dump(hPrev2Sess,hPrev2File)
				
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



		
				
	
	






