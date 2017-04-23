import numpy as np
from collections import Counter
from collections import defaultdict
import pickle

with open("../data/processedDataTest.pickle",'r') as pd:
	testText = pickle.load(pd)
with open("../data/processedDataTrain.pickle",'r') as pd:
	trainText = pickle.load(pd)

outerDict = defaultdict(lambda:Counter())
uniqueChar = ''.join(set(trainText))
for i in uniqueChar:
    for j in range(len(trainText)-1):
        if trainText[j] == i:
            outerDict[i][trainText[j+1]]+=1

outerDict = dict(outerDict)
wordDict =dict((k, dict(v)) for k, v in outerDict.items())
for k,v in wordDict.items():
    sum1=sum(v.values())
    for k1,v1 in v.items():
        wordDict[k][k1] = 1.* wordDict[k][k1]/sum1




ch = testText[0]
text = []
text.append(ch)
for i in range(99):
    ch = np.random.choice(list(wordDict[ch].keys()),p=list(wordDict[ch].values()))
    text.append(ch)
print "".join(text)
print len(text)


acc = []
for i,t in enumerate(testText):
    ch = testText[i]
    nextChar = np.random.choice(list(wordDict[ch].keys()),p=list(wordDict[ch].values()))
    if (i+1) < len(testText):
        if nextChar == testText[i+1]:
            acc.append(1)
        else:
            acc.append(0)
print acc
print np.mean(acc)
 
    
       
