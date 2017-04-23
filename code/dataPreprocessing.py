import pickle

with open("../data/limericks.txt", "r") as myfile:
	    text=myfile.read()
text = text[:1500000]
#textTrain = text[:-10000]
#textTest = text[-10000:]
unique_char = set(text)
uniqueCharToInt = {s : i for i,s in enumerate(unique_char)}
intToUniqueChar = {i : s for i,s in enumerate(unique_char)}

print len(text)
#print len(textTrain)
#print len(textTest)
print len(unique_char)
print unique_char
print uniqueCharToInt
print intToUniqueChar

with open("../data/limericksShort.txt",'w') as pd:
	pd.write(text)

#with open("../data/processedDataTrain.pickle",'w') as pd:
	#pickle.dump(textTrain,pd)

#with open("../data/processedDataTest.pickle",'w') as pd:
	#pickle.dump(textTest,pd)

with open("../data/uniqueChar.pickle",'w') as uc:
	pickle.dump(unique_char,uc)

with open("../data/uniqueCharToInt.pickle",'w') as uc1:
	pickle.dump(uniqueCharToInt,uc1)

with open("../data/intToUniqueChar.pickle",'w') as uc2:
	pickle.dump(intToUniqueChar,uc2)
