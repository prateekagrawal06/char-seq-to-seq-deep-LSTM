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
print len(unique_char)
print unique_char
print intToUniqueChar
print uniqueCharToInt