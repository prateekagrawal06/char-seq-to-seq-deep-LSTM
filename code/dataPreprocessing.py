import pickle
with open("../data/input_william.rtf", "rb") as myfile:
	    text=myfile.read()
text = text[347:]
unique_char = set(text)
uniqueCharToInt = {s : i for i,s in enumerate(unique_char)}
intToUniqueChar = {i : s for i,s in enumerate(unique_char)}

print len(text)
print len(unique_char)
print unique_char
print uniqueCharToInt
print intToUniqueChar

with open("../data/processedData.pickle",'w') as pd:
	pickle.dump(text,pd)

with open("../data/uniqueChar.pickle",'w') as uc:
	pickle.dump(unique_char,uc)

with open("../data/uniqueCharToInt.pickle",'w') as uc1:
	pickle.dump(uniqueCharToInt,uc1)

with open("../data/intToUniqueChar.pickle",'w') as uc2:
	pickle.dump(intToUniqueChar,uc2)
