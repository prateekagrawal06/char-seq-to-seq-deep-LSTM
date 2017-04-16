import matplotlib.pyplot as plt

fileDir = "../hidden_2/epochLossFile.txt"

with open(fileDir,'r') as g:
	text = g.readlines()

plt.plot(range(len(text)),text,'-bo', label = "lr : 0.001")
plt.ylabel("loss")
plt.xlabel("epoch")
plt.legend()
plt.savefig("../hidden_2/lr-0.001.png")
