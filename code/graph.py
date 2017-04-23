import matplotlib.pyplot as plt

path = "../hidden_1_limericks/"
fileDir = path + "/batchLossFile.txt"

with open(fileDir,'r') as g:
	text = g.readlines()

plt.plot(range(len(text)),text,label = "Batch Loss")
plt.ylabel("loss")
plt.xlabel("batch")
plt.legend()
plt.savefig(path + "batchLoss.png", dpi = 500)
