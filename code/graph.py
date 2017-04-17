import matplotlib.pyplot as plt

fileDir = "../hidden_3/batchLossFile.txt"

with open(fileDir,'r') as g:
	text = g.readlines()

plt.plot(range(len(text)),text,label = "lr_0.001 : clip_10 : steps_128")
plt.ylabel("loss")
plt.xlabel("batch")
plt.legend()
plt.savefig("../hidden_3/result_2.png", dpi = 500)
