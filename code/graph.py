import matplotlib.pyplot as plt

fileDir = "../hidden_1_lr_0.001_clip_100_steps_128/batchLossFile.txt"

with open(fileDir,'r') as g:
	text = g.readlines()

plt.plot(range(len(text)),text,label = "hidden_1_lr_0.001_clip_100_steps_128")
plt.ylabel("loss")
plt.xlabel("batch")
plt.legend()
plt.savefig("../hidden_1_lr_0.001_clip_100_steps_128/batchLoss.png", dpi = 500)
