import matplotlib.pyplot as plt

fileDir = "../hidden_3_lr_0.001_clip_100_step_128/epochLossFile.txt"

with open(fileDir,'r') as g:
	text = g.readlines()

plt.plot(range(len(text)),text,label = "hidden_3_lr_0.001_clip_100_step_128")
plt.ylabel("loss")
plt.xlabel("epoch")
plt.legend()
plt.savefig("../hidden_3_lr_0.001_clip_100_step_128/result_2.png", dpi = 500)
