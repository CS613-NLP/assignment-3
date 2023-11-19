import math
import matplotlib.pyplot as plt

loss = [
    87.23,
    31.60,
    19.16,
    12.09,
    10.88,
]

perplexity = [math.exp(l) for l in loss]

plt.plot(perplexity)
plt.xlabel("Epoch")
plt.ylabel("Perplexity")
plt.title("Perplexity vs Epoch while PreTraining")
plt.legend(["Perplexity"])
plt.savefig("perplexity_vs_epoch.png")
plt.show()

plt.plot(loss)
plt.xlabel("Epoch")
plt.ylabel("Log(Perplexity)/Loss")
plt.title("Log(Perplexity)/Loss vs Epoch while PreTraining")
plt.legend(["Log(Perplexity)/Loss"])
plt.savefig("loss_vs_epoch.png")
plt.show()
