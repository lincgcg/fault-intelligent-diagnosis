import numpy as np
import matplotlib.pyplot as plt
x = np.linspace(10,10000)
y = 20 * (1 -  np.log2(x)/20)
plt.plot(x, y)

plt.xlabel('model size in KB')
plt.ylabel('S(score)')

plt.savefig("/Users/cglin/Desktop/fault-intelligent-diagnosis/Sscore.png")
plt.show()
