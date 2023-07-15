import numpy as np
import matplotlib.pyplot as plt
x1 = [17.9 * 1024, 9.5 * 1024, 2.1*1024, 684, 532, 205]
x2 = [39/96, 39/96, 25/96, 25/96, 9/96, 22/96]
y = 20 * (1 -  np.log2(x1)/20) + 80 * x2
print(y)

plt.xlabel('model size in KB')
plt.ylabel('S(score)')

plt.show()
