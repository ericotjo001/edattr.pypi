import numpy as np
import matplotlib.pyplot as plt



x = np.random.normal(1,5,size=500) + 16
font = {'size': 17}
plt.rc('font', **font)

plt.figure(figsize=(5,7))
plt.hist(x,bins=16, color=(0,0.47,0), alpha=0.27)
plt.gca().set_xlabel('partition size')
plt.gca().set_ylabel('no. of partitions')
plt.gca().set_title('Expected partition profile')
plt.tight_layout()
plt.show()