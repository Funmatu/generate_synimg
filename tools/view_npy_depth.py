import sys
import numpy as np
import matplotlib.pyplot as plt

if len(sys.argv) != 2:
    print("Need target npy file to visualize")

npy_file = sys.argv[1]
depth = np.load(npy_file)
plt.imshow(depth)
plt.show()
