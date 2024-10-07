import matplotlib.pyplot as plt
import numpy as np

# create a 2D plot with a "custom" aspect ratio
plt.imshow(np.reshape(np.arange(9), (3,3)), aspect="auto")
plt.gca().set_box_aspect(0.5)

# add a colorbar
# in this simple form, it does not properly scale to the 2D plot
plt.colorbar()

plt.gcf().patch.set_facecolor("grey")
