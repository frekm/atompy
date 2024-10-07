import matplotlib.pyplot as plt
import numpy as np
import atompy as ap

# create a 2D plot with a "custom" aspect ratio
image = plt.imshow(np.reshape(np.arange(9), (3,3)), aspect="auto")
plt.gca().set_box_aspect(0.5)

# Add a colorbar that now scales correctly to the 2D plot
ap.add_colorbar(image)

ap.make_me_nice()

plt.gcf().patch.set_facecolor("grey")