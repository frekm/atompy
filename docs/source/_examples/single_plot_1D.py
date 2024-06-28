import atompy as ap
import numpy as np
import matplotlib.pyplot as plt

ax = plt.subplot()
ax.set_box_aspect(3/4)

x = np.linspace(0, 1, 100)
y = x**2
ax.plot(x, y)

ax.set_xlabel("x-label")
ax.set_ylabel("y-label")

ax.set_title("Title")

ap.make_me_nice()
