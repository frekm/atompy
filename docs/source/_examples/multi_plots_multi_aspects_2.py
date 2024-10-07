import matplotlib.pyplot as plt
import atompy as ap

plt.rcParams["figure.facecolor"] = "grey"

fig, axs = plt.subplots(2, 1, figsize=(3.2, 3.2))

axs[0].set_box_aspect(1.)

axs[1].set_box_aspect(1./1.618)

# align top left axes to left edge
tmp = axs[0].get_position()
axs[0].set_position([axs[1].get_position().x0,
                       tmp.y0,
                       tmp.width,
                       tmp.height])

ap.make_me_nice()
