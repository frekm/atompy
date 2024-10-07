import matplotlib.pyplot as plt
import atompy as ap

plt.rcParams["figure.facecolor"] = "grey"

fh, ah = 3.2, 1.5

fig, axs = plt.subplots(2, 1, figsize=(fh, fh))

ap.set_axes_size(ah, ah, axs[0], anchor="left")
ap.set_axes_size(ah*1.618, ah, axs[1], anchor="left")

axs[0].set_ylim(0.0, 0.1)

ap.make_me_nice(fix_figwidth=False, max_figwidth=fh)
