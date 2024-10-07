import matplotlib.pyplot as plt
import atompy as ap

plt.rcParams["figure.facecolor"] = "grey"

fig, axs = plt.subplots(2, 3, figsize=(3.2*3, 3.2*2))

for ax in axs.flatten():
    ax.set_box_aspect(1/1.618)

ap.make_me_nice()
