import matplotlib.pyplot as plt

plt.rcParams["figure.constrained_layout.use"] = True
plt.rcParams["figure.facecolor"] = "grey"

fig, axs = plt.subplots(2, 3, figsize=(3.2*3, 3.2*2))

for ax in axs.flatten():
    ax.set_box_aspect(1/1.618)
