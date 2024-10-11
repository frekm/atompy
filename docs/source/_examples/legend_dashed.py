import matplotlib.pyplot as plt
import atompy as ap

_, (ax0, ax1) = plt.subplots(1, 2)

ax0.plot([0, 1], ls="dashed", label="incorrect width")
ax0.plot([1, 2], ls="-", label="reference width")
ax1.plot([0, 1], ls=ap.dashed(), label="correct width")
ax1.plot([1, 2], ls="-", label="reference width")

for ax in (ax0, ax1):
    ax.set_box_aspect(1./1.618)
    ax.legend()

ap.make_me_nice()
