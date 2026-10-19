import matplotlib.pyplot as plt

import atompy as ap

xedges = (0, 1, 2, 3)
yedges = (0, 1, 2)
z = ((11, 12), (21, 22), (31, 32))

h1 = ap.Hist2d(z, xedges, yedges)

h1.save("tests/hist2d_save.tmp.txt")
h2 = ap.Hist2d.load("tests/hist2d_save.tmp.txt")

fig, axs = plt.subplots(2, 1, layout="compressed")

h1.plot(axs[0])
h2.plot(axs[1])

fig.savefig("tests/hist2d_save_load.pdf")
