import numpy as np

import atompy as ap

xedges = (0, 1, 2, 3)
yedges = (0, 1, 2)
z = ((11, 12), (21, 22), (31, 32))

h = ap.Hist2d(z, xedges, yedges)

h.save("tests/hist2d_save.tmp.txt")
