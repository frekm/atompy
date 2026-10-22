import matplotlib.pyplot as plt

import atompy as ap

d0 = ap.DataXYZ.from_table(
    (1, 1, 2, 2, 3, 3),
    (1, 2, 1, 2, 1, 2),
    (11, 12, 21, 22, 31, 32),
)

fname = "tests/data_xyz_save_load.tmp.txt"
d0.save(fname)

d1 = ap.DataXYZ.from_txt(fname)

fig, axs = plt.subplots(2, 1)

d0.plot(ax=axs[0])
d1.plot(ax=axs[1])

fig.savefig("tests/dataxyz_save_load.pdf")
