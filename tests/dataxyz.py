import matplotlib.pyplot as plt
import numpy as np

import atompy as ap

data_raw_table_xincr_yincr_xfirst = np.array(
    (
        (1, 1, 11),
        (2, 1, 21),
        (3, 1, 31),
        (1, 2, 12),
        (2, 2, 22),
        (3, 2, 32),
    )
).T

data_raw_table_xincr_yincr_yfirst = np.array(
    (
        (1, 1, 11),
        (1, 2, 12),
        (2, 1, 21),
        (2, 2, 22),
        (3, 1, 31),
        (3, 2, 32),
    )
).T

data_raw_table_xdecr_yincr_xfirst = np.array(
    (
        (3, 1, 31),
        (2, 1, 21),
        (1, 1, 11),
        (3, 2, 32),
        (2, 2, 22),
        (1, 2, 12),
    )
).T

d = ap.DataXYZ.from_table(*data_raw_table_xdecr_yincr_xfirst)

d.plot(fname="tests/dataxyz.pdf")
