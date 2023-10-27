import os, sys
import numpy as np
import matplotlib.pyplot as plt

# sys.path.append(os.path.abspath('../src/atompy'))

import atompy as ap


def main():
    z1, x1, y1 = ap.load_ascii_hist2d("test_yxz.txt", order="xy", permuting="y")
    z2, x2, y2 = ap.load_ascii_hist2d("test_xyz.txt", order="yx", permuting="y")

    fig, ax = ap.subplots(1, 2, ratio=1)

    ax[0][0].pcolormesh(x1, y1, z1, rasterized=True, cmap=ap.cm_lmf2root)
    ax[0][1].pcolormesh(x2, y2, z2, rasterized=True, cmap=ap.cm_lmf2root)


    fig.savefig("test.pdf", dpi=600)


if __name__ == "__main__":
    main()