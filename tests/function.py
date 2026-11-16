import matplotlib.pyplot as plt
import numpy as np

import atompy as ap

plt.rcParams["figure.constrained_layout.use"] = True

xlog = False
ylog = False

f = ap.Function(ap.gauss, (-0.1, 20), xlog=xlog, ylog=ylog, steps=100)

plt.subplot(121)
f.plot(ax=plt.gca(), fmt="o")
(f + f).plot(ax=plt.gca(), fmt="s")

plt.subplot(122)
fi = f.inverted()
fi.plot(ax=plt.gca(), fmt="o")

plt.savefig("tests/function.pdf")
plt.close()

x = np.logspace(-1, 10, 10)
y = x**3

f2 = ap.Function.from_xy(x, y, xlog=True, ylog=True)
# f2.plot(fmt="o")
plt.xscale("log")

(f + f).plot()

plt.savefig("tests/function2.pdf")
