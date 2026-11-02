import matplotlib.pyplot as plt
import numpy as np

import atompy as ap

plt.rcParams["figure.constrained_layout.use"] = True

xlog = True
ylog = True

f = ap.Function(ap.gauss, (0.1, 20), xlog=xlog, ylog=ylog, steps=30)

plt.subplot(121)
f.plot(ax=plt.gca(), fmt="o")

plt.subplot(122)
fi = f.inverted()
fi.plot(ax=plt.gca(), fmt="o")

plt.savefig("tests/function.pdf")
plt.close()

x = np.logspace(-1, 10, 10)
y = x**3

f2 = ap.Function.from_xy(x, y, xlog=True, ylog=True)
f2.plot(fmt="o")
plt.xscale("log")

plt.savefig("tests/function2.pdf")
