import matplotlib.pyplot as plt
import numpy as np

import atompy as ap

plt.style.use("atom")

x = np.linspace(-1, 1, 50)

for scale in ("pdf", "integral", "sum", "max"):
    plt.plot(x, ap.gauss(x, scale=scale), label=f"{scale=}")

plt.legend(loc="lower left", bbox_to_anchor=(0, 1), ncol=2)
plt.xlabel("x")
plt.ylabel("gauss(x)")
