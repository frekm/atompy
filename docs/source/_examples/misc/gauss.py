import matplotlib.pyplot as plt
import atompy as ap
import numpy as np

x = np.linspace(-2, 2, 50)

for scale in ("pdf", "integral", "sum", "max"):
    plt.plot(x, ap.gauss(x, scale=scale), label=f"{scale=}")

plt.legend(loc="lower left", bbox_to_anchor=(0, 1), ncol=2)
plt.xlabel("x")
plt.ylabel("gauss(x)")
