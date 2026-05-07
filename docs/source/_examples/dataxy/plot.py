import matplotlib.pyplot as plt
import numpy as np

import atompy as ap

plt.style.use("atom")

data = ap.DataXY.from_function(
    ap.gauss, np.linspace(-5, 5, 100), title="Normal distribution", xlabel="x"
)

data.plot()
