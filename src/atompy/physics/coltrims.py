import numpy as np
from numpy.typing import ArrayLike
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from typing import Sequence, Optional
import scipy.stats



def ion_tof_linear_fit(
        tof_vs_m_over_q_pairs: ArrayLike,
        show_plot: bool = True,
        names: Optional[Sequence[str]] = None,
        tof_unit = "ns",
        m_over_q_unit = "amu/a.u."
) -> tuple[float, float, float]:
    """
    Perform a linear fit of TOF vs m/Q pairs

    Parameters
    ----------
    tof_vs_m_over_q_pairs : [[float, float], ...]
        [(TOF_1, m_1/Q_1), (TOF_2, m_2/Q_2), ...]

    show_plot : bool, default True
        Plot data and fit and show it with ``plt.show()``.

    names : Sequence of str, optional
        Optionally, provide a list of names/chemical formulas for m/Q 
        ``len(names)`` must match ``len(tof_vs_m_over_q_pairs)``

        Only relevant if ``show_plot = True`` 
    
    tof_unit : str, default 'ns'
        Units of provided TOFs

        Only relevant if ``show_plot = True`` 

    m_over_q_unit : str, default "amu/a.u."
        Units of provided m/Q

        Only relevant if ``show_plot = True`` 
    """
    if not isinstance(tof_vs_m_over_q_pairs, np.ndarray):
        tof_vs_m_over_q_pairs = np.array(tof_vs_m_over_q_pairs)

    lin_regr = scipy.stats.linregress(tof_vs_m_over_q_pairs,
                                      alternative="greater")
    
    if show_plot:
        if(names is not None and (len(tof_vs_m_over_q_pairs) != len(names))):
            msg = (
                "If names is provided, its length must match "
                "tof_vs_m_over_q_pairs, but"
                f"{len(names)=} and {len(tof_vs_m_over_q_pairs)}"
            )
            raise ValueError(msg)

        _, ax = plt.subplots(layout = "constrained")
        ax: Axes

        ax.plot(tof_vs_m_over_q_pairs[:,0],
                tof_vs_m_over_q_pairs[:,1],
                "o",
                data=names
        )

        if names is not None:
            for i, name in enumerate(names):
                ax.annotate(name,
                            (tof_vs_m_over_q_pairs[i,0],
                             tof_vs_m_over_q_pairs[i,1]),
                            textcoords="offset points",
                            xytext=(0, 10),
                            ha="right")


        m = lin_regr.slope
        b = lin_regr.intercept
        xintercept = -b/m
        xlow = 0 if b < 0 else xintercept
        x = np.linspace(xlow, ax.get_xlim()[1], 2)
        ax.plot(x, m*x + b)

        ax.set_xlabel(f"time of flight ({tof_unit})")
        ax.set_ylabel(f"m/Q ({m_over_q_unit})")

        ax.axvline(xintercept, lw=0.5, color="k")
        ax.annotate(f"{xintercept:.1f} {tof_unit}", 
                    (xintercept, ax.get_ylim()[1]),
                    textcoords="offset points",
                    xytext=(5, -10),
                    va="top")

        plt.show()


if __name__ == "__main__":
    ion_tof_linear_fit(
        [[5600, 14], [3940, 10], [8100, 20]],
        names=["O", "C", "N22"]
    )

        






    

