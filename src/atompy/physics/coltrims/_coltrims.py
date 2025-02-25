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
    tof_unit: str = "ns",
    m_over_q_unit: str = "amu/a.u.",
    savefig_filename: Optional[str] = None,
) -> tuple[float, float, float, float]:
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

    tof_unit : str, default 'ns'
        Units of provided TOFs

    m_over_q_unit : str, default "amu/a.u."
        Units of provided m/Q

    savefig_filename : str, optional
        Optionally, provide a filename to save the figure shown with
        ``show_plot = True``.

    Returns
    -------
    slope : float
        slope of the linear regression

    intersept : float
        intersept (i.e., x=0)

    delta_slope : float
        standard error of the slope

    delta_intercept : float
        standard error of the intercept
    """
    if not isinstance(tof_vs_m_over_q_pairs, np.ndarray):
        tof_vs_m_over_q_pairs = np.array(tof_vs_m_over_q_pairs)

    lin_regr = scipy.stats.linregress(
        np.sqrt(tof_vs_m_over_q_pairs[:, 1]), tof_vs_m_over_q_pairs[:, 0]
    )

    if names is not None and (len(tof_vs_m_over_q_pairs) != len(names)):
        msg = (
            "If names is provided, its length must match "
            "tof_vs_m_over_q_pairs, but"
            f"{len(names)=} and {len(tof_vs_m_over_q_pairs)}"
        )
        raise ValueError(msg)

    fig, ax = plt.subplots(layout="constrained")
    ax: Axes

    ax.plot(
        np.sqrt(tof_vs_m_over_q_pairs[:, 1]),
        tof_vs_m_over_q_pairs[:, 0],
        "o",
        data=names,
    )

    m, b = lin_regr.slope, lin_regr.intercept  # type: ignore
    dm, db = lin_regr.stderr, lin_regr.intercept_stderr  # type: ignore

    xintercept = -b / m
    xlow = 0 if b < 0 else xintercept
    x = np.linspace(xlow, ax.get_xlim()[1], 2)
    ax.plot(x, m * x + b)

    if names is not None:
        for i, name in enumerate(names):
            ax.annotate(
                name,
                (np.sqrt(tof_vs_m_over_q_pairs[i, 1]), tof_vs_m_over_q_pairs[i, 0]),
                textcoords="offset points",
                xytext=(0, 5),
                ha="right",
            )

    ax.set_ylabel(f"time of flight ({tof_unit})")
    ax.set_xlabel(r"$\sqrt{\text{m/Q}}$ ($\sqrt{\text{" f"{m_over_q_unit}" r"}}$)")

    ax.axhline(b, lw=0.5, color="k")
    ax.axvline(0, lw=0.5, color="k")

    annotation = (
        f"$({m:.2f}" r"\pm" f"{dm:.2f})" r"x" f"{b:+.2f}" r"\pm" f"{db:.2f}$ {tof_unit}"
    )
    ax.annotate(
        annotation,
        (ax.get_xlim()[1], 0),
        textcoords="offset points",
        xytext=(-5, 5),
        va="bottom",
        ha="right",
    )

    if show_plot:
        plt.show()

    if savefig_filename is not None:
        fig.savefig(savefig_filename)

    return (m, b, dm, db)


if __name__ == "__main__":
    ion_tof_linear_fit(
        [
            [3964, 28 / 2],
            [5607, 28],
            [8100, 58],
        ],
        names=[r"CO$^{++}$", r"CO$^+$", r"C$_3$H$_6$O$^+$"],
    )
