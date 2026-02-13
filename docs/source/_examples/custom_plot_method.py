import atompy as ap
import mplutils as mplu
import matplotlib.pyplot as plt


MAX_ENERGY = 25.0
MAX_MOMENTUM = 1.5


class EnergyHist1d(ap.Hist1d):
    def plot(self, ax):
        mplu.set_axes_size(3, 1.0 / 1.618, ax=ax)
        fig, ax = super().plot(ax=ax, drawstyle="steps-mid")
        ax.set_ylim(bottom=0)
        ax.set_xlim(self.limits[0], MAX_ENERGY)
        return fig, ax


class EnergyVsAngleHist(ap.Hist2d):
    def plot(self, ax):
        mplu.set_axes_size(3, 1.0 / 1.618, ax=ax)
        fig, ax, _ = super().plot(ax=ax, cmap="viridis")
        ax.set_ylim(0, MAX_ENERGY)
        return fig, ax


class MomentumMap(ap.Hist2d):
    def plot(self, ax):
        mplu.set_axes_size(3, ax=ax)
        fig, ax, _ = super().plot(ax=ax, cmap="atom")
        ax.set_xlim(-MAX_MOMENTUM, MAX_MOMENTUM)
        ax.set_ylim(-MAX_MOMENTUM, MAX_MOMENTUM)
        return fig, ax


histos = []
fname = "example.root"

base = "He_Compton/electrons/momenta"
histos.append(MomentumMap.from_root(fname, f"{base}/px_vs_py"))
histos.append(MomentumMap.from_root(fname, f"{base}/px_vs_pz"))
histos.append(MomentumMap.from_root(fname, f"{base}/py_vs_pz"))
base = "He_Compton/electrons/energy"
histos.append(EnergyVsAngleHist.from_root(fname, f"{base}/phi_vs_electron_energy"))
histos.append(EnergyVsAngleHist.from_root(fname, f"{base}/ctheta_vs_electron_energy"))
histos.append(EnergyHist1d.from_root(fname, f"{base}/electron_energy"))

plt.style.use("atom")
fig, axs = plt.subplots(2, 3)

for histo, ax in zip(histos, axs.flat):
    histo.plot(ax)
