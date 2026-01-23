import atompy as ap

hist = ap.Hist1d((1, 2, 3, 4), (0, 1, 2, 3, 4))

hist.pad_with(0).plot(plot_kwargs=dict(drawstyle="steps-mid"))
