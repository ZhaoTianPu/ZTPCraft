import numpy as np

fig_width_pt = 246.0
inches_per_pt = 1.0 / 72.27
golden_mean = (np.sqrt(5) - 1.0) / 2.0
fig_width = fig_width_pt * inches_per_pt
fig_height = fig_width / 1.5
settings = {}
# plt.rcParams.update(settings)
settings["lines.linewidth"] = 1
settings["font.size"] = 14
settings["figure.figsize"] = [fig_width, fig_height]
settings["legend.fontsize"] = 10
settings["axes.labelsize"] = 10
settings["xtick.labelsize"] = 10
settings["ytick.labelsize"] = 10
settings["mathtext.fontset"] = "stix"
settings["font.family"] = "times"
