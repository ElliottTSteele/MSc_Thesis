'''
.py file for settings used throughout figures

includes particular dates, colour bar scales if necessary etc.
'''
import matplotlib.pyplot as plt

# Defining optimal width for figures to fit on A4 paper
text_width = 7.25

plt.rcParams.update({
    "font.size": 11,
    "axes.titlesize": 11,
    "axes.labelsize": 11,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "legend.fontsize": 11,
    "figure.titlesize": 11,
})

# defining text size (size 11)

from cycler import cycler
import matplotlib.pyplot as plt


# defining colour blind friendly palette options
tol_muted = [
    "#332288",  # indigo
    "#88CCEE",  # cyan
    "#44AA99",  # teal
    "#117733",  # green
    "#999933",  # olive
    "#DDCC77",  # sand
    "#CC6677",  # rose
    "#882255",  # wine
    "#AA4499",  # purple
]

plt.rcParams["axes.prop_cycle"] = cycler(color=tol_muted)


# defining plotting shapes

record_markers = {
    "ideal":'o',
    "resolved":'s',
    "background":'^',
    "median":'P'
}

record_plotting_params = {
    "marker_size":10,
    "transparency":0.5
}
