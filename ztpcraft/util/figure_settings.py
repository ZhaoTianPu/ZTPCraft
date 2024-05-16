import numpy as np
from typing import Any, Dict, Literal

# define a literal type that is made of strings of keys of MPL_PRESET
MPL_PRESET_KEY_TYPE = Literal["PhysRevOneCol", "PhysRevTwoCol"]

# preset
MPL_PRESET: Dict[str, Dict[str, Any]] = {}

# PhysRev
# compute the width and height of one column and two columns
PHYS_REV_ONE_COL_WIDTH_PT = 246.0
PHYS_REV_TWO_COL_WIDTH_PT = 510.0
PHYS_REV_INCHES_PER_PT = 1.0 / 72.27
PHYS_REV_GOLDEN_MEAN = (np.sqrt(5) - 1.0) / 2.0
PHYS_REV_ONE_COL_FIG_WIDTH = PHYS_REV_ONE_COL_WIDTH_PT * PHYS_REV_INCHES_PER_PT
PHYS_REV_ONE_COL_FIG_HEIGHT = PHYS_REV_ONE_COL_FIG_WIDTH / 1.5
MPL_PRESET["PhysRevOneCol"] = {
    "lines.linewidth": 1,
    "font.size": 10,
    "figure.figsize": [PHYS_REV_ONE_COL_FIG_WIDTH, PHYS_REV_ONE_COL_FIG_HEIGHT],
    "legend.fontsize": 10,
    "axes.labelsize": 10,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "mathtext.fontset": "stix",
    "font.family": "times",
    "figure.dpi": 300,
}
PHYS_REV_TWO_COL_FIG_WIDTH = PHYS_REV_TWO_COL_WIDTH_PT * PHYS_REV_INCHES_PER_PT
PHYS_REV_TWO_COL_FIG_HEIGHT = PHYS_REV_TWO_COL_FIG_WIDTH / 1.5
MPL_PRESET["PhysRevTwoCol"] = {
    "lines.linewidth": 1,
    "font.size": 10,
    "figure.figsize": [PHYS_REV_TWO_COL_FIG_WIDTH, PHYS_REV_TWO_COL_FIG_HEIGHT],
    "legend.fontsize": 10,
    "axes.labelsize": 10,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "mathtext.fontset": "stix",
    "font.family": "times",
    "figure.dpi": 300,
}

# COLOR ---------------------#
COLOR_PALETTE = {}
COLOR_PALETTE["RdYlBu_5"] = ["#d7191c", "#fdae61", "#ffffbf", "#abd9e9", "#2c7bb6"]
COLOR_PALETTE["RdYlBu_4"] = ["#d7191c", "#fdae61", "#abd9e9", "#2c7bb6"]
COLOR_PALETTE["RdYlBu_11"] = [
    "#a50026",
    "#d73027",
    "#f46d43",
    "#fdae61",
    "#fee090",
    "#ffffbf",
    "#e0f3f8",
    "#abd9e9",
    "#74add1",
    "#4575b4",
    "#313695",
]
COLOR_PALETTE["RdBu_9"] = [
    "#b2182b",
    "#d6604d",
    "#f4a582",
    "#fddbc7",
    "#f7f7f7",
    "#d1e5f0",
    "#92c5de",
    "#4393c3",
    "#2166ac",
]
COLOR_PALETTE["RdBu_11"] = [
    "#67001f",
    "#b2182b",
    "#d6604d",
    "#f4a582",
    "#fddbc7",
    "#f7f7f7",
    "#d1e5f0",
    "#92c5de",
    "#4393c3",
    "#2166ac",
    "#053061",
]
