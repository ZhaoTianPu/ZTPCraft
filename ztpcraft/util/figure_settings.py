import numpy as np
from typing import Any, Dict

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
}
