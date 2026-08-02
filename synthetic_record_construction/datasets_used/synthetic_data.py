# present wave data for demo mode number

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.msc_thesis.paths import *
import chaosmagpy as cp
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
from src.msc_thesis.synSetup import *

# choose mode number and reference time step


# load in data (uphi, utheta, br, SV)

# evaluate all at the same demo time step and plot

# then separately - plot phasor for SV, set of consecutive time step snapshots

# potentially a plot of signal period