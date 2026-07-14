# %% SETTING UP AUTOUPDATES

from IPython import get_ipython

ipython = get_ipython()
if ipython is not None:
    ipython.run_line_magic("load_ext", "autoreload")
    ipython.run_line_magic("autoreload", "2")

# %% FILE SYSTEM AND DEPENDENCY SETUP

# forcing root location such that the notebook can access everything 
import sys
from pathlib import Path
print(sys.path)
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import matplotlib.pyplot as plt
import h5py
import gc
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import chaosmagpy as cp
from tqdm import tqdm
from matplotlib.colors import LogNorm
from scipy.signal import periodogram
import chaosmagpy as cp

# import paths.py to establish file structure and directories
from src.msc_thesis.paths import *
# import library made for these synthetic tests
from pyscripts.Cov_tests.cov_tests import *

# %%

file_path = Path(f"{CHAOS_COV_DIR}/CHAOS_Cov_1997_2026_0806_MF.h5")

with h5py.File(file_path, "r") as cov_file:

    print(cov_file.keys())
    plt.imshow(cov_file["Cnm"][1,:,:], vmin=0, vmax=0.00001)
    plt.show()
    plt.imshow(cov_file["Cnm"][72,:,:], vmin=0, vmax=0.00001)

# %%
