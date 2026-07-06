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

PROJECT_ROOT = Path.cwd().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# dependencies
from pathlib import Path
import sys
import numpy as np
import matplotlib.pyplot as plt
import chaosmagpy as cp
import pydmd
import cmath
import copy
import h5py
import math
from scipy.signal import periodogram
from matplotlib.animation import FuncAnimation
from IPython.display import Video
from tqdm.notebook import tqdm
import pickle
from pydmd import FbDMD
from pydmd import HankelDMD
from pydmd import DMD
import gc
from scipy.interpolate import make_interp_spline
from chaosmagpy.chaos import BaseModel
import gc
import ctypes

# import paths.py to establish file structure and directories
from src.msc_thesis.paths import * # use one single python file for all paths

# importing all of the files from thesis utilities for use
from src.msc_thesis.sgdmd import * # use one single python file for all functions

# %% Defining experiment setup parameters

# defining time parameters

# note: times relative to start of synthetic time series (i.e. t=0)
t_start = 0.0 # start time
t_end = 25.0 # end time
t_record = t_end - t_start # record length
f_bin = 1 / t_record # associatied fourier resolution

dt_years = 0.5   # 6-month sampling (same as CHAOS independent information)
f_sample = 1 / dt_years    # samples per year

# getting decimal year and julian date formats
times_dyear = np.arange(t_start, t_end, dt_years)
n_times = int(len(times_dyear)) # number of total samples
print(n_times)
# choosing if decay on or not (globally)
decay = False
noise_flag = True # toggles whatever noise is on

if decay:
    sigma='default'
else:
    sigma=0


# %% Setting up the input magnetic data used for the test

# define array to loop over each input mode from Felix
mode_numbers = np.arange(1, 63, 1) # all 62 modes

# retrieve data
# ------------------- MAIN DATA RETRIEVAL -------------------------------

input_wave_data = []

# for each input mode, add it to total data storage
for i, mode_number in enumerate(tqdm(mode_numbers)):

    mode_i_data = [mode_number]

    gnm_mf, gnm_sv, br, sv, utheta, uphi, eigenvalue = \
        Felix_Wave_Obtain(times_dyear, mode_number,bin_choice='wave-centred', sigma=0)

    mode_i_data = {
        "mode_number": mode_number,
        "gnm_mf":gnm_mf,
        "gnm_sv":gnm_sv,
        "br": br,
        "sv": sv,
        "utheta": utheta,
        "uphi": uphi,
        "eigenvalue": eigenvalue,
        # filled later
        "best_match_idx": { # list as need to store for each noise realisation
            "br": [],
            "sv": [],
            "utheta": [],
            "uphi": []
        }
    }

    input_wave_data.append(mode_i_data) # input_wave_data = data dictionary for mode i



# %% applying the resolution matrix


# %% visualising the results of the resolution matrix application