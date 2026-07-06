'''
gauss_transform_check.py

Purpose:
To understand how Felix performs the physical -> spherical harmonic transform
by comparing it with other approaches

notes:
- decided to deal only with the 'arbitrary' i.e. non scaled
  synthetic wave data, as this is invariant regardless of scaling strategy
'''

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

# %% looping through each Felix wave mode

# define array to loop over each input mode from Felix
mode_numbers = np.arange(1, 63, 10) # all 62 modes

# retrieve data
# ------------------- MAIN DATA RETRIEVAL -------------------------------

input_wave_data = []

for i, mode_number in enumerate(mode_numbers):

    gnm, br, utheta, uphi, eigenval = Felix_Component_Load(mode_number, FELIX_DIR)

    # truncating felix gnm to 14 max degree
    gnm_f_mf_truncated = Truncate_Gauss_Coeffs(gnm, 14)

    sv = br * eigenval
    gnm_f_sv_truncated = gnm_f_mf_truncated * eigenval

    # manually projecting physical fields to SH max degree 14
    br_m14_manual = SH_L_Project(br, 'r', nmax=14)
    sv_m14_manual = SH_L_Project(sv, 'r', nmax=14)

    # forward model of Felix truncated gnm to physical space
    Ar = A_Truncated('r', nmax=14, A_dict=A_dict)
    br_m14_felix = Ar @ gnm_f_mf_truncated
    sv_m14_felix = Ar @ gnm_f_sv_truncated

    print(np.allclose(br_m14_manual, br_m14_felix.reshape(state_shape)))
    print(np.allclose(sv_m14_manual, sv_m14_felix.reshape(state_shape)))





# %% applying the resolution matrix


# %% visualising the results of the resolution matrix application