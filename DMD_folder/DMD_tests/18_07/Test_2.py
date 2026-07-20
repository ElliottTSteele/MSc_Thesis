'''
Plan:

performs the 'easiest'/ ideal test case recovery (post resolution)
using gauss coefficient space DMD 

outcome: doing DMD on gauss coefficient does in fact make it significantly easier
therefore biased by construction in gauss space
'''

# PREAMBLE
# %% SETTING UP AUTOUPDATES

from IPython import get_ipython

ipython = get_ipython()
if ipython is not None:
    ipython.run_line_magic("load_ext", "autoreload")
    ipython.run_line_magic("autoreload", "2")

# %% FILE SYSTEM AND DEPENDENCY SETUP

import sys
from pathlib import Path
print(sys.path)
PROJECT_ROOT = Path(__file__).resolve().parents[3]
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
from matplotlib import cm


import numpy as np
import matplotlib.pyplot as plt
import chaosmagpy as cp

import os
import sys
import pydmd
import cmath
import copy
import h5py
import math
import pickle
import gc
import pydmd

from pathlib import Path
from scipy.signal import periodogram
from matplotlib.animation import FuncAnimation
from IPython.display import Video
from tqdm import tqdm
from scipy.interpolate import make_interp_spline
from chaosmagpy.chaos import BaseModel

# importing from pyDMD
from pydmd import DMD

# import paths.py to establish file structure and directories
from src.msc_thesis.paths import *

# import simulation setup
from src.msc_thesis.synSetup import *
from src.msc_thesis.synUtils import *
from src.msc_thesis.synDMD import *

# ---------------------------------------------------------
# SIMULATION SETUP
# ---------------------------------------------------------

# put list of mode numbers used
mode_numbers = np.arange(1, 63, 1)

# if including perturbation ensemble
ensemble_flag = False

# if windowing to 'high quality' record
high_q_flag = True

#  %% -----------------------------------------------------
# OBTAIN INPUT DATA
# ---------------------------------------------------------

file_path = Path(FELIX_DIR) / "R_splines_arbitrary.h5"

# record dmd recovery results
DMD_recovery = {}

with h5py.File(file_path, "r") as h5_file:

    for mode_number in tqdm(mode_numbers):

        mode_number = str(mode_number)
        DMD_recovery[mode_number] = {}

        synthetic_info = {}

        mode_data = Component_Load_SV(mode_number)
        eigenvalue = mode_data["eigenvalue"]
        sv_phasor = mode_data["sv"]

        # Period implied by the provided eigenvalue
        true_period = 2 * np.pi / np.abs(eigenvalue.imag)

        # appending true period
        DMD_recovery[mode_number]["true_period"] = true_period

        # getting resolved spline gnm (precomputed)
        gnm_spl = np.asarray(
            h5_file[f"mode_{mode_number}/without_decay"][()]
        )

        # recovering scaling factor
        amp_scaler = mode_amp_scalings[str(mode_number)]

        gnm_spl_scaled = amp_scaler * gnm_spl

        gnm_mode_res = H_sv @ gnm_spl_scaled

        # if only examining behaviour over 'high quality' data range
        if high_q_flag:

            gnm_input_total_windowed = gnm_mode_res[good_record_slice, :]

        else:

            gnm_input_total_windowed = gnm_mode_res

        # ---------------------------------------------------------
        # NO GRID CONVERSION - JUST GAUSS TIME SERIES DIRECTLY
        # ---------------------------------------------------------

        sv_input_total = gnm_input_total_windowed.T

        # ------------------------------------------------------
        # APPLY DMD AND RECOVER MODES
        # ---------------------------------------------------------

        dmd = DMD(svd_rank=2)
        dmd.fit(sv_input_total)

        dmd_eigs = dmd.eigs
        dmd_eigs_compare = D_To_C_Eigenvalue_Converter(dmd_eigs, dt=dt_years)


        # first check dmd eigs are conjugate (i.e. oscillatory motion recovered)
        if dmd_eigs_compare[0] == np.conj(dmd_eigs_compare[1]):

            # can choose either eigenvalue (conjugates)
            dmd_eig = dmd_eigs_compare[0]
            eig_period = np.abs((2*np.pi)/dmd_eig.imag)
            eig_growth = np.abs(dmd_eig.real)

            DMD_recovery[mode_number]["DMD"] = {}

            DMD_recovery[mode_number]["DMD"]["period"] = eig_period
            DMD_recovery[mode_number]["DMD"]["growth"] = eig_growth
        else:
            DMD_recovery[mode_number]["DMD"] = False
# %%
for key in DMD_recovery:
    true_period = DMD_recovery[key]["true_period"]
    if DMD_recovery[key]["DMD"]:

        rec_period = DMD_recovery[key]["DMD"]["period"]
        rec_growth = DMD_recovery[key]["DMD"]["growth"]
        plt.scatter(true_period, 0, marker='o')
        plt.scatter(rec_period, rec_growth, marker='^')
    else:
        plt.scatter(true_period, 0, marker='x')

    plt.ylim((-0.02, 0.02))
    plt.xscale("log")




# %%
for key in DMD_recovery:
    true_period = DMD_recovery[key]["true_period"]
    if DMD_recovery[key]["DMD"]:

        rec_period = DMD_recovery[key]["DMD"]["period"]
        rec_growth = DMD_recovery[key]["DMD"]["growth"]
        plt.scatter(true_period, rec_period, marker='x')
    else:
        plt.scatter(true_period, true_period, marker='o')

plt.plot([0,100], [0,100])
plt.xlim(0,100)
plt.ylim(0,100)


# %%
