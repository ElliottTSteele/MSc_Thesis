'''
Plan:

seeing how long it takes to run DMD at each svd truncation

outcome: less svd truncation does not effect anything, can 
just include full rank as this is ideal for realism
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
import time


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

import numpy as np

# function for comparing 2 complex phasors of same shape
def Complex_Phasor_Compare(x, y):
    x = np.asarray(x).ravel()
    y = np.asarray(y).ravel()

    denominator = np.linalg.norm(x) * np.linalg.norm(y)

    if denominator == 0:
        return np.nan

    return np.abs(np.vdot(x, y)) / denominator


# %%

# ---------------------------------------------------------
# SIMULATION SETUP
# ---------------------------------------------------------

svd_truncation_list = [1, 2, 4, 8]

# put list of mode numbers used
mode_numbers = [5]

# if including perturbation ensemble
ensemble_flag = False

# if windowing to 'high quality' record
high_q_flag = True
n_skip = 5

# deciding on spherical harmonic truncation degree
Nmax = 10
A_r_20 = A_20_dict["r"]
A_r = Truncate_Gauss_Coeffs(A_r_20, tmax=Nmax)

noise_generated = False


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
        sv_phasor_exact = mode_data["sv"]

        # getting sv phasor for comparison
        gnm_phasor = mode_data["gnm"]
        gnm_phasor_20 = Truncate_Gauss_Coeffs(gnm_phasor, tmax=Nmax)
        sv_phasor = A_r @ gnm_phasor_20

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

        # just doing same noise realisation for all, generate once
        if not noise_generated:

            noise = Perturbation_Generate(gnm_mode_res, 
                                                n_realisations=1,
                                                temporal_z="ar1",
                                                dt=dt_years,
                                                just_noise=True)
            
            noise = np.squeeze(noise)
        
        gnm_mode_res += noise
        
        # if only examining behaviour over 'high quality' data range
        if high_q_flag:

            gnm_input_total_windowed = gnm_mode_res[good_record_slice, :]

        else:

            gnm_input_total_windowed = gnm_mode_res

        # ---------------------------------------------------------
        # GAUSS TO GRID CONVERSION
        # ---------------------------------------------------------

        # truncating if necessary
        gnm_input_total_windowed = Truncate_Gauss_Coeffs(gnm_input_total_windowed, tmax=Nmax)

        sv_input_total_all_steps = A_r @ gnm_input_total_windowed.T

        sv_input_total = sv_input_total_all_steps[:,::n_skip]


        # ------------------------------------------------------
        # APPLY DMD AND RECOVER MODES
        # ---------------------------------------------------------

        n_repeats = 5

        rank_timings = {
            rank: []
            for rank in svd_truncation_list
        }

        # Optional warm-up run
        DMD(svd_rank=svd_truncation_list[0]).fit(sv_input_total)

        total_start = time.perf_counter()

        for rank in svd_truncation_list:

            for repeat in range(n_repeats):

                start = time.perf_counter()

                dmd = DMD(svd_rank=rank)
                dmd.fit(sv_input_total)

                elapsed = time.perf_counter() - start
                rank_timings[rank].append(elapsed)

        total_elapsed = time.perf_counter() - total_start


# %%

print(
    f"{'Rank':>6} "
    f"{'Mean [s]':>12} "
    f"{'Median [s]':>12} "
    f"{'Std [s]':>12} "
    f"{'Min [s]':>12}"
)
print("-" * 58)

for rank in svd_truncation_list:
    times = np.asarray(rank_timings[rank])

    print(
        f"{rank:>6} "
        f"{times.mean():>12.5f} "
        f"{np.median(times):>12.5f} "
        f"{times.std(ddof=1):>12.5f} "
        f"{times.min():>12.5f}"
    )

print(f"\nTotal benchmark time: {total_elapsed:.2f} s")


# %%
