'''
Plan:

Notebook to outline an initial DMD setup using the synthetic wave environment created
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
mode_numbers =[5] 

# if including perturbation ensemble
ensemble_flag = False

# if windowing to 'high quality' record
high_q_flag = True

#  %% -----------------------------------------------------
# OBTAIN INPUT DATA
# ---------------------------------------------------------

file_path = Path(FELIX_DIR) / "R_splines_arbitrary.h5"

synthetic_suite_info = {}

gnm_spl_input_resolved_list = []
exact_input_sv_phasor_list = []

with h5py.File(file_path, "r") as h5_file:

    for mode_number in tqdm(mode_numbers):
        synthetic_info = {}

        mode_data = Component_Load_SV(mode_number)
        eigenvalue = mode_data["eigenvalue"]
        sv_phasor = mode_data["sv"]

        # Period implied by the provided eigenvalue
        period = 2 * np.pi / np.abs(eigenvalue.imag)

        # getting resolved spline gnm (precomputed)
        gnm_spl = np.asarray(
            h5_file[f"mode_{mode_number}/without_decay"][()]
        )

        # recovering scaling factor
        amp_scaler = mode_amp_scalings[str(mode_number)]

        gnm_spl_input_resolved_list.append(amp_scaler * gnm_spl)

        synthetic_info["period"] = period
        synthetic_info["sv"] = amp_scaler * sv_phasor
        synthetic_suite_info[str(mode_number)] = synthetic_info


# flattening list of resolved gnm inputs to get total resolved input
gnm_spl_input_resolved = np.sum(gnm_spl_input_resolved_list, axis=0)

gnm_input_resolved = H_sv @ gnm_spl_input_resolved

# ---------------------------------------------------------
# OPTIONAL: ADD NOISE AND WINDOW TO HIGH QUALITY RECORD
# ---------------------------------------------------------

# if examining ensemble recovery behaviour
if ensemble_flag:

    gnm_input_total = Perturbation_Generate(gnm_input_resolved, 
                                            n_realisations=10,
                                            temporal_z="ar1")

else:

    gnm_input_total = gnm_input_resolved

# if only examining behaviour over 'high quality' data range
if high_q_flag:

    gnm_input_total_windowed = gnm_input_total[good_record_slice, :]

else:

    gnm_input_total_windowed = gnm_input_total

# ---------------------------------------------------------
# CONVERT TO GRIDDED SV DATA
# ---------------------------------------------------------

sv_input_total = A_r @ gnm_input_total_windowed.T

# %% ------------------------------------------------------
# APPLY DMD AND RECOVER MODES
# ---------------------------------------------------------

dmd = DMD(svd_rank=0)
dmd.fit(sv_input_total)

dmd_eigs = dmd.eigs
dmd_eigs_compare = D_To_C_Eigenvalue_Converter(dmd_eigs, dt=dt_years)

# %%

periods = [((2*np.pi)/eig.imag) for eig in dmd_eigs_compare]
print(periods)
for mode_num in mode_numbers:
    print(synthetic_suite_info[str(mode_num)]["period"])

# EXAMINE OUTPUT
# Cube_Movie(sv_input_total, name="resolved wave solo with noise")
# %%
