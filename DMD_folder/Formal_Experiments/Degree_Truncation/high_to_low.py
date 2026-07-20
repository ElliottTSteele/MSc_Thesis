'''
Plan:

this presents the base case with exact dmd + uncertainty ensemble
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

import numpy as np

# ---------------------------------------------------------
# SIMULATION SETUP
# ---------------------------------------------------------

# put list of mode numbers used
mode_numbers = np.arange(1,63, 15)
mode_numbers = [str(num) for num in mode_numbers]

# if including perturbation ensemble
ensemble_flag = False

# if windowing to 'high quality' record
high_q_flag = True
n_skip = 3

# deciding on spherical harmonic truncation degree
Nmax = 20
A_r_20 = A_20_dict["r"]
A_r = Truncate_Gauss_Coeffs(A_r_20, tmax=Nmax)

noise_generated = True

# explicitly define degrees locally
# step size of 2 in truncation - due to even/ odd structure in waves n spectra
degree_truncations_tested = np.arange(16, -1, -4)


#  %% -----------------------------------------------------
# OBTAIN INPUT DATA
# ---------------------------------------------------------

file_path = Path(FELIX_DIR) / "R_splines_arbitrary.h5"

# record dmd recovery results
DMD_recovery = {}

with h5py.File(file_path, "r") as h5_file:

    gnm_total_res_list = []
    synthetic_suite_info = {}

    for mode_number in tqdm(mode_numbers):

        mode_info = {}

        mode_number = str(mode_number)
        DMD_recovery[mode_number] = {}

        synthetic_info = {}

        mode_data = Component_Load_SV(mode_number)
        eigenvalue = mode_data["eigenvalue"]
        sv_phasor_exact = mode_data["sv"]


        # getting sv phasor for comparison
        gnm_phasor = mode_data["gnm"]
        gnm_phasor_T = Truncate_Gauss_Coeffs(gnm_phasor, tmax=Nmax)
        sv_phasor = A_r @ gnm_phasor_T

        # Period implied by the provided eigenvalue
        true_period = 2 * np.pi / np.abs(eigenvalue.imag)
        mode_info["true_period"]=true_period

        # appending true period
        DMD_recovery[mode_number]["true_period"] = true_period

        # getting resolved spline gnm (precomputed)
        gnm_spl = np.asarray(
            h5_file[f"mode_{mode_number}/without_decay"][()]
        )

        # recovering scaling factor
        amp_scaler = mode_amp_scalings[str(mode_number)]

        mode_info["sv_phasor"]=sv_phasor * amp_scaler
        mode_info["gnm"] = gnm_phasor_T * amp_scaler

        gnm_spl_scaled = amp_scaler * gnm_spl

        gnm_mode_res = H_sv @ gnm_spl_scaled

        gnm_total_res_list.append(gnm_mode_res)

        synthetic_suite_info[mode_number] = mode_info

    gnm_total_res = np.sum(gnm_total_res_list, axis=0)


    if ensemble_flag:
        # noised ~ (nrealisations, Nt, Ng)
        noised = Perturbation_Generate(gnm_total_res, 
                                            n_realisations=10,
                                            temporal_z="ar1",
                                            tau=1.5,
                                            dt=dt_years)
        
        perturbed_signals = list(noised)
        queue = [gnm_total_res] + perturbed_signals
    else:
        queue = [gnm_total_res]

    # ---------------------------------------------------------
    # INITIALISE STORAGE -- do this before truncation loop
    # ---------------------------------------------------------

    for mode_number in mode_numbers:

        DMD_recovery[mode_number]["DMD"] = {}
        DMD_recovery[mode_number]["DMD_noised"] = {}

        for truncation_degree in degree_truncations_tested:

            DMD_recovery[mode_number]["DMD_noised"][truncation_degree] = {
                "similarity": [],
                "eigenvalue": [],
            }

    for truncation_degree in degree_truncations_tested:
        
        for queue_position, gnm_mode_res in enumerate(queue):
            if queue_position == 0:
                result="no_noise"
            else:
                result="noised"

            # if only examining behaviour over 'high quality' data range
            if high_q_flag:

                gnm_input_total_windowed = gnm_mode_res[good_record_slice, :]

            else:

                gnm_input_total_windowed = gnm_mode_res

            # ---------------------------------------------------------
            # GAUSS TO GRID CONVERSION
            # ---------------------------------------------------------

            # truncating if necessary
            gnm_input_total_windowed = Truncate_Gauss_Coeffs(gnm_input_total_windowed, 
                                                             tmax=Nmax,
                                                             tmin=truncation_degree)
            
            A_r = Truncate_Gauss_Coeffs(A_20_dict["r"], 
                                        tmax=Nmax,
                                        tmin=truncation_degree)

            sv_input_total_all_steps = A_r @ gnm_input_total_windowed.T


            sv_input_total = sv_input_total_all_steps[:,::n_skip]


            # ------------------------------------------------------
            # APPLY DMD AND RECOVER MODES
            # ---------------------------------------------------------

            dmd = DMD()
            dmd.fit(sv_input_total)

            for mode_number in synthetic_suite_info:

                mode_info = synthetic_suite_info[mode_number]

                target_period = mode_info["true_period"]
                gnm_phasor = mode_info["gnm"]
                gnm_phasor_truncation_degree = Truncate_Gauss_Coeffs(gnm_phasor, 
                                                             tmax=Nmax,
                                                             tmin=truncation_degree)
                sv_phasor = A_r @ gnm_phasor_truncation_degree.T

                # getting recovered modes
                recovered_dict = DMD_Mode_Pair(dmd, n_skip * dt_years)

                # find optimal fit
                candidate_eigs = recovered_dict["continuous_eigenvalues"]
                candidate_indices = np.arange(0, len(candidate_eigs))
                candidate_periods = np.asarray([(2*np.pi)/eig.imag for eig in candidate_eigs])
                relevant_period_mask = (candidate_periods > 1e-5) & (candidate_periods < 1000)

                candidate_modes = recovered_dict["phasors"]

                similarity_high_score = 0
                match_made = False

                for j, idx in enumerate(candidate_indices):
                    mode = candidate_modes[:, idx]
                    sim_score_idx = Complex_Phasor_Compare(mode, sv_phasor)
                    if sim_score_idx > similarity_high_score:
                        match_idx = idx
                        similarity_high_score = sim_score_idx
                        match_made = True

                if match_made and result == "no_noise":

                    DMD_recovery[mode_number]["DMD"][truncation_degree] = {
                            "similarity": similarity_high_score,
                            "eigenvalue": candidate_eigs[match_idx],
                        }

                elif match_made and result == "noised":

                    DMD_recovery[mode_number]["DMD_noised"][truncation_degree][
                        "similarity"
                    ].append(similarity_high_score)

                    DMD_recovery[mode_number]["DMD_noised"][truncation_degree][
                        "eigenvalue"
                    ].append(candidate_eigs[match_idx])

#

# %% ------------------------------------------------------
# PLOT DMD RECOVERY AS FUNCTION OF DEGREE TRUNCATION
# ---------------------------------------------------------

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import BoundaryNorm

# Ensure sorted integer truncation degrees
degrees = np.asarray(sorted(degree_truncations_tested), dtype=int)

# ---------------------------------------------------------
# DISCRETE VIRIDIS COLOUR MAPPING
# ---------------------------------------------------------

# One distinct colour for each tested truncation degree
cmap = plt.colormaps["viridis"].resampled(len(degrees))

# Boundaries centred between degree values, giving a genuinely discrete
# colourbar rather than implying a continuous variable.
if len(degrees) > 1:
    degree_step = np.median(np.diff(degrees))
else:
    degree_step = 1

boundaries = np.concatenate([
    [degrees[0] - degree_step / 2],
    (degrees[:-1] + degrees[1:]) / 2,
    [degrees[-1] + degree_step / 2],
])

norm = BoundaryNorm(boundaries, cmap.N)


# =========================================================
# FIGURE 1: SIMILARITY VS TRUE PERIOD
# =========================================================

fig, ax = plt.subplots(figsize=(7.5, 5.5))

for degree in degrees:

    true_periods = []
    similarities = []

    for mode_number, results in DMD_recovery.items():

        # Skip missing results
        if (
            not results.get("DMD")
            or degree not in results["DMD"]
        ):
            continue

        true_period = results["true_period"]
        similarity = results["DMD"][degree]["similarity"]

        true_periods.append(true_period)
        similarities.append(similarity)

    if len(true_periods) == 0:
        continue

    true_periods = np.asarray(true_periods)
    similarities = np.asarray(similarities)

    # Sort from HIGH period to LOW period before joining
    sort_idx = np.argsort(true_periods)[::-1]

    ax.plot(
        true_periods[sort_idx],
        similarities[sort_idx],
        marker="o",
        linestyle="-",
        linewidth=1.2,
        markersize=5,
        color=cmap(norm(degree)),
    )

ax.set_xlabel("True period (years)")
ax.set_ylabel("Spatial similarity")
ax.set_ylim(0, 1.05)

ax.grid(alpha=0.25)

# Discrete colourbar
sm = cm.ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])

cbar = fig.colorbar(
    sm,
    ax=ax,
    boundaries=boundaries,
    ticks=degrees,
    spacing="uniform",
)

cbar.set_label("Spherical harmonic truncation degree")

fig.tight_layout()
plt.show()


# =========================================================
# FIGURE 2: SIGNED PERIOD ERROR VS TRUE PERIOD
# =========================================================

fig, ax = plt.subplots(figsize=(7.5, 5.5))

for degree in degrees:

    true_periods = []
    period_errors = []

    for mode_number, results in DMD_recovery.items():

        # Skip missing results
        if (
            not results.get("DMD")
            or degree not in results["DMD"]
        ):
            continue

        true_period = results["true_period"]

        eig = results["DMD"][degree]["eigenvalue"]

        # DMD recovered period
        recovered_period = 2 * np.pi / np.abs(eig.imag)

        # SIGNED percentage error:
        # positive = recovered period too long
        # negative = recovered period too short
        period_error_pct = (
            100
            * (recovered_period - true_period)
            / true_period
        )

        true_periods.append(true_period)
        period_errors.append(period_error_pct)

    if len(true_periods) == 0:
        continue

    true_periods = np.asarray(true_periods)
    period_errors = np.asarray(period_errors)

    # Sort from HIGH period to LOW period before joining
    sort_idx = np.argsort(true_periods)[::-1]

    ax.plot(
        true_periods[sort_idx],
        period_errors[sort_idx],
        marker="o",
        linestyle="-",
        linewidth=1.2,
        markersize=5,
        color=cmap(norm(degree)),
    )

# Zero-error reference
ax.axhline(
    0,
    color="black",
    linestyle="--",
    linewidth=1,
)

ax.set_xlabel("True period (years)")
ax.set_ylabel("Recovered period error (%)")

ax.grid(alpha=0.25)

# Discrete colourbar
sm = cm.ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])

cbar = fig.colorbar(
    sm,
    ax=ax,
    boundaries=boundaries,
    ticks=degrees,
    spacing="uniform",
)

cbar.set_label("Spherical harmonic truncation degree")

fig.tight_layout()
plt.show()

