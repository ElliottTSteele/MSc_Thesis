'''
Plan:

see how to actually recover the best match to a given mode,
when no SVD truncation used
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

import numpy as np

# function for comparing 2 complex phasors of same shape
def Complex_Phasor_Compare(x, y):
    x = np.asarray(x).ravel()
    y = np.asarray(y).ravel()

    denominator = np.linalg.norm(x) * np.linalg.norm(y)

    if denominator == 0:
        return np.nan

    return np.abs(np.vdot(x, y)) / denominator


# function to pair dmd modes, output list of them
import warnings


def DMD_Mode_Pair(
    dmd,
    dt,
    atol=1e-8,
    rtol=1e-6,
    include_unpaired=False, # dont include unpaired modes
):

    eigenvalues = np.asarray(dmd.eigs, dtype=complex)
    modes = np.asarray(dmd.modes, dtype=complex)
    amplitudes = np.asarray(dmd.amplitudes, dtype=complex)

    if modes.shape[1] != eigenvalues.size:
        raise ValueError(
            "The number of columns in dmd.modes must equal the number "
            "of eigenvalues."
        )

    if amplitudes.size != eigenvalues.size:
        raise ValueError(
            "dmd.amplitudes must contain one amplitude per DMD mode."
        )

    # Complete dynamically scaled modal coefficients b_j phi_j.
    scaled_modes = modes * amplitudes[np.newaxis, :]

    used = set()
    output_phasors = []
    output_eigenvalues = []
    mode_information = []

    for i, eig_i in enumerate(eigenvalues):

        if i in used:
            continue

        # ---------------------------------------------------------
        # Real/non-oscillatory eigenvalue
        # ---------------------------------------------------------
        if np.isclose(eig_i.imag, 0.0, atol=atol, rtol=rtol):

            output_phasors.append(scaled_modes[:, i])
            output_eigenvalues.append(eig_i)

            mode_information.append({
                "type": "real",
                "indices": (i,),
                "pair_error": np.nan,
            })

            used.add(i)
            continue

        # ---------------------------------------------------------
        # Find the closest unused conjugate eigenvalue
        # ---------------------------------------------------------
        candidate_indices = [
            j
            for j in range(eigenvalues.size)
            if j != i and j not in used
        ]

        if candidate_indices:

            conjugate_target = np.conj(eig_i)

            errors = np.array([
                np.abs(eigenvalues[j] - conjugate_target)
                for j in candidate_indices
            ])

            best_position = np.argmin(errors)
            j = candidate_indices[best_position]
            pair_error = errors[best_position]

            is_pair = np.isclose(
                eigenvalues[j],
                conjugate_target,
                atol=atol,
                rtol=rtol,
            )

        else:
            j = None
            pair_error = np.inf
            is_pair = False

        # ---------------------------------------------------------
        # Combine a conjugate pair
        # ---------------------------------------------------------
        if is_pair:

            # Positive frequency corresponds to positive discrete angle.
            if np.angle(eig_i) > 0:
                positive_idx = i
                negative_idx = j
            else:
                positive_idx = j
                negative_idx = i

            eig_positive = eigenvalues[positive_idx]

            # If the pair were exact:
            #
            # scaled_negative = conj(scaled_positive)
            #
            # and therefore this equals 2 * scaled_positive.
            phasor = (
                scaled_modes[:, positive_idx]
                + np.conj(scaled_modes[:, negative_idx])
            )

            output_phasors.append(phasor)
            output_eigenvalues.append(eig_positive)

            mode_information.append({
                "type": "conjugate_pair",
                "indices": (positive_idx, negative_idx),
                "pair_error": pair_error,
            })

            used.add(i)
            used.add(j)

        # ---------------------------------------------------------
        # Complex mode without a conjugate partner
        # ---------------------------------------------------------
        else:

            used.add(i)

            if include_unpaired:

                output_phasors.append(scaled_modes[:, i])
                output_eigenvalues.append(eig_i)

                mode_information.append({
                    "type": "unpaired_complex",
                    "indices": (i,),
                    "pair_error": pair_error,
                })

                warnings.warn(
                    f"DMD mode {i} with eigenvalue {eig_i} has no "
                    "conjugate partner.",
                    RuntimeWarning,
                )

    output_phasors = np.column_stack(output_phasors)
    output_eigenvalues = np.asarray(output_eigenvalues)

    continuous_eigenvalues = np.asarray(
        D_To_C_Eigenvalue_Converter(
            output_eigenvalues,
            dt=dt,
        )
    )

    return {
        "phasors": output_phasors,
        "discrete_eigenvalues": output_eigenvalues,
        "continuous_eigenvalues": continuous_eigenvalues,
        "mode_information": mode_information,
    }

# ---------------------------------------------------------
# SIMULATION SETUP
# ---------------------------------------------------------

# put list of mode numbers used
mode_numbers = [13, 23, 45, 48, 52, 54, 57, 61]
mode_numbers = [str(num) for num in mode_numbers]

# if including perturbation ensemble
ensemble_flag = False

# if windowing to 'high quality' record
high_q_flag = True
n_skip = 3

# deciding on spherical harmonic truncation degree
Nmax = 15
A_r_20 = A_20_dict["r"]
A_r = Truncate_Gauss_Coeffs(A_r_20, tmax=Nmax)

noise_generated = True


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

        mode_info["sv_phasor"]=sv_phasor*amp_scaler

        gnm_spl_scaled = amp_scaler * gnm_spl

        gnm_mode_res = H_sv @ gnm_spl_scaled

        gnm_total_res_list.append(gnm_mode_res)

        synthetic_suite_info[mode_number] = mode_info

    gnm_total_res = np.sum(gnm_total_res_list, axis=0)


    # noised ~ (nrealisations, Nt, Ng)
    noised = Perturbation_Generate(gnm_total_res, 
                                        n_realisations=20,
                                        temporal_z="ar1",
                                        tau=1.5,
                                        dt=dt_years)
    
    perturbed_signals = list(noised)
    queue = [gnm_total_res] + perturbed_signals

    for mode_number in mode_numbers:

        DMD_recovery[mode_number]["DMD_noised"] = {}
        DMD_recovery[mode_number]["DMD_noised"]["similarity"] = []
        DMD_recovery[mode_number]["DMD_noised"]["eigenvalue"] = []
    
    for queue_position, gnm_mode_res in tqdm(enumerate(queue)):
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
        gnm_input_total_windowed = Truncate_Gauss_Coeffs(gnm_input_total_windowed, tmax=Nmax)

        sv_input_total_all_steps = A_r @ gnm_input_total_windowed.T


        sv_input_total = sv_input_total_all_steps[:,::n_skip]


        # ------------------------------------------------------
        # APPLY DMD AND RECOVER MODES
        # ---------------------------------------------------------

        dmd = DMD(svd_rank=-1)
        dmd.fit(sv_input_total)

        for mode_number in synthetic_suite_info:

            mode_info = synthetic_suite_info[mode_number]

            target_period = mode_info["true_period"]
            sv_phasor = mode_info["sv_phasor"]

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

            if match_made and result=="no_noise":
                DMD_recovery[mode_number]["DMD"] = {}
                DMD_recovery[mode_number]["DMD"]["similarity"] = similarity_high_score
                DMD_recovery[mode_number]["DMD"]["eigenvalue"] = candidate_eigs[match_idx]
            elif match_made and result=="noised":
                DMD_recovery[mode_number]["DMD_noised"]["similarity"].append(similarity_high_score)
                DMD_recovery[mode_number]["DMD_noised"]["eigenvalue"].append(candidate_eigs[match_idx])
            elif result=="no_noise":
                DMD_recovery[mode_number]["DMD"] = False


# %% visualising:
plt.scatter((2*np.pi)/DMD_recovery[mode_number]["DMD"]["eigenvalue"].imag, target_period)
sim = DMD_recovery[mode_number]["DMD"]["similarity"]
print(f"similarity is: {sim}")      
# %%
from matplotlib.lines import Line2D

fig, ax = plt.subplots(figsize=(8, 6))

mode_keys = list(DMD_recovery.keys())
cmap = plt.get_cmap("tab20")
mode_colours = {
    mode_number: cmap(i % cmap.N)
    for i, mode_number in enumerate(mode_keys)
}

for mode_number, results in DMD_recovery.items():

    colour = mode_colours[mode_number]
    true_period = results["true_period"]

    # Exact mode: circle
    ax.scatter(
        true_period,
        1,
        color=colour,
        marker="o",
        s=55,
        zorder=4,
    )

    # DMD without noise: square
    dmd_result = results.get("DMD", False)

    if dmd_result:
        eig = dmd_result["eigenvalue"]
        period = 2 * np.pi / np.abs(eig.imag)

        ax.scatter(
            period,
            dmd_result["similarity"],
            color=colour,
            marker="s",
            s=55,
            zorder=4,
        )

    # Noised DMD ensemble members: crosses
    noised_result = results.get("DMD_noised", {})
    noised_eigs = noised_result.get("eigenvalue", [])
    noised_similarities = np.asarray(
        noised_result.get("similarity", [])
    )

    if len(noised_eigs) > 0:

        noised_periods = np.asarray([
            2 * np.pi / np.abs(eig.imag)
            for eig in noised_eigs
        ])

        ax.scatter(
            noised_periods,
            noised_similarities,
            color=colour,
            marker="x",
            s=35,
            alpha=0.45,
            zorder=2,
        )

        # Noised ensemble median: triangle
        ax.scatter(
            np.median(noised_periods),
            np.median(noised_similarities),
            color=colour,
            edgecolor="black",
            linewidth=0.7,
            marker="^",
            s=85,
            zorder=5,
        )


# Legend describing marker meanings
marker_handles = [
    Line2D(
        [0], [0],
        marker="o", linestyle="none",
        markerfacecolor="grey", markeredgecolor="grey",
        markersize=7, label="Exact",
    ),
    Line2D(
        [0], [0],
        marker="s", linestyle="none",
        markerfacecolor="grey", markeredgecolor="grey",
        markersize=7, label="DMD, no noise",
    ),
    Line2D(
        [0], [0],
        marker="x", linestyle="none",
        color="grey",
        markersize=7, label="DMD, noised",
    ),
    Line2D(
        [0], [0],
        marker="^", linestyle="none",
        markerfacecolor="grey", markeredgecolor="black",
        markersize=8, label="Noised median",
    ),
]

marker_legend = ax.legend(
    handles=marker_handles,
    title="Result",
    loc="lower right",
)
ax.add_artist(marker_legend)


# Legend describing mode colours
mode_handles = [
    Line2D(
        [0], [0],
        marker="o",
        linestyle="none",
        markerfacecolor=mode_colours[mode_number],
        markeredgecolor=mode_colours[mode_number],
        markersize=7,
        label=f"Mode {mode_number}",
    )
    for mode_number in mode_keys
]

ax.legend(
    handles=mode_handles,
    title="Mode",
    loc="center left",
    bbox_to_anchor=(1.02, 0.5),
)

ax.set_xlabel("Recovered period [yr]")
ax.set_ylabel("Phasor similarity")
ax.set_ylim(0, 1.05)
#ax.set_xlim(0, 100)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
# %%
