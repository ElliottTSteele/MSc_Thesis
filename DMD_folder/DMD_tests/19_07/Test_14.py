'''
Plan:

investigating simple optimised dmd
on a single wave case
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
from pydmd import BOPDMD

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


            # positive imaginary part means positive frequency.
            if eig_i.imag > 0:
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

    # BOPDMD eigenvalues are already continuous-time eigenvalues.
    continuous_eigenvalues = output_eigenvalues

    return {
        "phasors": output_phasors,
        "continuous_eigenvalues": continuous_eigenvalues,
        "mode_information": mode_information,
    }

# ---------------------------------------------------------
# SIMULATION SETUP
# ---------------------------------------------------------

# put list of mode numbers used
mode_numbers = [ 6, 13, 21, 45, 48, 54, 57, 61]
mode_numbers = [str(num) for num in mode_numbers]

# if including perturbation ensemble
ensemble_flag = False


# if windowing to 'high quality' record
high_q_flag = False
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

    if ensemble_flag:
        # noised ~ (nrealisations, Nt, Ng)
        noised = Perturbation_Generate(gnm_total_res, 
                                            n_realisations=20,
                                            temporal_z="ar1",
                                            tau=1.5,
                                            dt=dt_years)
        
        perturbed_signals = list(noised)
        queue = [gnm_total_res] + perturbed_signals
    else:
        queue = [gnm_total_res]

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

        # Time coordinates corresponding to the retained snapshots.
        dt_dmd = n_skip * dt_years

        t_dmd = (
            np.arange(sv_input_total.shape[1])
            * dt_dmd
        )

        # Two DMD modes are required for each real oscillatory input:
        # one positive- and one negative-frequency mode.
        bop_rank = 2 * len(mode_numbers)

        # num_trials=0 gives optimized DMD without bagging.
        dmd = BOPDMD(
            svd_rank=bop_rank,
            num_trials=0,
            #trial_size=0.8,
            #remove_bad_bags=True,
            eig_constraints={
                "imag",
                "conjugate_pairs",
            },
            varpro_opts_dict={
                "maxiter": 1000,
                "tol": 0.1,
                "verbose": True,
                "maxlam": 200
            },
        )

        t_dmd = (
            np.arange(sv_input_total.shape[1])
            * n_skip
            * dt_years
        )

        # Time translation changes amplitudes/phases, but not eigenvalues.
        t_dmd = t_dmd - np.mean(t_dmd)

        # Global scaling preserves spatial relationships.
        data_scale = np.linalg.norm(sv_input_total)
        X_bop = sv_input_total / data_scale

        dmd.fit(
            X_bop,
            t_dmd,
        )

        for mode_number in synthetic_suite_info:

            mode_info = synthetic_suite_info[mode_number]

            target_period = mode_info["true_period"]
            sv_phasor = mode_info["sv_phasor"]

            # getting recovered modes
            recovered_dict = DMD_Mode_Pair(dmd, n_skip * dt_years)

            # find optimal fit
            candidate_eigs = recovered_dict["continuous_eigenvalues"]
            candidate_periods = np.asarray([
                (2 * np.pi) / np.abs(eig.imag)
                if not np.isclose(eig.imag, 0)
                else np.inf
                for eig in candidate_eigs
            ])

            relevant_period_mask = (
                (candidate_periods > 1e-5)
                & (candidate_periods < 1000)
            )

            candidate_indices = np.where(
                relevant_period_mask
            )[0]

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
ax.set_xlim(0, 100)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
# %%
# %% -----------------------------------------------------
# CROSS-MODE SPATIAL/PHASOR SIMILARITY
# ---------------------------------------------------------

mode_labels = list(synthetic_suite_info.keys())
n_modes = len(mode_labels)

similarity_matrix = np.full(
    (n_modes, n_modes),
    np.nan,
    dtype=float,
)

for i, mode_i in enumerate(mode_labels):
    phasor_i = synthetic_suite_info[mode_i]["sv_phasor"]

    for j, mode_j in enumerate(mode_labels):
        phasor_j = synthetic_suite_info[mode_j]["sv_phasor"]

        similarity_matrix[i, j] = Complex_Phasor_Compare(
            phasor_i,
            phasor_j,
        )


# ---------------------------------------------------------
# Identify potentially pathological mode pairs
# ---------------------------------------------------------

similarity_threshold = 0.80

pathology_pairs = []

for i in range(1, n_modes):
    for j in range(i):

        similarity = similarity_matrix[i, j]

        if similarity >= similarity_threshold:
            pathology_pairs.append({
                "mode_1": mode_labels[j],
                "mode_2": mode_labels[i],
                "period_1": synthetic_suite_info[
                    mode_labels[j]
                ]["true_period"],
                "period_2": synthetic_suite_info[
                    mode_labels[i]
                ]["true_period"],
                "similarity": similarity,
            })

# Sort from most to least similar
pathology_pairs = sorted(
    pathology_pairs,
    key=lambda result: result["similarity"],
    reverse=True,
)

print(
    f"Mode pairs with phasor similarity >= "
    f"{similarity_threshold:.2f}:"
)

if pathology_pairs:
    for result in pathology_pairs:
        print(
            f"Modes {result['mode_1']:>2} and "
            f"{result['mode_2']:>2}: "
            f"similarity={result['similarity']:.3f}, "
            f"periods={result['period_1']:.2f} and "
            f"{result['period_2']:.2f} yr"
        )
else:
    print("None found.")


# ---------------------------------------------------------
# Lower-triangular heat map
# ---------------------------------------------------------

# Mask only the upper triangle; retain the diagonal.
upper_triangle_mask = np.triu(
    np.ones_like(similarity_matrix, dtype=bool),
    k=1,
)

plot_matrix = np.ma.array(
    similarity_matrix,
    mask=upper_triangle_mask,
)

# Include both mode number and period in the tick labels.
tick_labels = [
    (
        f"Mode {mode}\n"
        f"{synthetic_suite_info[mode]['true_period']:.2f} yr"
    )
    for mode in mode_labels
]

fig, ax = plt.subplots(
    figsize=(max(8, 1.15 * n_modes),
             max(7, 1.00 * n_modes))
)

cmap = plt.colormaps["magma"].copy()
cmap.set_bad(color="white")

image = ax.imshow(
    plot_matrix,
    origin="upper",
    cmap=cmap,
    vmin=0,
    vmax=1,
    interpolation="nearest",
)

ax.set_xticks(np.arange(n_modes))
ax.set_yticks(np.arange(n_modes))

ax.set_xticklabels(
    tick_labels,
    rotation=45,
    ha="right",
)

ax.set_yticklabels(tick_labels)

ax.set_xlabel("Synthetic input mode")
ax.set_ylabel("Synthetic input mode")
ax.set_title(
    "Cross-mode spatial phasor similarity"
)

# Annotate each visible cell.
for i in range(n_modes):
    for j in range(i + 1):

        similarity = similarity_matrix[i, j]

        text_colour = (
            "black" if similarity > 0.65 else "white"
        )

        ax.text(
            j,
            i,
            f"{similarity:.2f}",
            ha="center",
            va="center",
            color=text_colour,
            fontsize=9,
        )

# Draw cell boundaries.
ax.set_xticks(
    np.arange(-0.5, n_modes, 1),
    minor=True,
)
ax.set_yticks(
    np.arange(-0.5, n_modes, 1),
    minor=True,
)

ax.grid(
    which="minor",
    color="white",
    linewidth=0.7,
)

ax.tick_params(which="minor", bottom=False, left=False)

colourbar = fig.colorbar(
    image,
    ax=ax,
    fraction=0.046,
    pad=0.04,
)

colourbar.set_label(
    r"$|\phi_i^H\phi_j|/"
    r"(\|\phi_i\|_2\|\phi_j\|_2)$"
)

fig.tight_layout()
plt.show()