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
mode_numbers = [58]

# if including perturbation ensemble
ensemble_flag = False

# if windowing to 'high quality' record
high_q_flag = True
n_skip = 3

# deciding on spherical harmonic truncation degree
Nmax = 12
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

    for mode_number in tqdm(mode_numbers):

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


        # noised ~ (nrealisations, Nt, Ng)
        noised = Perturbation_Generate(gnm_mode_res, 
                                            n_realisations=10,
                                            temporal_z="ar1",
                                            dt=dt_years)
        
        perturbed_signals = list(noised)
        queue = [gnm_mode_res] + perturbed_signals

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

            target_period = true_period

            # getting recovered modes
            recovered_dict = DMD_Mode_Pair(dmd, n_skip * dt_years)

            # find optimal fit
            candidate_eigs = recovered_dict["continuous_eigenvalues"]
            candidate_indices = np.arange(0, len(candidate_eigs))
            candidate_periods = np.asarray([(2*np.pi)/eig.imag for eig in candidate_eigs])

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
            else:
                DMD_recovery[mode_number]["DMD"] = False


# %% visualising:
plt.scatter((2*np.pi)/DMD_recovery[mode_number]["DMD"]["eigenvalue"].imag, target_period)
sim = DMD_recovery[mode_number]["DMD"]["similarity"]
print(f"similarity is: {sim}")      

# %%
fig, ax = plt.subplots(figsize=(7, 5))

for mode_number, results in DMD_recovery.items():

    true_period = results["true_period"]

    # Exact result
    ax.scatter(
        true_period, 1,
        color="black", marker="o",
        label="Exact" if mode_number == list(DMD_recovery)[0] else None,
    )

    # DMD without noise
    if results["DMD"]:
        eig = results["DMD"]["eigenvalue"]
        period = 2 * np.pi / np.abs(eig.imag)

        ax.scatter(
            period, results["DMD"]["similarity"],
            color="black", marker="s",
            label="DMD, no noise" if mode_number == list(DMD_recovery)[0] else None,
        )



    # Noised DMD realisations
    noised_eigs = results["DMD_noised"]["eigenvalue"]
    noised_similarities = np.asarray(
        results["DMD_noised"]["similarity"]
    )

    if len(noised_eigs) > 0:

        noised_periods = np.array([
            2 * np.pi / np.abs(eig.imag)
            for eig in noised_eigs
        ])

        ax.scatter(
            noised_periods,
            noised_similarities,
            color="blue",
            marker="x",
            alpha=0.4,
            label="DMD, noised",
        )

        # Centre of the noised ensemble
        ax.scatter(
            np.median(noised_periods),
            np.median(noised_similarities),
            color="blue",
            edgecolor="black",
            marker="o",
            s=70,
            label="Noised median",
            zorder=5,
        )
ax.set_xlabel("Period [yr]")
ax.set_ylabel("Phasor similarity")
ax.set_ylim(0, 1.05)
ax.grid(True, alpha=0.3)
ax.legend()

plt.tight_layout()
plt.show()
# %%
