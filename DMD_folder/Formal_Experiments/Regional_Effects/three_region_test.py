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
mode_numbers = np.arange(40, 63, 1)
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
degree_truncations_tested = np.arange(4, 21, 4)

all_at_once = False


#  %% -----------------------------------------------------
# OBTAIN INPUT DATA
# ---------------------------------------------------------

if all_at_once:
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
                gnm_input_total_windowed = Truncate_Gauss_Coeffs(gnm_input_total_windowed, tmax=truncation_degree,
                                                                tmin = truncation_degree-3)
                A_r = Truncate_Gauss_Coeffs(A_20_dict["r"], tmax=truncation_degree,
                                            tmin = truncation_degree-3)

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
                    gnm_phasor_truncation_degree = Truncate_Gauss_Coeffs(gnm_phasor, tmax=truncation_degree,
                                                                        tmin = truncation_degree-3)
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
else:
    # ------------------------------------------------------
    # REGIONAL DEGREE-BAND DMD RECOVERY
    # ------------------------------------------------------

    file_path = Path(FELIX_DIR) / "R_splines_arbitrary.h5"

    # Assumes A_r grid ordering is (latitude, longitude), flattened
    nlon = 360
    nlat = A_20_dict["r"].shape[0] // nlon
    latitudes = np.linspace(90, -90, nlat)

    lat_grid = np.repeat(latitudes, nlon)

    region_masks = {
        "North": lat_grid > 30,
        "Equator": (lat_grid >= -30) & (lat_grid <= 30),
        "South": lat_grid < -30,
    }

    DMD_recovery_region = {
        region: {}
        for region in region_masks
    }


    with h5py.File(file_path, "r") as h5_file:

        for mode_number in tqdm(mode_numbers, desc="Modes"):

            mode_number = str(mode_number)

            mode_data = Component_Load_SV(mode_number)

            eigenvalue = mode_data["eigenvalue"]
            true_period = 2 * np.pi / np.abs(eigenvalue.imag)

            gnm_phasor = Truncate_Gauss_Coeffs(
                mode_data["gnm"],
                tmax=Nmax,
            )

            amp_scaler = mode_amp_scalings[mode_number]
            gnm_phasor *= amp_scaler

            gnm_spl = np.asarray(
                h5_file[
                    f"mode_{mode_number}/without_decay"
                ][()]
            )

            gnm_mode_res = H_sv @ (
                amp_scaler * gnm_spl
            )

            if ensemble_flag:

                noised = Perturbation_Generate(
                    gnm_mode_res,
                    n_realisations=10,
                    temporal_z="ar1",
                    tau=1.5,
                    dt=dt_years,
                )

                queue = [("no_noise", gnm_mode_res)]
                queue += [
                    ("noised", x)
                    for x in noised
                ]

            else:

                queue = [
                    ("no_noise", gnm_mode_res)
                ]


            for region in region_masks:

                DMD_recovery_region[
                    region
                ][mode_number] = {
                    "true_period": true_period,
                    "DMD": {},
                    "DMD_noised": {},
                }

                for degree in degree_truncations_tested:

                    DMD_recovery_region[
                        region
                    ][mode_number][
                        "DMD_noised"
                    ][degree] = {
                        "similarity": [],
                        "eigenvalue": [],
                    }


            # --------------------------------------------------
            # DEGREE BANDS: 1-4, 5-8, 9-12, 13-16, 17-20
            # --------------------------------------------------

            for truncation_degree in degree_truncations_tested:

                tmax_current = truncation_degree
                tmin_current = max(
                    1,
                    truncation_degree - 3,
                )

                A_r_current = Truncate_Gauss_Coeffs(
                    A_20_dict["r"],
                    tmax=tmax_current,
                    tmin=tmin_current,
                )

                gnm_phasor_current = Truncate_Gauss_Coeffs(
                    gnm_phasor,
                    tmax=tmax_current,
                    tmin=tmin_current,
                )

                # Full-grid theoretical phasor
                sv_phasor_full = (
                    A_r_current
                    @ gnm_phasor_current.T
                )


                for result_type, gnm_input in queue:

                    if high_q_flag:
                        gnm_input = gnm_input[
                            good_record_slice, :
                        ]

                    gnm_input_current = Truncate_Gauss_Coeffs(
                        gnm_input,
                        tmax=tmax_current,
                        tmin=tmin_current,
                    )

                    # Full-grid resolved time series
                    sv_input_full = (
                        A_r_current
                        @ gnm_input_current.T
                    )

                    sv_input_full = (
                        sv_input_full[:, ::n_skip]
                    )


                    # ==========================================
                    # RUN DMD SEPARATELY IN EACH REGION
                    # ==========================================

                    for region, region_mask in region_masks.items():

                        sv_input_region = (
                            sv_input_full[
                                region_mask, :
                            ]
                        )

                        sv_phasor_region = (
                            sv_phasor_full[
                                region_mask
                            ]
                        )

                        dmd = DMD()
                        dmd.fit(sv_input_region)

                        recovered_dict = DMD_Mode_Pair(
                            dmd,
                            n_skip * dt_years,
                        )

                        candidate_eigs = np.asarray(
                            recovered_dict[
                                "continuous_eigenvalues"
                            ]
                        )

                        candidate_modes = (
                            recovered_dict["phasors"]
                        )

                        candidate_periods = np.asarray([
                            2 * np.pi / np.abs(eig.imag)
                            if np.abs(eig.imag) > 0
                            else np.inf
                            for eig in candidate_eigs
                        ])

                        candidate_indices = np.where(
                            (candidate_periods > 1e-5)
                            & (candidate_periods < 1000)
                        )[0]

                        similarity_high_score = -np.inf
                        match_idx = None

                        for idx in candidate_indices:

                            sim_score = Complex_Phasor_Compare(
                                candidate_modes[:, idx],
                                sv_phasor_region,
                            )

                            if sim_score > similarity_high_score:

                                similarity_high_score = sim_score
                                match_idx = idx


                        if match_idx is None:
                            continue

                        results = DMD_recovery_region[
                            region
                        ][mode_number]

                        if result_type == "no_noise":

                            results["DMD"][
                                truncation_degree
                            ] = {
                                "similarity":
                                    similarity_high_score,

                                "eigenvalue":
                                    candidate_eigs[match_idx],

                                "recovered_period":
                                    candidate_periods[match_idx],
                            }

                        else:

                            results[
                                "DMD_noised"
                            ][truncation_degree][
                                "similarity"
                            ].append(
                                similarity_high_score
                            )

                            results[
                                "DMD_noised"
                            ][truncation_degree][
                                "eigenvalue"
                            ].append(
                                candidate_eigs[match_idx]
                            )
#
# %% ------------------------------------------------------
# PLOT REGIONAL RECOVERY
# ------------------------------------------------------

from matplotlib import cm
from matplotlib.colors import BoundaryNorm

degrees = np.asarray(
    sorted(degree_truncations_tested),
    dtype=int,
)

cmap = plt.colormaps[
    "viridis"
].resampled(len(degrees))

degree_step = np.median(
    np.diff(degrees)
)

boundaries = np.concatenate([
    [degrees[0] - degree_step / 2],
    (degrees[:-1] + degrees[1:]) / 2,
    [degrees[-1] + degree_step / 2],
])

norm = BoundaryNorm(
    boundaries,
    cmap.N,
)

regions = [
    "North",
    "Equator",
    "South",
]


# %% ------------------------------------------------------
# SIMILARITY
# ------------------------------------------------------

fig, axes = plt.subplots(
    1, 3,
    figsize=(15, 4.5),
    sharex=True,
    sharey=True,
)

for ax, region in zip(
    axes,
    regions,
):

    for degree in degrees:

        periods = []
        similarities = []

        for mode_number, results in (
            DMD_recovery_region[
                region
            ].items()
        ):

            if degree not in results["DMD"]:
                continue

            periods.append(
                results["true_period"]
            )

            similarities.append(
                results["DMD"][
                    degree
                ]["similarity"]
            )

        periods = np.asarray(periods)
        similarities = np.asarray(
            similarities
        )

        sort_idx = np.argsort(
            periods
        )[::-1]

        ax.plot(
            periods[sort_idx],
            similarities[sort_idx],
            "o-",
            linewidth=1.2,
            markersize=4,
            color=cmap(norm(degree)),
        )

    ax.set_title(region)
    ax.set_xlabel(
        "True period (years)"
    )
    ax.grid(alpha=0.25)

axes[0].set_ylabel(
    "Spatial similarity"
)

axes[0].set_ylim(
    0, 1.05
)

sm = cm.ScalarMappable(
    cmap=cmap,
    norm=norm,
)

cbar = fig.colorbar(
    sm,
    ax=axes,
    boundaries=boundaries,
    ticks=degrees,
    spacing="uniform",
    pad=0.02,
)

cbar.set_label(
    "Maximum degree of 4-degree band"
)

fig.subplots_adjust(
    right=0.90,
    wspace=0.10,
)

plt.show()


# %% ------------------------------------------------------
# PERIOD ERROR
# ------------------------------------------------------

fig, axes = plt.subplots(
    1, 3,
    figsize=(15, 4.5),
    sharex=True,
    sharey=True,
)

for ax, region in zip(
    axes,
    regions,
):

    for degree in degrees:

        periods = []
        period_errors = []

        for mode_number, results in (
            DMD_recovery_region[
                region
            ].items()
        ):

            if degree not in results["DMD"]:
                continue

            true_period = (
                results["true_period"]
            )

            recovered_period = (
                results["DMD"][
                    degree
                ]["recovered_period"]
            )

            error = (
                100
                * (
                    recovered_period
                    - true_period
                )
                / true_period
            )

            periods.append(
                true_period
            )

            period_errors.append(
                error
            )

        periods = np.asarray(periods)

        period_errors = np.asarray(
            period_errors
        )

        sort_idx = np.argsort(
            periods
        )[::-1]

        ax.plot(
            periods[sort_idx],
            period_errors[sort_idx],
            "o-",
            linewidth=1.2,
            markersize=4,
            color=cmap(norm(degree)),
        )

    ax.axhline(
        0,
        color="black",
        linestyle="--",
        linewidth=1,
    )

    ax.set_title(region)

    ax.set_xlabel(
        "True period (years)"
    )
    ax.set_ylim(-50, 50)

    ax.grid(alpha=0.25)

axes[0].set_ylabel(
    "Recovered period error (%)"
)

sm = cm.ScalarMappable(
    cmap=cmap,
    norm=norm,
)

cbar = fig.colorbar(
    sm,
    ax=axes,
    boundaries=boundaries,
    ticks=degrees,
    spacing="uniform",
    pad=0.02,
)

cbar.set_label(
    "Maximum degree of 4-degree band"
)

fig.subplots_adjust(
    right=0.90,
    wspace=0.10,
)

plt.show()
# %%
