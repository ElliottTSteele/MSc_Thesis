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

# ---------------------------------------------------------
# SIMULATION SETUP
# ---------------------------------------------------------

# put list of mode numbers used
mode_numbers = [ 1, 6, 13, 21, 32, 45, 53, 60]
mode_numbers = [str(num) for num in mode_numbers]

# if including covariance-based perturbation ensemble
ensemble_flag = True
n_realisations = 10

# Temporal correlation model passed to Perturbation_Generate.
# This function is assumed to use the CHAOS covariance matrices already
# available through the synthetic setup utilities.
noise_temporal_model = "independent"
noise_tau_years = 1.5

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

all_at_once = True


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
                                                n_realisations=n_realisations,
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
                    "recovered_period": [],
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

                dmd = DMD(svd_rank=-1)
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
                    candidate_eigs = np.asarray(
                        recovered_dict["continuous_eigenvalues"]
                    )

                    candidate_modes = recovered_dict["phasors"]

                    # Calculate finite oscillatory periods
                    candidate_periods = np.full(
                        len(candidate_eigs),
                        np.nan,
                        dtype=float,
                    )

                    for idx, eig in enumerate(candidate_eigs):

                        if np.abs(eig.imag) > 0:

                            candidate_periods[idx] = (
                                2 * np.pi
                                / np.abs(eig.imag)
                            )

                    # Only admit sensible oscillatory DMD modes
                    relevant_period_mask = (
                        np.isfinite(candidate_periods)
                        & (candidate_periods > 1e-5)
                        & (candidate_periods < 1000)
                    )

                    candidate_indices = np.where(
                        relevant_period_mask
                    )[0]

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
                            "recovered_period": candidate_periods[match_idx],
                        }

                    elif match_made and result == "noised":

                        DMD_recovery[mode_number]["DMD_noised"][truncation_degree][
                            "similarity"
                        ].append(similarity_high_score)

                        DMD_recovery[mode_number]["DMD_noised"][truncation_degree][
                            "eigenvalue"
                        ].append(candidate_eigs[match_idx])

                        DMD_recovery[mode_number]["DMD_noised"][truncation_degree][
                            "recovered_period"
                        ].append(candidate_periods[match_idx])
else:
    # ------------------------------------------------------
    # OBTAIN INPUT DATA AND RUN EACH MODE INDEPENDENTLY
    # ---------------------------------------------------------

    file_path = Path(FELIX_DIR) / "R_splines_arbitrary.h5"

    # ---------------------------------------------------------
    # STORAGE
    # ---------------------------------------------------------

    DMD_recovery = {}

    with h5py.File(file_path, "r") as h5_file:

        # =====================================================
        # LOOP OVER MODES
        #
        # IMPORTANT:
        # Each mode is now completely independent.
        #
        # We:
        #   1. Load one theoretical mode
        #   2. Apply its resolution
        #   3. Optionally add noise to that mode alone
        #   4. Run DMD on that mode alone
        #   5. Match DMD modes only to that theoretical mode
        #
        # Only the final recovery metrics are combined later.
        # =====================================================

        for mode_number in tqdm(mode_numbers, desc="Modes"):

            mode_number = str(mode_number)

            # -------------------------------------------------
            # INITIALISE MODE STORAGE
            # -------------------------------------------------

            DMD_recovery[mode_number] = {
                "DMD": {},
                "DMD_noised": {},
            }

            for truncation_degree in degree_truncations_tested:

                DMD_recovery[mode_number]["DMD_noised"][
                    truncation_degree
                ] = {
                    "similarity": [],
                    "eigenvalue": [],
                    "recovered_period": [],
                }

            # -------------------------------------------------
            # LOAD THIS MODE ONLY
            # -------------------------------------------------

            mode_data = Component_Load_SV(mode_number)

            eigenvalue = mode_data["eigenvalue"]
            gnm_phasor = mode_data["gnm"]

            # True theoretical period
            true_period = (
                2 * np.pi
                / np.abs(eigenvalue.imag)
            )

            DMD_recovery[mode_number][
                "true_period"
            ] = true_period

            # -------------------------------------------------
            # THEORETICAL PHASOR
            #
            # Keep full n <= Nmax representation here.
            # Degree selection happens separately inside
            # the truncation loop.
            # -------------------------------------------------

            gnm_phasor_Nmax = Truncate_Gauss_Coeffs(
                gnm_phasor,
                tmax=Nmax,
                tmin=1,
            )

            # -------------------------------------------------
            # LOAD RESOLVED SPLINE FOR THIS MODE ONLY
            # -------------------------------------------------

            gnm_spl = np.asarray(
                h5_file[
                    f"mode_{mode_number}/without_decay"
                ][()]
            )

            # Apply amplitude scaling
            amp_scaler = mode_amp_scalings[
                mode_number
            ]

            gnm_phasor_Nmax = (
                amp_scaler
                * gnm_phasor_Nmax
            )

            gnm_spl_scaled = (
                amp_scaler
                * gnm_spl
            )

            # -------------------------------------------------
            # APPLY CHAOS RESOLUTION TO THIS MODE ONLY
            # -------------------------------------------------

            gnm_mode_res = (
                H_sv
                @ gnm_spl_scaled
            )

            # -------------------------------------------------
            # CREATE MODE-SPECIFIC QUEUE
            #
            # Crucially, noise is also generated independently
            # around this one mode rather than around a sum of
            # all modes.
            # -------------------------------------------------

            if ensemble_flag:

                noised = Perturbation_Generate(
                    gnm_mode_res,
                    n_realisations=n_realisations,
                    temporal_z=noise_temporal_model,
                    tau=noise_tau_years,
                    dt=dt_years,
                )

                queue = [
                    ("no_noise", gnm_mode_res)
                ]

                queue += [
                    ("noised", realisation)
                    for realisation in noised
                ]

            else:

                queue = [
                    ("no_noise", gnm_mode_res)
                ]

            # =================================================
            # DEGREE-SELECTION LOOP
            # =================================================

            for truncation_degree in degree_truncations_tested:

                # -------------------------------------------------
                # CURRENT DEGREE BAND
                #
                # Your present experiment is using four-degree
                # bands:
                #
                #   1-4, 5-8, 9-12, ...
                #
                # because:
                #
                #   tmax = truncation_degree
                #   tmin = truncation_degree - 3
                # -------------------------------------------------

                tmax_current = truncation_degree
                tmin_current = max(
                    1,
                    truncation_degree - 3,
                )

                # ---------------------------------------------
                # SYNTHESIS MATRIX FOR THIS DEGREE BAND
                # ---------------------------------------------

                A_r_current = Truncate_Gauss_Coeffs(
                    A_20_dict["r"],
                    tmax=tmax_current,
                    tmin=tmin_current,
                )

                # ---------------------------------------------
                # THEORETICAL SPATIAL PHASOR FOR THIS BAND
                # ---------------------------------------------

                gnm_phasor_current = (
                    Truncate_Gauss_Coeffs(
                        gnm_phasor_Nmax,
                        tmax=tmax_current,
                        tmin=tmin_current,
                    )
                )

                sv_phasor_current = (
                    A_r_current
                    @ gnm_phasor_current.T
                )

                # =============================================
                # RUN EACH REALISATION INDEPENDENTLY
                # =============================================

                for result_type, gnm_input in queue:

                    # -----------------------------------------
                    # TIME WINDOW
                    # -----------------------------------------

                    if high_q_flag:

                        gnm_input_windowed = (
                            gnm_input[
                                good_record_slice,
                                :
                            ]
                        )

                    else:

                        gnm_input_windowed = (
                            gnm_input
                        )

                    # -----------------------------------------
                    # SELECT DEGREE BAND
                    # -----------------------------------------

                    gnm_input_current = (
                        Truncate_Gauss_Coeffs(
                            gnm_input_windowed,
                            tmax=tmax_current,
                            tmin=tmin_current,
                        )
                    )

                    # -----------------------------------------
                    # GAUSS -> SV GRID
                    # -----------------------------------------

                    sv_input_all_steps = (
                        A_r_current
                        @ gnm_input_current.T
                    )

                    # Temporal subsampling
                    sv_input = (
                        sv_input_all_steps[
                            :,
                            ::n_skip
                        ]
                    )

                    # -----------------------------------------
                    # DMD
                    #
                    # This DMD sees ONLY ONE theoretical
                    # input mode.
                    # -----------------------------------------

                    dmd = DMD(svd_rank=-1)
                    dmd.fit(sv_input)

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

                    # -----------------------------------------
                    # HANDLE EMPTY DMD RESULT
                    # -----------------------------------------

                    if len(candidate_eigs) == 0:
                        continue

                    # -----------------------------------------
                    # CANDIDATE PERIODS
                    # -----------------------------------------

                    candidate_periods = np.full(
                        len(candidate_eigs),
                        np.nan,
                        dtype=float,
                    )

                    for idx, eig in enumerate(
                        candidate_eigs
                    ):

                        if np.abs(eig.imag) > 0:

                            candidate_periods[idx] = (
                                2 * np.pi
                                / np.abs(eig.imag)
                            )

                    # Only sensible oscillatory periods
                    relevant_period_mask = (
                        np.isfinite(
                            candidate_periods
                        )
                        & (
                            candidate_periods
                            > 1e-5
                        )
                        & (
                            candidate_periods
                            < 1000
                        )
                    )

                    candidate_indices = np.where(
                        relevant_period_mask
                    )[0]

                    # -----------------------------------------
                    # FIND BEST SPATIAL MATCH
                    # -----------------------------------------

                    similarity_high_score = -np.inf
                    match_idx = None

                    for idx in candidate_indices:

                        recovered_mode = (
                            candidate_modes[:, idx]
                        )

                        sim_score = (
                            Complex_Phasor_Compare(
                                recovered_mode,
                                sv_phasor_current,
                            )
                        )

                        if (
                            sim_score
                            > similarity_high_score
                        ):

                            similarity_high_score = (
                                sim_score
                            )

                            match_idx = idx

                    # -----------------------------------------
                    # STORE RESULT
                    # -----------------------------------------

                    if match_idx is None:
                        continue

                    if result_type == "no_noise":

                        DMD_recovery[
                            mode_number
                        ]["DMD"][
                            truncation_degree
                        ] = {

                            "similarity":
                                similarity_high_score,

                            "eigenvalue":
                                candidate_eigs[
                                    match_idx
                                ],

                            "recovered_period":
                                candidate_periods[
                                    match_idx
                                ],

                            "degree_min":
                                tmin_current,

                            "degree_max":
                                tmax_current,
                        }

                    elif result_type == "noised":

                        DMD_recovery[
                            mode_number
                        ]["DMD_noised"][
                            truncation_degree
                        ]["similarity"].append(
                            similarity_high_score
                        )

                        DMD_recovery[
                            mode_number
                        ]["DMD_noised"][
                            truncation_degree
                        ]["eigenvalue"].append(
                            candidate_eigs[
                                match_idx
                            ]
                        )

                        DMD_recovery[
                            mode_number
                        ]["DMD_noised"][
                            truncation_degree
                        ]["recovered_period"].append(
                            candidate_periods[
                                match_idx
                            ]
                        )


    print(
        f"Completed independent DMD tests "
        f"for {len(DMD_recovery)} modes."
    )
#

# %% ------------------------------------------------------
# PLOT DMD RECOVERY AS FUNCTION OF DEGREE BAND
# ---------------------------------------------------------

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import BoundaryNorm
from matplotlib.lines import Line2D

degrees = np.asarray(
    sorted(degree_truncations_tested),
    dtype=int,
)

# ---------------------------------------------------------
# DISCRETE VIRIDIS COLOUR MAPPING
# ---------------------------------------------------------

cmap = plt.colormaps["viridis"].resampled(
    len(degrees)
)

if len(degrees) > 1:
    degree_step = np.median(
        np.diff(degrees)
    )
else:
    degree_step = 1

boundaries = np.concatenate([
    [degrees[0] - degree_step / 2],
    (degrees[:-1] + degrees[1:]) / 2,
    [degrees[-1] + degree_step / 2],
])

norm = BoundaryNorm(
    boundaries,
    cmap.N,
)


# ---------------------------------------------------------
# DEGREE-DEPENDENT DRAWING WEIGHTS
#
# Lowest-degree bands are plotted first and most prominently.
# Higher-degree bands are drawn later (on top), but with
# progressively thinner uncertainty lines and smaller symbols.
# ---------------------------------------------------------

degree_rank = {
    degree: rank
    for rank, degree in enumerate(degrees)
}

n_degree_levels = max(len(degrees) - 1, 1)


def degree_style(degree):
    """Return linewidth/marker sizes that decrease with degree."""
    fraction = degree_rank[degree] / n_degree_levels

    return {
        "interval_lw": 3.0 - 1.8 * fraction,
        "median_size": 70.0 - 38.0 * fraction,
        "cross_size": 85.0 - 45.0 * fraction,
        "cross_lw": 2.2 - 0.8 * fraction,
        "outline_lw": 4.2 - 1.4 * fraction,
    }


def finite_quantiles(values, quantiles=(0.05, 0.5, 0.95)):
    """
    Return requested quantiles after dropping non-finite values.

    Returns None if no finite ensemble members remain.
    """
    values = np.asarray(
        values,
        dtype=float,
    )

    values = values[
        np.isfinite(values)
    ]

    if values.size == 0:
        return None

    return np.quantile(
        values,
        quantiles,
    )


# =========================================================
# FIGURE 1: SPATIAL SIMILARITY VS TRUE PERIOD
#
# Drawing order:
#   1. Lowest degree first, highest degree last
#   2. p05-p95 interval
#   3. ensemble median square
#   4. exact/no-perturbation cross on top
#
# Degree styling:
#   lower degree = thicker interval + larger symbols
#   higher degree = thinner interval + smaller symbols
# =========================================================

fig, ax = plt.subplots(
    figsize=(7.5, 5.5)
)

for degree in degrees:

    colour = cmap(
        norm(degree)
    )

    style = degree_style(
        degree
    )

    for mode_number, results in DMD_recovery.items():

        true_period = results[
            "true_period"
        ]

        # ---------------------------------------------
        # COVARIANCE-PERTURBED ENSEMBLE FIRST
        # ---------------------------------------------
        noised_result = results.get(
            "DMD_noised",
            {},
        ).get(
            degree,
            {},
        )

        similarity_q = finite_quantiles(
            noised_result.get(
                "similarity",
                [],
            )
        )

        if similarity_q is not None:

            p05, median, p95 = similarity_q

            ax.vlines(
                true_period,
                p05,
                p95,
                color=colour,
                linewidth=style["interval_lw"],
                alpha=0.85,
                zorder=1 + degree_rank[degree],
            )

            ax.scatter(
                true_period,
                median,
                marker="s",
                s=style["median_size"],
                color=colour,
                edgecolors="black",
                linewidths=0.35,
                zorder=20 + degree_rank[degree],
            )

        # ---------------------------------------------
        # EXACT / NO-PERTURBATION RESULT ON TOP
        # ---------------------------------------------
        exact_result = results.get(
            "DMD",
            {},
        ).get(
            degree,
            None,
        )

        if exact_result is not None:

            exact_similarity = exact_result[
                "similarity"
            ]

            if np.isfinite(
                exact_similarity
            ):

                # White under-stroke gives the cross a visible outline
                # against either the median square or another degree colour.
                ax.scatter(
                    true_period,
                    exact_similarity,
                    marker="x",
                    s=style["cross_size"],
                    linewidths=style["outline_lw"],
                    color="white",
                    zorder=40 + degree_rank[degree],
                )

                ax.scatter(
                    true_period,
                    exact_similarity,
                    marker="x",
                    s=style["cross_size"],
                    linewidths=style["cross_lw"],
                    color=colour,
                    zorder=41 + degree_rank[degree],
                )


ax.set_xlabel(
    "True period (years)"
)

ax.set_ylabel(
    "Spatial similarity"
)

ax.set_ylim(
    0,
    1.05,
)

ax.grid(
    alpha=0.25
)

marker_handles = [
    Line2D(
        [0],
        [0],
        marker="x",
        linestyle="None",
        markeredgecolor="black",
        markeredgewidth=2.0,
        color="white",
        markersize=8,
        label="No perturbation",
    ),
    Line2D(
        [0],
        [0],
        marker="s",
        linestyle="None",
        markerfacecolor="black",
        markeredgecolor="black",
        markersize=6,
        label="Perturbed median",
    ),
    Line2D(
        [0],
        [0],
        linestyle="-",
        color="black",
        linewidth=2.0,
        label="Perturbed p05-p95",
    ),
]

ax.legend(
    handles=marker_handles,
    loc="best",
)

sm = cm.ScalarMappable(
    cmap=cmap,
    norm=norm,
)

sm.set_array([])

cbar = fig.colorbar(
    sm,
    ax=ax,
    boundaries=boundaries,
    ticks=degrees,
    spacing="uniform",
)

cbar.set_label(
    "Upper spherical harmonic degree of 4-degree band"
)

fig.tight_layout()
plt.show()


# =========================================================
# FIGURE 2: SIGNED PERIOD ERROR VS TRUE PERIOD
#
# Same layering convention as Figure 1.
# =========================================================

fig, ax = plt.subplots(
    figsize=(7.5, 5.5)
)

for degree in degrees:

    colour = cmap(
        norm(degree)
    )

    style = degree_style(
        degree
    )

    for mode_number, results in DMD_recovery.items():

        true_period = results[
            "true_period"
        ]

        # ---------------------------------------------
        # COVARIANCE-PERTURBED ENSEMBLE FIRST
        # ---------------------------------------------
        noised_result = results.get(
            "DMD_noised",
            {},
        ).get(
            degree,
            {},
        )

        recovered_periods = np.asarray(
            noised_result.get(
                "recovered_period",
                [],
            ),
            dtype=float,
        )

        if recovered_periods.size == 0:

            noised_eigs = noised_result.get(
                "eigenvalue",
                [],
            )

            recovered_periods = np.asarray([
                (
                    2 * np.pi
                    / np.abs(eig.imag)
                )
                if np.abs(eig.imag) > 0
                else np.nan
                for eig in noised_eigs
            ])

        period_errors = (
            100
            * (
                recovered_periods
                - true_period
            )
            / true_period
        )

        period_q = finite_quantiles(
            period_errors
        )

        if period_q is not None:

            p05, median, p95 = period_q

            ax.vlines(
                true_period,
                p05,
                p95,
                color=colour,
                linewidth=style["interval_lw"],
                alpha=0.85,
                zorder=1 + degree_rank[degree],
            )

            ax.scatter(
                true_period,
                median,
                marker="s",
                s=style["median_size"],
                color=colour,
                edgecolors="black",
                linewidths=0.35,
                zorder=20 + degree_rank[degree],
            )

        # ---------------------------------------------
        # EXACT / NO-PERTURBATION RESULT ON TOP
        # ---------------------------------------------
        exact_result = results.get(
            "DMD",
            {},
        ).get(
            degree,
            None,
        )

        if exact_result is not None:

            if (
                "recovered_period"
                in exact_result
            ):

                recovered_period = exact_result[
                    "recovered_period"
                ]

            else:

                eig = exact_result[
                    "eigenvalue"
                ]

                if np.abs(
                    eig.imag
                ) > 0:

                    recovered_period = (
                        2
                        * np.pi
                        / np.abs(
                            eig.imag
                        )
                    )

                else:

                    recovered_period = np.nan

            if np.isfinite(
                recovered_period
            ):

                exact_period_error = (
                    100
                    * (
                        recovered_period
                        - true_period
                    )
                    / true_period
                )

                ax.scatter(
                    true_period,
                    exact_period_error,
                    marker="x",
                    s=style["cross_size"],
                    linewidths=style["outline_lw"],
                    color="white",
                    zorder=40 + degree_rank[degree],
                )

                ax.scatter(
                    true_period,
                    exact_period_error,
                    marker="x",
                    s=style["cross_size"],
                    linewidths=style["cross_lw"],
                    color=colour,
                    zorder=41 + degree_rank[degree],
                )


ax.axhline(
    0,
    color="black",
    linestyle="--",
    linewidth=1,
)

ax.set_xlabel(
    "True period (years)"
)

ax.set_ylabel(
    "Recovered period error (%)"
)

ax.set_ylim(-100, 100)

ax.grid(
    alpha=0.25
)

ax.legend(
    handles=marker_handles,
    loc="best",
)

sm = cm.ScalarMappable(
    cmap=cmap,
    norm=norm,
)

sm.set_array([])

cbar = fig.colorbar(
    sm,
    ax=ax,
    boundaries=boundaries,
    ticks=degrees,
    spacing="uniform",
)

cbar.set_label(
    "Upper spherical harmonic degree of 4-degree band"
)

fig.tight_layout()
plt.show()


# =========================================================
# OPTIONAL NUMERICAL ENSEMBLE SUMMARY
# =========================================================

print(
    "\nCovariance-perturbation ensemble summary"
)

print(
    f"Requested realisations: {n_realisations}"
)

for degree in degrees:

    n_similarity_matches = 0
    n_period_matches = 0

    for results in DMD_recovery.values():

        noised_result = results.get(
            "DMD_noised",
            {},
        ).get(
            degree,
            {},
        )

        n_similarity_matches += len(
            noised_result.get(
                "similarity",
                [],
            )
        )

        n_period_matches += np.sum(
            np.isfinite(
                np.asarray(
                    noised_result.get(
                        "recovered_period",
                        [],
                    ),
                    dtype=float,
                )
            )
        )

    print(
        f"Degree band "
        f"{max(1, degree - 3)}-{degree}: "
        f"{n_similarity_matches} matched similarity results, "
        f"{n_period_matches} finite period results"
    )

# %%