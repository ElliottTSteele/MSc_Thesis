"""
DMD generic template
"""

# %% FILE SYSTEM AND DEPENDENCY SETUP

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

import h5py
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm
from matplotlib.colors import BoundaryNorm
from matplotlib.lines import Line2D
from pydmd import HankelDMD
from tqdm import tqdm

# Project paths / utilities
from src.msc_thesis.paths import *
from src.msc_thesis.synSetup import *
from src.msc_thesis.synUtils import *
from src.msc_thesis.synDMD import *


# %% ------------------------------------------------------
# USER SETTINGS
# ------------------------------------------------------

# Every mode listed here is included simultaneously in the synthetic dataset.
mode_numbers = [1, 6, 13, 21, 32, 45, 53, 60]
mode_numbers = [str(mode_number) for mode_number in mode_numbers]


# ---------------------------------------------------------
# PERTURBATION ENSEMBLE
# ---------------------------------------------------------

ensemble_flag = True
n_realisations = 10

noise_temporal_model = "independent"
noise_tau_years = 1.5


# ---------------------------------------------------------
# TIME / DEGREE SETTINGS
# ---------------------------------------------------------

# True = just use high quality record
high_q_flag = True
# indexing of which time sample spacing to use
n_skip = 3

# max spherical harmonic degree used
Nmax = 12

# %% ------------------------------------------------------
# BUILD ONE COMBINED SYNTHETIC DATASET
# ------------------------------------------------------

file_path = Path(FELIX_DIR) / "R_splines_arbitrary.h5"

A_r_Nmax = Truncate_Gauss_Coeffs(
    A_20_dict["r"],
    tmax=Nmax,
)

DMD_recovery = {}
synthetic_suite_info = {}
gnm_total_res_list = []

with h5py.File(file_path, "r") as h5_file:

    for mode_number in tqdm(
        mode_numbers,
        desc="Loading combined modes",
    ):

        mode_data = Component_Load_SV(
            mode_number
        )

        eigenvalue = mode_data["eigenvalue"]
        gnm_phasor = mode_data["gnm"]

        true_period = (
            2.0
            * np.pi
            / np.abs(eigenvalue.imag)
        )

        gnm_phasor_Nmax = Truncate_Gauss_Coeffs(
            gnm_phasor,
            tmax=Nmax,
        )

        try:
            amp_scaler = mode_amp_scalings[
                str(mode_number)
            ]
        except KeyError as exc:
            raise KeyError(
                f"No amplitude scaling found for mode {mode_number}."
            ) from exc

        gnm_phasor_scaled = (
            amp_scaler
            * gnm_phasor_Nmax
        )

        dataset_name = (
            f"mode_{mode_number}/without_decay"
        )

        if dataset_name not in h5_file:
            raise KeyError(
                f"Missing HDF5 dataset: {dataset_name}"
            )

        gnm_spline = np.asarray(
            h5_file[dataset_name][()]
        )

        gnm_spline_scaled = (
            amp_scaler
            * gnm_spline
        )

        # Existing project resolution mapping.
        gnm_mode_res = (
            H_sv
            @ gnm_spline_scaled
        )

        gnm_total_res_list.append(
            gnm_mode_res
        )

        synthetic_suite_info[
            mode_number
        ] = {
            "true_period": true_period,
            "true_eigenvalue": eigenvalue,
            "gnm_phasor": gnm_phasor_scaled,
        }

        DMD_recovery[
            mode_number
        ] = {
            "true_period": true_period,
            "true_eigenvalue": eigenvalue,
            "DMD": {},
            "DMD_noised": {},
        }

gnm_total_res = np.sum(
    np.asarray(gnm_total_res_list),
    axis=0,
)

del gnm_total_res_list


# %% ------------------------------------------------------
# GENERATE PERTURBED COMBINED SIGNALS
# ------------------------------------------------------

if ensemble_flag:

    noised = Perturbation_Generate(
        gnm_total_res,
        n_realisations=n_realisations,
        temporal_z=noise_temporal_model,
        tau=noise_tau_years,
        dt=dt_years,
    )

    queue = [
        ("no_noise", gnm_total_res)
    ]

    queue.extend(
        ("noised", realisation)
        for realisation in noised
    )

else:

    queue = [
        ("no_noise", gnm_total_res)
    ]


for mode_number in mode_numbers:

    DMD_recovery[
        mode_number
    ]["DMD_noised"]= {
        "similarity": [],
        "eigenvalue": [],
        "recovered_period": [],
    }


# %% ------------------------------------------------------
# DEGREE-BAND HANKEL DMD TESTS
# ------------------------------------------------------

# skipping time steps (as very dense) - ensure DMD understands
# equivalent real time between steps
dt_snapshot = (
    n_skip
    * dt_years
)

# Getting the A_r projection operator for up to Nmax
A_r_current = Truncate_Gauss_Coeffs(
    A_20_dict["r"],
    tmax=Nmax,
)

# Precompute every target theoretical phasor once for this degree band.
target_phasors = {}

for mode_number in mode_numbers:

    gnm_target = (
        synthetic_suite_info[
            mode_number
        ]["gnm_phasor"]
    )

    gnm_target_band = Truncate_Gauss_Coeffs(
        gnm_target,
        tmax=Nmax,
    )

    target_phasors[
        mode_number
    ] = (
        A_r_current
        @ gnm_target_band.T
    )

for result_type, gnm_input in tqdm(
    queue,
    desc=f"Degrees up to Nmax = {Nmax}",
    leave=False,
):

    # ---------------------------------------------
    # TIME WINDOW
    # ---------------------------------------------

    if high_q_flag:

        gnm_input_windowed = (
            gnm_input[
                good_record_slice,
                :
            ]
        )

    else:

        gnm_input_windowed = gnm_input

    # ---------------------------------------------
    # SELECT CURRENT DEGREE BAND
    # ---------------------------------------------

    gnm_input_band = Truncate_Gauss_Coeffs(
        gnm_input_windowed,
        tmax=Nmax,
    )

    # ---------------------------------------------
    # GAUSS -> SV GRID
    # ---------------------------------------------

    sv_input_all_steps = (
        A_r_current
        @ gnm_input_band.T
    )

    # Temporal subsampling.
    sv_input = (
        sv_input_all_steps[
            :,
            ::n_skip
        ]
    )

    n_physical = (
        sv_input.shape[0]
    )

    n_snapshots = (
        sv_input.shape[1]
    )

    # ---------------------------------------------
    # FIT HANKEL DMD
    # ---------------------------------------------

    dmd = build_hankel_dmd()

    dmd.fit(
        sv_input
    )

    # Extract / pair modes ONCE per fitted dataset, then compare that
    # recovered candidate set to every known theoretical input mode.

    # HANKEL REQUIRES SPATIAL DEAGUMENTATION!!!
    # OPDMD SHOULD NOT HAVE D-C CONVERSION!!!
    (
        candidate_eigs,
        candidate_modes,
        candidate_periods,
    ) = extract_hankel_candidates(
        dmd=dmd,
        dt_snapshot=dt_snapshot,
        n_physical=n_physical,
    )

    # ---------------------------------------------
    # MATCH TO EVERY KNOWN INPUT MODE
    # ---------------------------------------------

    for mode_number in mode_numbers:

        match = best_spatial_match(
            candidate_modes=candidate_modes,
            candidate_eigs=candidate_eigs,
            candidate_periods=candidate_periods,
            target_phasor=target_phasors[
                mode_number
            ],
        )

        # No eligible oscillatory candidate / no finite similarity.
        if match is None:
            continue

        if result_type == "no_noise":

            DMD_recovery[
                mode_number
            ]["DMD"]= {
                **match,
                "degree_max": Nmax,
            }

        else:

            noised_store = (
                DMD_recovery[
                    mode_number
                ]["DMD_noised"]
            )

            noised_store[
                "similarity"
            ].append(
                match["similarity"]
            )

            noised_store[
                "eigenvalue"
            ].append(
                match["eigenvalue"]
            )

            noised_store[
                "recovered_period"
            ].append(
                match["recovered_period"]
            )

