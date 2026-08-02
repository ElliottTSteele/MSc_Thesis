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
from matplotlib.ticker import MaxNLocator
from pydmd.utils import pseudo_hankel_matrix
from tqdm import tqdm

# Project paths / utilities
from src.msc_thesis.paths import *
from src.msc_thesis.synSetup import *
from src.msc_thesis.synUtils import *
from src.msc_thesis.synDMD import *
from src.msc_thesis.synVideo import (
    prepare_recovery_video_data,
    make_dmd_recovery_video,
)


# %% ------------------------------------------------------
# USER SETTINGS
# ------------------------------------------------------

# Every mode listed here is included simultaneously in the synthetic dataset.

# sparse case
# mode_numbers = [1, 6, 13, 21, 32, 45, 53, 60]

# dense case
mode_numbers = [ 3,  4,  6, 14, 18, 21, 25, 29, 32,\
                 36, 40, 45, 48, 54, 59, 60, 62]

mode_numbers = [ 1,  2,  3,  4,  6, 11, 15, 16, 18,\
                 20, 21, 25, 29, 30, 32, 33, 34, 36,\
                 37, 38, 39, 41, 43, 45, 48, 53, 54,\
                 57, 59, 60, 61, 62]


# mode_numbers = np.arange(50, 63, 1)

mode_numbers = [str(mode_number) for mode_number in mode_numbers]

DMD_algorithm = "opdmd"

# Optional time-delay embedding applied independently of the DMD algorithm.
hankel_embedding_flag = True
hankel_d = 10
hankel_reconstruction_method = "first"

# Choose whether the DMD rank is selected from the existing noise threshold
# or passed directly to PyDMD using its native svd_rank conventions.
svd_rank_method = "noise_threshold"
# Applied only in noise-threshold mode, then rounded to the nearest integer
# (with half-integer results rounded up).
noise_rank_multiplier = 1.0
fixed_svd_rank = 18

# Plot the cumulative variance explained before any SVD truncation is applied.
# In fixed-rank mode these spectra are diagnostic only and do not alter the
# PyDMD rank argument. Ideal, resolved, and perturbed runs are compared.
singular_value_plot_flag = True

video_plot=True
# Above this number of active inputs, group duplicate best-match candidates
# into compact video rows and append retained candidates with no input match.
video_compact_input_threshold = 8

up_lim_yr_plot = 100

filter_flag = False

# Include analytical ideal candidates in the all-modes period-power plot.
# Disabled by default because the extra candidate set can clutter the figure.
show_ideal_all_modes = False

# Optional finite-period window for matching, retained DMD candidates, and
# plotting. The DMD fit itself always uses the complete combined signal.
period_limit_flag = False
period_lower_bound = 1.01
period_upper_bound = 10

if period_limit_flag:
    if not (
        np.isfinite(period_lower_bound)
        and np.isfinite(period_upper_bound)
        and 1.0 < period_lower_bound
        < period_upper_bound
    ):
        raise ValueError(
            "With period_limit_flag=True, bounds must be finite and "
            "satisfy 1 < period_lower_bound < period_upper_bound. "
            "The lower bound must exceed 1 year so the requested "
            "one-year plot margin remains positive on logarithmic axes."
        )

# ---------------------------------------------------------
# PERTURBATION ENSEMBLE
# ---------------------------------------------------------

ensemble_flag = False
n_realisations = 1

# Number of noise-only realisations used when the noise threshold is selected
# or its singular-value diagnostics are plotted.
n_noise_rank_realisations = n_realisations
noise_rank_quantile = 0.95

noise_temporal_model = "independent"
noise_tau_years = 6


# ---------------------------------------------------------
# TIME / DEGREE SETTINGS
# ---------------------------------------------------------

# True = just use high quality record
high_q_flag = True
# indexing of which time sample spacing to use
n_skip = 1

# max spherical harmonic degree used
Nmax = 10

# When enabled, run cumulative spherical-harmonic truncations 1:n for every
# integer n up to Nmax. The final Nmax run also supplies all established plots.
degree_sweep_flag = False

if (
    not isinstance(
        Nmax,
        (int, np.integer),
    )
    or Nmax < 1
):
    raise ValueError(
        "Nmax must be a positive integer."
    )

degree_truncations_tested = np.arange(
    1,
    Nmax + 1,
    dtype=int,
)

if svd_rank_method not in (
    "noise_threshold",
    "fixed",
):
    raise ValueError(
        "svd_rank_method must be either "
        "'noise_threshold' or 'fixed'. "
        f"Received {svd_rank_method!r}."
    )

if (
    isinstance(
        noise_rank_multiplier,
        (bool, np.bool_),
    )
    or not isinstance(
        noise_rank_multiplier,
        (
            int,
            float,
            np.integer,
            np.floating,
        ),
    )
    or not np.isfinite(
        noise_rank_multiplier
    )
    or noise_rank_multiplier <= 0.0
):
    raise ValueError(
        "noise_rank_multiplier must be a finite positive number. "
        f"Received {noise_rank_multiplier!r}."
    )

if svd_rank_method == "fixed":
    fixed_rank_is_integer = (
        isinstance(
            fixed_svd_rank,
            (int, np.integer),
        )
        and not isinstance(
            fixed_svd_rank,
            (bool, np.bool_),
        )
    )
    fixed_rank_is_energy_fraction = (
        isinstance(
            fixed_svd_rank,
            (float, np.floating),
        )
        and np.isfinite(
            fixed_svd_rank
        )
        and 0.0 < fixed_svd_rank < 1.0
    )

    if not (
        (
            fixed_rank_is_integer
            and fixed_svd_rank >= -1
        )
        or fixed_rank_is_energy_fraction
    ):
        raise ValueError(
            "With svd_rank_method='fixed', fixed_svd_rank must "
            "follow the PyDMD conventions: -1 for no truncation, "
            "0 for the PyDMD optimal rank, a positive integer, or "
            "a floating-point energy fraction strictly between 0 and 1. "
            f"Received {fixed_svd_rank!r}."
        )
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
gnm_total_ideal_list = []
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

        gnm_phasor_degree_20 = Truncate_Gauss_Coeffs(
            gnm_phasor,
            tmax=20,
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

        # Analytical ideal SV: evaluate the scaled SV phasor directly.
        # This path deliberately applies no P, R, or H_sv operator.
        gnm_mode_ideal = G_Time_Series_Eval(
            amp_scaler
            * gnm_phasor_degree_20,
            eigenvalue,
        )

        gnm_total_ideal_list.append(
            gnm_mode_ideal
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
            "gnm_ideal": gnm_mode_ideal,
            "gnm_resolved": gnm_mode_res,
        }

        DMD_recovery[
            mode_number
        ] = {
            "true_period": true_period,
            "true_eigenvalue": eigenvalue,
            "DMD_ideal": {},
            "DMD": {},
            "DMD_noised": {},
        }

gnm_total_ideal = np.sum(
    np.asarray(gnm_total_ideal_list),
    axis=0,
)

gnm_total_res = np.sum(
    np.asarray(gnm_total_res_list),
    axis=0,
)

del gnm_total_ideal_list
del gnm_total_res_list


# %% ------------------------------------------------------
# GENERATE THE REQUIRED NOISE BANK
# ------------------------------------------------------

noise_rank_diagnostics_required = (
    svd_rank_method == "noise_threshold"
    or singular_value_plot_flag
)

if noise_rank_diagnostics_required:
    if (
        not isinstance(
            n_noise_rank_realisations,
            (int, np.integer),
        )
        or n_noise_rank_realisations < 1
    ):
        raise ValueError(
            "n_noise_rank_realisations must be a positive integer "
            "when noise-threshold rank selection or singular-value "
            "plotting is enabled."
        )

    if not (
        np.isfinite(noise_rank_quantile)
        and 0.0 < noise_rank_quantile < 1.0
    ):
        raise ValueError(
            "noise_rank_quantile must be finite and lie strictly "
            "between zero and one."
        )

if ensemble_flag and (
    not isinstance(
        n_realisations,
        (int, np.integer),
    )
    or n_realisations < 1
):
    raise ValueError(
        "n_realisations must be a positive integer when "
        "ensemble_flag=True."
    )

noise_bank_size = max(
    (
        n_noise_rank_realisations
        if noise_rank_diagnostics_required
        else 0
    ),
    n_realisations if ensemble_flag else 0,
)

if noise_bank_size > 0:
    noise_only_bank = Perturbation_Generate(
        gnm_total_res,
        n_realisations=noise_bank_size,
        temporal_z=noise_temporal_model,
        tau=noise_tau_years,
        dt=dt_years,
        just_noise=True,
    )
else:
    noise_only_bank = np.empty(
        (
            0,
            *gnm_total_res.shape,
        ),
        dtype=float,
    )

noise_only_rank_bank = noise_only_bank[
    :(
        n_noise_rank_realisations
        if noise_rank_diagnostics_required
        else 0
    )
]

queue = [
    ("ideal", gnm_total_ideal),
    ("no_noise", gnm_total_res),
]

if ensemble_flag:
    noised = (
        noise_only_bank[:n_realisations]
        + gnm_total_res[
            None,
            :,
            :,
        ]
    )
    queue.extend(
        ("noised", realisation)
        for realisation in noised
    )


for mode_number in mode_numbers:

    DMD_recovery[
        mode_number
    ]["DMD_noised"]= {
        "similarity": [],
        "eigenvalue": [],
        "recovered_period": [],
    }


# %% ------------------------------------------------------
# DMD TESTS
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

if period_limit_flag:
    matching_mode_numbers = [
        mode_number
        for mode_number in mode_numbers
        if (
            period_lower_bound
            <= synthetic_suite_info[
                mode_number
            ]["true_period"]
            <= period_upper_bound
        )
    ]
else:
    matching_mode_numbers = (
        mode_numbers.copy()
    )

clean_candidate_eigs = None
clean_candidate_modes = None
clean_candidate_periods = None
clean_candidate_ids = None
video_sum_candidate_eigs = None
video_sum_candidate_modes = None
video_sum_candidate_ids = None
ideal_candidate_eigs = None
ideal_candidate_modes = None
ideal_candidate_periods = None
effective_svd_ranks = {
    "ideal": None,
    "resolved": None,
    "perturbed": [],
}
singular_value_cumulative_variance = {
    "ideal": None,
    "resolved": None,
    "perturbed": [],
    "noise_only": [],
}
singular_value_magnitudes = {
    "ideal": None,
    "resolved": None,
    "perturbed": [],
    "noise_only": [],
}

if DMD_algorithm not in (
    "exact",
    "fbdmd",
    "opdmd",
):
    raise ValueError(
        "DMD_algorithm must be one of "
        "'exact', 'fbdmd', or 'opdmd'. "
        f"Received {DMD_algorithm!r}."
    )

if (
    hankel_embedding_flag
    and (
        not isinstance(
            hankel_d,
            (int, np.integer),
        )
        or hankel_d < 1
    )
):
    raise ValueError(
        "hankel_d must be a positive integer when "
        "Hankel embedding is enabled."
    )


def prepare_sv_input_for_dmd(
    gnm_input,
    truncation_degree,
    projection_operator,
):
    """Apply the common temporal, degree, grid, and sampling operations."""

    if high_q_flag:
        gnm_input_windowed = (
            gnm_input[
                good_record_slice,
                :
            ]
        )
    else:
        gnm_input_windowed = gnm_input

    if filter_flag:
        gnm_input_windowed, _, _ = (
            Long_Period_Taper_Filter(
                gnm_input_windowed,
                dt=dt_years,
                pass_period=10.0,
                stop_period=20.0,
                axis=0,
            )
        )

    gnm_input_band = Truncate_Gauss_Coeffs(
        gnm_input_windowed,
        tmax=truncation_degree,
    )

    sv_input_all_steps = (
        projection_operator
        @ gnm_input_band.T
    )

    return (
        sv_input_all_steps[
            :,
            ::n_skip
        ]
    )


def calculate_pretruncation_spectrum(
    sv_input,
    result_label,
):
    """Calculate the spectrum used for rank selection or diagnostics."""

    n_physical = (
        sv_input.shape[0]
    )

    n_snapshots = (
        sv_input.shape[1]
    )

    if (
        hankel_embedding_flag
        and (
            not isinstance(
                hankel_d,
                (int, np.integer),
            )
            or hankel_d < 1
        )
    ):
        raise ValueError(
            "hankel_d must be a positive integer when "
            "Hankel embedding is enabled."
        )

    if (
        hankel_embedding_flag
        and hankel_d > n_snapshots
    ):
        raise ValueError(
            f"hankel_d={hankel_d} exceeds the "
            f"{n_snapshots} available snapshots."
        )

    if hankel_embedding_flag:
        svd_input = pseudo_hankel_matrix(
            sv_input,
            d=hankel_d,
        )
    else:
        svd_input = sv_input

    if DMD_algorithm in ("exact", "fbdmd"):
        # These estimators choose the truncation rank from the leading
        # snapshot matrix X. FbDMD's backward calculation uses the
        # shifted Y matrix at that already-selected rank.
        svd_input = svd_input[:, :-1]

    singular_value_magnitude = np.abs(
        np.linalg.svd(
            svd_input,
            compute_uv=False,
        )
    )
    singular_value_variance = np.square(
        singular_value_magnitude
    )
    total_singular_value_variance = np.sum(
        singular_value_variance
    )

    if (
        not np.isfinite(
            total_singular_value_variance
        )
        or total_singular_value_variance <= 0.0
    ):
        raise ValueError(
            "Cannot calculate a singular spectrum for "
            f"{result_label!r} because its total squared "
            "singular-value magnitude is not finite and positive."
        )

    cumulative_variance = np.cumsum(
        singular_value_variance
    ) / total_singular_value_variance

    return (
        singular_value_magnitude,
        cumulative_variance,
    )


def fit_dmd_candidate_suite(
    sv_input,
    requested_svd_rank,
):
    """Fit DMD and return its candidates and effective fitted rank."""

    n_physical = (
        sv_input.shape[0]
    )
    n_snapshots = (
        sv_input.shape[1]
    )

    if DMD_algorithm == "exact":
        base_dmd = build_exact_dmd(
            svd_rank=requested_svd_rank
        )

    elif DMD_algorithm == "fbdmd":
        base_dmd = build_fbdmd(
            svd_rank=requested_svd_rank
        )

    elif DMD_algorithm == "opdmd":
        base_dmd = build_bopdmd(
            svd_rank=requested_svd_rank
        )

    else:
        raise ValueError(
            "DMD_algorithm must be one of "
            "'exact', 'fbdmd', or 'opdmd'. "
            f"Received {DMD_algorithm!r}."
        )

    dmd = apply_hankel_embedding(
        base_dmd,
        enabled=hankel_embedding_flag,
        d=hankel_d,
        reconstruction_method=
            hankel_reconstruction_method,
    )

    embedding_d = (
        hankel_d
        if hankel_embedding_flag
        else None
    )

    if DMD_algorithm == "opdmd":
        n_fit_snapshots = (
            n_snapshots
            - hankel_d
            + 1
            if hankel_embedding_flag
            else n_snapshots
        )

        fit_times = (
            np.arange(
                n_fit_snapshots
            )
            * dt_snapshot
        )

        dmd.fit(
            sv_input,
            fit_times,
        )

        effective_svd_rank = int(
            np.asarray(
                dmd.modes
            ).shape[1]
        )
        candidate_suite = extract_optimized_dmd_candidates(
            dmd=dmd,
            n_physical=n_physical,
            embedding_d=embedding_d,
        )

    else:
        dmd.fit(
            sv_input
        )

        effective_svd_rank = int(
            np.asarray(
                dmd.modes
            ).shape[1]
        )
        candidate_suite = extract_standard_dmd_candidates(
            dmd=dmd,
            dt_snapshot=dt_snapshot,
            n_physical=n_physical,
            embedding_d=embedding_d,
        )

    if effective_svd_rank < 1:
        raise RuntimeError(
            "The fitted DMD estimator returned no modes, so an "
            "effective SVD rank could not be recorded."
        )

    return (
        *candidate_suite,
        effective_svd_rank,
    )


def select_degree_svd_rank(
    truncation_degree,
    projection_operator,
):
    """Apply the existing noise-threshold rank rule at one degree."""

    resolved_sv_input = prepare_sv_input_for_dmd(
        gnm_total_res,
        truncation_degree=truncation_degree,
        projection_operator=projection_operator,
    )
    (
        resolved_magnitudes,
        _,
    ) = calculate_pretruncation_spectrum(
        resolved_sv_input,
        result_label=(
            "clean resolved signal at spherical-harmonic "
            f"truncation {truncation_degree}"
        ),
    )

    noise_leading_values = []

    for noise_index, noise_realisation in enumerate(
        noise_only_rank_bank
    ):
        noise_sv_input = prepare_sv_input_for_dmd(
            noise_realisation,
            truncation_degree=truncation_degree,
            projection_operator=projection_operator,
        )
        (
            noise_magnitudes,
            _,
        ) = calculate_pretruncation_spectrum(
            noise_sv_input,
            result_label=(
                "noise-only realisation "
                f"{noise_index} at spherical-harmonic "
                f"truncation {truncation_degree}"
            ),
        )
        noise_leading_values.append(
            noise_magnitudes[0]
        )

    noise_threshold = float(
        np.quantile(
            np.asarray(
                noise_leading_values,
                dtype=float,
            ),
            noise_rank_quantile,
        )
    )

    base_noise_svd_rank = int(
        np.count_nonzero(
            resolved_magnitudes
            > noise_threshold
        )
    )

    if base_noise_svd_rank == 0:
        raise RuntimeError(
            "The noise-threshold rank criterion retained no resolved "
            "singular values at spherical-harmonic truncation "
            f"n={truncation_degree}. No DMD fit was attempted for "
            f"that degree. The threshold was {noise_threshold:.6e}, "
            f"derived from the q={noise_rank_quantile:.3f} quantile "
            f"of {n_noise_rank_realisations} noise-only leading "
            "singular values."
        )

    selected_svd_rank = int(
        np.floor(
            base_noise_svd_rank
            * noise_rank_multiplier
            + 0.5
        )
    )

    if selected_svd_rank < 1:
        raise RuntimeError(
            "The noise-selected SVD rank became zero after applying "
            f"noise_rank_multiplier={noise_rank_multiplier:g} at "
            f"spherical-harmonic truncation n={truncation_degree}. "
            f"The unmultiplied rank was {base_noise_svd_rank}."
        )

    return (
        selected_svd_rank,
        base_noise_svd_rank,
        noise_threshold,
    )


def make_degree_sweep_candidate_store(
    raw_candidate_periods,
    retained_candidate_periods,
):
    """Store retained periods and the raw infinite-period candidates."""

    raw_candidate_periods = np.asarray(
        raw_candidate_periods,
        dtype=float,
    )

    return {
        "recovered_period": np.asarray(
            retained_candidate_periods,
            dtype=float,
        ).copy(),
        "static_recovered_period": (
            raw_candidate_periods[
                np.isinf(
                    raw_candidate_periods
                )
            ].copy()
        ),
    }


degree_sweep_results = None

if degree_sweep_flag:
    degree_sweep_results = {
        "metadata": {
            "degree_definition": (
                "cumulative spherical-harmonic degrees 1:n"
            ),
            "period_limit_enabled": bool(
                period_limit_flag
            ),
            "period_lower_bound": (
                float(period_lower_bound)
                if period_limit_flag
                else None
            ),
            "period_upper_bound": (
                float(period_upper_bound)
                if period_limit_flag
                else None
            ),
            "svd_rank_selection": (
                (
                    "number of clean resolved singular values strictly "
                    "greater than the selected quantile of noise-only "
                    "leading singular values, multiplied by "
                    "noise_rank_multiplier and evaluated independently "
                    "at each degree truncation"
                )
                if svd_rank_method == "noise_threshold"
                else (
                    "fixed PyDMD svd_rank argument passed unchanged "
                    "at every degree truncation"
                )
            ),
            "svd_rank_method": svd_rank_method,
            "noise_rank_multiplier": float(
                noise_rank_multiplier
            ),
            "fixed_svd_rank": (
                fixed_svd_rank
                if svd_rank_method == "fixed"
                else None
            ),
            "ensemble_enabled": bool(
                ensemble_flag
            ),
            "n_realisations": (
                int(n_realisations)
                if ensemble_flag
                else 0
            ),
            "noise_rank_quantile": (
                float(
                    noise_rank_quantile
                )
                if noise_rank_diagnostics_required
                else None
            ),
            "noise_rank_realisations": (
                int(
                    n_noise_rank_realisations
                )
                if noise_rank_diagnostics_required
                else 0
            ),
        },
        "degree_truncations": (
            degree_truncations_tested.copy()
        ),
        "by_degree": {},
    }

    # Run every lower truncation here. The existing main workflow below is
    # retained as the sole Nmax run and supplies all established diagnostics.
    for truncation_degree in tqdm(
        degree_truncations_tested[:-1],
        desc="Cumulative degree sweep",
    ):
        truncation_degree = int(
            truncation_degree
        )

        projection_operator = Truncate_Gauss_Coeffs(
            A_20_dict["r"],
            tmax=truncation_degree,
        )

        if svd_rank_method == "noise_threshold":
            (
                degree_requested_svd_rank,
                degree_base_noise_svd_rank,
                degree_noise_threshold,
            ) = select_degree_svd_rank(
                truncation_degree,
                projection_operator,
            )
            degree_noise_selected_svd_rank = int(
                degree_requested_svd_rank
            )
        else:
            degree_requested_svd_rank = (
                fixed_svd_rank
            )
            degree_noise_selected_svd_rank = None
            degree_base_noise_svd_rank = None
            degree_noise_threshold = None

        degree_store = {
            # Retained for compatibility: this is the argument supplied to
            # PyDMD, not necessarily the effective fitted rank for 0 or -1.
            "svd_rank": degree_requested_svd_rank,
            "requested_svd_rank": (
                degree_requested_svd_rank
            ),
            "noise_selected_svd_rank": (
                degree_noise_selected_svd_rank
            ),
            "noise_base_svd_rank": (
                degree_base_noise_svd_rank
            ),
            "effective_svd_rank": {
                "ideal": None,
                "resolved": None,
                "perturbed": [],
            },
            "noise_singular_value_threshold": (
                float(
                    degree_noise_threshold
                )
                if degree_noise_threshold is not None
                else None
            ),
            "ideal": None,
            "resolved": None,
            "perturbed": [],
        }

        perturbed_realisation_index = 0

        for result_type, gnm_input in tqdm(
            queue,
            desc=(
                "Degrees 1-"
                f"{truncation_degree}, "
                "PyDMD rank argument="
                f"{degree_requested_svd_rank}"
            ),
            leave=False,
        ):
            sv_input = prepare_sv_input_for_dmd(
                gnm_input,
                truncation_degree=truncation_degree,
                projection_operator=projection_operator,
            )

            (
                candidate_eigs,
                candidate_modes,
                raw_candidate_periods,
                fitted_effective_svd_rank,
            ) = fit_dmd_candidate_suite(
                sv_input,
                requested_svd_rank=
                    degree_requested_svd_rank,
            )

            candidate_periods = (
                raw_candidate_periods
            )

            if period_limit_flag:
                (
                    candidate_eigs,
                    candidate_modes,
                    candidate_periods,
                ) = filter_candidate_period_range(
                    candidate_eigs,
                    candidate_modes,
                    candidate_periods,
                    lower_period=period_lower_bound,
                    upper_period=period_upper_bound,
                )

            candidate_store = (
                make_degree_sweep_candidate_store(
                    raw_candidate_periods,
                    candidate_periods,
                )
            )

            if result_type == "ideal":
                degree_store[
                    "ideal"
                ] = candidate_store
                degree_store[
                    "effective_svd_rank"
                ][
                    "ideal"
                ] = fitted_effective_svd_rank

            elif result_type == "no_noise":
                degree_store[
                    "resolved"
                ] = candidate_store
                degree_store[
                    "effective_svd_rank"
                ][
                    "resolved"
                ] = fitted_effective_svd_rank

            elif result_type == "noised":
                candidate_store[
                    "realisation_index"
                ] = int(
                    perturbed_realisation_index
                )
                degree_store[
                    "perturbed"
                ].append(
                    candidate_store
                )
                degree_store[
                    "effective_svd_rank"
                ][
                    "perturbed"
                ].append(
                    fitted_effective_svd_rank
                )
                perturbed_realisation_index += 1

            else:
                raise ValueError(
                    f"Unknown DMD result type: {result_type}"
                )

        degree_sweep_results[
            "by_degree"
        ][truncation_degree] = degree_store


# %% ------------------------------------------------------
# MAIN SVD-RANK RESOLUTION
# ------------------------------------------------------

noise_singular_value_threshold = None
noise_selected_svd_rank = None
noise_base_svd_rank = None

if noise_rank_diagnostics_required:
    for (
        result_key,
        result_label,
        gnm_input,
    ) in (
        (
            "ideal",
            "ideal signal",
            gnm_total_ideal,
        ),
        (
            "resolved",
            "clean resolved signal",
            gnm_total_res,
        ),
    ):
        sv_input = prepare_sv_input_for_dmd(
            gnm_input,
            truncation_degree=Nmax,
            projection_operator=A_r_current,
        )
        (
            singular_value_magnitudes[
                result_key
            ],
            singular_value_cumulative_variance[
                result_key
            ],
        ) = calculate_pretruncation_spectrum(
            sv_input,
            result_label=result_label,
        )
        del sv_input

    for noise_index, noise_realisation in enumerate(
        tqdm(
            noise_only_rank_bank,
            desc="Noise-only SVD rank bank",
            leave=False,
        )
    ):
        noise_sv_input = prepare_sv_input_for_dmd(
            noise_realisation,
            truncation_degree=Nmax,
            projection_operator=A_r_current,
        )
        (
            noise_magnitude,
            noise_cumulative_variance,
        ) = calculate_pretruncation_spectrum(
            noise_sv_input,
            result_label=(
                "noise-only realisation "
                f"{noise_index}"
            ),
        )
        singular_value_magnitudes[
            "noise_only"
        ].append(
            noise_magnitude
        )
        singular_value_cumulative_variance[
            "noise_only"
        ].append(
            noise_cumulative_variance
        )
        del noise_sv_input

    noise_leading_singular_values = np.asarray([
        spectrum[0]
        for spectrum in singular_value_magnitudes[
            "noise_only"
        ]
    ])

    noise_singular_value_threshold = float(
        np.quantile(
            noise_leading_singular_values,
            noise_rank_quantile,
        )
    )

if svd_rank_method == "noise_threshold":
    resolved_singular_value_magnitude = (
        singular_value_magnitudes[
            "resolved"
        ]
    )

    noise_base_svd_rank = int(
        np.count_nonzero(
            resolved_singular_value_magnitude
            > noise_singular_value_threshold
        )
    )

    if noise_base_svd_rank == 0:
        raise RuntimeError(
            "The noise-threshold rank criterion retained no resolved "
            "singular values. No DMD fit was attempted. The threshold "
            f"was {noise_singular_value_threshold:.6e}, derived from "
            f"the q={noise_rank_quantile:.3f} quantile of "
            f"{n_noise_rank_realisations} noise-only leading singular "
            "values."
        )

    noise_selected_svd_rank = int(
        np.floor(
            noise_base_svd_rank
            * noise_rank_multiplier
            + 0.5
        )
    )

    if noise_selected_svd_rank < 1:
        raise RuntimeError(
            "The noise-selected SVD rank became zero after applying "
            f"noise_rank_multiplier={noise_rank_multiplier:g}. "
            f"The unmultiplied rank was {noise_base_svd_rank}."
        )

    requested_svd_rank = (
        noise_selected_svd_rank
    )

    print(
        "\nNOISE-THRESHOLD SVD RANK SELECTION\n"
        f"Noise-only realisations: {n_noise_rank_realisations}\n"
        f"Leading-noise quantile: q={noise_rank_quantile:.3f}\n"
        f"Noise singular-value threshold: "
        f"{noise_singular_value_threshold:.6e}\n"
        f"Base noise-selected rank: {noise_base_svd_rank}\n"
        f"Noise-rank multiplier: {noise_rank_multiplier:g}\n"
        f"PyDMD rank argument: {requested_svd_rank}"
    )
else:
    requested_svd_rank = fixed_svd_rank

    print(
        "\nFIXED PYDMD SVD RANK\n"
        f"PyDMD rank argument: {requested_svd_rank}"
    )

    if noise_singular_value_threshold is not None:
        print(
            "Noise singular-value threshold retained for diagnostics: "
            f"{noise_singular_value_threshold:.6e}"
        )

requested_svd_rank_metadata = (
    requested_svd_rank.item()
    if isinstance(
        requested_svd_rank,
        np.generic,
    )
    else requested_svd_rank
)

if degree_sweep_flag:
    degree_sweep_results[
        "by_degree"
    ][int(Nmax)] = {
        "svd_rank": requested_svd_rank_metadata,
        "requested_svd_rank": (
            requested_svd_rank_metadata
        ),
        "noise_selected_svd_rank": (
            noise_selected_svd_rank
        ),
        "noise_base_svd_rank": (
            noise_base_svd_rank
        ),
        "effective_svd_rank": {
            "ideal": None,
            "resolved": None,
            "perturbed": [],
        },
        "noise_singular_value_threshold": (
            float(
                noise_singular_value_threshold
            )
            if noise_singular_value_threshold is not None
            else None
        ),
        "ideal": None,
        "resolved": None,
        "perturbed": [],
    }

# Area weights and time coordinates used by the existing record-mean power
# definition. Candidate powers are evaluated immediately after each perturbed
# DMD fit so complete spatial candidate suites do not need to be retained.
sv_grid_weights = W2D.ravel()

if high_q_flag:
    analysis_times_absolute = (
        times_absolute[
            good_record_slice
        ]
    )
else:
    analysis_times_absolute = (
        times_absolute.copy()
    )

analysis_times_absolute = (
    analysis_times_absolute[
        ::n_skip
    ]
)

truth_power_times = (
    analysis_times_absolute
    - times_absolute[0]
)

dmd_power_times = (
    analysis_times_absolute
    - analysis_times_absolute[0]
)

ensemble_candidate_modal_power_results = {
    "true_period": [],
    "true_power": [],
    "true_mode_number": [],
    "recovered_period": [],
    "recovered_power": [],
    "realisation_index": [],
}

for mode_number in mode_numbers:
    true_eigenvalue = synthetic_suite_info[
        mode_number
    ]["true_eigenvalue"]

    ensemble_candidate_modal_power_results[
        "true_period"
    ].append(
        synthetic_suite_info[
            mode_number
        ]["true_period"]
    )
    ensemble_candidate_modal_power_results[
        "true_power"
    ].append(
        SV_Grid_Phasor_Record_Power(
            target_phasors[
                mode_number
            ],
            true_eigenvalue,
            evaluation_times=truth_power_times,
            spatial_weights=sv_grid_weights,
        )
    )
    ensemble_candidate_modal_power_results[
        "true_mode_number"
    ].append(
        mode_number
    )

noised_realisation_index = 0


# %% ------------------------------------------------------
# DMD FITS
# ------------------------------------------------------

for result_type, gnm_input in tqdm(
    queue,
    desc=(
        f"Degrees up to Nmax = {Nmax}, "
        f"PyDMD rank argument = {requested_svd_rank}"
    ),
    leave=False,
):
    sv_input = prepare_sv_input_for_dmd(
        gnm_input,
        truncation_degree=Nmax,
        projection_operator=A_r_current,
    )
    n_physical = (
        sv_input.shape[0]
    )
    n_snapshots = (
        sv_input.shape[1]
    )

    if analysis_times_absolute.size != n_snapshots:
        raise ValueError(
            "Power-evaluation time count does not match the "
            f"{result_type!r} snapshots supplied to DMD: "
            f"{analysis_times_absolute.size} vs {n_snapshots}."
        )

    if sv_grid_weights.size != n_physical:
        raise ValueError(
            "Spatial power-weight count does not match the DMD "
            f"physical dimension: {sv_grid_weights.size} vs "
            f"{n_physical}."
        )

    if result_type == "noised" and singular_value_plot_flag:
        (
            perturbed_magnitude,
            perturbed_cumulative_variance,
        ) = calculate_pretruncation_spectrum(
            sv_input,
            result_label="perturbed signal",
        )
        singular_value_magnitudes[
            "perturbed"
        ].append(
            perturbed_magnitude
        )
        singular_value_cumulative_variance[
            "perturbed"
        ].append(
            perturbed_cumulative_variance
        )

    (
        candidate_eigs,
        candidate_modes,
        raw_candidate_periods,
        fitted_effective_svd_rank,
    ) = fit_dmd_candidate_suite(
        sv_input,
        requested_svd_rank=requested_svd_rank,
    )

    if result_type == "ideal":
        effective_svd_ranks[
            "ideal"
        ] = fitted_effective_svd_rank
    elif result_type == "no_noise":
        effective_svd_ranks[
            "resolved"
        ] = fitted_effective_svd_rank
    elif result_type == "noised":
        effective_svd_ranks[
            "perturbed"
        ].append(
            fitted_effective_svd_rank
        )

    candidate_periods = (
        raw_candidate_periods
    )
    candidate_ids = np.arange(
        raw_candidate_periods.size,
        dtype=int,
    )

    # The video total follows the finite period window but deliberately retains
    # static candidates. Keep this suite separate so the established matching
    # and plotting behaviour of period_limit_flag is unchanged.
    if result_type == "no_noise":
        if period_limit_flag:
            video_sum_candidate_keep = (
                np.isinf(
                    raw_candidate_periods
                )
                | (
                    np.isfinite(
                        raw_candidate_periods
                    )
                    & (
                        raw_candidate_periods
                        >= period_lower_bound
                    )
                    & (
                        raw_candidate_periods
                        <= period_upper_bound
                    )
                )
            )
        else:
            video_sum_candidate_keep = np.ones(
                raw_candidate_periods.shape,
                dtype=bool,
            )

        video_sum_candidate_eigs = (
            candidate_eigs[
                video_sum_candidate_keep
            ].copy()
        )
        video_sum_candidate_modes = (
            candidate_modes[
                :,
                video_sum_candidate_keep
            ].copy()
        )
        video_sum_candidate_ids = (
            candidate_ids[
                video_sum_candidate_keep
            ].copy()
        )

    if period_limit_flag:
        matching_candidate_keep = (
            np.isfinite(
                candidate_periods
            )
            & (
                candidate_periods
                >= period_lower_bound
            )
            & (
                candidate_periods
                <= period_upper_bound
            )
        )
        (
            candidate_eigs,
            candidate_modes,
            candidate_periods,
        ) = filter_candidate_period_range(
            candidate_eigs,
            candidate_modes,
            candidate_periods,
            lower_period=period_lower_bound,
            upper_period=period_upper_bound,
        )
        candidate_ids = candidate_ids[
            matching_candidate_keep
        ]

    if degree_sweep_flag:
        candidate_store = (
            make_degree_sweep_candidate_store(
                raw_candidate_periods,
                candidate_periods,
            )
        )
        degree_store = degree_sweep_results[
            "by_degree"
        ][int(Nmax)]

        if result_type == "ideal":
            degree_store[
                "ideal"
            ] = candidate_store
            degree_store[
                "effective_svd_rank"
            ][
                "ideal"
            ] = fitted_effective_svd_rank

        elif result_type == "no_noise":
            degree_store[
                "resolved"
            ] = candidate_store
            degree_store[
                "effective_svd_rank"
            ][
                "resolved"
            ] = fitted_effective_svd_rank

        elif result_type == "noised":
            candidate_store[
                "realisation_index"
            ] = int(
                noised_realisation_index
            )
            degree_store[
                "perturbed"
            ].append(
                candidate_store
            )
            degree_store[
                "effective_svd_rank"
            ][
                "perturbed"
            ].append(
                fitted_effective_svd_rank
            )

    if result_type == "noised":
        candidate_powers = np.asarray([
            SV_Grid_Phasor_Record_Power(
                candidate_modes[:, candidate_index],
                candidate_eigs[candidate_index],
                evaluation_times=dmd_power_times,
                spatial_weights=sv_grid_weights,
            )
            for candidate_index in range(
                candidate_eigs.size
            )
        ])

        ensemble_candidate_modal_power_results[
            "recovered_period"
        ].extend(
            candidate_periods.tolist()
        )
        ensemble_candidate_modal_power_results[
            "recovered_power"
        ].extend(
            candidate_powers.tolist()
        )
        ensemble_candidate_modal_power_results[
            "realisation_index"
        ].extend(
            [
                noised_realisation_index
            ]
            * candidate_eigs.size
        )
        noised_realisation_index += 1

    # candidates found, store for no perturbation case
    if result_type == "no_noise":

        clean_candidate_eigs = (
            candidate_eigs.copy()
        )

        clean_candidate_modes = (
            candidate_modes.copy()
        )

        clean_candidate_periods = (
            candidate_periods.copy()
        )

        clean_candidate_ids = (
            candidate_ids.copy()
        )

    elif result_type == "ideal":

        ideal_candidate_eigs = (
            candidate_eigs.copy()
        )

        ideal_candidate_modes = (
            candidate_modes.copy()
        )

        ideal_candidate_periods = (
            candidate_periods.copy()
        )


    # ---------------------------------------------
    # MATCH TO EVERY KNOWN INPUT MODE
    # ---------------------------------------------

    for mode_number in matching_mode_numbers:

        match = best_spatial_match(
            candidate_modes=candidate_modes,
            candidate_eigs=candidate_eigs,
            candidate_periods=candidate_periods,
            target_phasor=target_phasors[
                mode_number
            ],
        )

        # No eligible physical candidate / no finite similarity.
        if match is None:
            continue

        if result_type == "no_noise":

            DMD_recovery[
                mode_number
            ]["DMD"]= {
                **match,
                "degree_max": Nmax,
            }

        elif result_type == "ideal":

            DMD_recovery[
                mode_number
            ]["DMD_ideal"] = {
                **match,
                "degree_max": Nmax,
            }

        elif result_type == "noised":

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

        else:
            raise ValueError(
                f"Unknown DMD result type: {result_type}"
            )

print(
    "\nDMD SVD RANK SUMMARY\n"
    f"Rank method: {svd_rank_method}\n"
    f"PyDMD rank argument: {requested_svd_rank_metadata}\n"
    f"Ideal effective fitted rank: "
    f"{effective_svd_ranks['ideal']}\n"
    f"Resolved effective fitted rank: "
    f"{effective_svd_ranks['resolved']}"
)

# %% ------------------------------------------------------
# STORE TRUE / IDEAL / RESOLVED MODAL POWER RESULTS
# ------------------------------------------------------

modal_power_results = {
    "metadata": {
        "DMD_algorithm": DMD_algorithm,
        "svd_rank_selection": (
            (
                "number of clean resolved singular values strictly "
                "greater than the selected quantile of noise-only "
                "leading singular values, multiplied by "
                "noise_rank_multiplier"
            )
            if svd_rank_method == "noise_threshold"
            else (
                "fixed PyDMD svd_rank argument passed unchanged "
                "to every DMD fit"
            )
        ),
        "svd_rank_method": svd_rank_method,
        "noise_rank_multiplier": float(
            noise_rank_multiplier
        ),
        # Retained for compatibility: this is the argument passed to PyDMD.
        "svd_rank": requested_svd_rank_metadata,
        "requested_svd_rank": (
            requested_svd_rank_metadata
        ),
        "noise_selected_svd_rank": (
            noise_selected_svd_rank
        ),
        "noise_base_svd_rank": (
            noise_base_svd_rank
        ),
        "effective_svd_rank": {
            "ideal": effective_svd_ranks[
                "ideal"
            ],
            "resolved": effective_svd_ranks[
                "resolved"
            ],
            "perturbed": effective_svd_ranks[
                "perturbed"
            ].copy(),
        },
        "fixed_svd_rank": (
            requested_svd_rank_metadata
            if svd_rank_method == "fixed"
            else None
        ),
        "noise_rank_quantile": (
            float(
                noise_rank_quantile
            )
            if noise_rank_diagnostics_required
            else None
        ),
        "noise_rank_realisations": (
            int(
                n_noise_rank_realisations
            )
            if noise_rank_diagnostics_required
            else 0
        ),
        "noise_singular_value_threshold": (
            float(
                noise_singular_value_threshold
            )
            if noise_singular_value_threshold is not None
            else None
        ),
        "period_limit_enabled": bool(
            period_limit_flag
        ),
        "period_lower_bound": (
            float(period_lower_bound)
            if period_limit_flag
            else None
        ),
        "period_upper_bound": (
            float(period_upper_bound)
            if period_limit_flag
            else None
        ),
        "hankel_embedding": bool(
            hankel_embedding_flag
        ),
        "hankel_d": (
            int(hankel_d)
            if hankel_embedding_flag
            else None
        ),
        "analysis_times_absolute": (
            analysis_times_absolute.copy()
        ),
        "truth_reference_year": float(
            times_absolute[0]
        ),
        "dmd_reference_year": float(
            analysis_times_absolute[0]
        ),
        "power_definition": (
            "time mean of the area-weighted spatial mean square "
            "of Re[phasor * exp(eigenvalue * relative_time)]"
        ),
    },
    "true": {},
    "ideal": {},
    "resolved": {},
}

for mode_number in matching_mode_numbers:

    true_eigenvalue = synthetic_suite_info[
        mode_number
    ]["true_eigenvalue"]

    true_phasor = target_phasors[
        mode_number
    ]

    modal_power_results[
        "true"
    ][mode_number] = {
        "eigenvalue": true_eigenvalue,
        "period": synthetic_suite_info[
            mode_number
        ]["true_period"],
        "power": SV_Grid_Phasor_Record_Power(
            true_phasor,
            true_eigenvalue,
            evaluation_times=truth_power_times,
            spatial_weights=sv_grid_weights,
        ),
        "quality_factor": Mode_Quality_Factor(
            [true_eigenvalue]
        )[0],
    }

for (
    result_name,
    candidate_eigs_store,
    candidate_modes_store,
    candidate_periods_store,
    match_key,
) in (
    (
        "ideal",
        ideal_candidate_eigs,
        ideal_candidate_modes,
        ideal_candidate_periods,
        "DMD_ideal",
    ),
    (
        "resolved",
        clean_candidate_eigs,
        clean_candidate_modes,
        clean_candidate_periods,
        "DMD",
    ),
):

    if (
        candidate_eigs_store is None
        or candidate_modes_store is None
        or candidate_periods_store is None
    ):
        raise RuntimeError(
            f"Missing cached {result_name} DMD candidate suite."
        )

    matched_input_modes = [
        []
        for _ in range(
            candidate_eigs_store.size
        )
    ]

    for mode_number in matching_mode_numbers:
        candidate_index = DMD_recovery[
            mode_number
        ].get(
            match_key,
            {},
        ).get(
            "candidate_index"
        )

        if candidate_index is None:
            continue

        candidate_index = int(
            candidate_index
        )

        if not (
            0
            <= candidate_index
            < candidate_eigs_store.size
        ):
            raise IndexError(
                f"{result_name} candidate index "
                f"{candidate_index} is outside the retained "
                f"candidate set of size "
                f"{candidate_eigs_store.size}."
            )

        matched_input_modes[
            candidate_index
        ].append(
            mode_number
        )

    candidate_powers = np.asarray([
        SV_Grid_Phasor_Record_Power(
            candidate_modes_store[:, idx],
            candidate_eigs_store[idx],
            evaluation_times=dmd_power_times,
            spatial_weights=sv_grid_weights,
        )
        for idx in range(
            candidate_eigs_store.size
        )
    ])

    spatial_similarity = np.empty(
        (
            candidate_eigs_store.size,
            len(matching_mode_numbers),
        ),
        dtype=float,
    )

    for candidate_idx in range(
        candidate_eigs_store.size
    ):
        for input_idx, mode_number in enumerate(
            matching_mode_numbers
        ):
            spatial_similarity[
                candidate_idx,
                input_idx,
            ] = Complex_Phasor_Compare(
                candidate_modes_store[
                    :,
                    candidate_idx,
                ],
                target_phasors[
                    mode_number
                ],
            )

    modal_power_results[
        result_name
    ] = {
        "eigenvalue": candidate_eigs_store.copy(),
        "period": candidate_periods_store.copy(),
        "power": candidate_powers,
        "quality_factor": Mode_Quality_Factor(
            candidate_eigs_store
        ),
        "matched_input_modes": matched_input_modes,
        "input_mode_numbers": matching_mode_numbers.copy(),
        "spatial_similarity": spatial_similarity,
    }


def plot_ensemble_candidate_period_power(
    results,
    figsize=(9, 6),
):
    """
    Plot all perturbed-ensemble DMD candidates without mode matching.

    True inputs are shown for reference. Recovered candidates are treated as
    one population irrespective of their spatial similarity to any input.
    """

    true_periods = np.asarray(
        results[
            "true_period"
        ],
        dtype=float,
    )
    true_powers = np.asarray(
        results[
            "true_power"
        ],
        dtype=float,
    )
    recovered_periods = np.asarray(
        results[
            "recovered_period"
        ],
        dtype=float,
    )
    recovered_powers = np.asarray(
        results[
            "recovered_power"
        ],
        dtype=float,
    )
    realisation_indices = np.asarray(
        results[
            "realisation_index"
        ],
        dtype=int,
    )

    if true_periods.size != true_powers.size:
        raise ValueError(
            "True period and power arrays must have equal lengths."
        )

    if not (
        recovered_periods.size
        == recovered_powers.size
        == realisation_indices.size
    ):
        raise ValueError(
            "Recovered period, power, and realisation-index arrays "
            "must have equal lengths."
        )

    true_valid = (
        np.isfinite(
            true_periods
        )
        & (
            true_periods > 0.0
        )
        & np.isfinite(
            true_powers
        )
        & (
            true_powers > 0.0
        )
    )

    recovered_valid = (
        np.isfinite(
            recovered_periods
        )
        & (
            recovered_periods > 0.0
        )
        & np.isfinite(
            recovered_powers
        )
        & (
            recovered_powers > 0.0
        )
    )

    if period_limit_flag:
        recovered_valid &= (
            recovered_periods
            >= period_lower_bound
        ) & (
            recovered_periods
            <= period_upper_bound
        )

    fig, ax = plt.subplots(
        figsize=figsize
    )

    ax.scatter(
        true_periods[
            true_valid
        ],
        true_powers[
            true_valid
        ],
        marker="o",
        s=60,
        facecolor="0.65",
        edgecolor="0.3",
        linewidth=0.8,
        alpha=0.9,
        label="True inputs",
        zorder=5,
    )

    if np.any(
        recovered_valid
    ):
        ax.scatter(
            recovered_periods[
                recovered_valid
            ],
            recovered_powers[
                recovered_valid
            ],
            marker=".",
            s=20,
            color="tab:purple",
            edgecolors="none",
            alpha=0.3,
            label=(
                "All perturbed-ensemble "
                "recovered modes"
            ),
            rasterized=True,
            zorder=2,
        )

    ax.set_xscale(
        "log"
    )
    ax.set_yscale(
        "log"
    )
    ax.set_xlabel(
        "Period (years)"
    )
    ax.set_ylabel(
        r"Record-mean area-weighted SV power "
        r"[(nT/yr)$^2$]"
    )
    ax.set_title(
        "Perturbed-ensemble DMD candidate population\n"
        "all recovered modes shown without input matching"
    )
    ax.grid(
        alpha=0.25,
        which="both",
    )
    ax.legend(
        loc="best"
    )
    fig.tight_layout()

    return fig, ax


def plot_degree_sweep_recovered_periods(
    results,
    true_periods,
    figsize=(10, 12),
):
    """Plot every finite recovered period in three aligned sweep panels."""

    degrees = np.asarray(
        results[
            "degree_truncations"
        ],
        dtype=int,
    )
    true_periods = np.asarray(
        true_periods,
        dtype=float,
    )
    true_periods = np.unique(
        true_periods[
            np.isfinite(
                true_periods
            )
            & (
                true_periods > 0.0
            )
        ]
    )

    panel_definitions = (
        (
            "ideal",
            "Ideal (no resolution mapping)",
            "tab:blue",
            "o",
        ),
        (
            "resolved",
            "Resolved (resolution mapping applied)",
            "tab:orange",
            "x",
        ),
        (
            "perturbed",
            "Perturbed ensemble",
            "tab:green",
            ".",
        ),
    )

    fig, axes = plt.subplots(
        nrows=3,
        ncols=1,
        figsize=figsize,
        sharex=True,
        sharey=True,
    )

    for (
        ax,
        (
            result_key,
            panel_title,
            color,
            marker,
        ),
    ) in zip(
        axes,
        panel_definitions,
    ):
        true_label_used = False

        for true_period in true_periods:
            ax.axvline(
                true_period,
                color="0.45",
                linestyle="--",
                linewidth=0.9,
                alpha=0.55,
                label=(
                    "True input periods"
                    if not true_label_used
                    else None
                ),
                zorder=1,
            )
            true_label_used = True

        recovered_label_used = False

        for degree in degrees:
            degree_store = results[
                "by_degree"
            ][int(degree)]

            if result_key == "perturbed":
                candidate_stores = degree_store[
                    "perturbed"
                ]
            else:
                candidate_store = degree_store[
                    result_key
                ]
                candidate_stores = (
                    []
                    if candidate_store is None
                    else [candidate_store]
                )

            for candidate_store in candidate_stores:
                periods = np.asarray(
                    candidate_store[
                        "recovered_period"
                    ],
                    dtype=float,
                )
                finite_periods = periods[
                    np.isfinite(
                        periods
                    )
                    & (
                        periods > 0.0
                    )
                ]

                if finite_periods.size == 0:
                    continue

                ax.scatter(
                    finite_periods,
                    np.full(
                        finite_periods.size,
                        degree,
                        dtype=float,
                    ),
                    color=color,
                    marker=marker,
                    s=(
                        28
                        if result_key != "perturbed"
                        else 18
                    ),
                    linewidths=(
                        1.0
                        if marker == "x"
                        else None
                    ),
                    edgecolors=(
                        "none"
                        if marker == "."
                        else None
                    ),
                    alpha=(
                        0.8
                        if result_key != "perturbed"
                        else 0.28
                    ),
                    label=(
                        "Recovered modes"
                        if not recovered_label_used
                        else None
                    ),
                    rasterized=(
                        result_key == "perturbed"
                    ),
                    zorder=3,
                )
                recovered_label_used = True

        if (
            result_key == "perturbed"
            and not results[
                "metadata"
            ][
                "ensemble_enabled"
            ]
        ):
            ax.text(
                0.5,
                0.5,
                "Perturbation ensemble disabled",
                transform=ax.transAxes,
                ha="center",
                va="center",
                color="0.4",
            )

        ax.set_title(
            panel_title
        )
        ax.set_ylabel(
            "Truncation degree n"
        )
        ax.set_yticks(
            degrees
        )
        ax.grid(
            alpha=0.25,
            which="both",
        )
        ax.legend(
            loc="best"
        )

    axes[-1].set_xscale(
        "log"
    )
    axes[-1].set_xlabel(
        "Recovered period (years)"
    )

    if period_limit_flag:
        axes[-1].set_xlim(
            period_lower_bound,
            period_upper_bound,
        )

    fig.suptitle(
        "Recovered DMD periods across cumulative "
        "spherical-harmonic degree truncations"
    )
    fig.tight_layout()

    return fig, axes


def plot_degree_sweep_static_counts(
    results,
    figsize=(9, 6),
):
    """Plot counts of raw infinite-period candidates at every degree."""

    degrees = np.asarray(
        results[
            "degree_truncations"
        ],
        dtype=int,
    )

    ideal_counts = []
    resolved_counts = []

    for degree in degrees:
        degree_store = results[
            "by_degree"
        ][int(degree)]

        ideal_counts.append(
            degree_store[
                "ideal"
            ][
                "static_recovered_period"
            ].size
        )
        resolved_counts.append(
            degree_store[
                "resolved"
            ][
                "static_recovered_period"
            ].size
        )

    fig, ax = plt.subplots(
        figsize=figsize
    )

    ax.plot(
        degrees,
        ideal_counts,
        color="tab:blue",
        marker="o",
        linewidth=1.8,
        label="Ideal",
        zorder=5,
    )
    ax.plot(
        degrees,
        resolved_counts,
        color="tab:orange",
        marker="x",
        linewidth=1.8,
        label="Resolved",
        zorder=6,
    )

    perturbed_label_used = False
    perturbed_median = []

    for degree in degrees:
        perturbed_stores = results[
            "by_degree"
        ][int(degree)][
            "perturbed"
        ]
        perturbed_counts = np.asarray([
            candidate_store[
                "static_recovered_period"
            ].size
            for candidate_store in perturbed_stores
        ], dtype=float)

        if perturbed_counts.size == 0:
            perturbed_median.append(
                np.nan
            )
            continue

        ax.scatter(
            np.full(
                perturbed_counts.size,
                degree,
                dtype=float,
            ),
            perturbed_counts,
            color="tab:green",
            marker=".",
            s=28,
            alpha=0.3,
            label=(
                "Perturbed realisations"
                if not perturbed_label_used
                else None
            ),
            zorder=2,
        )
        perturbed_label_used = True
        perturbed_median.append(
            np.median(
                perturbed_counts
            )
        )

    perturbed_median = np.asarray(
        perturbed_median,
        dtype=float,
    )
    finite_median = np.isfinite(
        perturbed_median
    )

    if np.any(
        finite_median
    ):
        ax.plot(
            degrees[
                finite_median
            ],
            perturbed_median[
                finite_median
            ],
            color="tab:green",
            marker="s",
            linewidth=1.8,
            label="Perturbed median",
            zorder=4,
        )

    ax.set_xlabel(
        "Cumulative spherical-harmonic truncation degree n"
    )
    ax.set_ylabel(
        "Number of recovered static modes"
    )
    ax.set_xticks(
        degrees
    )
    ax.yaxis.set_major_locator(
        MaxNLocator(
            integer=True
        )
    )
    ax.set_title(
        "Infinite-period DMD recovery across degree truncations"
    )
    ax.grid(
        alpha=0.25
    )
    ax.legend(
        loc="best"
    )
    fig.tight_layout()

    return fig, ax


def plot_degree_sweep_svd_rank(
    results,
    figsize=(8, 5.5),
):
    """Plot the resolved fit's effective SVD rank by degree."""

    degrees = np.asarray(
        results[
            "degree_truncations"
        ],
        dtype=int,
    )
    ranks = np.asarray([
        results[
            "by_degree"
        ][int(degree)][
            "effective_svd_rank"
        ][
            "resolved"
        ]
        for degree in degrees
    ], dtype=int)

    fig, ax = plt.subplots(
        figsize=figsize
    )
    ax.plot(
        degrees,
        ranks,
        color="tab:purple",
        marker="o",
        linewidth=1.8,
    )
    ax.set_xlabel(
        "Cumulative spherical-harmonic truncation degree n"
    )
    ax.set_ylabel(
        "Effective resolved-fit SVD rank"
    )
    ax.set_xticks(
        degrees
    )
    ax.yaxis.set_major_locator(
        MaxNLocator(
            integer=True
        )
    )
    if (
        results[
            "metadata"
        ][
            "svd_rank_method"
        ] == "noise_threshold"
    ):
        rank_plot_title = (
            "Noise-threshold SVD rank selected independently "
            "at each degree truncation"
        )
    else:
        rank_plot_title = (
            "Effective resolved-fit SVD rank across degree truncations\n"
            "fixed PyDMD rank argument = "
            f"{results['metadata']['fixed_svd_rank']}"
        )

    ax.set_title(
        rank_plot_title
    )
    ax.grid(
        alpha=0.25
    )
    fig.tight_layout()

    return fig, ax


# %% video creation
if video_plot:
    video_data = prepare_recovery_video_data(

        # Restrict the exact and resolved truth rows to the same active true
        # period window used by the recovery diagnostics.
        mode_numbers=matching_mode_numbers,

        DMD_recovery=DMD_recovery,

        synthetic_suite_info=
            synthetic_suite_info,

        target_phasors=
            target_phasors,

        clean_candidate_eigs=
            clean_candidate_eigs,

        clean_candidate_modes=
            clean_candidate_modes,

        clean_candidate_ids=
            clean_candidate_ids,

        recovered_sum_candidate_eigs=
            video_sum_candidate_eigs,

        recovered_sum_candidate_modes=
            video_sum_candidate_modes,

        recovered_sum_candidate_ids=
            video_sum_candidate_ids,

        A_r_current=
            A_r_current,

        # Absolute decimal-year vector corresponding exactly to the
        # rows of gnm_total_res. The synthetic series starts at 1997.1.
        model_time_years=
            times_absolute,

        dt_years=
            dt_years,

        Nmax=
            Nmax,

        truncate_gauss_coeffs=
            Truncate_Gauss_Coeffs,

        high_q_flag=
            high_q_flag,

        good_record_slice=
            good_record_slice,

        # IMPORTANT:
        # Must be the same t=0 phase origin used when the synthetic
        # R_splines_arbitrary.h5 series were originally generated.
        #
        # The synthetic phase origin is the first sample at 1997.1.
        truth_t0_year=
            times_absolute[0],

        # Your HDF5 dataset is explicitly called "without_decay",
        # so this should normally remain False.
        truth_include_growth=
            False,

        # One video frame at every 0.2-year source sample.
        frame_spacing_years=
            0.2,

        compact_input_threshold=
            video_compact_input_threshold,
    )

    nlon = 360
    nlat = n_physical // nlon

    assert (
        nlat * nlon
        == n_physical
    )

    output_path = make_dmd_recovery_video(

        video_data=
            video_data,

        output_path=
            PROJECT_ROOT
            / "outputs"
            / (
                f"{DMD_algorithm}"
                + (
                    f"_hankel_d{hankel_d}"
                    if hankel_embedding_flag
                    else ""
                )
                + "_recovery_video.mp4"
            ),

        DMD_algorithm=
            DMD_algorithm,

        Nmax=
            Nmax,

        svd_rank=
            effective_svd_ranks[
                "resolved"
            ],

        nlat=
            nlat,

        nlon=
            nlon,

        hankel_embedding_d=(
            hankel_d
            if hankel_embedding_flag
            else None
        ),

        high_q_flag=
            high_q_flag,

        n_skip=
            n_skip,

        # Requested settings
        fps=5,
        cmap="seismic",

        dpi=100,
    )

# %% plotting results

if period_limit_flag:
    plot_lower_lim_yr = (
        period_lower_bound - 1.0
    )
    plot_upper_lim_yr = (
        period_upper_bound + 1.0
    )
    modal_period_xlim = (
        plot_lower_lim_yr,
        plot_upper_lim_yr,
    )
else:
    plot_lower_lim_yr = 1.0
    plot_upper_lim_yr = up_lim_yr_plot
    modal_period_xlim = None

DMD_recovery_plot = {
    mode_number: DMD_recovery[
        mode_number
    ]
    for mode_number in matching_mode_numbers
}

fig_q, ax_q = plot_quality_factor_recovery(
    DMD_recovery_plot,
    lower_lim_yr=plot_lower_lim_yr,
    upper_lim_yr=plot_upper_lim_yr,
)

fig_eigenvalue, ax_eigenvalue = plot_continuous_eigenvalue_recovery(
    DMD_recovery_plot,
    lower_lim_yr=plot_lower_lim_yr,
    upper_lim_yr=plot_upper_lim_yr,
)

fig_similarity, ax_similarity = plot_similarity_vs_recovered_period(
    DMD_recovery_plot,
    lower_lim_yr=plot_lower_lim_yr,
    upper_lim_yr=plot_upper_lim_yr,
)

fig_modal_power, ax_modal_power = (
    plot_modal_period_power(
        modal_power_results,
        show_ideal=show_ideal_all_modes,
        period_xlim=modal_period_xlim,
    )
)

(
    fig_ensemble_candidate_power,
    ax_ensemble_candidate_power,
) = plot_ensemble_candidate_period_power(
    ensemble_candidate_modal_power_results
)

(
    fig_similarity_heatmap,
    ax_similarity_heatmap,
    similarity_heatmap_row_order,
) = plot_resolved_mode_similarity_heatmap(
    modal_power_results
)

fig_cumulative_variance = None
ax_cumulative_variance = None
perturbed_cumulative_variance_median = None
noise_only_cumulative_variance_median = None
fig_singular_value_magnitude = None
ax_singular_value_magnitude = None
perturbed_singular_value_magnitude_median = None
noise_only_singular_value_magnitude_median = None

if singular_value_plot_flag:
    ideal_cumulative_variance = (
        singular_value_cumulative_variance[
            "ideal"
        ]
    )
    resolved_cumulative_variance = (
        singular_value_cumulative_variance[
            "resolved"
        ]
    )
    perturbed_cumulative_variance = (
        singular_value_cumulative_variance[
            "perturbed"
        ]
    )
    noise_only_cumulative_variance = (
        singular_value_cumulative_variance[
            "noise_only"
        ]
    )

    if (
        ideal_cumulative_variance is None
        or resolved_cumulative_variance is None
    ):
        raise RuntimeError(
            "The ideal and resolved cumulative-variance "
            "curves must both be available before plotting."
        )

    singular_value_indices = np.arange(
        1,
        ideal_cumulative_variance.size + 1,
    )
    marker_interval = max(
        1,
        singular_value_indices.size // 12,
    )

    fig_cumulative_variance, ax_cumulative_variance = (
        plt.subplots(
            figsize=(8, 6)
        )
    )

    ax_cumulative_variance.plot(
        singular_value_indices,
        ideal_cumulative_variance,
        color="tab:blue",
        marker="+",
        markevery=marker_interval,
        markersize=7,
        linewidth=1.8,
        label="Ideal (no resolution mapping)",
        zorder=5,
    )

    ax_cumulative_variance.plot(
        singular_value_indices,
        resolved_cumulative_variance,
        color="tab:orange",
        marker="x",
        markevery=marker_interval,
        markersize=6,
        linewidth=1.8,
        label="Resolved (resolution mapping applied)",
        zorder=5,
    )

    if perturbed_cumulative_variance:
        perturbed_cumulative_variance = np.stack(
            perturbed_cumulative_variance,
            axis=0,
        )

        for perturbation_index, perturbation_curve in enumerate(
            perturbed_cumulative_variance
        ):
            ax_cumulative_variance.plot(
                singular_value_indices,
                perturbation_curve,
                color="tab:green",
                marker=".",
                markevery=marker_interval,
                markersize=3,
                linewidth=0.8,
                alpha=0.2,
                label=(
                    "Perturbed ensemble members"
                    if perturbation_index == 0
                    else None
                ),
                zorder=1,
            )

        perturbed_cumulative_variance_median = np.median(
            perturbed_cumulative_variance,
            axis=0,
        )

        perturbation_label = (
            f"Perturbed median ({noise_temporal_model}"
        )
        if noise_temporal_model == "ar1":
            perturbation_label += (
                f", tau={noise_tau_years:g} years"
            )
        perturbation_label += ")"

        ax_cumulative_variance.plot(
            singular_value_indices,
            perturbed_cumulative_variance_median,
            color="tab:green",
            marker="s",
            markevery=marker_interval,
            markersize=5,
            linewidth=2.2,
            alpha=1.0,
            label=perturbation_label,
            zorder=6,
        )

    if noise_only_cumulative_variance:
        noise_only_cumulative_variance = np.stack(
            noise_only_cumulative_variance,
            axis=0,
        )

        for noise_index, noise_curve in enumerate(
            noise_only_cumulative_variance
        ):
            ax_cumulative_variance.plot(
                singular_value_indices,
                noise_curve,
                color="tab:purple",
                marker="v",
                markevery=marker_interval,
                markersize=3,
                linewidth=0.8,
                alpha=0.2,
                label=(
                    "Perturbation-only ensemble members"
                    if noise_index == 0
                    else None
                ),
                zorder=1,
            )

        noise_only_cumulative_variance_median = np.median(
            noise_only_cumulative_variance,
            axis=0,
        )

        noise_only_label = (
            "Perturbation-only median "
            f"({noise_temporal_model}"
        )
        if noise_temporal_model == "ar1":
            noise_only_label += (
                f", tau={noise_tau_years:g} years"
            )
        noise_only_label += ")"

        ax_cumulative_variance.plot(
            singular_value_indices,
            noise_only_cumulative_variance_median,
            color="tab:purple",
            marker="D",
            markevery=marker_interval,
            markersize=5,
            linewidth=2.2,
            alpha=1.0,
            label=noise_only_label,
            zorder=6,
        )

    embedding_plot_label = (
        f"Hankel embedding d={hankel_d}"
        if hankel_embedding_flag
        else "no embedding"
    )
    svd_input_plot_label = (
        "leading snapshot matrix X"
        if DMD_algorithm in ("exact", "fbdmd")
        else "full snapshot matrix"
    )

    ax_cumulative_variance.set_xlabel(
        "Number of singular values retained, k"
    )
    ax_cumulative_variance.set_ylabel(
        "Cumulative variance explained (fraction)"
    )
    ax_cumulative_variance.set_xlim(
        1,
        singular_value_indices[-1],
    )

    ax_cumulative_variance.set_title(
        "Pre-truncation cumulative explained variance\n"
        f"{DMD_algorithm}; {embedding_plot_label}; "
        f"{svd_input_plot_label}; Nmax={Nmax}"
    )
    ax_cumulative_variance.grid(
        alpha=0.25
    )
    ax_cumulative_variance.legend(
        loc="best"
    )
    fig_cumulative_variance.tight_layout()

    ideal_singular_value_magnitude = (
        singular_value_magnitudes[
            "ideal"
        ]
    )
    resolved_singular_value_magnitude = (
        singular_value_magnitudes[
            "resolved"
        ]
    )
    perturbed_singular_value_magnitudes = (
        singular_value_magnitudes[
            "perturbed"
        ]
    )
    noise_only_singular_value_magnitudes = (
        singular_value_magnitudes[
            "noise_only"
        ]
    )

    if (
        ideal_singular_value_magnitude is None
        or resolved_singular_value_magnitude is None
    ):
        raise RuntimeError(
            "The ideal and resolved singular-value magnitude "
            "curves must both be available before plotting."
        )

    if (
        ideal_singular_value_magnitude.size
        != singular_value_indices.size
        or resolved_singular_value_magnitude.size
        != singular_value_indices.size
    ):
        raise ValueError(
            "The ideal, resolved, and cumulative-variance "
            "singular-value indices must have equal lengths."
        )

    fig_singular_value_magnitude, ax_singular_value_magnitude = (
        plt.subplots(
            figsize=(8, 6)
        )
    )

    spectrum_display_rank = (
        effective_svd_ranks[
            "resolved"
        ]
    )

    if (
        spectrum_display_rank is None
        or spectrum_display_rank < 1
        or spectrum_display_rank
        > singular_value_indices.size
    ):
        raise ValueError(
            "The resolved fit's effective SVD rank must lie between "
            "1 and the number of pre-truncation singular values. "
            f"Received {spectrum_display_rank!r} for "
            f"{singular_value_indices.size} singular values."
        )

    retained_indices = singular_value_indices[
        :spectrum_display_rank
    ]
    truncated_indices = singular_value_indices[
        spectrum_display_rank:
    ]

    def scatter_signal_spectrum(
        values,
        color,
        label,
        retained_alpha=1.0,
        truncated_alpha=0.75,
        retained_size=34,
        truncated_size=18,
        zorder=5,
    ):
        """Plot retained signal values as crosses and truncated values as dots."""

        ax_singular_value_magnitude.scatter(
            retained_indices,
            values[:spectrum_display_rank],
            color=color,
            marker="x",
            s=retained_size,
            linewidths=1.2,
            alpha=retained_alpha,
            label=f"{label}: retained",
            zorder=zorder,
        )

        if truncated_indices.size:
            ax_singular_value_magnitude.scatter(
                truncated_indices,
                values[spectrum_display_rank:],
                color=color,
                marker=".",
                s=truncated_size,
                alpha=truncated_alpha,
                label=f"{label}: truncated",
                zorder=zorder,
            )

    scatter_signal_spectrum(
        ideal_singular_value_magnitude,
        color="tab:blue",
        label="Ideal",
        zorder=7,
    )

    scatter_signal_spectrum(
        resolved_singular_value_magnitude,
        color="tab:orange",
        label="Resolved",
        zorder=8,
    )

    if perturbed_singular_value_magnitudes:
        perturbed_singular_value_magnitudes = np.stack(
            perturbed_singular_value_magnitudes,
            axis=0,
        )

        for perturbation_index, perturbation_curve in enumerate(
            perturbed_singular_value_magnitudes
        ):
            ax_singular_value_magnitude.scatter(
                retained_indices,
                perturbation_curve[
                    :spectrum_display_rank
                ],
                color="tab:green",
                marker="x",
                s=15,
                linewidths=0.7,
                alpha=0.15,
                zorder=1,
            )
            if truncated_indices.size:
                ax_singular_value_magnitude.scatter(
                    truncated_indices,
                    perturbation_curve[
                        spectrum_display_rank:
                    ],
                    color="tab:green",
                    marker=".",
                    s=8,
                    alpha=0.15,
                    zorder=1,
                )

        perturbed_singular_value_magnitude_median = np.median(
            perturbed_singular_value_magnitudes,
            axis=0,
        )

        scatter_signal_spectrum(
            perturbed_singular_value_magnitude_median,
            color="tab:green",
            label=perturbation_label,
            retained_size=38,
            truncated_size=22,
            zorder=6,
        )

    if noise_only_singular_value_magnitudes:
        noise_only_singular_value_magnitudes = np.stack(
            noise_only_singular_value_magnitudes,
            axis=0,
        )

        for noise_index, noise_curve in enumerate(
            noise_only_singular_value_magnitudes
        ):
            ax_singular_value_magnitude.scatter(
                singular_value_indices,
                noise_curve,
                color="tab:purple",
                marker=".",
                s=7,
                alpha=0.06,
                label=(
                    "Noise-only realisations"
                    if noise_index == 0
                    else None
                ),
                zorder=1,
            )

        noise_only_singular_value_magnitude_median = np.median(
            noise_only_singular_value_magnitudes,
            axis=0,
        )

        ax_singular_value_magnitude.scatter(
            singular_value_indices,
            noise_only_singular_value_magnitude_median,
            color="tab:purple",
            marker=".",
            s=20,
            alpha=1.0,
            label=noise_only_label,
            zorder=6,
        )

    threshold_percent = (
        100.0
        * noise_rank_quantile
    )
    ax_singular_value_magnitude.axhline(
        noise_singular_value_threshold,
        color="black",
        linestyle="--",
        linewidth=1.4,
        label=(
            f"q{threshold_percent:g} of noise-only "
            r"$\sigma_1$"
        ),
        zorder=4,
    )
    if svd_rank_method == "noise_threshold":
        spectrum_rank_annotation = (
            f"Base noise rank {noise_base_svd_rank} × "
            f"{noise_rank_multiplier:g} = "
            f"PyDMD rank {noise_selected_svd_rank}; "
            f"resolved effective rank = {spectrum_display_rank}"
        )
    else:
        spectrum_rank_annotation = (
            "Fixed PyDMD rank argument = "
            f"{requested_svd_rank_metadata}; resolved effective "
            f"rank = {spectrum_display_rank}"
        )

    ax_singular_value_magnitude.annotate(
        spectrum_rank_annotation,
        xy=(
            min(
                spectrum_display_rank + 0.5,
                singular_value_indices[-1],
            ),
            noise_singular_value_threshold,
        ),
        xytext=(8, 8),
        textcoords="offset points",
        ha="left",
        va="bottom",
        fontsize=9,
        color="black",
        bbox={
            "facecolor": "white",
            "edgecolor": "none",
            "alpha": 0.75,
            "pad": 1.5,
        },
        zorder=10,
    )

    ax_singular_value_magnitude.set_xlabel(
        "Singular value index, k"
    )
    ax_singular_value_magnitude.set_ylabel(
        "Singular value magnitude"
    )
    ax_singular_value_magnitude.set_xlim(
        1,
        singular_value_indices[-1],
    )
    ax_singular_value_magnitude.set_yscale(
        "log"
    )

    ax_singular_value_magnitude.set_title(
        "Pre-truncation singular-value magnitude\n"
        f"{DMD_algorithm}; {embedding_plot_label}; "
        f"{svd_input_plot_label}; Nmax={Nmax}; "
        f"rank method={svd_rank_method}; "
        f"resolved effective rank={spectrum_display_rank}"
    )
    ax_singular_value_magnitude.grid(
        alpha=0.25,
        which="both",
    )
    ax_singular_value_magnitude.legend(
        loc="best"
    )
    fig_singular_value_magnitude.tight_layout()

fig_degree_sweep_periods = None
ax_degree_sweep_periods = None
fig_degree_sweep_static = None
ax_degree_sweep_static = None
fig_degree_sweep_svd_rank = None
ax_degree_sweep_svd_rank = None

if degree_sweep_flag:
    active_true_periods = np.asarray([
        synthetic_suite_info[
            mode_number
        ][
            "true_period"
        ]
        for mode_number in matching_mode_numbers
    ], dtype=float)

    (
        fig_degree_sweep_periods,
        ax_degree_sweep_periods,
    ) = plot_degree_sweep_recovered_periods(
        degree_sweep_results,
        true_periods=active_true_periods,
    )

    (
        fig_degree_sweep_static,
        ax_degree_sweep_static,
    ) = plot_degree_sweep_static_counts(
        degree_sweep_results
    )

    (
        fig_degree_sweep_svd_rank,
        ax_degree_sweep_svd_rank,
    ) = plot_degree_sweep_svd_rank(
        degree_sweep_results
    )

plt.show()
# %%
if video_plot:
    print("\nVIDEO RECOVERY AMPLITUDES")

    if video_data.get(
        "layout",
        "full",
    ) == "compact":
        summary = video_data["summary"]
        print(
            "Compact video layout\n"
            f"Input modes: {summary['input_count']}\n"
            "Inputs with no finite DMD match: "
            f"{summary['unmatched_input_count']} "
            f"{summary['unmatched_input_modes']}\n"
            "Unique matched DMD candidates: "
            f"{summary['unique_matched_dmd_count']}\n"
            "Additional inputs sharing a candidate: "
            f"{summary['shared_input_count']}\n"
            "Unmatched retained DMD candidates: "
            f"{summary['unmatched_dmd_count']} "
            f"{summary['unmatched_dmd_candidate_ids']}"
        )

        for row in video_data[
            "compact_rows"
        ]:
            if row["kind"] != "matched":
                continue
            print(
                f"DMD candidate "
                f"{row['candidate_id']}: "
                "best match for inputs "
                f"{row['input_modes']}, "
                "similarities="
                f"{row['similarities']}"
            )
    else:
        for i, mode_number in enumerate(
            matching_mode_numbers
        ):

            idx = DMD_recovery[
                mode_number
            ]["DMD"].get(
                "candidate_index"
            )

            exact_max = np.max(
                np.abs(
                    video_data["exact"][i]
                )
            )

            resolved_max = np.max(
                np.abs(
                    video_data["resolved"][i]
                )
            )

            recovered_max = np.max(
                np.abs(
                    video_data["recovered"][i]
                )
            )

            print(
                f"Mode {mode_number}: "
                f"candidate={idx}, "
                f"exact={exact_max:.4e}, "
                f"resolved={resolved_max:.4e}, "
                f"recovered={recovered_max:.4e}, "
                "rec/res="
                f"{recovered_max / resolved_max if resolved_max else np.nan:.4e}"
            )
# %%
import inspect

print(inspect.getsource(best_spatial_match))
# %%
