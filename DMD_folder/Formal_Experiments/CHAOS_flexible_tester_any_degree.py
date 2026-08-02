"""
Flexible DMD analysis of CHAOS-8.6 radial secular variation.

This script:

1. evaluates the CHAOS-8.6 SV Gauss-coefficient time series;
2. optionally adds paired covariance perturbations;
3. converts each coefficient series to gridded radial SV at the CMB;
4. optionally applies time-delay embedding;
5. fits Exact DMD, Forward-Backward DMD, or Optimized DMD;
6. selects the SVD rank either directly or from a noise-only threshold;
7. treats the unperturbed CHAOS modes as empirical baseline candidates;
8. matches perturbed candidates one-to-one to that baseline;
9. optionally repeats the analysis over cumulative spherical-harmonic degree
   truncations; and
10. reports modal properties, uncertainty stability, reconstruction, spatial
    phasors, singular-value diagnostics, and optional reconstruction and
    individual-mode videos.

Unlike the synthetic tester, this script has no known true modes. Agreement
with the unperturbed CHAOS fit measures uncertainty stability, not recovery
accuracy against ground truth.
"""

# %% FILE SYSTEM AND DEPENDENCY SETUP

import gc
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

import chaosmagpy as cp
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import Normalize
from matplotlib.ticker import MaxNLocator
from pydmd.utils import pseudo_hankel_matrix
from scipy.optimize import linear_sum_assignment
from tqdm import tqdm

from src.msc_thesis.paths import *
from src.msc_thesis.synSetup import *
from src.msc_thesis.synUtils import *
from src.msc_thesis.synDMD import *
from src.msc_thesis.synVideo import (
    prepare_chaos_reconstruction_video_data,
    make_chaos_reconstruction_video,
    prepare_chaos_all_modes_video_data,
    make_chaos_all_modes_video,
)


# %% ------------------------------------------------------
# USER SETTINGS
# ------------------------------------------------------

DMD_algorithm = "opdmd"

# Optional time-delay embedding, independent of the base DMD algorithm.
hankel_embedding_flag = True
hankel_d = 10
hankel_reconstruction_method = "first"

# Rank selection can either retain the historical fixed rank or use the same
# noise-threshold rule as the synthetic tester. The selected Nmax rank is
# assigned to svd_rank before any main-analysis estimator is constructed.

# "noise_threshold" or "fixed"
svd_rank_selection = "noise_threshold"
# Applied only in noise-threshold mode, then rounded to the nearest integer
# (with half-integer results rounded up).
noise_rank_multiplier = 1.0
fixed_svd_rank = 45
svd_rank = None

# Calculate and plot both cumulative explained variance and raw singular-value
# magnitude for every pre-truncation input matrix.
singular_value_plot_flag = True

# Optional removal of the temporal mean at every SV grid point. This is a
# methodological choice and is disabled by default.
remove_temporal_mean_flag = False

# Optional long-period removal before temporal subsampling and gridding.
filter_flag = False
filter_pass_period_years = 10.0
filter_stop_period_years = 20.0

# Optional finite-period range applied after DMD fitting. DMD itself always
# sees the complete input record.
period_limit_flag = True
period_lower_bound = 1.01
period_upper_bound = 25

# Plot the real and imaginary spatial phasors of every retained baseline mode.
mode_map_plot_flag = True

# Plot the reconstruction made from the retained period-limited modes.
reconstruction_plot_flag = True

# Plot all perturbed candidates without applying ensemble mode matching.
ensemble_candidate_plot_flag = True

# Plot the complete baseline-versus-perturbed spatial-similarity matrix for
# each covariance realisation, with accepted one-to-one matches highlighted.
similarity_heatmap_plot_flag = True

# Animate processed CHAOS input, the unique retained-mode reconstruction, and
# reconstruction minus input.
video_plot = True

# Animate every unique retained oscillatory and static DMD mode in its own
# panel. Finite candidates obey the active period window.
all_modes_video_plot = True

video_frame_spacing_years = 0.2
video_fps = 5


# ---------------------------------------------------------
# UNCERTAINTY ENSEMBLE
# ---------------------------------------------------------

ensemble_flag = False
n_realisations = 1
noise_seed = 42

noise_temporal_model = "independent"
noise_tau_years = 6.0

# The noise-only bank used for rank selection is always generated, independently
# of whether perturbed CHAOS realisations are subsequently fitted.
n_noise_rank_realisations = n_realisations
noise_rank_quantile = 0.95


# ---------------------------------------------------------
# TIME / DEGREE SETTINGS
# ---------------------------------------------------------

# Use the higher-quality satellite interval defined in synSetup.py.
high_q_flag = True

# Temporal subsampling after record selection and optional filtering.
n_skip = 1

# Maximum spherical-harmonic degree supplied to DMD.
Nmax = 10

# Run cumulative degree truncations 1:n through Nmax. The Nmax case remains the
# sole source for the established CHAOS analysis and plots.
degree_sweep_flag = False
degree_truncations_tested = np.arange(
    1,
    Nmax + 1,
    dtype=int,
)

# CHAOS and the covariance product both contain complete degree-20 vectors.
# Loading degree 20 before perturbation keeps their coefficient conventions
# aligned; the requested Nmax truncation is applied afterwards.
CHAOS_covariance_Nmax = 20


# ---------------------------------------------------------
# ONE-TO-ONE ENSEMBLE MODE MATCHING
# ---------------------------------------------------------

# Cost = spatial_weight * (1 - spatial_similarity)
#      + period_weight * relative_period_error
#      + growth_weight * growth_difference / baseline_frequency
matching_spatial_weight = 1.0
matching_period_weight = 1.0
matching_growth_weight = 0.25

# Candidate pairs outside either eligibility threshold are not matched.
matching_max_relative_period_error = 0.25
matching_min_spatial_similarity = 0.50


# ---------------------------------------------------------
# PLOTTING SETTINGS
# ---------------------------------------------------------

mode_map_colour_percentile = 99.0


# %% ------------------------------------------------------
# VALIDATE SETTINGS
# ------------------------------------------------------

if DMD_algorithm not in ("exact", "fbdmd", "opdmd"):
    raise ValueError(
        "DMD_algorithm must be one of 'exact', 'fbdmd', or 'opdmd'. "
        f"Received {DMD_algorithm!r}."
    )

if svd_rank_selection not in (
    "fixed",
    "noise_threshold",
):
    raise ValueError(
        "svd_rank_selection must be 'fixed' or 'noise_threshold'."
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

if (
    not isinstance(
        fixed_svd_rank,
        (int, np.integer),
    )
    or fixed_svd_rank < 1
):
    raise ValueError(
        "fixed_svd_rank must be a positive integer."
    )

if (
    not isinstance(
        n_noise_rank_realisations,
        (int, np.integer),
    )
    or n_noise_rank_realisations < 1
):
    raise ValueError(
        "n_noise_rank_realisations must be a positive integer."
    )

if not (
    np.isfinite(
        noise_rank_quantile
    )
    and 0.0
    < noise_rank_quantile
    < 1.0
):
    raise ValueError(
        "noise_rank_quantile must lie strictly between zero and one."
    )

if not isinstance(n_skip, (int, np.integer)) or n_skip < 1:
    raise ValueError("n_skip must be a positive integer.")

if (
    video_plot
    or all_modes_video_plot
) and (
    not isinstance(
        video_fps,
        (
            int,
            float,
            np.integer,
            np.floating,
        ),
    )
    or not np.isfinite(
        video_fps
    )
    or video_fps <= 0
):
    raise ValueError(
        "video_fps must be a finite positive number."
    )

if (
    video_plot
    or all_modes_video_plot
) and (
    not isinstance(
        video_frame_spacing_years,
        (
            int,
            float,
            np.integer,
            np.floating,
        ),
    )
    or not np.isfinite(
        video_frame_spacing_years
    )
    or video_frame_spacing_years <= 0.0
):
    raise ValueError(
        "video_frame_spacing_years must be finite and positive."
    )

if not isinstance(Nmax, (int, np.integer)) or not (
    1 <= Nmax <= CHAOS_covariance_Nmax
):
    raise ValueError(
        "Nmax must be an integer between 1 and "
        f"{CHAOS_covariance_Nmax}."
    )

if ensemble_flag and (
    not isinstance(n_realisations, (int, np.integer))
    or n_realisations < 1
):
    raise ValueError(
        "n_realisations must be a positive integer when "
        "ensemble_flag=True."
    )

if hankel_embedding_flag and (
    not isinstance(hankel_d, (int, np.integer))
    or hankel_d < 1
):
    raise ValueError(
        "hankel_d must be a positive integer when embedding is enabled."
    )

if period_limit_flag and not (
    np.isfinite(period_lower_bound)
    and np.isfinite(period_upper_bound)
    and 0.0 < period_lower_bound < period_upper_bound
):
    raise ValueError(
        "Period bounds must be finite and satisfy "
        "0 < period_lower_bound < period_upper_bound."
    )

if not (
    matching_spatial_weight >= 0.0
    and matching_period_weight >= 0.0
    and matching_growth_weight >= 0.0
    and (
        matching_spatial_weight
        + matching_period_weight
        + matching_growth_weight
    ) > 0.0
):
    raise ValueError(
        "Mode-matching weights must be non-negative and at least one "
        "weight must be positive."
    )

if not (
    matching_max_relative_period_error > 0.0
    and 0.0 <= matching_min_spatial_similarity <= 1.0
):
    raise ValueError(
        "Matching thresholds require a positive relative-period tolerance "
        "and a spatial-similarity threshold between 0 and 1."
    )


# %% ------------------------------------------------------
# HELPERS
# ------------------------------------------------------

def build_selected_dmd(
    selected_svd_rank,
):
    """Construct the configured base DMD estimator."""

    if DMD_algorithm == "exact":
        return build_exact_dmd(
            svd_rank=selected_svd_rank
        )

    if DMD_algorithm == "fbdmd":
        return build_fbdmd(
            svd_rank=selected_svd_rank
        )

    return build_bopdmd(
        svd_rank=selected_svd_rank
    )


def pre_truncation_svd_input(sv_input):
    """
    Return the matrix on which the selected estimator chooses its SVD rank.

    Optimized DMD uses the complete snapshot matrix. Exact and
    Forward-Backward DMD select their forward rank from the leading snapshot
    matrix X. Optional Hankel preprocessing is reproduced exactly first.
    """

    if hankel_embedding_flag:
        matrix = pseudo_hankel_matrix(
            sv_input,
            d=hankel_d,
        )
    else:
        matrix = sv_input

    if DMD_algorithm in ("exact", "fbdmd"):
        matrix = matrix[:, :-1]

    return matrix


def calculate_singular_diagnostic(sv_input):
    """Return raw magnitudes and cumulative squared-singular-value energy."""

    matrix = pre_truncation_svd_input(
        sv_input
    )

    magnitudes = np.abs(
        np.linalg.svd(
            matrix,
            compute_uv=False,
        )
    )

    variance = np.square(
        magnitudes
    )
    total_variance = np.sum(
        variance
    )

    if (
        not np.isfinite(total_variance)
        or total_variance <= 0.0
    ):
        raise ValueError(
            "The pre-truncation input must have finite, positive "
            "squared singular-value magnitude."
        )

    cumulative_variance = np.cumsum(
        variance
    ) / total_variance

    return {
        "magnitude": magnitudes,
        "cumulative_variance": cumulative_variance,
    }


def prepare_chaos_sv_input(
    gnm_input,
    truncation_degree,
    projection_operator,
):
    """Apply the common CHAOS preprocessing used by DMD and rank selection."""

    if high_q_flag:
        gnm_windowed = (
            gnm_input[
                good_record_slice,
                :
            ]
        )
    else:
        gnm_windowed = gnm_input

    if filter_flag:
        gnm_windowed, _, _ = (
            Long_Period_Taper_Filter(
                gnm_windowed,
                dt=dt_years,
                pass_period=
                    filter_pass_period_years,
                stop_period=
                    filter_stop_period_years,
                axis=0,
            )
        )

    gnm_band = Truncate_Gauss_Coeffs(
        gnm_windowed,
        tmax=truncation_degree,
    )
    expected_coefficients = (
        (truncation_degree + 1)**2
        - 1
    )

    if gnm_band.shape[1] != expected_coefficients:
        raise ValueError(
            "Degree truncation produced an unexpected coefficient "
            f"count at n={truncation_degree}: "
            f"{gnm_band.shape[1]} vs {expected_coefficients}."
        )
    if projection_operator.shape != (
        state_shape[0]
        * state_shape[1],
        expected_coefficients,
    ):
        raise ValueError(
            "Unexpected radial SV design-matrix shape at "
            f"n={truncation_degree}: {projection_operator.shape}."
        )

    sv_input = (
        projection_operator
        @ gnm_band.T
    )
    sv_input = (
        sv_input[
            :,
            ::n_skip,
        ]
    )

    if sv_input.shape != (
        projection_operator.shape[0],
        analysis_times_absolute.size,
    ):
        raise ValueError(
            "Gridded SV input and selected time coordinates do not "
            f"align at n={truncation_degree}: "
            f"{sv_input.shape} vs "
            f"{analysis_times_absolute.size} times."
        )

    if remove_temporal_mean_flag:
        sv_input = (
            sv_input
            - np.mean(
                sv_input,
                axis=1,
                keepdims=True,
            )
        )

    if not np.all(
        np.isfinite(
            sv_input
        )
    ):
        raise ValueError(
            "A preprocessed CHAOS input contains non-finite values "
            f"at n={truncation_degree}."
        )

    return sv_input


def select_degree_svd_rank(
    truncation_degree,
    projection_operator,
    collect_diagnostics=False,
):
    """Select or retrieve the DMD rank independently at one degree."""

    need_spectra = (
        svd_rank_selection
        == "noise_threshold"
        or collect_diagnostics
    )

    if not need_spectra:
        return (
            int(fixed_svd_rank),
            None,
            None,
            None,
            [],
        )

    baseline_sv_input = prepare_chaos_sv_input(
        chaos_sv_full,
        truncation_degree,
        projection_operator,
    )
    baseline_diagnostic = (
        calculate_singular_diagnostic(
            baseline_sv_input
        )
    )
    del baseline_sv_input

    noise_diagnostics = []

    for noise_realisation in (
        noise_only_rank_bank
    ):
        noise_sv_input = (
            prepare_chaos_sv_input(
                noise_realisation,
                truncation_degree,
                projection_operator,
            )
        )
        noise_diagnostics.append(
            calculate_singular_diagnostic(
                noise_sv_input
            )
        )
        del noise_sv_input

    if svd_rank_selection == "fixed":
        return (
            int(fixed_svd_rank),
            None,
            None,
            baseline_diagnostic,
            noise_diagnostics,
        )

    noise_leading_singular_values = (
        np.asarray([
            diagnostic[
                "magnitude"
            ][0]
            for diagnostic
            in noise_diagnostics
        ])
    )
    noise_threshold = float(
        np.quantile(
            noise_leading_singular_values,
            noise_rank_quantile,
        )
    )
    base_noise_rank = int(
        np.count_nonzero(
            baseline_diagnostic[
                "magnitude"
            ]
            > noise_threshold
        )
    )

    if base_noise_rank == 0:
        raise RuntimeError(
            "The noise-threshold rank criterion retained no CHAOS "
            "singular values at spherical-harmonic truncation "
            f"n={truncation_degree}. The threshold was "
            f"{noise_threshold:.6e}, derived from the "
            f"q={noise_rank_quantile:.3f} quantile of "
            f"{n_noise_rank_realisations} noise-only leading "
            "singular values."
        )

    selected_rank = int(
        np.floor(
            base_noise_rank
            * noise_rank_multiplier
            + 0.5
        )
    )

    if selected_rank < 1:
        raise RuntimeError(
            "The noise-selected SVD rank became zero after applying "
            f"noise_rank_multiplier={noise_rank_multiplier:g} at "
            f"spherical-harmonic truncation n={truncation_degree}. "
            f"The unmultiplied rank was {base_noise_rank}."
        )

    return (
        selected_rank,
        base_noise_rank,
        noise_threshold,
        baseline_diagnostic,
        noise_diagnostics,
    )


def make_degree_sweep_candidate_store(
    candidate_suite,
):
    """Store retained finite periods and raw static periods compactly."""

    return {
        "effective_svd_rank": int(
            candidate_suite[
                "effective_svd_rank"
            ]
        ),
        "recovered_period": (
            candidate_suite[
                "period"
            ].copy()
        ),
        "static_recovered_period": (
            candidate_suite[
                "static"
            ][
                "period"
            ].copy()
        ),
    }


def sort_candidate_suite(
    eigenvalues,
    modes,
    periods,
):
    """Sort aligned oscillatory candidates by increasing period."""

    eigenvalues = np.asarray(
        eigenvalues,
        dtype=complex,
    ).ravel()
    modes = np.asarray(
        modes,
        dtype=complex,
    )
    periods = np.asarray(
        periods,
        dtype=float,
    ).ravel()

    if not (
        modes.ndim == 2
        and modes.shape[1] == eigenvalues.size
        and periods.size == eigenvalues.size
    ):
        raise ValueError(
            "Candidate eigenvalues, mode columns, and periods must align."
        )

    order = np.argsort(
        periods
    )

    return (
        eigenvalues[order],
        modes[:, order],
        periods[order],
    )


def fit_candidate_suite(
    sv_input,
    selected_svd_rank,
    dt_snapshot,
    evaluation_times,
    spatial_weights,
):
    """
    Fit the selected DMD model and return physical oscillatory candidates.

    Returned mode columns are amplitude-scaled physical-space phasors and
    eigenvalues are continuous-time values in inverse years.
    """

    n_physical, n_snapshots = sv_input.shape

    if hankel_embedding_flag and hankel_d >= n_snapshots:
        raise ValueError(
            f"hankel_d={hankel_d} leaves fewer than two embedded "
            f"snapshots from the {n_snapshots} available snapshots."
        )

    base_dmd = build_selected_dmd(
        selected_svd_rank
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
                n_fit_snapshots,
                dtype=float,
            )
            * dt_snapshot
        )

        dmd.fit(
            sv_input,
            fit_times,
        )

        (
            candidate_eigenvalues,
            candidate_modes,
            candidate_periods,
        ) = extract_optimized_dmd_candidates(
            dmd=dmd,
            n_physical=n_physical,
            embedding_d=embedding_d,
        )

    else:
        dmd.fit(
            sv_input
        )

        (
            candidate_eigenvalues,
            candidate_modes,
            candidate_periods,
        ) = extract_standard_dmd_candidates(
            dmd=dmd,
            dt_snapshot=dt_snapshot,
            n_physical=n_physical,
            embedding_d=embedding_d,
        )

    effective_svd_rank = int(
        np.asarray(
            dmd.modes
        ).shape[1]
    )

    if effective_svd_rank < 1:
        raise RuntimeError(
            "The fitted CHAOS DMD estimator returned no modes, so "
            "an effective SVD rank could not be recorded."
        )

    raw_candidate_eigenvalues = (
        candidate_eigenvalues.copy()
    )
    raw_candidate_modes = (
        candidate_modes.copy()
    )
    raw_candidate_periods = (
        candidate_periods.copy()
    )

    # Preserve static candidates separately. The established modal analysis
    # and uncertainty matching remain finite/oscillatory, while reconstruction,
    # video, and degree-sweep diagnostics can include static contributions.
    static = np.isinf(
        raw_candidate_periods
    )
    static_suite = {
        "eigenvalue": (
            raw_candidate_eigenvalues[
                static
            ].copy()
        ),
        "mode": (
            raw_candidate_modes[
                :,
                static,
            ].copy()
        ),
        "period": (
            raw_candidate_periods[
                static
            ].copy()
        ),
    }

    oscillatory = (
        np.isfinite(
            raw_candidate_periods
        )
        & (
            raw_candidate_periods
            > 0.0
        )
    )

    candidate_eigenvalues = (
        raw_candidate_eigenvalues[
            oscillatory
        ]
    )
    candidate_modes = (
        raw_candidate_modes[
            :,
            oscillatory,
        ]
    )
    candidate_periods = (
        raw_candidate_periods[
            oscillatory
        ]
    )

    if period_limit_flag:
        (
            candidate_eigenvalues,
            candidate_modes,
            candidate_periods,
        ) = filter_candidate_period_range(
            candidate_eigenvalues,
            candidate_modes,
            candidate_periods,
            lower_period=period_lower_bound,
            upper_period=period_upper_bound,
        )

    (
        candidate_eigenvalues,
        candidate_modes,
        candidate_periods,
    ) = sort_candidate_suite(
        candidate_eigenvalues,
        candidate_modes,
        candidate_periods,
    )

    candidate_power = np.asarray([
        SV_Grid_Phasor_Record_Power(
            candidate_modes[:, index],
            candidate_eigenvalues[index],
            evaluation_times=evaluation_times,
            spatial_weights=spatial_weights,
        )
        for index in range(
            candidate_eigenvalues.size
        )
    ])

    reconstruction_suite = {
        "eigenvalue": np.concatenate([
            candidate_eigenvalues,
            static_suite[
                "eigenvalue"
            ],
        ]),
        "mode": np.concatenate(
            [
                candidate_modes,
                static_suite[
                    "mode"
                ],
            ],
            axis=1,
        ),
        "period": np.concatenate([
            candidate_periods,
            static_suite[
                "period"
            ],
        ]),
    }

    return {
        "effective_svd_rank": (
            effective_svd_rank
        ),
        "eigenvalue": candidate_eigenvalues,
        "mode": candidate_modes,
        "period": candidate_periods,
        "growth_rate": candidate_eigenvalues.real.copy(),
        "quality_factor": Mode_Quality_Factor(
            candidate_eigenvalues
        ),
        "power": candidate_power,
        "static": static_suite,
        "reconstruction_suite": (
            reconstruction_suite
        ),
        "raw_period": (
            raw_candidate_periods
        ),
    }


def match_candidate_suites(
    baseline_suite,
    perturbed_suite,
):
    """
    Match perturbed candidates one-to-one to baseline CHAOS candidates.

    The Hungarian assignment minimises the configured combined cost. Pairings
    that fail the period or spatial thresholds remain unrecovered.
    """

    baseline_eigenvalues = baseline_suite[
        "eigenvalue"
    ]
    baseline_modes = baseline_suite[
        "mode"
    ]
    baseline_periods = baseline_suite[
        "period"
    ]

    perturbed_eigenvalues = perturbed_suite[
        "eigenvalue"
    ]
    perturbed_modes = perturbed_suite[
        "mode"
    ]
    perturbed_periods = perturbed_suite[
        "period"
    ]

    n_baseline = baseline_eigenvalues.size
    n_perturbed = perturbed_eigenvalues.size

    output = {
        "recovered": np.zeros(
            n_baseline,
            dtype=bool,
        ),
        "candidate_index": np.full(
            n_baseline,
            -1,
            dtype=int,
        ),
        "cost": np.full(
            n_baseline,
            np.nan,
            dtype=float,
        ),
        "spatial_similarity": np.full(
            n_baseline,
            np.nan,
            dtype=float,
        ),
        "relative_period_error": np.full(
            n_baseline,
            np.nan,
            dtype=float,
        ),
        "eigenvalue": np.full(
            n_baseline,
            np.nan + 1j * np.nan,
            dtype=complex,
        ),
        "period": np.full(
            n_baseline,
            np.nan,
            dtype=float,
        ),
        "growth_rate": np.full(
            n_baseline,
            np.nan,
            dtype=float,
        ),
        "quality_factor": np.full(
            n_baseline,
            np.nan,
            dtype=float,
        ),
        "power": np.full(
            n_baseline,
            np.nan,
            dtype=float,
        ),
        "pairwise_spatial_similarity": np.full(
            (
                n_baseline,
                n_perturbed,
            ),
            np.nan,
            dtype=float,
        ),
        "pairwise_eligible": np.zeros(
            (
                n_baseline,
                n_perturbed,
            ),
            dtype=bool,
        ),
    }

    if n_baseline == 0 or n_perturbed == 0:
        return output

    invalid_cost = 1e12
    cost = np.full(
        (
            n_baseline,
            n_perturbed,
        ),
        invalid_cost,
        dtype=float,
    )
    similarities = output[
        "pairwise_spatial_similarity"
    ]
    pairwise_eligible = output[
        "pairwise_eligible"
    ]
    period_errors = np.full_like(
        cost,
        np.nan,
    )

    for baseline_index in range(
        n_baseline
    ):
        baseline_frequency = max(
            abs(
                baseline_eigenvalues[
                    baseline_index
                ].imag
            ),
            np.finfo(float).eps,
        )

        for perturbed_index in range(
            n_perturbed
        ):
            similarity = Complex_Phasor_Compare(
                baseline_modes[
                    :,
                    baseline_index,
                ],
                perturbed_modes[
                    :,
                    perturbed_index,
                ],
            )

            relative_period_error = (
                abs(
                    perturbed_periods[
                        perturbed_index
                    ]
                    - baseline_periods[
                        baseline_index
                    ]
                )
                / baseline_periods[
                    baseline_index
                ]
            )

            growth_difference = (
                abs(
                    perturbed_eigenvalues[
                        perturbed_index
                    ].real
                    - baseline_eigenvalues[
                        baseline_index
                    ].real
                )
                / baseline_frequency
            )

            similarities[
                baseline_index,
                perturbed_index,
            ] = similarity
            period_errors[
                baseline_index,
                perturbed_index,
            ] = relative_period_error

            eligible = (
                np.isfinite(similarity)
                and np.isfinite(
                    relative_period_error
                )
                and np.isfinite(
                    growth_difference
                )
                and similarity
                >= matching_min_spatial_similarity
                and relative_period_error
                <= matching_max_relative_period_error
            )

            if not eligible:
                continue

            pairwise_eligible[
                baseline_index,
                perturbed_index,
            ] = True
            cost[
                baseline_index,
                perturbed_index,
            ] = (
                matching_spatial_weight
                * (1.0 - similarity)
                + matching_period_weight
                * relative_period_error
                + matching_growth_weight
                * growth_difference
            )

    row_indices, column_indices = (
        linear_sum_assignment(
            cost
        )
    )

    for baseline_index, perturbed_index in zip(
        row_indices,
        column_indices,
    ):
        if (
            cost[
                baseline_index,
                perturbed_index,
            ]
            >= invalid_cost
        ):
            continue

        output[
            "recovered"
        ][baseline_index] = True
        output[
            "candidate_index"
        ][baseline_index] = perturbed_index
        output[
            "cost"
        ][baseline_index] = cost[
            baseline_index,
            perturbed_index,
        ]
        output[
            "spatial_similarity"
        ][baseline_index] = similarities[
            baseline_index,
            perturbed_index,
        ]
        output[
            "relative_period_error"
        ][baseline_index] = period_errors[
            baseline_index,
            perturbed_index,
        ]

        for key in (
            "eigenvalue",
            "period",
            "growth_rate",
            "quality_factor",
            "power",
        ):
            output[
                key
            ][baseline_index] = perturbed_suite[
                key
            ][perturbed_index]

    return output


def finite_mode_quantiles(
    values,
    quantiles=(0.05, 0.50, 0.95),
):
    """Calculate per-baseline-mode quantiles over finite ensemble values."""

    values = np.asarray(
        values,
        dtype=float,
    )

    if values.ndim != 2:
        raise ValueError(
            "Ensemble summary input must have shape "
            "(n_realisations, n_baseline_modes)."
        )

    output = np.full(
        (
            values.shape[1],
            len(quantiles),
        ),
        np.nan,
        dtype=float,
    )

    for mode_index in range(
        values.shape[1]
    ):
        finite_values = values[
            :,
            mode_index,
        ]
        finite_values = finite_values[
            np.isfinite(
                finite_values
            )
        ]

        if finite_values.size:
            output[
                mode_index,
                :,
            ] = np.quantile(
                finite_values,
                quantiles,
            )

    return output


def reconstruct_candidate_suite(
    candidate_suite,
    evaluation_times,
    n_physical,
):
    """
    Reconstruct the real signal represented by retained physical phasors.

    The caller chooses whether this is the finite oscillatory analysis suite
    or the complete reconstruction suite that also includes static candidates.
    """

    reconstruction = np.zeros(
        (
            n_physical,
            evaluation_times.size,
        ),
        dtype=float,
    )

    for mode_index, eigenvalue in enumerate(
        candidate_suite[
            "eigenvalue"
        ]
    ):
        with np.errstate(
            over="ignore",
            invalid="ignore",
        ):
            temporal_dynamics = np.exp(
                eigenvalue
                * evaluation_times
            )
            contribution = np.real(
                candidate_suite[
                    "mode"
                ][
                    :,
                    mode_index,
                ][
                    :,
                    None,
                ]
                * temporal_dynamics[
                    None,
                    :,
                ]
            )

        if not np.all(
            np.isfinite(
                contribution
            )
        ):
            raise FloatingPointError(
                "A retained DMD mode produced a non-finite reconstruction."
            )

        reconstruction += contribution

    return reconstruction


def area_weighted_rms(
    snapshots,
    spatial_weights,
):
    """Return area-weighted RMS amplitude at each snapshot time."""

    snapshots = np.asarray(
        snapshots,
        dtype=float,
    )
    weights = np.asarray(
        spatial_weights,
        dtype=float,
    ).ravel()

    if (
        snapshots.ndim != 2
        or snapshots.shape[0] != weights.size
    ):
        raise ValueError(
            "Snapshots and spatial weights must align along space."
        )

    return np.sqrt(
        np.einsum(
            "i,ij->j",
            weights,
            np.square(
                snapshots
            ),
        )
        / np.sum(
            weights
        )
    )


def plot_singular_diagnostic(
    diagnostic_store,
    value_key,
    ylabel,
    title,
    log_y=False,
):
    """Plot CHAOS, perturbed members/median, and noise-only members/median."""

    chaos_values = diagnostic_store[
        "chaos"
    ][value_key]
    perturbed_values = [
        diagnostic[value_key]
        for diagnostic in diagnostic_store[
            "perturbed"
        ]
    ]
    noise_only_values = [
        diagnostic[value_key]
        for diagnostic in diagnostic_store[
            "noise_only"
        ]
    ]

    indices = np.arange(
        1,
        chaos_values.size + 1,
    )
    marker_interval = max(
        1,
        indices.size // 12,
    )

    fig, ax = plt.subplots(
        figsize=(8, 6)
    )

    if value_key == "magnitude":
        retained_count = min(
            int(
                baseline_suite[
                    "effective_svd_rank"
                ]
            ),
            indices.size,
        )

        ax.plot(
            indices,
            chaos_values,
            color="tab:blue",
            linewidth=1.6,
            alpha=0.65,
            zorder=5,
        )

        if retained_count:
            ax.scatter(
                indices[:retained_count],
                chaos_values[:retained_count],
                color="tab:blue",
                marker="x",
                s=34,
                label="CHAOS-8.6 retained by SVD rank",
                zorder=7,
            )

        if retained_count < indices.size:
            ax.scatter(
                indices[retained_count:],
                chaos_values[retained_count:],
                color="tab:blue",
                marker=".",
                s=20,
                alpha=0.55,
                label="CHAOS-8.6 truncated by SVD rank",
                zorder=6,
            )
    else:
        ax.plot(
            indices,
            chaos_values,
            color="tab:blue",
            marker="x",
            markevery=marker_interval,
            linewidth=2.0,
            markersize=6,
            label="CHAOS-8.6",
            zorder=6,
        )

    medians = {
        "perturbed": None,
        "noise_only": None,
    }

    if perturbed_values:
        perturbed_values = np.stack(
            perturbed_values,
            axis=0,
        )

        if perturbed_values.shape[1] != indices.size:
            raise ValueError(
                "Perturbed singular diagnostic lengths do not "
                "match the CHAOS baseline."
            )

        for member_index, member_values in enumerate(
            perturbed_values
        ):
            ax.plot(
                indices,
                member_values,
                color="tab:green",
                marker=".",
                markevery=marker_interval,
                linewidth=0.8,
                markersize=3,
                alpha=0.20,
                label=(
                    "CHAOS + perturbation members"
                    if member_index == 0
                    else None
                ),
                zorder=1,
            )

        medians[
            "perturbed"
        ] = np.median(
            perturbed_values,
            axis=0,
        )

        ax.plot(
            indices,
            medians[
                "perturbed"
            ],
            color="tab:green",
            marker="s",
            markevery=marker_interval,
            linewidth=2.2,
            markersize=5,
            label="CHAOS + perturbation median",
            zorder=7,
        )

    if noise_only_values:
        noise_only_values = np.stack(
            noise_only_values,
            axis=0,
        )

        if noise_only_values.shape[1] != indices.size:
            raise ValueError(
                "Perturbation-only singular diagnostic lengths do not "
                "match the CHAOS baseline."
            )

        for member_index, member_values in enumerate(
            noise_only_values
        ):
            ax.plot(
                indices,
                member_values,
                color="tab:purple",
                marker="v",
                markevery=marker_interval,
                linewidth=0.8,
                markersize=3,
                alpha=0.20,
                label=(
                    "Perturbation-only members"
                    if member_index == 0
                    else None
                ),
                zorder=1,
            )

        medians[
            "noise_only"
        ] = np.median(
            noise_only_values,
            axis=0,
        )

        ax.plot(
            indices,
            medians[
                "noise_only"
            ],
            color="tab:purple",
            marker="D",
            markevery=marker_interval,
            linewidth=2.2,
            markersize=5,
            label="Perturbation-only median",
            zorder=7,
        )

    embedding_label = (
        f"Hankel embedding d={hankel_d}"
        if hankel_embedding_flag
        else "no embedding"
    )
    matrix_label = (
        "leading snapshot matrix X"
        if DMD_algorithm in ("exact", "fbdmd")
        else "full snapshot matrix"
    )

    ax.set_xlabel(
        "Singular value index, k"
        if value_key == "magnitude"
        else "Number of singular values retained, k"
    )
    ax.set_ylabel(
        ylabel
    )
    ax.set_xlim(
        1,
        indices[-1],
    )

    if log_y:
        ax.set_yscale(
            "log"
        )

    if value_key == "magnitude":
        if noise_singular_value_threshold is not None:
            threshold_percent = (
                100.0
                * noise_rank_quantile
            )
            ax.axhline(
                noise_singular_value_threshold,
                color="black",
                linestyle="--",
                linewidth=1.3,
                label=(
                    f"q{threshold_percent:g} of noise-only "
                    r"$\sigma_1$"
                ),
                zorder=5,
            )

        if retained_count < indices.size:
            if (
                svd_rank_selection
                == "noise_threshold"
            ):
                rank_line_label = (
                    f"Base noise rank {noise_base_svd_rank} × "
                    f"{noise_rank_multiplier:g} = "
                    f"PyDMD rank {svd_rank}; effective rank "
                    f"{retained_count}"
                )
            else:
                rank_line_label = (
                    f"Fixed PyDMD rank {svd_rank}; "
                    f"effective rank {retained_count}"
                )

            ax.axvline(
                retained_count + 0.5,
                color="0.35",
                linestyle=":",
                linewidth=1.1,
                label=rank_line_label,
                zorder=5,
            )

    ax.set_title(
        f"{title}\n"
        f"{DMD_algorithm}; {embedding_label}; "
        f"{matrix_label}; Nmax={Nmax}"
    )
    ax.grid(
        alpha=0.25,
        which="both",
    )
    ax.legend(
        loc="best"
    )
    fig.tight_layout()

    return fig, ax, medians


def plot_degree_sweep_periods(
    results,
    figsize=(10, 8),
):
    """Plot unperturbed and perturbed finite periods at every degree."""

    degrees = np.asarray(
        results[
            "degree_truncations"
        ],
        dtype=int,
    )
    reference_periods = np.asarray(
        results[
            "by_degree"
        ][int(degrees[-1])][
            "baseline"
        ][
            "recovered_period"
        ],
        dtype=float,
    )
    reference_periods = np.unique(
        reference_periods[
            np.isfinite(
                reference_periods
            )
            & (
                reference_periods
                > 0.0
            )
        ]
    )

    fig, axes = plt.subplots(
        2,
        1,
        figsize=figsize,
        sharex=True,
        sharey=True,
    )

    for ax, result_key, title, color, marker in (
        (
            axes[0],
            "baseline",
            "Unperturbed CHAOS",
            "tab:blue",
            "x",
        ),
        (
            axes[1],
            "perturbed",
            "Covariance-perturbed CHAOS ensemble",
            "tab:green",
            ".",
        ),
    ):
        reference_label_used = False

        for reference_period in reference_periods:
            ax.axvline(
                reference_period,
                color="0.45",
                linestyle="--",
                linewidth=0.9,
                alpha=0.5,
                label=(
                    "Nmax unperturbed CHAOS "
                    "reference periods"
                    if not reference_label_used
                    else None
                ),
                zorder=1,
            )
            reference_label_used = True

        recovered_label_used = False

        for degree in degrees:
            degree_store = results[
                "by_degree"
            ][int(degree)]
            candidate_stores = (
                [degree_store["baseline"]]
                if result_key == "baseline"
                else degree_store["perturbed"]
            )

            for candidate_store in candidate_stores:
                if candidate_store is None:
                    continue

                periods = np.asarray(
                    candidate_store[
                        "recovered_period"
                    ],
                    dtype=float,
                )
                valid = (
                    np.isfinite(periods)
                    & (
                        periods > 0.0
                    )
                )

                if not np.any(valid):
                    continue

                ax.scatter(
                    periods[valid],
                    np.full(
                        np.count_nonzero(
                            valid
                        ),
                        degree,
                        dtype=float,
                    ),
                    color=color,
                    marker=marker,
                    s=(
                        30
                        if result_key
                        == "baseline"
                        else 18
                    ),
                    alpha=(
                        0.85
                        if result_key
                        == "baseline"
                        else 0.28
                    ),
                    label=(
                        "Recovered candidates"
                        if not recovered_label_used
                        else None
                    ),
                    rasterized=(
                        result_key
                        == "perturbed"
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
            title
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

        handles, labels = (
            ax.get_legend_handles_labels()
        )
        if handles:
            ax.legend(
                loc="best"
            )

    axes[-1].set_xscale(
        "log"
    )
    axes[-1].set_xlabel(
        "Recovered period [yr]"
    )

    if results[
        "metadata"
    ][
        "period_limit_enabled"
    ]:
        axes[-1].set_xlim(
            results[
                "metadata"
            ][
                "period_lower_bound"
            ],
            results[
                "metadata"
            ][
                "period_upper_bound"
            ],
        )

    fig.suptitle(
        "CHAOS DMD candidates across cumulative "
        "spherical-harmonic degree truncations"
    )
    fig.tight_layout()

    return fig, axes


def plot_degree_sweep_static_counts(
    results,
    figsize=(9, 6),
):
    """Plot raw static-candidate counts at every degree."""

    degrees = np.asarray(
        results[
            "degree_truncations"
        ],
        dtype=int,
    )
    baseline_counts = np.asarray([
        results[
            "by_degree"
        ][int(degree)][
            "baseline"
        ][
            "static_recovered_period"
        ].size
        for degree in degrees
    ])

    fig, ax = plt.subplots(
        figsize=figsize
    )
    ax.plot(
        degrees,
        baseline_counts,
        color="tab:blue",
        marker="x",
        linewidth=1.8,
        label="Unperturbed CHAOS",
        zorder=5,
    )

    perturbed_medians = []
    perturbed_label_used = False

    for degree in degrees:
        perturbed_stores = results[
            "by_degree"
        ][int(degree)][
            "perturbed"
        ]
        counts = np.asarray([
            candidate_store[
                "static_recovered_period"
            ].size
            for candidate_store
            in perturbed_stores
        ], dtype=float)

        if counts.size == 0:
            perturbed_medians.append(
                np.nan
            )
            continue

        ax.scatter(
            np.full(
                counts.size,
                degree,
                dtype=float,
            ),
            counts,
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
        perturbed_medians.append(
            np.median(
                counts
            )
        )

    perturbed_medians = np.asarray(
        perturbed_medians,
        dtype=float,
    )
    valid_median = np.isfinite(
        perturbed_medians
    )

    if np.any(
        valid_median
    ):
        ax.plot(
            degrees[
                valid_median
            ],
            perturbed_medians[
                valid_median
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
        "CHAOS infinite-period DMD candidates across degree truncations"
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
    """Plot the independently selected or configured rank at each degree."""

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
            "svd_rank"
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
        "Selected SVD rank"
    )
    ax.set_xticks(
        degrees
    )
    ax.yaxis.set_major_locator(
        MaxNLocator(
            integer=True
        )
    )
    rank_selection_method = results[
        "metadata"
    ][
        "svd_rank_selection"
    ]
    ax.set_title(
        (
            "CHAOS DMD SVD rank evaluated independently "
            "at each degree truncation\n"
            f"noise-rank multiplier = "
            f"{results['metadata']['noise_rank_multiplier']:g}"
        )
        if rank_selection_method == "noise_threshold"
        else (
            "Configured fixed CHAOS DMD SVD rank "
            "across degree truncations"
        )
    )
    ax.grid(
        alpha=0.25
    )
    fig.tight_layout()

    return fig, ax


def plot_ensemble_candidate_period_power(
    baseline_suite,
    perturbed_suites,
    figsize=(8, 6),
):
    """Plot all perturbed candidates without one-to-one mode matching."""

    fig, ax = plt.subplots(
        figsize=figsize
    )

    baseline_periods = np.asarray(
        baseline_suite[
            "period"
        ],
        dtype=float,
    )
    baseline_power = np.asarray(
        baseline_suite[
            "power"
        ],
        dtype=float,
    )
    baseline_valid = (
        np.isfinite(
            baseline_periods
        )
        & (
            baseline_periods > 0.0
        )
        & np.isfinite(
            baseline_power
        )
        & (
            baseline_power > 0.0
        )
    )

    ax.scatter(
        baseline_periods[
            baseline_valid
        ],
        baseline_power[
            baseline_valid
        ],
        color="tab:blue",
        marker="o",
        s=55,
        edgecolor="black",
        linewidth=0.5,
        label="Unperturbed CHAOS candidates",
        zorder=5,
    )

    perturbed_label_used = False

    for perturbed_suite in perturbed_suites:
        periods = np.asarray(
            perturbed_suite[
                "period"
            ],
            dtype=float,
        )
        power = np.asarray(
            perturbed_suite[
                "power"
            ],
            dtype=float,
        )
        valid = (
            np.isfinite(periods)
            & (
                periods > 0.0
            )
            & np.isfinite(power)
            & (
                power > 0.0
            )
        )

        if not np.any(valid):
            continue

        ax.scatter(
            periods[valid],
            power[valid],
            color="tab:green",
            marker=".",
            s=20,
            alpha=0.28,
            edgecolors="none",
            label=(
                "All perturbed candidates"
                if not perturbed_label_used
                else None
            ),
            rasterized=True,
            zorder=2,
        )
        perturbed_label_used = True

    ax.set_xscale(
        "log"
    )
    ax.set_yscale(
        "log"
    )
    ax.set_xlabel(
        "Period [yr]"
    )
    ax.set_ylabel(
        r"Record-mean area-weighted SV power [(nT/yr)$^2$]"
    )
    ax.set_title(
        (
            "CHAOS covariance-ensemble candidate population\n"
            "all retained candidates shown without matching"
        )
        if perturbed_suites
        else (
            "Unperturbed CHAOS candidate population\n"
            "perturbation ensemble disabled"
        )
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


def plot_pairwise_similarity_heatmaps(
    baseline_suite,
    perturbed_suites,
    ensemble_matches,
):
    """Plot one complete baseline-versus-perturbed matrix per realisation."""

    figures = []
    baseline_periods = baseline_suite[
        "period"
    ]

    for realisation_index, (
        perturbed_suite,
        match,
    ) in enumerate(
        zip(
            perturbed_suites,
            ensemble_matches,
        )
    ):
        similarities = match[
            "pairwise_spatial_similarity"
        ]
        fig, ax = plt.subplots(
            figsize=(10, 7)
        )

        if (
            similarities.shape[0] == 0
            or similarities.shape[1] == 0
        ):
            ax.text(
                0.5,
                0.5,
                "No finite oscillatory candidates available",
                transform=ax.transAxes,
                ha="center",
                va="center",
            )
            ax.set_axis_off()
        else:
            cmap = plt.colormaps[
                "viridis"
            ].copy()
            cmap.set_bad(
                "0.75"
            )
            image = ax.imshow(
                np.ma.masked_invalid(
                    similarities
                ),
                origin="upper",
                aspect="auto",
                cmap=cmap,
                vmin=0.0,
                vmax=1.0,
            )
            fig.colorbar(
                image,
                ax=ax,
                label="Complex-phasor spatial similarity",
            )

            n_baseline, n_perturbed = (
                similarities.shape
            )

            if similarities.size <= 400:
                for baseline_index in range(
                    n_baseline
                ):
                    for candidate_index in range(
                        n_perturbed
                    ):
                        similarity = similarities[
                            baseline_index,
                            candidate_index,
                        ]

                        if not np.isfinite(
                            similarity
                        ):
                            continue

                        ax.text(
                            candidate_index,
                            baseline_index,
                            f"{similarity:.2f}",
                            ha="center",
                            va="center",
                            fontsize=7,
                            color=(
                                "white"
                                if similarity < 0.45
                                else "black"
                            ),
                        )

            y_step = max(
                1,
                n_baseline // 12,
            )
            x_step = max(
                1,
                n_perturbed // 12,
            )
            y_ticks = np.arange(
                0,
                n_baseline,
                y_step,
            )
            x_ticks = np.arange(
                0,
                n_perturbed,
                x_step,
            )
            ax.set_yticks(
                y_ticks
            )
            ax.set_yticklabels([
                (
                    f"B{index + 1}\n"
                    f"{baseline_periods[index]:.2g} yr"
                )
                for index in y_ticks
            ])
            ax.set_xticks(
                x_ticks
            )
            ax.set_xticklabels(
                [
                    (
                        f"P{index + 1}\n"
                        f"{perturbed_suite['period'][index]:.2g} yr"
                    )
                    for index in x_ticks
                ],
                rotation=45,
                ha="right",
            )

            accepted_rows = np.where(
                match[
                    "recovered"
                ]
            )[0]

            for baseline_index in accepted_rows:
                candidate_index = int(
                    match[
                        "candidate_index"
                    ][baseline_index]
                )
                ax.scatter(
                    candidate_index,
                    baseline_index,
                    marker="s",
                    s=120,
                    facecolors="none",
                    edgecolors="red",
                    linewidths=1.4,
                )

            ax.set_xlabel(
                "Perturbed candidate index and period"
            )
            ax.set_ylabel(
                "Unperturbed baseline mode and period"
            )

        ax.set_title(
            "CHAOS baseline-versus-perturbed spatial similarity\n"
            f"covariance realisation {realisation_index}"
        )
        fig.tight_layout()
        figures.append(
            (
                fig,
                ax,
            )
        )

    return figures


# %% ------------------------------------------------------
# LOAD CHAOS-8.6 SV GAUSS COEFFICIENTS
# ------------------------------------------------------

chaos_path = (
    Path(CHAOS_DIR)
    / "CHAOS-8.6.mat"
)

if not chaos_path.exists():
    raise FileNotFoundError(
        f"Could not find CHAOS-8.6 model at {chaos_path}."
    )

chaos_model = cp.load_CHAOS_matfile(
    str(
        chaos_path
    )
)

chaos_sv_full = np.asarray(
    chaos_model.synth_coeffs_tdep(
        times_mjd2000,
        nmax=CHAOS_covariance_Nmax,
        deriv=1,
        extrapolate="off",
    ),
    dtype=float,
)

expected_full_coefficients = (
    (CHAOS_covariance_Nmax + 1)**2
    - 1
)

if chaos_sv_full.shape != (
    times_mjd2000.size,
    expected_full_coefficients,
):
    raise ValueError(
        "Unexpected CHAOS SV coefficient shape. Expected "
        f"{(times_mjd2000.size, expected_full_coefficients)}, "
        f"received {chaos_sv_full.shape}."
    )

if not np.all(
    np.isfinite(
        chaos_sv_full
    )
):
    raise ValueError(
        "CHAOS SV coefficients contain non-finite values."
    )


# %% ------------------------------------------------------
# BUILD PAIRED UNCERTAINTY INPUTS
# ------------------------------------------------------

queue = [
    (
        "chaos",
        chaos_sv_full,
    )
]

if Cov_full.shape != (
    chaos_sv_full.shape[0],
    chaos_sv_full.shape[1],
    chaos_sv_full.shape[1],
):
    raise ValueError(
        "CHAOS covariance and coefficient arrays do not align. "
        f"Covariance shape {Cov_full.shape}; coefficient shape "
        f"{chaos_sv_full.shape}."
    )

noise_bank_size = max(
    n_noise_rank_realisations,
    n_realisations
    if ensemble_flag
    else 0,
)

noise_only_bank = Perturbation_Generate(
    chaos_sv_full,
    n_realisations=
        noise_bank_size,
    seed=noise_seed,
    temporal_z=
        noise_temporal_model,
    tau=noise_tau_years,
    dt=dt_years,
    covariance_used=Cov_full,
    just_noise=True,
)

noise_only_rank_bank = (
    noise_only_bank[
        :n_noise_rank_realisations
    ]
)

if ensemble_flag:
    perturbed_chaos = (
        chaos_sv_full[
            None,
            :,
            :,
        ]
        + noise_only_bank[
            :n_realisations
        ]
    )

    queue.extend(
        (
            "perturbed",
            realisation,
        )
        for realisation in perturbed_chaos
    )


# %% ------------------------------------------------------
# TIME AND GAUSS-TO-GRID OPERATORS
# ------------------------------------------------------

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

if analysis_times_absolute.size < 2:
    raise ValueError(
        "At least two selected CHAOS snapshots are required."
    )

if (
    hankel_embedding_flag
    and hankel_d
    >= analysis_times_absolute.size
):
    raise ValueError(
        f"hankel_d={hankel_d} leaves fewer than two embedded "
        f"snapshots from the "
        f"{analysis_times_absolute.size} selected snapshots."
    )

snapshot_differences = np.diff(
    analysis_times_absolute
)
dt_snapshot = float(
    np.median(
        snapshot_differences
    )
)

if not np.allclose(
    snapshot_differences,
    dt_snapshot,
    rtol=1e-8,
    atol=1e-10,
):
    raise ValueError(
        "The selected CHAOS snapshots are not uniformly sampled."
    )

if (
    video_plot
    or all_modes_video_plot
):
    video_frame_step = int(
        round(
            video_frame_spacing_years
            / dt_snapshot
        )
    )

    if (
        video_frame_step < 1
        or not np.isclose(
            video_frame_step
            * dt_snapshot,
            video_frame_spacing_years,
            rtol=1e-10,
            atol=1e-10,
        )
    ):
        raise ValueError(
            "video_frame_spacing_years must be an integer multiple "
            "of the selected CHAOS snapshot spacing. "
            f"Received video_frame_spacing_years="
            f"{video_frame_spacing_years}, "
            f"dt_snapshot={dt_snapshot}."
        )

dmd_evaluation_times = (
    analysis_times_absolute
    - analysis_times_absolute[0]
)

A_r_current = Truncate_Gauss_Coeffs(
    A_20_dict[
        "r"
    ],
    tmax=Nmax,
)

expected_band_coefficients = (
    (Nmax + 1)**2
    - 1
)

if A_r_current.shape != (
    state_shape[0]
    * state_shape[1],
    expected_band_coefficients,
):
    raise ValueError(
        "Unexpected radial SV design-matrix shape. Expected "
        f"{(state_shape[0] * state_shape[1], expected_band_coefficients)}, "
        f"received {A_r_current.shape}."
    )

sv_grid_weights = (
    W2D.ravel()
)


# %% ------------------------------------------------------
# PER-DEGREE RANK SELECTION AND OPTIONAL LOWER-DEGREE SWEEP
# ------------------------------------------------------

(
    svd_rank,
    noise_base_svd_rank,
    noise_singular_value_threshold,
    nmax_baseline_singular_diagnostic,
    nmax_noise_singular_diagnostics,
) = select_degree_svd_rank(
    Nmax,
    A_r_current,
    collect_diagnostics=
        singular_value_plot_flag,
)

print(
    "\nCHAOS SVD RANK SELECTION\n"
    f"Method: {svd_rank_selection}\n"
    f"PyDMD Nmax rank argument: {svd_rank}"
)
if noise_singular_value_threshold is not None:
    print(
        "Noise-only realisations: "
        f"{n_noise_rank_realisations}\n"
        "Leading-noise quantile: "
        f"q={noise_rank_quantile:.3f}\n"
        "Noise singular-value threshold: "
        f"{noise_singular_value_threshold:.6e}\n"
        f"Base noise-selected rank: {noise_base_svd_rank}\n"
        f"Noise-rank multiplier: {noise_rank_multiplier:g}"
    )

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
                svd_rank_selection
            ),
            "noise_rank_multiplier": float(
                noise_rank_multiplier
            ),
            "ensemble_enabled": bool(
                ensemble_flag
            ),
            "n_realisations": (
                int(n_realisations)
                if ensemble_flag
                else 0
            ),
            "noise_rank_quantile": float(
                noise_rank_quantile
            ),
            "noise_rank_realisations": int(
                n_noise_rank_realisations
            ),
        },
        "degree_truncations": (
            degree_truncations_tested.copy()
        ),
        "by_degree": {},
    }

    for truncation_degree in tqdm(
        degree_truncations_tested[:-1],
        desc="CHAOS cumulative degree sweep",
    ):
        truncation_degree = int(
            truncation_degree
        )
        projection_operator = (
            Truncate_Gauss_Coeffs(
                A_20_dict[
                    "r"
                ],
                tmax=truncation_degree,
            )
        )
        (
            degree_rank,
            degree_base_noise_rank,
            degree_threshold,
            _,
            _,
        ) = select_degree_svd_rank(
            truncation_degree,
            projection_operator,
            collect_diagnostics=False,
        )

        degree_store = {
            "svd_rank": int(
                degree_rank
            ),
            "noise_base_svd_rank": (
                int(
                    degree_base_noise_rank
                )
                if degree_base_noise_rank
                is not None
                else None
            ),
            "noise_singular_value_threshold": (
                float(degree_threshold)
                if degree_threshold
                is not None
                else None
            ),
            "baseline": None,
            "perturbed": [],
        }
        perturbed_index = 0

        for result_type, gnm_input in tqdm(
            queue,
            desc=(
                f"CHAOS degrees 1-{truncation_degree}, "
                f"rank={degree_rank}"
            ),
            leave=False,
        ):
            sv_input = prepare_chaos_sv_input(
                gnm_input,
                truncation_degree,
                projection_operator,
            )
            candidate_suite = fit_candidate_suite(
                sv_input,
                selected_svd_rank=
                    degree_rank,
                dt_snapshot=dt_snapshot,
                evaluation_times=
                    dmd_evaluation_times,
                spatial_weights=
                    sv_grid_weights,
            )
            candidate_store = (
                make_degree_sweep_candidate_store(
                    candidate_suite
                )
            )

            if result_type == "chaos":
                degree_store[
                    "baseline"
                ] = candidate_store
            elif result_type == "perturbed":
                candidate_store[
                    "realisation_index"
                ] = int(
                    perturbed_index
                )
                degree_store[
                    "perturbed"
                ].append(
                    candidate_store
                )
                perturbed_index += 1
            else:
                raise ValueError(
                    "Unknown lower-degree input category "
                    f"{result_type!r}."
                )

            del sv_input
            del candidate_suite
            gc.collect()

        degree_sweep_results[
            "by_degree"
        ][truncation_degree] = degree_store

    degree_sweep_results[
        "by_degree"
    ][int(Nmax)] = {
        "svd_rank": int(
            svd_rank
        ),
        "noise_base_svd_rank": (
            int(
                noise_base_svd_rank
            )
            if noise_base_svd_rank
            is not None
            else None
        ),
        "noise_singular_value_threshold": (
            float(
                noise_singular_value_threshold
            )
            if noise_singular_value_threshold
            is not None
            else None
        ),
        "baseline": None,
        "perturbed": [],
    }


# %% ------------------------------------------------------
# PREPROCESS AND FIT EVERY INPUT
# ------------------------------------------------------

baseline_suite = None
baseline_sv_input = None
perturbed_suites = []

singular_diagnostics = {
    "chaos": (
        nmax_baseline_singular_diagnostic
        if singular_value_plot_flag
        else None
    ),
    "perturbed": [],
    "noise_only": (
        nmax_noise_singular_diagnostics
        if singular_value_plot_flag
        else []
    ),
}

for result_type, gnm_input in tqdm(
    queue,
    desc=(
        "CHAOS DMD and uncertainty "
        f"(Nmax={Nmax})"
    ),
):
    sv_input = prepare_chaos_sv_input(
        gnm_input,
        Nmax,
        A_r_current,
    )

    if (
        singular_value_plot_flag
        and result_type
        == "perturbed"
    ):
        diagnostic = calculate_singular_diagnostic(
            sv_input
        )
        singular_diagnostics[
            "perturbed"
        ].append(
            diagnostic
        )

    candidate_suite = fit_candidate_suite(
        sv_input,
        selected_svd_rank=svd_rank,
        dt_snapshot=dt_snapshot,
        evaluation_times=
            dmd_evaluation_times,
        spatial_weights=
            sv_grid_weights,
    )

    if result_type == "chaos":
        baseline_suite = candidate_suite
        baseline_sv_input = (
            sv_input.copy()
        )
    elif result_type == "perturbed":
        perturbed_suites.append(
            candidate_suite
        )
    else:
        raise ValueError(
            f"Unknown input category {result_type!r}."
        )

    if degree_sweep_flag:
        degree_candidate_store = (
            make_degree_sweep_candidate_store(
                candidate_suite
            )
        )
        nmax_degree_store = (
            degree_sweep_results[
                "by_degree"
            ][int(Nmax)]
        )

        if result_type == "chaos":
            nmax_degree_store[
                "baseline"
            ] = degree_candidate_store
        elif result_type == "perturbed":
            degree_candidate_store[
                "realisation_index"
            ] = len(
                nmax_degree_store[
                    "perturbed"
                ]
            )
            nmax_degree_store[
                "perturbed"
            ].append(
                degree_candidate_store
            )

    del sv_input
    gc.collect()


# %% ------------------------------------------------------
# MATCH UNCERTAINTY MODES TO THE CHAOS BASELINE
# ------------------------------------------------------

if (
    baseline_suite is None
    or baseline_sv_input is None
):
    raise RuntimeError(
        "The unperturbed CHAOS DMD fit did not produce a baseline result."
    )

if baseline_sv_input.shape != (
    A_r_current.shape[0],
    analysis_times_absolute.size,
):
    raise ValueError(
        "Stored baseline SV grid does not align with analysis times."
    )

n_baseline_modes = baseline_suite[
    "eigenvalue"
].size

if n_baseline_modes == 0:
    raise RuntimeError(
        "No oscillatory baseline modes were retained. Change the rank, "
        "embedding, filter, or period settings before analysis."
    )

baseline_suite[
    "mode_number"
] = np.arange(
    1,
    n_baseline_modes + 1,
)

ensemble_matches = [
    match_candidate_suites(
        baseline_suite,
        perturbed_suite,
    )
    for perturbed_suite in perturbed_suites
]

if ensemble_matches:
    recovered_matrix = np.stack([
        match[
            "recovered"
        ]
        for match in ensemble_matches
    ])

    ensemble_summary = {
        "recovery_fraction": np.mean(
            recovered_matrix,
            axis=0,
        ),
    }

    for key in (
        "period",
        "growth_rate",
        "quality_factor",
        "power",
        "spatial_similarity",
        "relative_period_error",
    ):
        values = np.stack([
            match[key]
            for match in ensemble_matches
        ])

        ensemble_summary[
            key
        ] = {
            "values": values,
            "p05_median_p95":
                finite_mode_quantiles(
                    values
                ),
        }
else:
    ensemble_summary = {
        "recovery_fraction": np.full(
            n_baseline_modes,
            np.nan,
        ),
    }


# %% ------------------------------------------------------
# STORE COMPLETE ANALYSIS RESULTS
# ------------------------------------------------------

CHAOS_DMD_results = {
    "metadata": {
        "model": "CHAOS-8.6",
        "data_component": "radial secular variation at the CMB",
        "coefficient_units": "CHAOSMagPy SV output",
        "DMD_algorithm": DMD_algorithm,
        "svd_rank_selection":
            svd_rank_selection,
        "noise_rank_multiplier": float(
            noise_rank_multiplier
        ),
        "fixed_svd_rank": int(
            fixed_svd_rank
        ),
        "svd_rank": int(
            svd_rank
        ),
        "noise_base_svd_rank": (
            int(
                noise_base_svd_rank
            )
            if noise_base_svd_rank
            is not None
            else None
        ),
        "effective_svd_rank": {
            "baseline": int(
                baseline_suite[
                    "effective_svd_rank"
                ]
            ),
            "perturbed": [
                int(
                    suite[
                        "effective_svd_rank"
                    ]
                )
                for suite in perturbed_suites
            ],
        },
        "noise_rank_quantile": float(
            noise_rank_quantile
        ),
        "noise_rank_realisations": int(
            n_noise_rank_realisations
        ),
        "noise_singular_value_threshold": (
            float(
                noise_singular_value_threshold
            )
            if noise_singular_value_threshold
            is not None
            else None
        ),
        "Nmax": Nmax,
        "high_quality_record": bool(
            high_q_flag
        ),
        "remove_temporal_mean": bool(
            remove_temporal_mean_flag
        ),
        "temporal_filter": bool(
            filter_flag
        ),
        "hankel_embedding": bool(
            hankel_embedding_flag
        ),
        "hankel_d": (
            int(hankel_d)
            if hankel_embedding_flag
            else None
        ),
        "period_limit": (
            (
                float(period_lower_bound),
                float(period_upper_bound),
            )
            if period_limit_flag
            else None
        ),
        "analysis_times_absolute":
            analysis_times_absolute.copy(),
        "dt_snapshot_years":
            dt_snapshot,
        "matching": {
            "spatial_weight":
                matching_spatial_weight,
            "period_weight":
                matching_period_weight,
            "growth_weight":
                matching_growth_weight,
            "max_relative_period_error":
                matching_max_relative_period_error,
            "min_spatial_similarity":
                matching_min_spatial_similarity,
        },
    },
    "baseline": baseline_suite,
    "perturbed_suites": perturbed_suites,
    "ensemble_matches": ensemble_matches,
    "ensemble_summary": ensemble_summary,
    "singular_diagnostics": singular_diagnostics,
    "degree_sweep": degree_sweep_results,
}


# %% ------------------------------------------------------
# PRINT BASELINE AND UNCERTAINTY SUMMARY
# ------------------------------------------------------

print(
    "\nCHAOS-8.6 DMD BASELINE"
)
print(
    f"Input shape: {baseline_sv_input.shape}; "
    f"dt={dt_snapshot:.4f} yr; "
    f"record={dmd_evaluation_times[-1]:.2f} yr"
)
print(
    f"Retained oscillatory modes: {n_baseline_modes}"
)
print(
    "Retained static modes for reconstruction: "
    f"{baseline_suite['static']['eigenvalue'].size}"
)

for mode_index in range(
    n_baseline_modes
):
    print(
        f"Mode {mode_index + 1:2d}: "
        f"period={baseline_suite['period'][mode_index]:7.3f} yr, "
        f"growth={baseline_suite['growth_rate'][mode_index]:+.4e} yr^-1, "
        f"Q={baseline_suite['quality_factor'][mode_index]:.3g}, "
        f"power={baseline_suite['power'][mode_index]:.4e}, "
        f"ensemble recovery="
        f"{ensemble_summary['recovery_fraction'][mode_index]:.1%}"
        if ensemble_matches
        else
        f"Mode {mode_index + 1:2d}: "
        f"period={baseline_suite['period'][mode_index]:7.3f} yr, "
        f"growth={baseline_suite['growth_rate'][mode_index]:+.4e} yr^-1, "
        f"Q={baseline_suite['quality_factor'][mode_index]:.3g}, "
        f"power={baseline_suite['power'][mode_index]:.4e}"
    )


# %% ------------------------------------------------------
# PLOT BASELINE CONTINUOUS EIGENVALUES
# ------------------------------------------------------

baseline_periods = baseline_suite[
    "period"
]
baseline_eigenvalues = baseline_suite[
    "eigenvalue"
]
baseline_power = baseline_suite[
    "power"
]
baseline_quality = baseline_suite[
    "quality_factor"
]

period_norm = Normalize(
    vmin=np.min(
        baseline_periods
    ),
    vmax=np.max(
        baseline_periods
    ),
)

fig_eigenvalues, ax_eigenvalues = plt.subplots(
    figsize=(7.5, 6)
)

eigenvalue_scatter = ax_eigenvalues.scatter(
    baseline_eigenvalues.real,
    baseline_eigenvalues.imag,
    c=baseline_periods,
    cmap="viridis",
    norm=period_norm,
    s=75,
    edgecolor="black",
    linewidth=0.6,
)

for mode_index, eigenvalue in enumerate(
    baseline_eigenvalues
):
    ax_eigenvalues.annotate(
        str(
            mode_index + 1
        ),
        (
            eigenvalue.real,
            eigenvalue.imag,
        ),
        xytext=(4, 4),
        textcoords="offset points",
        fontsize=8,
    )

ax_eigenvalues.axhline(
    0.0,
    color="black",
    linewidth=0.8,
)
ax_eigenvalues.axvline(
    0.0,
    color="black",
    linewidth=0.8,
)
ax_eigenvalues.set_xlabel(
    r"Growth/decay rate, Re$(\lambda)$ [yr$^{-1}$]"
)
ax_eigenvalues.set_ylabel(
    r"Angular frequency, Im$(\lambda)$ [rad yr$^{-1}$]"
)
ax_eigenvalues.set_title(
    "CHAOS-8.6 retained continuous-time DMD eigenvalues"
)
ax_eigenvalues.grid(
    alpha=0.25
)
fig_eigenvalues.colorbar(
    eigenvalue_scatter,
    ax=ax_eigenvalues,
    label="Period [yr]",
)
fig_eigenvalues.tight_layout()


# %% ------------------------------------------------------
# PLOT BASELINE PERIOD, POWER, GROWTH, AND QUALITY
# ------------------------------------------------------

growth_limit = max(
    np.max(
        np.abs(
            baseline_eigenvalues.real
        )
    ),
    np.finfo(float).eps,
)

fig_period_power, ax_period_power = plt.subplots(
    figsize=(7.5, 6)
)

valid_power = (
    np.isfinite(
        baseline_power
    )
    & (baseline_power > 0.0)
)

power_scatter = ax_period_power.scatter(
    baseline_periods[
        valid_power
    ],
    baseline_power[
        valid_power
    ],
    c=baseline_eigenvalues.real[
        valid_power
    ],
    cmap="coolwarm",
    norm=Normalize(
        vmin=-growth_limit,
        vmax=growth_limit,
    ),
    s=75,
    edgecolor="black",
    linewidth=0.6,
)

for mode_index in np.where(
    valid_power
)[0]:
    ax_period_power.annotate(
        str(
            mode_index + 1
        ),
        (
            baseline_periods[
                mode_index
            ],
            baseline_power[
                mode_index
            ],
        ),
        xytext=(4, 4),
        textcoords="offset points",
        fontsize=8,
    )

ax_period_power.set_xscale(
    "log"
)
ax_period_power.set_yscale(
    "log"
)
ax_period_power.set_xlabel(
    "Period [yr]"
)
ax_period_power.set_ylabel(
    r"Record-mean area-weighted SV power [(nT/yr)$^2$]"
)
ax_period_power.set_title(
    "CHAOS-8.6 retained DMD mode power"
)
ax_period_power.grid(
    alpha=0.25,
    which="both",
)
fig_period_power.colorbar(
    power_scatter,
    ax=ax_period_power,
    label=r"Growth/decay rate [yr$^{-1}$]",
)
fig_period_power.tight_layout()

fig_quality, ax_quality = plt.subplots(
    figsize=(7.5, 6)
)

valid_quality = (
    np.isfinite(
        baseline_quality
    )
    & (baseline_quality > 0.0)
)

ax_quality.scatter(
    baseline_periods[
        valid_quality
    ],
    baseline_quality[
        valid_quality
    ],
    color="tab:blue",
    marker="x",
    s=70,
)

for mode_index in np.where(
    valid_quality
)[0]:
    ax_quality.annotate(
        str(
            mode_index + 1
        ),
        (
            baseline_periods[
                mode_index
            ],
            baseline_quality[
                mode_index
            ],
        ),
        xytext=(4, 4),
        textcoords="offset points",
        fontsize=8,
    )

ax_quality.set_xscale(
    "log"
)
ax_quality.set_yscale(
    "log"
)
ax_quality.set_xlabel(
    "Period [yr]"
)
ax_quality.set_ylabel(
    "Quality factor"
)
ax_quality.set_title(
    "CHAOS-8.6 retained DMD mode quality factors"
)
ax_quality.grid(
    alpha=0.25,
    which="both",
)
fig_quality.tight_layout()


# %% ------------------------------------------------------
# PLOT ENSEMBLE STABILITY
# ------------------------------------------------------

fig_stability = None
axes_stability = None

if ensemble_matches:
    period_quantiles = ensemble_summary[
        "period"
    ][
        "p05_median_p95"
    ]
    similarity_quantiles = ensemble_summary[
        "spatial_similarity"
    ][
        "p05_median_p95"
    ]

    fig_stability, axes_stability = plt.subplots(
        3,
        1,
        figsize=(8, 12),
        sharex=True,
    )

    valid_period_summary = np.all(
        np.isfinite(
            period_quantiles
        ),
        axis=1,
    )

    period_error_lower = (
        period_quantiles[
            valid_period_summary,
            1,
        ]
        - period_quantiles[
            valid_period_summary,
            0,
        ]
    )
    period_error_upper = (
        period_quantiles[
            valid_period_summary,
            2,
        ]
        - period_quantiles[
            valid_period_summary,
            1,
        ]
    )

    axes_stability[0].errorbar(
        baseline_periods[
            valid_period_summary
        ],
        period_quantiles[
            valid_period_summary,
            1,
        ],
        yerr=np.vstack([
            period_error_lower,
            period_error_upper,
        ]),
        color="tab:green",
        marker="s",
        linestyle="None",
        capsize=3,
        label="Perturbed p05–median–p95",
    )

    reference_period_min = np.min(
        baseline_periods
    )
    reference_period_max = np.max(
        baseline_periods
    )
    axes_stability[0].plot(
        [
            reference_period_min,
            reference_period_max,
        ],
        [
            reference_period_min,
            reference_period_max,
        ],
        color="black",
        linestyle="--",
        linewidth=1.0,
        label="Unchanged period",
    )
    axes_stability[0].set_ylabel(
        "Matched period [yr]"
    )
    axes_stability[0].set_yscale(
        "log"
    )
    axes_stability[0].legend(
        loc="best"
    )

    valid_similarity_summary = np.all(
        np.isfinite(
            similarity_quantiles
        ),
        axis=1,
    )

    similarity_error_lower = (
        similarity_quantiles[
            valid_similarity_summary,
            1,
        ]
        - similarity_quantiles[
            valid_similarity_summary,
            0,
        ]
    )
    similarity_error_upper = (
        similarity_quantiles[
            valid_similarity_summary,
            2,
        ]
        - similarity_quantiles[
            valid_similarity_summary,
            1,
        ]
    )

    axes_stability[1].errorbar(
        baseline_periods[
            valid_similarity_summary
        ],
        similarity_quantiles[
            valid_similarity_summary,
            1,
        ],
        yerr=np.vstack([
            similarity_error_lower,
            similarity_error_upper,
        ]),
        color="tab:green",
        marker="s",
        linestyle="None",
        capsize=3,
    )
    axes_stability[1].set_ylabel(
        "Spatial phasor similarity"
    )
    axes_stability[1].set_ylim(
        0.0,
        1.01,
    )

    axes_stability[2].plot(
        baseline_periods,
        ensemble_summary[
            "recovery_fraction"
        ],
        color="tab:green",
        marker="s",
        linewidth=1.5,
    )
    axes_stability[2].set_xlabel(
        "Baseline CHAOS mode period [yr]"
    )
    axes_stability[2].set_ylabel(
        "Ensemble recovery fraction"
    )
    axes_stability[2].set_ylim(
        0.0,
        1.01,
    )

    for axis in axes_stability:
        axis.set_xscale(
            "log"
        )
        axis.grid(
            alpha=0.25,
            which="both",
        )

    fig_stability.suptitle(
        "CHAOS-8.6 DMD mode stability under covariance perturbations"
    )
    fig_stability.tight_layout()


# %% ------------------------------------------------------
# ALL-CANDIDATE ENSEMBLE AND PAIRWISE-SIMILARITY DIAGNOSTICS
# ------------------------------------------------------

fig_ensemble_candidate_power = None
ax_ensemble_candidate_power = None

if ensemble_candidate_plot_flag:
    (
        fig_ensemble_candidate_power,
        ax_ensemble_candidate_power,
    ) = plot_ensemble_candidate_period_power(
        baseline_suite,
        perturbed_suites,
    )

similarity_heatmap_figures = []

if (
    similarity_heatmap_plot_flag
    and ensemble_matches
):
    similarity_heatmap_figures = (
        plot_pairwise_similarity_heatmaps(
            baseline_suite,
            perturbed_suites,
            ensemble_matches,
        )
    )


# %% ------------------------------------------------------
# RECONSTRUCTION DIAGNOSTIC
# ------------------------------------------------------

baseline_reconstruction = None
baseline_residual = None
fig_reconstruction = None
ax_reconstruction = None

if (
    reconstruction_plot_flag
    or video_plot
):
    baseline_reconstruction = reconstruct_candidate_suite(
        baseline_suite[
            "reconstruction_suite"
        ],
        evaluation_times=
            dmd_evaluation_times,
        n_physical=
            baseline_sv_input.shape[0],
    )

if reconstruction_plot_flag:
    baseline_residual = (
        baseline_reconstruction
        - baseline_sv_input
    )

    input_rms = area_weighted_rms(
        baseline_sv_input,
        sv_grid_weights,
    )
    reconstruction_rms = area_weighted_rms(
        baseline_reconstruction,
        sv_grid_weights,
    )
    residual_rms = area_weighted_rms(
        baseline_residual,
        sv_grid_weights,
    )

    weighted_relative_residual = (
        np.linalg.norm(
            np.sqrt(
                sv_grid_weights
            )[
                :,
                None,
            ]
            * baseline_residual
        )
        / np.linalg.norm(
            np.sqrt(
                sv_grid_weights
            )[
                :,
                None,
            ]
            * baseline_sv_input
        )
    )

    fig_reconstruction, ax_reconstruction = plt.subplots(
        figsize=(8, 6)
    )
    ax_reconstruction.plot(
        analysis_times_absolute,
        input_rms,
        color="black",
        linewidth=1.8,
        label="CHAOS input",
    )
    ax_reconstruction.plot(
        analysis_times_absolute,
        reconstruction_rms,
        color="tab:blue",
        linewidth=1.5,
        label="Retained-mode reconstruction",
    )
    ax_reconstruction.plot(
        analysis_times_absolute,
        residual_rms,
        color="tab:red",
        linewidth=1.5,
        label="Residual",
    )
    ax_reconstruction.set_xlabel(
        "Decimal year"
    )
    ax_reconstruction.set_ylabel(
        "Area-weighted global RMS SV [nT/yr]"
    )
    ax_reconstruction.set_title(
        "CHAOS-8.6 retained-mode reconstruction\n"
        f"Weighted relative residual = "
        f"{weighted_relative_residual:.3f}"
    )
    ax_reconstruction.grid(
        alpha=0.25
    )
    ax_reconstruction.legend(
        loc="best"
    )
    fig_reconstruction.tight_layout()

    CHAOS_DMD_results[
        "reconstruction"
    ] = {
        "reconstructed_sv":
            baseline_reconstruction,
        "residual_sv":
            baseline_residual,
        "weighted_relative_residual":
            weighted_relative_residual,
        "input_rms":
            input_rms,
        "reconstruction_rms":
            reconstruction_rms,
        "residual_rms":
            residual_rms,
    }


# %% ------------------------------------------------------
# CHAOS VIDEO CONFIGURATION
# ------------------------------------------------------

reconstruction_suite = None
chaos_video_parameter_summary = None
chaos_video_configuration_stem = None

if (
    video_plot
    or all_modes_video_plot
):
    reconstruction_suite = baseline_suite[
        "reconstruction_suite"
    ]
    embedding_label = (
        f"Hankel d={hankel_d}"
        if hankel_embedding_flag
        else "no Hankel embedding"
    )
    period_label = (
        (
            f"periods {period_lower_bound:g}-"
            f"{period_upper_bound:g} yr + static"
        )
        if period_limit_flag
        else "all finite periods + static"
    )
    processing_labels = [
        str(DMD_algorithm).upper(),
        f"Nmax={Nmax}",
        f"rank={svd_rank}",
        embedding_label,
        f"n_skip={n_skip}",
        (
            "high-quality window"
            if high_q_flag
            else "full window"
        ),
        (
            "temporal filter"
            if filter_flag
            else "no temporal filter"
        ),
        (
            "temporal mean removed"
            if remove_temporal_mean_flag
            else "temporal mean retained"
        ),
        period_label,
        (
            "candidates="
            f"{reconstruction_suite['eigenvalue'].size} "
            f"({baseline_suite['static']['eigenvalue'].size} static)"
        ),
    ]
    chaos_video_parameter_summary = (
        " | ".join(
            processing_labels[
                :5
            ]
        )
        + "\n"
        + " | ".join(
            processing_labels[
                5:
            ]
        )
    )

    period_file_label = (
        (
            f"periods{period_lower_bound:g}-"
            f"{period_upper_bound:g}yr"
        )
        if period_limit_flag
        else "allperiods"
    )
    filter_file_label = (
        (
            f"filter{filter_pass_period_years:g}-"
            f"{filter_stop_period_years:g}yr"
        )
        if filter_flag
        else "nofilter"
    )
    chaos_video_configuration_stem = "_".join(
        [
            "CHAOS",
            str(DMD_algorithm),
            f"N{Nmax}",
            f"r{svd_rank}",
            (
                f"hankel{hankel_d}"
                if hankel_embedding_flag
                else "nohankel"
            ),
            (
                "highq"
                if high_q_flag
                else "fullrecord"
            ),
            f"skip{n_skip}",
            filter_file_label,
            (
                "meanremoved"
                if remove_temporal_mean_flag
                else "meanretained"
            ),
            period_file_label,
            f"frames{video_frame_spacing_years:g}yr",
            f"fps{video_fps:g}",
        ]
    ).replace(
        ".",
        "p",
    )


# %% ------------------------------------------------------
# CHAOS RETAINED-MODE RECONSTRUCTION VIDEO
# ------------------------------------------------------

chaos_video_data = None
chaos_video_output_path = None

if video_plot:
    chaos_video_data = (
        prepare_chaos_reconstruction_video_data(
            baseline_sv_input=
                baseline_sv_input,
            analysis_times_absolute=
                analysis_times_absolute,
            candidate_eigenvalues=
                reconstruction_suite[
                    "eigenvalue"
                ],
            candidate_modes=
                reconstruction_suite[
                    "mode"
                ],
            frame_spacing_years=
                video_frame_spacing_years,
            reconstruction_sv=
                baseline_reconstruction,
        )
    )
    chaos_video_filename = (
        chaos_video_configuration_stem
        + "_reconstruction_video.mp4"
    )
    chaos_video_output_path = (
        make_chaos_reconstruction_video(
            video_data=
                chaos_video_data,
            output_path=(
                PROJECT_ROOT
                / "outputs"
                / chaos_video_filename
            ),
            parameter_summary=
                chaos_video_parameter_summary,
            nlat=state_shape[0],
            nlon=state_shape[1],
            fps=video_fps,
            cmap="seismic",
            dpi=100,
        )
    )

    CHAOS_DMD_results[
        "video"
    ] = {
        "output_path":
            chaos_video_output_path,
        "frame_years": (
            chaos_video_data[
                "frame_years"
            ].copy()
        ),
        "frame_spacing_years": float(
            video_frame_spacing_years
        ),
        "fps": float(
            video_fps
        ),
    }


# %% ------------------------------------------------------
# CHAOS INDIVIDUAL RETAINED-MODE VIDEO
# ------------------------------------------------------

chaos_all_modes_video_data = None
chaos_all_modes_video_output_path = None

if all_modes_video_plot:
    chaos_all_modes_video_data = (
        prepare_chaos_all_modes_video_data(
            analysis_times_absolute=
                analysis_times_absolute,
            candidate_eigenvalues=
                reconstruction_suite[
                    "eigenvalue"
                ],
            candidate_modes=
                reconstruction_suite[
                    "mode"
                ],
            candidate_periods=
                reconstruction_suite[
                    "period"
                ],
            frame_spacing_years=
                video_frame_spacing_years,
        )
    )
    chaos_all_modes_video_filename = (
        chaos_video_configuration_stem
        + "_all_modes_video.mp4"
    )
    chaos_all_modes_video_output_path = (
        make_chaos_all_modes_video(
            video_data=
                chaos_all_modes_video_data,
            output_path=(
                PROJECT_ROOT
                / "outputs"
                / chaos_all_modes_video_filename
            ),
            parameter_summary=
                chaos_video_parameter_summary,
            nlat=state_shape[0],
            nlon=state_shape[1],
            fps=video_fps,
            cmap="seismic",
            dpi=100,
            max_columns=4,
        )
    )

    CHAOS_DMD_results[
        "all_modes_video"
    ] = {
        "output_path":
            chaos_all_modes_video_output_path,
        "frame_years": (
            chaos_all_modes_video_data[
                "frame_years"
            ].copy()
        ),
        "frame_spacing_years": float(
            video_frame_spacing_years
        ),
        "fps": float(
            video_fps
        ),
        "period": (
            reconstruction_suite[
                "period"
            ].copy()
        ),
        "growth_rate": (
            reconstruction_suite[
                "eigenvalue"
            ].real.copy()
        ),
        "mode_vmax": (
            chaos_all_modes_video_data[
                "mode_vmax"
            ].copy()
        ),
    }


# %% ------------------------------------------------------
# BASELINE MODE PHASOR MAPS
# ------------------------------------------------------

mode_map_figures = []

if mode_map_plot_flag:
    for mode_index in range(
        n_baseline_modes
    ):
        phasor = baseline_suite[
            "mode"
        ][
            :,
            mode_index,
        ].reshape(
            state_shape
        )

        colour_values = np.concatenate([
            np.abs(
                phasor.real
            ).ravel(),
            np.abs(
                phasor.imag
            ).ravel(),
        ])
        colour_limit = np.percentile(
            colour_values,
            mode_map_colour_percentile,
        )
        colour_limit = max(
            colour_limit,
            np.finfo(float).eps,
        )

        fig_mode, axes_mode = plt.subplots(
            1,
            2,
            figsize=(12, 4.8),
            sharex=True,
            sharey=True,
        )

        for axis, values, component_label in (
            (
                axes_mode[0],
                phasor.real,
                "Real phasor",
            ),
            (
                axes_mode[1],
                phasor.imag,
                "Imaginary phasor",
            ),
        ):
            image = axis.imshow(
                values,
                origin="lower",
                extent=(
                    longitude[0],
                    longitude[-1],
                    latitude[0],
                    latitude[-1],
                ),
                aspect="auto",
                cmap="seismic",
                vmin=-colour_limit,
                vmax=colour_limit,
            )
            axis.set_xlabel(
                "Longitude [degrees]"
            )
            axis.set_title(
                component_label
            )
            fig_mode.colorbar(
                image,
                ax=axis,
                label="Radial SV phasor [nT/yr]",
                shrink=0.85,
            )

        axes_mode[0].set_ylabel(
            "Latitude [degrees]"
        )
        fig_mode.suptitle(
            f"CHAOS DMD mode {mode_index + 1}: "
            f"period={baseline_periods[mode_index]:.3f} yr, "
            f"growth={baseline_eigenvalues[mode_index].real:+.3e} yr$^{{-1}}$"
        )
        fig_mode.tight_layout()
        mode_map_figures.append(
            (
                fig_mode,
                axes_mode,
            )
        )


# %% ------------------------------------------------------
# PRE-TRUNCATION SINGULAR-VALUE DIAGNOSTICS
# ------------------------------------------------------

fig_cumulative_variance = None
ax_cumulative_variance = None
cumulative_variance_medians = None

fig_singular_value_magnitude = None
ax_singular_value_magnitude = None
singular_value_magnitude_medians = None

if singular_value_plot_flag:
    if singular_diagnostics[
        "chaos"
    ] is None:
        raise RuntimeError(
            "Missing unperturbed CHAOS singular-value diagnostic."
        )

    (
        fig_cumulative_variance,
        ax_cumulative_variance,
        cumulative_variance_medians,
    ) = plot_singular_diagnostic(
        singular_diagnostics,
        value_key=
            "cumulative_variance",
        ylabel=
            "Cumulative variance explained (fraction)",
        title=
            "CHAOS-8.6 pre-truncation cumulative explained variance",
        log_y=False,
    )

    (
        fig_singular_value_magnitude,
        ax_singular_value_magnitude,
        singular_value_magnitude_medians,
    ) = plot_singular_diagnostic(
        singular_diagnostics,
        value_key=
            "magnitude",
        ylabel=
            "Singular value magnitude",
        title=
            "CHAOS-8.6 pre-truncation singular-value magnitude",
        log_y=True,
    )

fig_degree_sweep_periods = None
axes_degree_sweep_periods = None
fig_degree_sweep_static = None
ax_degree_sweep_static = None
fig_degree_sweep_svd_rank = None
ax_degree_sweep_svd_rank = None

if degree_sweep_flag:
    (
        fig_degree_sweep_periods,
        axes_degree_sweep_periods,
    ) = plot_degree_sweep_periods(
        degree_sweep_results
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
