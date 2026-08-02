"""
Flexible DMD analysis of CHAOS-8.6 radial secular variation.

The configured low and high recovery degrees are fitted independently. Each
degree has its own projection, rank selection, DMD fits, covariance matching,
reconstruction, diagnostic figures, and videos. No temporal filter is applied.
The configured period limits create additional zoomed plots only; they never
alter fitting, candidate retention, matching, reconstruction, or videos.
"""

# %% FILE SYSTEM AND DEPENDENCY SETUP

import gc
import sys
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

import chaosmagpy as cp
import matplotlib.pyplot as plt
import numpy as np
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

plt.rcParams.update({"font.size": 11})


# %% ------------------------------------------------------
# USER SETTINGS
# ------------------------------------------------------

DMD_algorithm = "opdmd"

# Optional time-delay embedding. No temporal filtering is applied.
hankel_embedding_flag = True
hankel_d = 10
hankel_reconstruction_method = "first"

# Rank selection is independent at each recovery degree.
# "noise_threshold" uses the unmodified threshold-selected rank.
svd_rank_selection = "noise_threshold"
fixed_svd_rank = 45
singular_value_plot_flag = True

# Optional mean removal is retained as a separate methodological choice.
remove_temporal_mean_flag = False

# Plot-only period window. Full-period versions are always produced too.
period_plot_lower_bound = 1.0
period_plot_upper_bound = 25.0

reconstruction_plot_flag = True
ensemble_candidate_plot_flag = True
similarity_heatmap_plot_flag = True

# Spatial information is output in videos only; there are no static maps.
video_plot = True
all_modes_video_plot = True
video_frame_spacing_years = 0.2
video_fps = 5

# Store one complete run beneath PROJECT_ROOT/results. Set run_label to a
# short descriptive string to add it to the automatically generated name.
save_run_outputs = True
show_figures_interactively = True
save_numerical_results = True
figure_output_formats = ("png", "pdf")
figure_output_dpi = 300
run_label = None


# ---------------------------------------------------------
# UNCERTAINTY ENSEMBLE
# ---------------------------------------------------------

ensemble_flag = False
n_realisations = 1
noise_seed = 42
noise_temporal_model = "independent"
noise_tau_years = 6.0
n_noise_rank_realisations = n_realisations
noise_rank_quantile = 0.95


# ---------------------------------------------------------
# TIME / DEGREE SETTINGS
# ---------------------------------------------------------

high_q_flag = True
n_skip = 1

# Two independent recovery analyses. No degree sweep is performed.
low_recovery_degree = 10
high_recovery_degree = 15
recovery_degrees = {
    "low": low_recovery_degree,
    "high": high_recovery_degree,
}

# CHAOS and the covariance product are loaded at their common full degree.
CHAOS_covariance_Nmax = 20


# ---------------------------------------------------------
# ONE-TO-ONE COVARIANCE MODE MATCHING
# ---------------------------------------------------------

matching_spatial_weight = 1.0
matching_period_weight = 1.0
matching_growth_weight = 0.25
matching_max_relative_period_error = 0.25
matching_min_spatial_similarity = 0.50


# %% ------------------------------------------------------
# VALIDATE SETTINGS
# ------------------------------------------------------

if DMD_algorithm not in ("exact", "fbdmd", "opdmd"):
    raise ValueError(
        "DMD_algorithm must be 'exact', 'fbdmd', or 'opdmd'."
    )

if svd_rank_selection not in ("fixed", "noise_threshold"):
    raise ValueError(
        "svd_rank_selection must be 'fixed' or 'noise_threshold'."
    )

if (
    not isinstance(fixed_svd_rank, (int, np.integer))
    or fixed_svd_rank < 1
):
    raise ValueError("fixed_svd_rank must be a positive integer.")

if (
    not isinstance(n_noise_rank_realisations, (int, np.integer))
    or n_noise_rank_realisations < 1
):
    raise ValueError(
        "n_noise_rank_realisations must be a positive integer."
    )

if not (
    np.isfinite(noise_rank_quantile)
    and 0.0 < noise_rank_quantile < 1.0
):
    raise ValueError(
        "noise_rank_quantile must lie strictly between zero and one."
    )

if not isinstance(n_skip, (int, np.integer)) or n_skip < 1:
    raise ValueError("n_skip must be a positive integer.")

if not (
    isinstance(low_recovery_degree, (int, np.integer))
    and isinstance(high_recovery_degree, (int, np.integer))
    and 1
    <= low_recovery_degree
    < high_recovery_degree
    <= CHAOS_covariance_Nmax
):
    raise ValueError(
        "Recovery degrees must satisfy "
        "1 <= low_recovery_degree < high_recovery_degree <= "
        f"{CHAOS_covariance_Nmax}."
    )

if not (
    np.isfinite(period_plot_lower_bound)
    and np.isfinite(period_plot_upper_bound)
    and 0.0 < period_plot_lower_bound < period_plot_upper_bound
):
    raise ValueError(
        "Plot period bounds must be finite, positive, and increasing."
    )

if ensemble_flag and (
    not isinstance(n_realisations, (int, np.integer))
    or n_realisations < 1
):
    raise ValueError(
        "n_realisations must be positive when ensemble_flag=True."
    )

if hankel_embedding_flag and (
    not isinstance(hankel_d, (int, np.integer))
    or hankel_d < 1
):
    raise ValueError(
        "hankel_d must be positive when embedding is enabled."
    )

if (
    video_plot or all_modes_video_plot
) and (
    not np.isfinite(video_frame_spacing_years)
    or video_frame_spacing_years <= 0.0
    or not np.isfinite(video_fps)
    or video_fps <= 0.0
):
    raise ValueError(
        "Video frame spacing and fps must be finite and positive."
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
        "Matching weights must be non-negative and not all zero."
    )

if not (
    matching_max_relative_period_error > 0.0
    and 0.0 <= matching_min_spatial_similarity <= 1.0
):
    raise ValueError("Invalid covariance matching thresholds.")

supported_figure_formats = {"png", "pdf", "svg"}
figure_output_formats = tuple(
    str(output_format).lower()
    for output_format in figure_output_formats
)
if (
    not figure_output_formats
    or not set(figure_output_formats).issubset(
        supported_figure_formats
    )
):
    raise ValueError(
        "figure_output_formats must contain one or more of "
        "'png', 'pdf', or 'svg'."
    )
if (
    not isinstance(figure_output_dpi, (int, np.integer))
    or figure_output_dpi < 1
):
    raise ValueError("figure_output_dpi must be a positive integer.")


# %% ------------------------------------------------------
# RUN OUTPUT DIRECTORY
# ------------------------------------------------------

run_started_at = datetime.now()


def filesystem_safe_label(label):
    """Return a compact filesystem-safe label."""

    return "".join(
        character
        if character.isalnum() or character in ("-", "_")
        else "_"
        for character in str(label).strip()
    ).strip("_")


run_output_directory = None
if save_run_outputs:
    hankel_run_label = (
        f"hankel_d{hankel_d}"
        if hankel_embedding_flag
        else "noHankel"
    )
    if svd_rank_selection == "noise_threshold":
        svd_run_label = "svdNoise"
    else:
        fixed_rank_label = (
            str(fixed_svd_rank)
            .replace("-", "Minus")
            .replace(".", "p")
        )
        svd_run_label = f"svdFixed{fixed_rank_label}"
    run_name_parts = [
        "CHAOS-8.6",
        f"N{low_recovery_degree}_{high_recovery_degree}",
        filesystem_safe_label(DMD_algorithm),
        hankel_run_label,
        svd_run_label,
    ]
    if run_label:
        safe_run_label = filesystem_safe_label(run_label)
        if safe_run_label:
            run_name_parts.append(safe_run_label)
    run_name_parts.append(
        run_started_at.strftime("%Y%m%d_%H%M%S")
    )
    run_output_directory = (
        PROJECT_ROOT
        / "results"
        / "_".join(run_name_parts)
    )
    collision_index = 1
    while run_output_directory.exists():
        run_output_directory = (
            run_output_directory.parent
            / (
                "_".join(run_name_parts)
                + f"_{collision_index:02d}"
            )
        )
        collision_index += 1
    run_output_directory.mkdir(parents=True)
    print(
        f"CHAOS run outputs: {run_output_directory}"
    )


# %% ------------------------------------------------------
# LOAD CHAOS-8.6 AND BUILD PAIRED UNCERTAINTY INPUTS
# ------------------------------------------------------

chaos_path = Path(CHAOS_DIR) / "CHAOS-8.6.mat"
if not chaos_path.exists():
    raise FileNotFoundError(
        f"Could not find CHAOS-8.6 model at {chaos_path}."
    )

chaos_model = cp.load_CHAOS_matfile(str(chaos_path))
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
    (CHAOS_covariance_Nmax + 1) ** 2 - 1
)
expected_full_shape = (
    times_mjd2000.size,
    expected_full_coefficients,
)
if chaos_sv_full.shape != expected_full_shape:
    raise ValueError(
        "Unexpected CHAOS coefficient shape: "
        f"{chaos_sv_full.shape} vs {expected_full_shape}."
    )
if not np.all(np.isfinite(chaos_sv_full)):
    raise ValueError("CHAOS SV coefficients contain non-finite values.")

if Cov_full.shape != (
    chaos_sv_full.shape[0],
    chaos_sv_full.shape[1],
    chaos_sv_full.shape[1],
):
    raise ValueError(
        "CHAOS covariance and coefficient arrays do not align: "
        f"{Cov_full.shape} vs {chaos_sv_full.shape}."
    )

noise_bank_size = max(
    n_noise_rank_realisations,
    n_realisations if ensemble_flag else 0,
)
noise_only_bank = Perturbation_Generate(
    chaos_sv_full,
    n_realisations=noise_bank_size,
    seed=noise_seed,
    temporal_z=noise_temporal_model,
    tau=noise_tau_years,
    dt=dt_years,
    covariance_used=Cov_full,
    just_noise=True,
)
noise_only_rank_bank = noise_only_bank[
    :n_noise_rank_realisations
]

fit_queue = [("chaos", chaos_sv_full)]
if ensemble_flag:
    fit_queue.extend(
        ("perturbed", chaos_sv_full + noise_realisation)
        for noise_realisation in noise_only_bank[:n_realisations]
    )


# %% ------------------------------------------------------
# TIME COORDINATES AND SHARED HELPERS
# ------------------------------------------------------

if high_q_flag:
    analysis_times_absolute = times_absolute[good_record_slice]
else:
    analysis_times_absolute = times_absolute.copy()
analysis_times_absolute = analysis_times_absolute[::n_skip]

if analysis_times_absolute.size < 2:
    raise ValueError("At least two selected CHAOS snapshots are required.")

snapshot_differences = np.diff(analysis_times_absolute)
dt_snapshot = float(np.median(snapshot_differences))
if not np.allclose(
    snapshot_differences,
    dt_snapshot,
    rtol=1e-8,
    atol=1e-10,
):
    raise ValueError("Selected CHAOS snapshots are not uniformly sampled.")

if hankel_embedding_flag and hankel_d >= analysis_times_absolute.size:
    raise ValueError(
        f"hankel_d={hankel_d} leaves too few embedded snapshots."
    )

if video_plot or all_modes_video_plot:
    frame_step = int(round(video_frame_spacing_years / dt_snapshot))
    if (
        frame_step < 1
        or not np.isclose(
            frame_step * dt_snapshot,
            video_frame_spacing_years,
            rtol=1e-10,
            atol=1e-10,
        )
    ):
        raise ValueError(
            "video_frame_spacing_years must be an integer multiple "
            f"of dt_snapshot={dt_snapshot}."
        )

dmd_evaluation_times = (
    analysis_times_absolute - analysis_times_absolute[0]
)
spatial_weights = np.asarray(W2D, dtype=float).ravel()


def prepare_chaos_sv_input(
    gnm_input,
    truncation_degree,
    projection_operator,
):
    """Window, degree-truncate, project, and subsample CHAOS coefficients."""

    if high_q_flag:
        gnm_windowed = gnm_input[good_record_slice, :]
    else:
        gnm_windowed = gnm_input

    gnm_band = Truncate_Gauss_Coeffs(
        gnm_windowed,
        tmax=truncation_degree,
    )
    expected_coefficients = (truncation_degree + 1) ** 2 - 1
    if gnm_band.shape[1] != expected_coefficients:
        raise ValueError(
            f"Unexpected coefficient count at n={truncation_degree}: "
            f"{gnm_band.shape[1]}."
        )

    sv_input = projection_operator @ gnm_band.T
    sv_input = sv_input[:, ::n_skip]
    expected_shape = (
        state_shape[0] * state_shape[1],
        analysis_times_absolute.size,
    )
    if sv_input.shape != expected_shape:
        raise ValueError(
            f"Unexpected gridded CHAOS shape at n={truncation_degree}: "
            f"{sv_input.shape} vs {expected_shape}."
        )

    if remove_temporal_mean_flag:
        sv_input = sv_input - np.mean(
            sv_input,
            axis=1,
            keepdims=True,
        )

    if not np.all(np.isfinite(sv_input)):
        raise ValueError(
            f"CHAOS input contains non-finite values at n={truncation_degree}."
        )
    return sv_input


def pre_truncation_svd_input(sv_input):
    if hankel_embedding_flag:
        matrix = pseudo_hankel_matrix(sv_input, d=hankel_d)
    else:
        matrix = sv_input
    if DMD_algorithm in ("exact", "fbdmd"):
        matrix = matrix[:, :-1]
    return matrix


def calculate_singular_diagnostic(sv_input):
    magnitudes = np.abs(
        np.linalg.svd(
            pre_truncation_svd_input(sv_input),
            compute_uv=False,
        )
    )
    variance = np.square(magnitudes)
    total_variance = np.sum(variance)
    if not np.isfinite(total_variance) or total_variance <= 0.0:
        raise ValueError(
            "The pre-truncation singular-value energy is not positive."
        )
    return {
        "magnitude": magnitudes,
        "cumulative_variance": np.cumsum(variance) / total_variance,
    }


def select_degree_svd_rank(
    truncation_degree,
    projection_operator,
):
    """Select the rank independently at one recovery degree."""

    baseline_sv_input = prepare_chaos_sv_input(
        chaos_sv_full,
        truncation_degree,
        projection_operator,
    )
    baseline_diagnostic = calculate_singular_diagnostic(
        baseline_sv_input
    )
    noise_diagnostics = [
        calculate_singular_diagnostic(
            prepare_chaos_sv_input(
                noise_realisation,
                truncation_degree,
                projection_operator,
            )
        )
        for noise_realisation in noise_only_rank_bank
    ]

    if svd_rank_selection == "fixed":
        return (
            int(fixed_svd_rank),
            None,
            None,
            baseline_diagnostic,
            noise_diagnostics,
        )

    leading_noise = np.asarray([
        diagnostic["magnitude"][0]
        for diagnostic in noise_diagnostics
    ])
    noise_threshold = float(
        np.quantile(leading_noise, noise_rank_quantile)
    )
    base_noise_rank = int(
        np.count_nonzero(
            baseline_diagnostic["magnitude"] > noise_threshold
        )
    )
    if base_noise_rank < 1:
        raise RuntimeError(
            "The noise threshold retained no CHAOS singular values at "
            f"n={truncation_degree}; threshold={noise_threshold:.6e}."
        )

    return (
        base_noise_rank,
        base_noise_rank,
        noise_threshold,
        baseline_diagnostic,
        noise_diagnostics,
    )


def build_selected_dmd(selected_svd_rank):
    if DMD_algorithm == "exact":
        return build_exact_dmd(svd_rank=selected_svd_rank)
    if DMD_algorithm == "fbdmd":
        return build_fbdmd(svd_rank=selected_svd_rank)
    return build_bopdmd(svd_rank=selected_svd_rank)


def fit_candidate_suite(
    sv_input,
    selected_svd_rank,
):
    """Fit DMD and retain every physical oscillatory or static candidate."""

    n_physical, n_snapshots = sv_input.shape
    base_dmd = build_selected_dmd(selected_svd_rank)
    dmd = apply_hankel_embedding(
        base_dmd,
        enabled=hankel_embedding_flag,
        d=hankel_d,
        reconstruction_method=hankel_reconstruction_method,
    )
    embedding_d = hankel_d if hankel_embedding_flag else None

    if DMD_algorithm == "opdmd":
        n_fit_snapshots = (
            n_snapshots - hankel_d + 1
            if hankel_embedding_flag
            else n_snapshots
        )
        dmd.fit(
            sv_input,
            np.arange(n_fit_snapshots, dtype=float) * dt_snapshot,
        )
        eigenvalues, modes, periods = extract_optimized_dmd_candidates(
            dmd=dmd,
            n_physical=n_physical,
            embedding_d=embedding_d,
        )
    else:
        dmd.fit(sv_input)
        eigenvalues, modes, periods = extract_standard_dmd_candidates(
            dmd=dmd,
            dt_snapshot=dt_snapshot,
            n_physical=n_physical,
            embedding_d=embedding_d,
        )

    eigenvalues = np.asarray(eigenvalues, dtype=complex).ravel()
    modes = np.asarray(modes, dtype=complex)
    periods = np.asarray(periods, dtype=float).ravel()
    if modes.shape != (n_physical, eigenvalues.size):
        raise ValueError("Extracted CHAOS candidates do not align.")

    oscillatory = np.isfinite(periods) & (periods > 0.0)
    static = np.isposinf(periods)
    oscillatory_indices = np.flatnonzero(oscillatory)
    oscillatory_indices = oscillatory_indices[
        np.argsort(periods[oscillatory_indices])
    ]
    static_indices = np.flatnonzero(static)
    retained_indices = np.concatenate(
        (oscillatory_indices, static_indices)
    )

    reconstruction_eigenvalues = eigenvalues[retained_indices]
    reconstruction_modes = modes[:, retained_indices]
    reconstruction_periods = periods[retained_indices]
    oscillatory_count = oscillatory_indices.size

    candidate_eigenvalues = reconstruction_eigenvalues[
        :oscillatory_count
    ]
    candidate_modes = reconstruction_modes[:, :oscillatory_count]
    candidate_periods = reconstruction_periods[:oscillatory_count]
    candidate_power = np.asarray([
        SV_Grid_Phasor_Record_Power(
            candidate_modes[:, index],
            candidate_eigenvalues[index],
            evaluation_times=dmd_evaluation_times,
            spatial_weights=spatial_weights,
        )
        for index in range(oscillatory_count)
    ])

    return {
        "effective_svd_rank": int(np.asarray(dmd.modes).shape[1]),
        "eigenvalue": candidate_eigenvalues,
        "mode": candidate_modes,
        "period": candidate_periods,
        "growth_rate": candidate_eigenvalues.real.copy(),
        "quality_factor": Mode_Quality_Factor(candidate_eigenvalues),
        "power": candidate_power,
        "static_count": int(static_indices.size),
        "reconstruction_eigenvalue": reconstruction_eigenvalues,
        "reconstruction_mode": reconstruction_modes,
        "reconstruction_period": reconstruction_periods,
    }


def match_candidate_suites(baseline_suite, perturbed_suite):
    """Match perturbed oscillatory candidates one-to-one to the baseline."""

    baseline_periods = baseline_suite["period"]
    perturbed_periods = perturbed_suite["period"]
    n_baseline = baseline_periods.size
    n_perturbed = perturbed_periods.size

    similarities = np.full(
        (n_baseline, n_perturbed),
        np.nan,
        dtype=float,
    )
    relative_period_error = np.full_like(similarities, np.inf)
    growth_difference = np.full_like(similarities, np.inf)
    eligible = np.zeros_like(similarities, dtype=bool)
    cost = np.full_like(similarities, 1e12)

    for baseline_index in range(n_baseline):
        baseline_eigenvalue = baseline_suite["eigenvalue"][baseline_index]
        baseline_frequency = max(
            np.abs(baseline_eigenvalue.imag),
            np.finfo(float).eps,
        )
        for candidate_index in range(n_perturbed):
            similarity = Complex_Phasor_Compare(
                baseline_suite["mode"][:, baseline_index],
                perturbed_suite["mode"][:, candidate_index],
            )
            period_error = (
                np.abs(
                    perturbed_periods[candidate_index]
                    - baseline_periods[baseline_index]
                )
                / baseline_periods[baseline_index]
            )
            growth_error = np.abs(
                perturbed_suite["eigenvalue"][candidate_index].real
                - baseline_eigenvalue.real
            )
            similarities[baseline_index, candidate_index] = similarity
            relative_period_error[
                baseline_index,
                candidate_index,
            ] = period_error
            growth_difference[
                baseline_index,
                candidate_index,
            ] = growth_error

            pair_is_eligible = (
                np.isfinite(similarity)
                and similarity >= matching_min_spatial_similarity
                and period_error <= matching_max_relative_period_error
            )
            eligible[baseline_index, candidate_index] = pair_is_eligible
            if pair_is_eligible:
                cost[baseline_index, candidate_index] = (
                    matching_spatial_weight * (1.0 - similarity)
                    + matching_period_weight * period_error
                    + matching_growth_weight
                    * growth_error
                    / baseline_frequency
                )

    recovered = np.zeros(n_baseline, dtype=bool)
    candidate_index = np.full(n_baseline, -1, dtype=int)
    matched_similarity = np.full(n_baseline, np.nan)
    recovered_period = np.full(n_baseline, np.nan)
    recovered_eigenvalue = np.full(
        n_baseline,
        np.nan + 1j * np.nan,
        dtype=complex,
    )

    if n_baseline and n_perturbed:
        rows, columns = linear_sum_assignment(cost)
        for row, column in zip(rows, columns):
            if not eligible[row, column]:
                continue
            recovered[row] = True
            candidate_index[row] = column
            matched_similarity[row] = similarities[row, column]
            recovered_period[row] = perturbed_periods[column]
            recovered_eigenvalue[row] = (
                perturbed_suite["eigenvalue"][column]
            )

    return {
        "recovered": recovered,
        "candidate_index": candidate_index,
        "similarity": matched_similarity,
        "recovered_period": recovered_period,
        "eigenvalue": recovered_eigenvalue,
        "similarity_matrix": similarities,
        "relative_period_error_matrix": relative_period_error,
        "growth_difference_matrix": growth_difference,
        "eligible_matrix": eligible,
        "cost_matrix": cost,
    }


def evaluate_candidate_sum(eigenvalues, modes, relative_times):
    if eigenvalues.size == 0:
        return np.zeros(
            (modes.shape[0], relative_times.size),
            dtype=float,
        )
    dynamics = np.exp(
        eigenvalues[:, None] * relative_times[None, :]
    )
    return np.real(modes @ dynamics)


def area_weighted_rms(records):
    records = np.asarray(records)
    return np.sqrt(
        np.sum(
            spatial_weights[:, None] * np.square(records),
            axis=0,
        )
        / np.sum(spatial_weights)
    )


def finite_period_limits(baseline_suite, perturbed_suites):
    period_groups = [baseline_suite["period"]]
    period_groups.extend(
        suite["period"] for suite in perturbed_suites
    )
    periods = np.concatenate([
        np.asarray(group, dtype=float).ravel()
        for group in period_groups
        if np.asarray(group).size
    ]) if any(np.asarray(group).size for group in period_groups) else np.array([])
    periods = periods[np.isfinite(periods) & (periods > 0.0)]
    if periods.size == 0:
        return (period_plot_lower_bound, period_plot_upper_bound)
    lower = max(np.min(periods) / 1.15, np.finfo(float).tiny)
    upper = np.max(periods) * 1.15
    if not lower < upper:
        upper = lower * 1.1
    return (float(lower), float(upper))


# %% ------------------------------------------------------
# INDEPENDENT LOW/HIGH ANALYSES
# ------------------------------------------------------

def analyse_chaos_degree(case_label, truncation_degree):
    print(
        f"\nCHAOS DMD: {case_label.upper()} RECOVERY "
        f"(n <= {truncation_degree})"
    )

    projection_operator = Truncate_Gauss_Coeffs(
        A_20_dict["r"],
        tmax=truncation_degree,
    )
    expected_projection_shape = (
        state_shape[0] * state_shape[1],
        (truncation_degree + 1) ** 2 - 1,
    )
    if projection_operator.shape != expected_projection_shape:
        raise ValueError(
            f"Unexpected projection shape at n={truncation_degree}: "
            f"{projection_operator.shape}."
        )

    (
        selected_svd_rank,
        base_noise_rank,
        noise_threshold,
        baseline_singular_diagnostic,
        noise_singular_diagnostics,
    ) = select_degree_svd_rank(
        truncation_degree,
        projection_operator,
    )

    baseline_sv_input = None
    baseline_suite = None
    perturbed_suites = []
    perturbed_singular_diagnostics = []

    for input_label, gnm_input in tqdm(
        fit_queue,
        desc=f"Fitting CHAOS n<={truncation_degree}",
    ):
        sv_input = prepare_chaos_sv_input(
            gnm_input,
            truncation_degree,
            projection_operator,
        )
        suite = fit_candidate_suite(
            sv_input,
            selected_svd_rank,
        )
        if input_label == "chaos":
            baseline_sv_input = sv_input
            baseline_suite = suite
        else:
            perturbed_suites.append(suite)
            if singular_value_plot_flag:
                perturbed_singular_diagnostics.append(
                    calculate_singular_diagnostic(sv_input)
                )
        if input_label != "chaos":
            del sv_input
        gc.collect()

    if baseline_suite is None or baseline_sv_input is None:
        raise RuntimeError("The unperturbed CHAOS fit was not produced.")

    matches = [
        match_candidate_suites(baseline_suite, perturbed_suite)
        for perturbed_suite in perturbed_suites
    ]

    reconstruction = evaluate_candidate_sum(
        baseline_suite["reconstruction_eigenvalue"],
        baseline_suite["reconstruction_mode"],
        dmd_evaluation_times,
    )
    residual = reconstruction - baseline_sv_input
    reconstruction_metrics = {
        "input_rms": area_weighted_rms(baseline_sv_input),
        "reconstruction_rms": area_weighted_rms(reconstruction),
        "residual_rms": area_weighted_rms(residual),
        "record_relative_rms_error": float(
            np.sqrt(
                np.sum(
                    spatial_weights[:, None] * np.square(residual)
                )
                / np.sum(
                    spatial_weights[:, None]
                    * np.square(baseline_sv_input)
                )
            )
        ),
    }

    print(
        f"Selected PyDMD rank: {selected_svd_rank}; "
        f"effective rank: {baseline_suite['effective_svd_rank']}; "
        f"oscillatory candidates: {baseline_suite['period'].size}; "
        f"static candidates: {baseline_suite['static_count']}; "
        f"relative reconstruction RMS error: "
        f"{reconstruction_metrics['record_relative_rms_error']:.4g}"
    )

    return {
        "label": case_label,
        "degree": int(truncation_degree),
        "projection_operator": projection_operator,
        "selected_svd_rank": int(selected_svd_rank),
        "base_noise_rank": base_noise_rank,
        "noise_threshold": noise_threshold,
        "singular_diagnostics": {
            "baseline": baseline_singular_diagnostic,
            "noise_only": noise_singular_diagnostics,
            "perturbed": perturbed_singular_diagnostics,
        },
        "baseline_sv_input": baseline_sv_input,
        "baseline_suite": baseline_suite,
        "perturbed_suites": perturbed_suites,
        "matches": matches,
        "reconstruction": reconstruction,
        "residual": residual,
        "reconstruction_metrics": reconstruction_metrics,
    }


chaos_degree_results = {
    case_label: analyse_chaos_degree(case_label, degree)
    for case_label, degree in recovery_degrees.items()
}


# %% ------------------------------------------------------
# SAVE RUN MATERIALS
# ------------------------------------------------------

def degree_material_directory(result, material_name):
    """Return and create one material directory for a degree result."""

    if run_output_directory is None:
        return None
    material_directory = (
        run_output_directory
        / f"{result['label']}_N{result['degree']}"
        / material_name
    )
    material_directory.mkdir(
        parents=True,
        exist_ok=True,
    )
    return material_directory


def figure_objects(figure_entry):
    """Yield Matplotlib figures from a stored figure entry."""

    if hasattr(figure_entry, "savefig"):
        yield figure_entry
        return
    if (
        isinstance(figure_entry, tuple)
        and figure_entry
        and hasattr(figure_entry[0], "savefig")
    ):
        yield figure_entry[0]
        return
    if isinstance(figure_entry, (tuple, list)):
        for nested_entry in figure_entry:
            yield from figure_objects(nested_entry)


def save_figure_collection(result):
    """Save every figure stored for one recovery degree."""

    figure_directory = degree_material_directory(
        result,
        "figures",
    )
    if figure_directory is None:
        return

    for figure_name, figure_entry in result["figures"].items():
        figures = list(figure_objects(figure_entry))
        for figure_index, figure in enumerate(figures):
            indexed_name = (
                figure_name
                if len(figures) == 1
                else f"{figure_name}_{figure_index + 1:03d}"
            )
            for output_format in figure_output_formats:
                figure.savefig(
                    figure_directory
                    / f"{indexed_name}.{output_format}",
                    dpi=figure_output_dpi,
                    bbox_inches="tight",
                )


def save_chaos_numerical_results(result):
    """Save CHAOS inputs, candidates, matches, and reconstruction arrays."""

    if not save_numerical_results:
        return
    numerical_directory = degree_material_directory(
        result,
        "numerical",
    )
    if numerical_directory is None:
        return

    suite = result["baseline_suite"]
    archive = {
        "analysis_times_absolute": analysis_times_absolute,
        "evaluation_times": dmd_evaluation_times,
        "chaos_input": result["baseline_sv_input"],
        "reconstruction": result["reconstruction"],
        "residual": result["residual"],
        "input_rms": result[
            "reconstruction_metrics"
        ]["input_rms"],
        "reconstruction_rms": result[
            "reconstruction_metrics"
        ]["reconstruction_rms"],
        "residual_rms": result[
            "reconstruction_metrics"
        ]["residual_rms"],
        "record_relative_rms_error": np.asarray(
            result[
                "reconstruction_metrics"
            ]["record_relative_rms_error"]
        ),
        "selected_svd_rank": np.asarray(
            result["selected_svd_rank"]
        ),
        "effective_svd_rank": np.asarray(
            suite["effective_svd_rank"]
        ),
        "baseline_eigenvalue": suite["eigenvalue"],
        "baseline_mode": suite["mode"],
        "baseline_period": suite["period"],
        "baseline_power": suite["power"],
        "baseline_quality_factor": suite["quality_factor"],
        "baseline_reconstruction_eigenvalue": suite[
            "reconstruction_eigenvalue"
        ],
        "baseline_reconstruction_mode": suite[
            "reconstruction_mode"
        ],
        "baseline_reconstruction_period": suite[
            "reconstruction_period"
        ],
    }

    baseline_diagnostic = result[
        "singular_diagnostics"
    ]["baseline"]
    archive["baseline_singular_magnitude"] = (
        baseline_diagnostic["magnitude"]
    )
    archive["baseline_cumulative_variance"] = (
        baseline_diagnostic["cumulative_variance"]
    )

    for realisation_index, diagnostic in enumerate(
        result["singular_diagnostics"]["noise_only"]
    ):
        key_prefix = f"noise_{realisation_index:03d}"
        archive[f"{key_prefix}_singular_magnitude"] = (
            diagnostic["magnitude"]
        )
        archive[f"{key_prefix}_cumulative_variance"] = (
            diagnostic["cumulative_variance"]
        )

    for realisation_index, (perturbed_suite, match) in enumerate(
        zip(result["perturbed_suites"], result["matches"])
    ):
        key_prefix = f"perturbed_{realisation_index:03d}"
        archive[f"{key_prefix}_eigenvalue"] = (
            perturbed_suite["eigenvalue"]
        )
        archive[f"{key_prefix}_mode"] = perturbed_suite["mode"]
        archive[f"{key_prefix}_period"] = perturbed_suite["period"]
        archive[f"{key_prefix}_power"] = perturbed_suite["power"]
        archive[f"{key_prefix}_recovered"] = match["recovered"]
        archive[f"{key_prefix}_candidate_index"] = (
            match["candidate_index"]
        )
        archive[f"{key_prefix}_similarity"] = match["similarity"]
        archive[f"{key_prefix}_recovered_period"] = (
            match["recovered_period"]
        )
        archive[f"{key_prefix}_recovered_eigenvalue"] = (
            match["eigenvalue"]
        )
        archive[f"{key_prefix}_similarity_matrix"] = (
            match["similarity_matrix"]
        )
        archive[f"{key_prefix}_relative_period_error_matrix"] = (
            match["relative_period_error_matrix"]
        )
        archive[f"{key_prefix}_growth_difference_matrix"] = (
            match["growth_difference_matrix"]
        )
        archive[f"{key_prefix}_eligible_matrix"] = (
            match["eligible_matrix"]
        )
        archive[f"{key_prefix}_cost_matrix"] = match["cost_matrix"]

    np.savez_compressed(
        numerical_directory / "numerical_results.npz",
        **archive,
    )


def write_run_configuration():
    """Write a human-readable record of the complete run configuration."""

    if run_output_directory is None:
        return

    lines = [
        "CHAOS-8.6 flexible DMD run",
        f"Started: {run_started_at.isoformat(timespec='seconds')}",
        f"Algorithm: {DMD_algorithm}",
        f"Low recovery degree: {low_recovery_degree}",
        f"High recovery degree: {high_recovery_degree}",
        f"Rank method: {svd_rank_selection}",
        f"Fixed rank setting: {fixed_svd_rank}",
        f"Hankel embedding: {hankel_embedding_flag}",
        f"Hankel d: {hankel_d}",
        f"Hankel reconstruction method: {hankel_reconstruction_method}",
        f"High-quality record: {high_q_flag}",
        f"n_skip: {n_skip}",
        f"Selected record start: {analysis_times_absolute[0]:.6f}",
        f"Selected record end: {analysis_times_absolute[-1]:.6f}",
        f"Selected snapshots: {analysis_times_absolute.size}",
        f"Snapshot spacing (years): {dt_snapshot:.12g}",
        "Temporal filter: none",
        f"Temporal mean removed: {remove_temporal_mean_flag}",
        f"Plot-only period lower bound (years): "
        f"{period_plot_lower_bound}",
        f"Plot-only period upper bound (years): "
        f"{period_plot_upper_bound}",
        f"Ensemble enabled: {ensemble_flag}",
        f"Ensemble realisations: {n_realisations}",
        f"Noise seed: {noise_seed}",
        f"Noise rank realisations: {n_noise_rank_realisations}",
        f"Noise rank quantile: {noise_rank_quantile}",
        f"Noise temporal model: {noise_temporal_model}",
        f"Noise tau (years): {noise_tau_years}",
        f"Matching spatial weight: {matching_spatial_weight}",
        f"Matching period weight: {matching_period_weight}",
        f"Matching growth weight: {matching_growth_weight}",
        "Matching maximum relative period error: "
        f"{matching_max_relative_period_error}",
        "Matching minimum spatial similarity: "
        f"{matching_min_spatial_similarity}",
        f"Figure formats: {', '.join(figure_output_formats)}",
        f"Figure dpi: {figure_output_dpi}",
        f"Reconstruction video enabled: {video_plot}",
        f"Individual-mode video enabled: {all_modes_video_plot}",
        "",
        "Degree-specific results:",
    ]
    for result in chaos_degree_results.values():
        suite = result["baseline_suite"]
        lines.extend([
            (
                f"{result['label']} n<={result['degree']}: "
                f"selected rank={result['selected_svd_rank']}, "
                f"effective rank={suite['effective_svd_rank']}, "
                f"oscillatory candidates={suite['period'].size}, "
                f"static candidates={suite['static_count']}, "
                "relative reconstruction RMS error="
                f"{result['reconstruction_metrics']['record_relative_rms_error']:.12g}"
            )
        ])

    (
        run_output_directory / "run_configuration.txt"
    ).write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )


write_run_configuration()


# %% ------------------------------------------------------
# DEGREE-SPECIFIC DIAGNOSTIC PLOTS
# ------------------------------------------------------

def period_view_title(view_label):
    if view_label == "zoom":
        return (
            f"{period_plot_lower_bound:g}-"
            f"{period_plot_upper_bound:g} year view"
        )
    return "full period range"


def matched_perturbed_candidate_mask(
    perturbed_suite,
    match,
):
    """Return candidates accepted by the one-to-one CHAOS matching."""

    matched_indices = match["candidate_index"][
        match["recovered"]
    ]
    return np.isin(
        np.arange(perturbed_suite["period"].size),
        matched_indices,
    )


def plot_period_power(result, period_limits, view_label):
    baseline = result["baseline_suite"]
    fig, ax = plt.subplots(
        figsize=(text_width, 0.65 * text_width)
    )
    ax.scatter(
        baseline["period"],
        baseline["power"],
        marker="o",
        facecolors="none",
        edgecolors="black",
        linewidths=1.5,
        s=65,
        label="Unperturbed CHAOS DMD",
        zorder=4,
    )
    if ensemble_candidate_plot_flag:
        for index, suite in enumerate(result["perturbed_suites"]):
            ax.scatter(
                suite["period"],
                suite["power"],
                marker="x",
                alpha=0.45,
                s=35,
                label="Perturbed candidates" if index == 0 else None,
            )
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(*period_limits)
    ax.set_xlabel("Period (years)")
    ax.set_ylabel(
        r"Record-mean area-weighted SV power [(nT/yr)$^2$]"
    )
    ax.set_title(
        f"{result['label'].capitalize()} recovery mode period and power "
        f"(n <= {result['degree']}; {period_view_title(view_label)})"
    )
    ax.grid(alpha=0.25, which="both")
    ax.legend()
    fig.tight_layout()
    return fig, ax


def plot_eigenvalue_recovery(
    result,
    period_limits,
    view_label,
):
    baseline = result["baseline_suite"]
    fig, ax = plt.subplots(
        figsize=(text_width, 0.65 * text_width)
    )
    ax.scatter(
        baseline["period"],
        baseline["growth_rate"],
        marker="o",
        facecolors="none",
        edgecolors="black",
        linewidths=1.5,
        s=65,
        label="Unperturbed baseline",
        zorder=4,
    )
    for realisation_index, (suite, match) in enumerate(
        zip(result["perturbed_suites"], result["matches"])
    ):
        matched = matched_perturbed_candidate_mask(
            suite,
            match,
        )
        if np.any(matched):
            ax.scatter(
                suite["period"][matched],
                suite["eigenvalue"][matched].real,
                marker="x",
                color="green",
                linewidths=1.8,
                alpha=0.65,
                s=55,
                label="Matched perturbed candidate"
                if realisation_index == 0
                else None,
            )
        if np.any(~matched):
            ax.scatter(
                suite["period"][~matched],
                suite["eigenvalue"][~matched].real,
                marker="x",
                color="red",
                linewidths=1.8,
                alpha=0.55,
                s=55,
                label="Unmatched perturbed candidate"
                if realisation_index == 0
                else None,
            )
    ax.set_xscale("log")
    ax.set_xlim(*period_limits)
    ax.axhline(0.0, color="0.5", linewidth=0.8)
    ax.set_xlabel("Period (years)")
    ax.set_ylabel(r"Growth/decay rate $\sigma$ (yr$^{-1}$)")
    ax.set_title(
        f"{result['label'].capitalize()} recovery continuous eigenvalues "
        f"(n <= {result['degree']}; {period_view_title(view_label)})"
    )
    ax.grid(alpha=0.25, which="both")
    ax.legend()
    fig.tight_layout()
    return fig, ax


def plot_similarity_recovery(
    result,
    period_limits,
    view_label,
):
    baseline = result["baseline_suite"]
    fig, ax = plt.subplots(
        figsize=(text_width, 0.65 * text_width)
    )
    ax.scatter(
        baseline["period"],
        np.ones(baseline["period"].size),
        marker="o",
        facecolors="none",
        edgecolors="black",
        linewidths=1.5,
        s=65,
        label="Unperturbed baseline",
        zorder=4,
    )
    green_line_label_used = False
    red_line_label_used = False
    for realisation_index, (suite, match) in enumerate(
        zip(result["perturbed_suites"], result["matches"])
    ):
        matched = matched_perturbed_candidate_mask(
            suite,
            match,
        )
        for period in suite["period"][matched]:
            ax.axvline(
                period,
                color="green",
                linestyle="--",
                linewidth=1.0,
                alpha=0.55,
                label=(
                    "Matched candidate period"
                    if not green_line_label_used
                    else None
                ),
            )
            green_line_label_used = True
        for period in suite["period"][~matched]:
            ax.axvline(
                period,
                color="red",
                linestyle="--",
                linewidth=1.0,
                alpha=0.4,
                label=(
                    "Unmatched candidate period"
                    if not red_line_label_used
                    else None
                ),
            )
            red_line_label_used = True

        recovered = match["recovered"]
        ax.scatter(
            match["recovered_period"][recovered],
            match["similarity"][recovered],
            marker="x",
            color="green",
            linewidths=1.8,
            alpha=0.55,
            s=55,
            label="Matched covariance recovery"
            if realisation_index == 0
            else None,
        )
    ax.set_xscale("log")
    ax.set_xlim(*period_limits)
    ax.set_ylim(0.0, 1.03)
    ax.set_xlabel("Recovered period (years)")
    ax.set_ylabel("Complex spatial-phasor similarity")
    ax.set_title(
        f"{result['label'].capitalize()} recovery period and similarity "
        f"(n <= {result['degree']}; {period_view_title(view_label)})"
    )
    ax.grid(alpha=0.25, which="both")
    ax.legend()
    fig.tight_layout()
    return fig, ax


def plot_singular_diagnostic(result):
    diagnostics = result["singular_diagnostics"]
    fig, ax = plt.subplots(
        figsize=(text_width, 0.48 * text_width),
    )
    baseline = diagnostics["baseline"]
    indices = np.arange(1, baseline["magnitude"].size + 1)
    ax.plot(
        indices,
        baseline["magnitude"],
        color="black",
        label="Unperturbed CHAOS",
    )

    for noise_index, diagnostic in enumerate(
        diagnostics["noise_only"]
    ):
        diagnostic_indices = np.arange(
            1,
            diagnostic["magnitude"].size + 1,
        )
        ax.plot(
            diagnostic_indices,
            diagnostic["magnitude"],
            color="tab:purple",
            alpha=0.45,
            label=(
                "Noise-only spectrum"
                if noise_index == 0
                else None
            ),
        )
    for perturbed_index, diagnostic in enumerate(
        diagnostics["perturbed"]
    ):
        diagnostic_indices = np.arange(
            1,
            diagnostic["magnitude"].size + 1,
        )
        ax.plot(
            diagnostic_indices,
            diagnostic["magnitude"],
            color="tab:blue",
            alpha=0.25,
            label=(
                "Perturbed CHAOS spectrum"
                if perturbed_index == 0
                else None
            ),
        )

    if result["selected_svd_rank"] > 0:
        ax.axvline(
            result["selected_svd_rank"],
            color="black",
            linestyle="-.",
            linewidth=1.0,
            label="Selected rank",
        )
    if result["noise_threshold"] is not None:
        ax.axhline(
            result["noise_threshold"],
            color="tab:red",
            linestyle="--",
            label="Noise threshold",
        )

    ax.set_yscale("log")
    ax.set_xlabel("Singular-value index")
    ax.set_ylabel("Singular-value magnitude")
    ax.legend(loc="lower right")
    ax.set_title(
        f"{result['label'].capitalize()} recovery SVD diagnostics "
        f"(n <= {result['degree']})"
    )
    fig.tight_layout()
    return fig, ax


def plot_reconstruction_diagnostic(result):
    metrics = result["reconstruction_metrics"]
    fig, ax = plt.subplots(
        figsize=(text_width, 0.48 * text_width)
    )
    ax.plot(
        analysis_times_absolute,
        metrics["input_rms"],
        color="black",
        label="Unperturbed CHAOS input",
    )
    ax.plot(
        analysis_times_absolute,
        metrics["reconstruction_rms"],
        color="tab:blue",
        label="DMD reconstruction (all recovered modes)",
    )
    ax.plot(
        analysis_times_absolute,
        metrics["residual_rms"],
        color="tab:red",
        label="Reconstruction - input",
    )
    ax.set_xlabel("Decimal year")
    ax.set_ylabel("Area-weighted spatial RMS (nT/yr)")
    ax.set_title(
        f"{result['label'].capitalize()} recovery signal reconstruction "
        f"(n <= {result['degree']}; relative RMS error "
        f"{metrics['record_relative_rms_error']:.3g})"
    )
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    return fig, ax


def plot_similarity_heatmaps(result):
    figures = []
    baseline_periods = result["baseline_suite"]["period"]
    for realisation_index, (suite, match) in enumerate(
        zip(result["perturbed_suites"], result["matches"])
    ):
        similarities = match["similarity_matrix"]
        fig, ax = plt.subplots(
            figsize=(text_width, 0.72 * text_width)
        )
        image = ax.imshow(
            similarities,
            origin="lower",
            aspect="auto",
            vmin=0.0,
            vmax=1.0,
            cmap="viridis",
        )
        recovered_rows = np.flatnonzero(match["recovered"])
        if recovered_rows.size:
            ax.scatter(
                match["candidate_index"][recovered_rows],
                recovered_rows,
                marker="s",
                s=80,
                facecolors="none",
                edgecolors="red",
                linewidths=1.2,
                label="Accepted match",
            )
            ax.legend()
        ax.set_xlabel("Perturbed candidate index")
        ax.set_ylabel("Unperturbed baseline candidate index")
        ax.set_title(
            f"{result['label'].capitalize()} recovery similarity matrix "
            f"(n <= {result['degree']}; realisation "
            f"{realisation_index + 1})"
        )
        colourbar = fig.colorbar(image, ax=ax)
        colourbar.set_label("Complex spatial-phasor similarity")
        fig.tight_layout()
        figures.append((fig, ax))
    return figures


for result in chaos_degree_results.values():
    full_period_limits = finite_period_limits(
        result["baseline_suite"],
        result["perturbed_suites"],
    )
    zoom_period_limits = (
        period_plot_lower_bound,
        period_plot_upper_bound,
    )
    figures = {}

    for view_label, period_limits in (
        ("full", full_period_limits),
        ("zoom", zoom_period_limits),
    ):
        figures[f"period_power_{view_label}"] = plot_period_power(
            result,
            period_limits,
            view_label,
        )
        figures[f"eigenvalue_{view_label}"] = (
            plot_eigenvalue_recovery(
                result,
                period_limits,
                view_label,
            )
        )
        figures[f"similarity_{view_label}"] = (
            plot_similarity_recovery(
                result,
                period_limits,
                view_label,
            )
        )

    if singular_value_plot_flag:
        figures["svd"] = plot_singular_diagnostic(result)
    if reconstruction_plot_flag:
        figures["reconstruction"] = (
            plot_reconstruction_diagnostic(result)
        )
    if similarity_heatmap_plot_flag:
        figures["similarity_heatmaps"] = (
            plot_similarity_heatmaps(result)
        )
    result["figures"] = figures
    save_figure_collection(result)
    save_chaos_numerical_results(result)

if show_figures_interactively:
    plt.show(block=False)
    plt.pause(0.1)


# %% ------------------------------------------------------
# DEGREE-SPECIFIC VIDEOS USING ALL RECOVERED MODES
# ------------------------------------------------------

if video_plot or all_modes_video_plot:
    for result in chaos_degree_results.values():
        output_directory = degree_material_directory(
            result,
            "videos",
        )
        if output_directory is None:
            output_directory = PROJECT_ROOT / "outputs"
            output_directory.mkdir(
                parents=True,
                exist_ok=True,
            )
        suite = result["baseline_suite"]
        parameter_summary = (
            f"CHAOS-8.6 | {DMD_algorithm} | "
            f"{result['label']} recovery n<={result['degree']} | "
            f"rank={result['selected_svd_rank']} | "
            + (
                f"Hankel d={hankel_d}"
                if hankel_embedding_flag
                else "no Hankel embedding"
            )
            + " | all recovered modes"
        )
        configuration_stem = (
            f"CHAOS_{DMD_algorithm}_{result['label']}_"
            f"N{result['degree']}_r{result['selected_svd_rank']}"
        )

        if video_plot:
            reconstruction_video_data = (
                prepare_chaos_reconstruction_video_data(
                    baseline_sv_input=result["baseline_sv_input"],
                    analysis_times_absolute=analysis_times_absolute,
                    candidate_eigenvalues=(
                        suite["reconstruction_eigenvalue"]
                    ),
                    candidate_modes=suite["reconstruction_mode"],
                    frame_spacing_years=video_frame_spacing_years,
                    reconstruction_sv=result["reconstruction"],
                )
            )
            result["reconstruction_video_output_path"] = (
                make_chaos_reconstruction_video(
                    video_data=reconstruction_video_data,
                    output_path=output_directory / (
                        configuration_stem
                        + "_all_modes_reconstruction.mp4"
                    ),
                    parameter_summary=parameter_summary,
                    nlat=state_shape[0],
                    nlon=state_shape[1],
                    fps=video_fps,
                    cmap="seismic",
                    dpi=100,
                )
            )

        if all_modes_video_plot:
            all_modes_video_data = prepare_chaos_all_modes_video_data(
                analysis_times_absolute=analysis_times_absolute,
                candidate_eigenvalues=(
                    suite["reconstruction_eigenvalue"]
                ),
                candidate_modes=suite["reconstruction_mode"],
                candidate_periods=suite["reconstruction_period"],
                frame_spacing_years=video_frame_spacing_years,
            )
            result["all_modes_video_output_path"] = (
                make_chaos_all_modes_video(
                    video_data=all_modes_video_data,
                    output_path=output_directory / (
                        configuration_stem
                        + "_individual_modes.mp4"
                    ),
                    parameter_summary=parameter_summary,
                    nlat=state_shape[0],
                    nlon=state_shape[1],
                    fps=video_fps,
                    cmap="seismic",
                    dpi=100,
                )
            )


# Public result object retained for interactive inspection.
CHAOS_DMD_results = {
    "metadata": {
        "model": "CHAOS-8.6",
        "algorithm": DMD_algorithm,
        "hankel_d": hankel_d if hankel_embedding_flag else None,
        "rank_method": svd_rank_selection,
        "recovery_degrees": recovery_degrees.copy(),
        "period_plot_window_years": (
            period_plot_lower_bound,
            period_plot_upper_bound,
        ),
        "period_window_is_plot_only": True,
        "remove_temporal_mean": remove_temporal_mean_flag,
    },
    "by_degree": chaos_degree_results,
}

if run_output_directory is not None:
    CHAOS_DMD_results["metadata"]["run_output_directory"] = (
        run_output_directory
    )

if show_figures_interactively:
    plt.show()
