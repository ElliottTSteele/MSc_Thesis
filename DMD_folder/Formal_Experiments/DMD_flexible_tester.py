"""
Flexible synthetic DMD recovery at two independently fitted degree truncations.

The low and high recovery degrees are separate experiments: each constructs
its own spatial projection, selects its own SVD rank, fits every requested DMD
input, matches all synthetic input modes, reconstructs the resolved signal
from every unique recovered candidate, and produces its own diagnostics.
Period bounds are used only for additional zoomed diagnostic figures.
"""

# %% FILE SYSTEM AND DEPENDENCY SETUP

import sys
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

import h5py
import matplotlib.pyplot as plt
import numpy as np
from pydmd.utils import pseudo_hankel_matrix
from tqdm import tqdm

from src.msc_thesis.paths import *
from src.msc_thesis.synSetup import *
from src.msc_thesis.synUtils import *
from src.msc_thesis.synDMD import *
from src.msc_thesis.synVideo import (
    prepare_recovery_video_data,
    make_dmd_recovery_video,
)

plt.rcParams.update({"font.size": 11})


# %% ------------------------------------------------------
# USER SETTINGS
# ------------------------------------------------------

# Every listed mode is included simultaneously in the synthetic record.
mode_numbers = [
    1, 2, 3, 4, 6, 11, 15, 16, 18, 20, 21, 25, 29, 30, 32, 33,
    34, 36, 37, 38, 39, 41, 43, 45, 48, 53, 54, 57, 59, 60, 61, 62,
]
mode_numbers = [str(mode_number) for mode_number in mode_numbers]

DMD_algorithm = "opdmd"

# Optional time-delay embedding. No temporal filtering is applied.
hankel_embedding_flag = True
hankel_d = 10
hankel_reconstruction_method = "first"

# "noise_threshold" uses the unmodified threshold-selected rank.
# "fixed" passes fixed_svd_rank directly to PyDMD.
svd_rank_method = "noise_threshold"
fixed_svd_rank = -1
singular_value_plot_flag = True

ensemble_flag = False
n_realisations = 1
n_noise_rank_realisations = n_realisations
noise_rank_quantile = 0.95
noise_temporal_model = "independent"
noise_tau_years = 6

# Record selection and temporal subsampling.
high_q_flag = True
n_skip = 1

# Two independent recovery analyses. No degree sweep is performed.
low_recovery_degree = 10
high_recovery_degree = 15
recovery_degrees = {
    "low": low_recovery_degree,
    "high": high_recovery_degree,
}

# Plot-only period window. It never changes fitting, matching, reconstruction,
# videos, or the candidate arrays retained in the results.
period_plot_lower_bound = 1.0
period_plot_upper_bound = 25.0

show_ideal_all_modes = False
video_plot = True
video_compact_input_threshold = 8
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


# %% ------------------------------------------------------
# VALIDATE SETTINGS
# ------------------------------------------------------

if DMD_algorithm not in ("exact", "fbdmd", "opdmd"):
    raise ValueError(
        "DMD_algorithm must be 'exact', 'fbdmd', or 'opdmd'."
    )

if svd_rank_method not in ("noise_threshold", "fixed"):
    raise ValueError(
        "svd_rank_method must be 'noise_threshold' or 'fixed'."
    )

if not isinstance(n_skip, (int, np.integer)) or n_skip < 1:
    raise ValueError("n_skip must be a positive integer.")

if not (
    isinstance(low_recovery_degree, (int, np.integer))
    and isinstance(high_recovery_degree, (int, np.integer))
    and 1 <= low_recovery_degree < high_recovery_degree <= 20
):
    raise ValueError(
        "Recovery degrees must satisfy "
        "1 <= low_recovery_degree < high_recovery_degree <= 20."
    )

if not (
    np.isfinite(period_plot_lower_bound)
    and np.isfinite(period_plot_upper_bound)
    and 0.0 < period_plot_lower_bound < period_plot_upper_bound
):
    raise ValueError(
        "Plot period bounds must be finite, positive, and increasing."
    )

if svd_rank_method == "fixed":
    fixed_rank_is_integer = (
        isinstance(fixed_svd_rank, (int, np.integer))
        and not isinstance(fixed_svd_rank, (bool, np.bool_))
        and fixed_svd_rank >= -1
    )
    fixed_rank_is_energy_fraction = (
        isinstance(fixed_svd_rank, (float, np.floating))
        and np.isfinite(fixed_svd_rank)
        and 0.0 < fixed_svd_rank < 1.0
    )
    if not (fixed_rank_is_integer or fixed_rank_is_energy_fraction):
        raise ValueError(
            "fixed_svd_rank must follow PyDMD conventions: -1, 0, a "
            "positive integer, or a floating-point energy fraction in (0, 1)."
        )

if (
    svd_rank_method == "noise_threshold"
    or singular_value_plot_flag
):
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
        "hankel_d must be a positive integer when embedding is enabled."
    )

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
    if svd_rank_method == "noise_threshold":
        svd_run_label = "svdNoise"
    else:
        fixed_rank_label = (
            str(fixed_svd_rank)
            .replace("-", "Minus")
            .replace(".", "p")
        )
        svd_run_label = f"svdFixed{fixed_rank_label}"
    run_name_parts = [
        "synthetic",
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
        f"Synthetic run outputs: {run_output_directory}"
    )


# %% ------------------------------------------------------
# BUILD THE COMBINED SYNTHETIC RECORD
# ------------------------------------------------------

file_path = Path(FELIX_DIR) / "R_splines_arbitrary.h5"
synthetic_suite_info = {}
gnm_total_ideal_list = []
gnm_total_resolved_list = []

with h5py.File(file_path, "r") as h5_file:
    for mode_number in tqdm(mode_numbers, desc="Loading combined modes"):
        mode_data = Component_Load_SV(mode_number)
        eigenvalue = complex(mode_data["eigenvalue"])
        gnm_phasor_degree_20 = Truncate_Gauss_Coeffs(
            mode_data["gnm"],
            tmax=20,
        )
        true_period = 2.0 * np.pi / np.abs(eigenvalue.imag)

        try:
            amplitude_scaler = mode_amp_scalings[mode_number]
        except KeyError as exc:
            raise KeyError(
                f"No amplitude scaling found for mode {mode_number}."
            ) from exc

        scaled_phasor = amplitude_scaler * gnm_phasor_degree_20
        gnm_mode_ideal = G_Time_Series_Eval(
            scaled_phasor,
            eigenvalue,
        )

        dataset_name = f"mode_{mode_number}/without_decay"
        if dataset_name not in h5_file:
            raise KeyError(f"Missing HDF5 dataset: {dataset_name}")

        gnm_spline = np.asarray(h5_file[dataset_name][()])
        gnm_mode_resolved = H_sv @ (
            amplitude_scaler * gnm_spline
        )

        synthetic_suite_info[mode_number] = {
            "true_period": float(true_period),
            "true_eigenvalue": eigenvalue,
            "gnm_phasor": scaled_phasor,
            "gnm_ideal": gnm_mode_ideal,
            "gnm_resolved": gnm_mode_resolved,
        }
        gnm_total_ideal_list.append(gnm_mode_ideal)
        gnm_total_resolved_list.append(gnm_mode_resolved)

gnm_total_ideal = np.sum(np.asarray(gnm_total_ideal_list), axis=0)
gnm_total_resolved = np.sum(
    np.asarray(gnm_total_resolved_list),
    axis=0,
)
del gnm_total_ideal_list
del gnm_total_resolved_list


# %% ------------------------------------------------------
# BUILD A COMMON NOISE BANK
# ------------------------------------------------------

noise_diagnostics_required = (
    svd_rank_method == "noise_threshold"
    or singular_value_plot_flag
)
noise_bank_size = max(
    n_noise_rank_realisations if noise_diagnostics_required else 0,
    n_realisations if ensemble_flag else 0,
)

if noise_bank_size:
    noise_only_bank = Perturbation_Generate(
        gnm_total_resolved,
        n_realisations=noise_bank_size,
        temporal_z=noise_temporal_model,
        tau=noise_tau_years,
        dt=dt_years,
        just_noise=True,
    )
else:
    noise_only_bank = np.empty(
        (0, *gnm_total_resolved.shape),
        dtype=float,
    )

noise_only_rank_bank = noise_only_bank[
    :n_noise_rank_realisations
] if noise_diagnostics_required else noise_only_bank[:0]

fit_queue = [
    ("ideal", gnm_total_ideal),
    ("resolved", gnm_total_resolved),
]
if ensemble_flag:
    fit_queue.extend(
        ("perturbed", gnm_total_resolved + noise_realisation)
        for noise_realisation in noise_only_bank[:n_realisations]
    )


# %% ------------------------------------------------------
# SHARED DMD HELPERS
# ------------------------------------------------------

if high_q_flag:
    analysis_times_absolute = times_absolute[good_record_slice]
else:
    analysis_times_absolute = times_absolute.copy()
analysis_times_absolute = analysis_times_absolute[::n_skip]

if analysis_times_absolute.size < 2:
    raise ValueError("At least two selected snapshots are required.")

snapshot_differences = np.diff(analysis_times_absolute)
dt_snapshot = float(np.median(snapshot_differences))
if not np.allclose(
    snapshot_differences,
    dt_snapshot,
    rtol=1e-8,
    atol=1e-10,
):
    raise ValueError("Selected snapshots are not uniformly sampled.")

if hankel_embedding_flag and hankel_d >= analysis_times_absolute.size:
    raise ValueError(
        f"hankel_d={hankel_d} leaves too few embedded snapshots."
    )

evaluation_times = (
    analysis_times_absolute
    - analysis_times_absolute[0]
)
spatial_weights = np.asarray(W2D, dtype=float).ravel()


def prepare_sv_input_for_dmd(
    gnm_input,
    truncation_degree,
    projection_operator,
):
    """Window, degree-truncate, project, and subsample one coefficient record."""

    if high_q_flag:
        gnm_windowed = gnm_input[good_record_slice, :]
    else:
        gnm_windowed = gnm_input

    gnm_band = Truncate_Gauss_Coeffs(
        gnm_windowed,
        tmax=truncation_degree,
    )
    sv_input = projection_operator @ gnm_band.T
    sv_input = sv_input[:, ::n_skip]

    expected_shape = (
        state_shape[0] * state_shape[1],
        analysis_times_absolute.size,
    )
    if sv_input.shape != expected_shape:
        raise ValueError(
            f"Unexpected SV input shape at n={truncation_degree}: "
            f"{sv_input.shape} vs {expected_shape}."
        )
    if not np.all(np.isfinite(sv_input)):
        raise ValueError(
            f"SV input contains non-finite values at n={truncation_degree}."
        )
    return sv_input


def pre_truncation_svd_input(sv_input):
    """Return the matrix used by the selected estimator for rank selection."""

    if hankel_embedding_flag:
        matrix = pseudo_hankel_matrix(sv_input, d=hankel_d)
    else:
        matrix = sv_input
    if DMD_algorithm in ("exact", "fbdmd"):
        matrix = matrix[:, :-1]
    return matrix


def calculate_singular_diagnostic(sv_input):
    """Return raw singular magnitudes and cumulative squared energy."""

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


def select_degree_svd_rank(truncation_degree, projection_operator):
    """Select the rank independently at one spherical-harmonic degree."""

    resolved_sv = prepare_sv_input_for_dmd(
        gnm_total_resolved,
        truncation_degree,
        projection_operator,
    )
    resolved_diagnostic = calculate_singular_diagnostic(resolved_sv)
    noise_diagnostics = [
        calculate_singular_diagnostic(
            prepare_sv_input_for_dmd(
                noise_realisation,
                truncation_degree,
                projection_operator,
            )
        )
        for noise_realisation in noise_only_rank_bank
    ]

    if svd_rank_method == "fixed":
        return (
            fixed_svd_rank,
            None,
            None,
            resolved_diagnostic,
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
            resolved_diagnostic["magnitude"] > noise_threshold
        )
    )
    if base_noise_rank < 1:
        raise RuntimeError(
            "The noise threshold retained no resolved singular values at "
            f"n={truncation_degree}; threshold={noise_threshold:.6e}."
        )

    return (
        base_noise_rank,
        base_noise_rank,
        noise_threshold,
        resolved_diagnostic,
        noise_diagnostics,
    )


def build_selected_dmd(selected_svd_rank):
    if DMD_algorithm == "exact":
        return build_exact_dmd(svd_rank=selected_svd_rank)
    if DMD_algorithm == "fbdmd":
        return build_fbdmd(svd_rank=selected_svd_rank)
    return build_bopdmd(svd_rank=selected_svd_rank)


def fit_candidate_suite(sv_input, selected_svd_rank):
    """Fit DMD and retain every physical candidate exactly once."""

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
        raise ValueError("Extracted DMD candidates do not align.")

    candidate_ids = np.arange(eigenvalues.size, dtype=int)
    oscillatory = np.isfinite(periods) & (periods > 0.0)
    static = np.isposinf(periods)
    oscillatory_order = np.argsort(periods[oscillatory])
    oscillatory_indices = np.flatnonzero(oscillatory)[oscillatory_order]
    static_indices = np.flatnonzero(static)
    retained_indices = np.concatenate(
        (oscillatory_indices, static_indices)
    )

    reconstruction_eigenvalues = eigenvalues[retained_indices]
    reconstruction_modes = modes[:, retained_indices]
    reconstruction_periods = periods[retained_indices]
    reconstruction_ids = candidate_ids[retained_indices]

    oscillatory_count = oscillatory_indices.size
    oscillatory_eigenvalues = reconstruction_eigenvalues[
        :oscillatory_count
    ]
    oscillatory_modes = reconstruction_modes[:, :oscillatory_count]
    oscillatory_periods = reconstruction_periods[:oscillatory_count]
    oscillatory_ids = reconstruction_ids[:oscillatory_count]
    powers = np.asarray([
        SV_Grid_Phasor_Record_Power(
            oscillatory_modes[:, index],
            oscillatory_eigenvalues[index],
            evaluation_times=evaluation_times,
            spatial_weights=spatial_weights,
        )
        for index in range(oscillatory_count)
    ])

    return {
        "effective_svd_rank": int(np.asarray(dmd.modes).shape[1]),
        "eigenvalue": oscillatory_eigenvalues,
        "mode": oscillatory_modes,
        "period": oscillatory_periods,
        "candidate_id": oscillatory_ids,
        "power": powers,
        "static_count": int(static_indices.size),
        "reconstruction_eigenvalue": reconstruction_eigenvalues,
        "reconstruction_mode": reconstruction_modes,
        "reconstruction_period": reconstruction_periods,
        "reconstruction_candidate_id": reconstruction_ids,
    }


def evaluate_candidate_sum(eigenvalues, modes, relative_times):
    """Evaluate all unique continuous-time candidates and sum them."""

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
    """Area-weighted RMS across the spatial grid at each time."""

    records = np.asarray(records)
    return np.sqrt(
        np.sum(
            spatial_weights[:, None] * np.square(records),
            axis=0,
        )
        / np.sum(spatial_weights)
    )


def candidate_matched_inputs(candidate_count, recovery, result_key):
    matched_inputs = [[] for _ in range(candidate_count)]
    for mode_number, mode_result in recovery.items():
        match = mode_result.get(result_key, {})
        if "candidate_index" in match:
            matched_inputs[int(match["candidate_index"])].append(mode_number)
    return matched_inputs


def finite_period_limits(recovery, suites):
    periods = [
        float(result["true_period"])
        for result in recovery.values()
    ]
    for suite in suites:
        periods.extend(
            np.asarray(suite["period"], dtype=float).tolist()
        )
    periods = np.asarray(periods, dtype=float)
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

def analyse_recovery_degree(case_label, truncation_degree):
    print(
        f"\nSYNTHETIC DMD: {case_label.upper()} RECOVERY "
        f"(n <= {truncation_degree})"
    )

    projection_operator = Truncate_Gauss_Coeffs(
        A_20_dict["r"],
        tmax=truncation_degree,
    )
    expected_coefficients = (truncation_degree + 1) ** 2 - 1
    expected_projection_shape = (
        state_shape[0] * state_shape[1],
        expected_coefficients,
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
        resolved_singular_diagnostic,
        noise_singular_diagnostics,
    ) = select_degree_svd_rank(
        truncation_degree,
        projection_operator,
    )

    suites = {"perturbed": []}
    singular_diagnostics = {
        "resolved": resolved_singular_diagnostic,
        "ideal": None,
        "perturbed": [],
        "noise_only": noise_singular_diagnostics,
    }
    sv_inputs = {}

    for input_label, gnm_input in fit_queue:
        sv_input = prepare_sv_input_for_dmd(
            gnm_input,
            truncation_degree,
            projection_operator,
        )
        suite = fit_candidate_suite(
            sv_input,
            selected_svd_rank,
        )

        if input_label == "perturbed":
            suites["perturbed"].append(suite)
            singular_diagnostics["perturbed"].append(
                calculate_singular_diagnostic(sv_input)
            )
        else:
            suites[input_label] = suite
            sv_inputs[input_label] = sv_input
            if input_label == "ideal":
                singular_diagnostics["ideal"] = (
                    calculate_singular_diagnostic(sv_input)
                )

    recovery = {}
    target_phasors = {}
    for mode_number in mode_numbers:
        source = synthetic_suite_info[mode_number]
        target_phasor = projection_operator @ (
            Truncate_Gauss_Coeffs(
                source["gnm_phasor"],
                tmax=truncation_degree,
            ).T
        )
        target_phasors[mode_number] = target_phasor

        recovery[mode_number] = {
            "true_period": source["true_period"],
            "true_eigenvalue": source["true_eigenvalue"],
            "DMD_ideal": {},
            "DMD": {},
            "DMD_noised": {
                "similarity": [],
                "eigenvalue": [],
                "recovered_period": [],
            },
        }

        ideal_match = best_spatial_match(
            suites["ideal"]["mode"],
            suites["ideal"]["eigenvalue"],
            suites["ideal"]["period"],
            target_phasor,
        )
        if ideal_match is not None:
            recovery[mode_number]["DMD_ideal"] = ideal_match

        resolved_match = best_spatial_match(
            suites["resolved"]["mode"],
            suites["resolved"]["eigenvalue"],
            suites["resolved"]["period"],
            target_phasor,
        )
        if resolved_match is not None:
            recovery[mode_number]["DMD"] = resolved_match

        for perturbed_suite in suites["perturbed"]:
            perturbed_match = best_spatial_match(
                perturbed_suite["mode"],
                perturbed_suite["eigenvalue"],
                perturbed_suite["period"],
                target_phasor,
            )
            if perturbed_match is not None:
                noised_result = recovery[mode_number]["DMD_noised"]
                noised_result["similarity"].append(
                    perturbed_match["similarity"]
                )
                noised_result["eigenvalue"].append(
                    perturbed_match["eigenvalue"]
                )
                noised_result["recovered_period"].append(
                    perturbed_match["recovered_period"]
                )

    modal_results = {
        "true": {},
        "ideal": {
            "period": suites["ideal"]["period"].copy(),
            "power": suites["ideal"]["power"].copy(),
            "matched_input_modes": candidate_matched_inputs(
                suites["ideal"]["period"].size,
                recovery,
                "DMD_ideal",
            ),
        },
        "resolved": {
            "period": suites["resolved"]["period"].copy(),
            "power": suites["resolved"]["power"].copy(),
            "matched_input_modes": candidate_matched_inputs(
                suites["resolved"]["period"].size,
                recovery,
                "DMD",
            ),
        },
    }
    for mode_number in mode_numbers:
        source = synthetic_suite_info[mode_number]
        modal_results["true"][mode_number] = {
            "period": source["true_period"],
            "power": SV_Grid_Phasor_Record_Power(
                target_phasors[mode_number],
                1j * source["true_eigenvalue"].imag,
                evaluation_times=evaluation_times,
                spatial_weights=spatial_weights,
            ),
        }

    resolved_suite = suites["resolved"]
    reconstruction = evaluate_candidate_sum(
        resolved_suite["reconstruction_eigenvalue"],
        resolved_suite["reconstruction_mode"],
        evaluation_times,
    )
    resolved_input = sv_inputs["resolved"]
    residual = reconstruction - resolved_input
    reconstruction_metrics = {
        "input_rms": area_weighted_rms(resolved_input),
        "reconstruction_rms": area_weighted_rms(reconstruction),
        "residual_rms": area_weighted_rms(residual),
        "record_relative_rms_error": float(
            np.sqrt(
                np.sum(
                    spatial_weights[:, None] * np.square(residual)
                )
                / np.sum(
                    spatial_weights[:, None] * np.square(resolved_input)
                )
            )
        ),
    }

    print(
        f"Selected PyDMD rank: {selected_svd_rank}; "
        f"effective resolved rank: "
        f"{resolved_suite['effective_svd_rank']}; "
        f"oscillatory candidates: {resolved_suite['period'].size}; "
        f"static candidates: {resolved_suite['static_count']}; "
        f"relative reconstruction RMS error: "
        f"{reconstruction_metrics['record_relative_rms_error']:.4g}"
    )

    return {
        "label": case_label,
        "degree": int(truncation_degree),
        "projection_operator": projection_operator,
        "selected_svd_rank": selected_svd_rank,
        "base_noise_rank": base_noise_rank,
        "noise_threshold": noise_threshold,
        "singular_diagnostics": singular_diagnostics,
        "sv_inputs": sv_inputs,
        "suites": suites,
        "target_phasors": target_phasors,
        "recovery": recovery,
        "modal_results": modal_results,
        "reconstruction": reconstruction,
        "residual": residual,
        "reconstruction_metrics": reconstruction_metrics,
    }


synthetic_degree_results = {
    case_label: analyse_recovery_degree(case_label, degree)
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


def save_synthetic_numerical_results(result):
    """Save the main input, candidate, matching, and reconstruction arrays."""

    if not save_numerical_results:
        return
    numerical_directory = degree_material_directory(
        result,
        "numerical",
    )
    if numerical_directory is None:
        return

    archive = {
        "analysis_times_absolute": analysis_times_absolute,
        "evaluation_times": evaluation_times,
        "resolved_input": result["sv_inputs"]["resolved"],
        "ideal_input": result["sv_inputs"]["ideal"],
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
        "effective_resolved_svd_rank": np.asarray(
            result["suites"]["resolved"][
                "effective_svd_rank"
            ]
        ),
        "mode_numbers": np.asarray(
            mode_numbers,
            dtype=str,
        ),
    }

    for suite_name in ("ideal", "resolved"):
        suite = result["suites"][suite_name]
        archive[f"{suite_name}_eigenvalue"] = suite[
            "eigenvalue"
        ]
        archive[f"{suite_name}_mode"] = suite["mode"]
        archive[f"{suite_name}_period"] = suite["period"]
        archive[f"{suite_name}_power"] = suite["power"]
        archive[
            f"{suite_name}_reconstruction_eigenvalue"
        ] = suite["reconstruction_eigenvalue"]
        archive[
            f"{suite_name}_reconstruction_mode"
        ] = suite["reconstruction_mode"]
        archive[
            f"{suite_name}_reconstruction_period"
        ] = suite["reconstruction_period"]

    for mode_number, mode_result in result["recovery"].items():
        key_prefix = f"input_mode_{mode_number}"
        archive[f"{key_prefix}_true_period"] = np.asarray(
            mode_result["true_period"]
        )
        archive[f"{key_prefix}_true_eigenvalue"] = np.asarray(
            mode_result["true_eigenvalue"]
        )
        archive[f"{key_prefix}_target_phasor"] = result[
            "target_phasors"
        ][mode_number]

        for result_name, output_name in (
            ("DMD_ideal", "ideal_match"),
            ("DMD", "resolved_match"),
        ):
            match = mode_result[result_name]
            archive[
                f"{key_prefix}_{output_name}_candidate_index"
            ] = np.asarray(
                match.get("candidate_index", -1)
            )
            archive[
                f"{key_prefix}_{output_name}_similarity"
            ] = np.asarray(
                match.get("similarity", np.nan)
            )
            archive[
                f"{key_prefix}_{output_name}_eigenvalue"
            ] = np.asarray(
                match.get(
                    "eigenvalue",
                    np.nan + 1j * np.nan,
                )
            )
            archive[
                f"{key_prefix}_{output_name}_period"
            ] = np.asarray(
                match.get("recovered_period", np.nan)
            )

        noised_result = mode_result["DMD_noised"]
        archive[
            f"{key_prefix}_perturbed_similarity"
        ] = np.asarray(
            noised_result["similarity"],
            dtype=float,
        )
        archive[
            f"{key_prefix}_perturbed_eigenvalue"
        ] = np.asarray(
            noised_result["eigenvalue"],
            dtype=complex,
        )
        archive[
            f"{key_prefix}_perturbed_period"
        ] = np.asarray(
            noised_result["recovered_period"],
            dtype=float,
        )

    np.savez_compressed(
        numerical_directory / "numerical_results.npz",
        **archive,
    )


def write_run_configuration():
    """Write a human-readable record of the complete run configuration."""

    if run_output_directory is None:
        return

    lines = [
        "Synthetic flexible DMD run",
        f"Started: {run_started_at.isoformat(timespec='seconds')}",
        f"Algorithm: {DMD_algorithm}",
        "",
        "Synthetic input modes:",
        f"Number of input modes: {len(mode_numbers)}",
        f"Input modes: {', '.join(mode_numbers)}",
        "",
        "Recovery configuration:",
        f"Low recovery degree: {low_recovery_degree}",
        f"High recovery degree: {high_recovery_degree}",
        f"Rank method: {svd_rank_method}",
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
        f"Plot-only period lower bound (years): "
        f"{period_plot_lower_bound}",
        f"Plot-only period upper bound (years): "
        f"{period_plot_upper_bound}",
        f"Ensemble enabled: {ensemble_flag}",
        f"Ensemble realisations: {n_realisations}",
        f"Noise rank realisations: {n_noise_rank_realisations}",
        f"Noise rank quantile: {noise_rank_quantile}",
        f"Noise temporal model: {noise_temporal_model}",
        f"Noise tau (years): {noise_tau_years}",
        f"Figure formats: {', '.join(figure_output_formats)}",
        f"Figure dpi: {figure_output_dpi}",
        f"Videos enabled: {video_plot}",
        "",
        "Degree-specific results:",
    ]
    for result in synthetic_degree_results.values():
        lines.extend([
            (
                f"{result['label']} n<={result['degree']}: "
                f"selected rank={result['selected_svd_rank']}, "
                "effective resolved rank="
                f"{result['suites']['resolved']['effective_svd_rank']}, "
                "oscillatory candidates="
                f"{result['suites']['resolved']['period'].size}, "
                "static candidates="
                f"{result['suites']['resolved']['static_count']}, "
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
# DIAGNOSTIC PLOTS
# ------------------------------------------------------

def plot_singular_diagnostic(result):
    diagnostics = result["singular_diagnostics"]
    fig, ax = plt.subplots(
        figsize=(text_width, 0.48 * text_width),
    )
    categories = (
        ("ideal", "Ideal", "tab:blue"),
        ("resolved", "Resolved", "tab:orange"),
    )
    for key, label, colour in categories:
        diagnostic = diagnostics[key]
        if diagnostic is None:
            continue
        indices = np.arange(1, diagnostic["magnitude"].size + 1)
        ax.plot(
            indices,
            diagnostic["magnitude"],
            label=label,
            color=colour,
        )

    for noise_index, diagnostic in enumerate(
        diagnostics["noise_only"]
    ):
        indices = np.arange(1, diagnostic["magnitude"].size + 1)
        ax.plot(
            indices,
            diagnostic["magnitude"],
            color="tab:purple",
            alpha=0.45,
            label=(
                "Noise-only spectrum"
                if noise_index == 0
                else None
            ),
        )

    rank = result["selected_svd_rank"]
    if isinstance(rank, (int, np.integer)) and rank > 0:
        ax.axvline(
            rank,
            color="black",
            linestyle="-.",
            linewidth=1.0,
            label="Selected rank",
        )
    threshold = result["noise_threshold"]
    if threshold is not None:
        ax.axhline(
            threshold,
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


def matched_candidate_indices(result):
    """Return resolved candidates selected by at least one synthetic input."""

    return np.asarray(
        sorted({
            int(mode_result["DMD"]["candidate_index"])
            for mode_result in result["recovery"].values()
            if "candidate_index" in mode_result["DMD"]
        }),
        dtype=int,
    )


def period_view_title(view_label):
    if view_label == "zoom":
        return (
            f"{period_plot_lower_bound:g}-"
            f"{period_plot_upper_bound:g} year view"
        )
    return "full period range"


def plot_candidate_eigenvalue_recovery(
    result,
    period_limits,
    view_label,
):
    """Plot every resolved candidate using matched/unmatched colours."""

    suite = result["suites"]["resolved"]
    candidate_indices = np.arange(suite["period"].size)
    matched_indices = matched_candidate_indices(result)
    matched = np.isin(candidate_indices, matched_indices)

    fig, ax = plt.subplots(
        figsize=(text_width, 0.72 * text_width)
    )
    true_periods = np.asarray([
        mode_result["true_period"]
        for mode_result in result["recovery"].values()
    ])
    true_growth = np.asarray([
        complex(mode_result["true_eigenvalue"]).real
        for mode_result in result["recovery"].values()
    ])
    ax.scatter(
        true_periods,
        true_growth,
        marker="o",
        facecolors="none",
        edgecolors="black",
        linewidths=1.5,
        s=65,
        label="True input",
        zorder=5,
    )
    if np.any(matched):
        ax.scatter(
            suite["period"][matched],
            suite["eigenvalue"][matched].real,
            marker="x",
            color="green",
            linewidths=1.8,
            s=65,
            label="Matched DMD candidate",
            zorder=4,
        )
    if np.any(~matched):
        ax.scatter(
            suite["period"][~matched],
            suite["eigenvalue"][~matched].real,
            marker="x",
            color="red",
            linewidths=1.8,
            s=65,
            label="Unmatched DMD candidate",
            zorder=3,
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


def plot_candidate_similarity_recovery(
    result,
    period_limits,
    view_label,
):
    """Show match similarities and every resolved candidate period."""

    suite = result["suites"]["resolved"]
    candidate_indices = np.arange(suite["period"].size)
    matched_indices = matched_candidate_indices(result)
    matched = np.isin(candidate_indices, matched_indices)

    fig, ax = plt.subplots(
        figsize=(text_width, 0.72 * text_width)
    )
    true_periods = np.asarray([
        mode_result["true_period"]
        for mode_result in result["recovery"].values()
    ])
    ax.scatter(
        true_periods,
        np.ones(true_periods.size),
        marker="o",
        facecolors="none",
        edgecolors="black",
        linewidths=1.5,
        s=65,
        label="True input",
        zorder=5,
    )

    for line_index, period in enumerate(
        suite["period"][matched]
    ):
        ax.axvline(
            period,
            color="green",
            linestyle="--",
            linewidth=1.0,
            alpha=0.55,
            label=(
                "Matched candidate period"
                if line_index == 0
                else None
            ),
        )
    for line_index, period in enumerate(
        suite["period"][~matched]
    ):
        ax.axvline(
            period,
            color="red",
            linestyle="--",
            linewidth=1.0,
            alpha=0.4,
            label=(
                "Unmatched candidate period"
                if line_index == 0
                else None
            ),
        )

    matched_periods = []
    matched_similarities = []
    for mode_result in result["recovery"].values():
        match = mode_result["DMD"]
        if "candidate_index" not in match:
            continue
        matched_periods.append(match["recovered_period"])
        matched_similarities.append(match["similarity"])
    if matched_periods:
        ax.scatter(
            matched_periods,
            matched_similarities,
            marker="x",
            color="green",
            linewidths=1.8,
            s=65,
            label="Matched output similarity",
            zorder=6,
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


def plot_reconstruction_diagnostic(result):
    metrics = result["reconstruction_metrics"]
    fig, ax = plt.subplots(
        figsize=(text_width, 0.48 * text_width)
    )
    ax.plot(
        analysis_times_absolute,
        metrics["input_rms"],
        label="Resolved synthetic input",
        color="black",
    )
    ax.plot(
        analysis_times_absolute,
        metrics["reconstruction_rms"],
        label="DMD reconstruction (all recovered modes)",
        color="tab:blue",
    )
    ax.plot(
        analysis_times_absolute,
        metrics["residual_rms"],
        label="Reconstruction - input",
        color="tab:red",
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


for result in synthetic_degree_results.values():
    full_period_limits = finite_period_limits(
        result["recovery"],
        (
            result["suites"]["ideal"],
            result["suites"]["resolved"],
        ),
    )
    zoom_period_limits = (
        period_plot_lower_bound,
        period_plot_upper_bound,
    )

    figures = {}
    figures["period_power_full"] = plot_modal_period_power(
        result["modal_results"],
        figsize=(text_width, 0.72 * text_width),
        show_ideal=show_ideal_all_modes,
        period_xlim=full_period_limits,
    )
    figures["period_power_zoom"] = plot_modal_period_power(
        result["modal_results"],
        figsize=(text_width, 0.72 * text_width),
        show_ideal=show_ideal_all_modes,
        period_xlim=zoom_period_limits,
    )
    figures["period_power_full"][1].set_title(
        f"{result['label'].capitalize()} recovery retained-mode "
        f"period and power (n <= {result['degree']}; full period range)"
    )
    figures["period_power_zoom"][1].set_title(
        f"{result['label'].capitalize()} recovery retained-mode "
        f"period and power (n <= {result['degree']}; "
        f"{period_view_title('zoom')})"
    )
    figures["period_power_full"][0].tight_layout()
    figures["period_power_zoom"][0].tight_layout()

    for view_label, limits in (
        ("full", full_period_limits),
        ("zoom", zoom_period_limits),
    ):
        figures[f"eigenvalue_{view_label}"] = (
            plot_candidate_eigenvalue_recovery(
                result,
                limits,
                view_label,
            )
        )
        figures[f"similarity_{view_label}"] = (
            plot_candidate_similarity_recovery(
                result,
                limits,
                view_label,
            )
        )

    figures["svd"] = plot_singular_diagnostic(result)
    figures["reconstruction"] = plot_reconstruction_diagnostic(result)
    result["figures"] = figures
    save_figure_collection(result)
    save_synthetic_numerical_results(result)

if show_figures_interactively:
    plt.show(block=False)
    plt.pause(0.1)


# %% ------------------------------------------------------
# DEGREE-SPECIFIC RECOVERY VIDEOS
# ------------------------------------------------------

if video_plot:
    for result in synthetic_degree_results.values():
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
        resolved_suite = result["suites"]["resolved"]
        video_data = prepare_recovery_video_data(
            mode_numbers=mode_numbers,
            DMD_recovery=result["recovery"],
            synthetic_suite_info=synthetic_suite_info,
            target_phasors=result["target_phasors"],
            clean_candidate_eigs=resolved_suite["eigenvalue"],
            clean_candidate_modes=resolved_suite["mode"],
            clean_candidate_ids=resolved_suite["candidate_id"],
            recovered_sum_candidate_eigs=(
                resolved_suite["reconstruction_eigenvalue"]
            ),
            recovered_sum_candidate_modes=(
                resolved_suite["reconstruction_mode"]
            ),
            recovered_sum_candidate_ids=(
                resolved_suite["reconstruction_candidate_id"]
            ),
            A_r_current=result["projection_operator"],
            model_time_years=times_absolute,
            dt_years=dt_years,
            Nmax=result["degree"],
            truncate_gauss_coeffs=Truncate_Gauss_Coeffs,
            high_q_flag=high_q_flag,
            good_record_slice=good_record_slice,
            truth_t0_year=times_absolute[0],
            truth_include_growth=False,
            frame_spacing_years=video_frame_spacing_years,
            compact_input_threshold=video_compact_input_threshold,
        )

        video_path = output_directory / (
            f"{DMD_algorithm}_"
            f"{result['label']}_N{result['degree']}_"
            "all_recovered_modes.mp4"
        )
        result["video_output_path"] = make_dmd_recovery_video(
            video_data=video_data,
            output_path=video_path,
            DMD_algorithm=DMD_algorithm,
            Nmax=result["degree"],
            svd_rank=resolved_suite["effective_svd_rank"],
            nlat=state_shape[0],
            nlon=state_shape[1],
            hankel_embedding_d=(
                hankel_d if hankel_embedding_flag else None
            ),
            high_q_flag=high_q_flag,
            n_skip=n_skip,
            fps=video_fps,
            cmap="seismic",
            dpi=100,
        )


# Public result object retained for interactive inspection.
DMD_recovery = {
    "metadata": {
        "algorithm": DMD_algorithm,
        "hankel_d": hankel_d if hankel_embedding_flag else None,
        "rank_method": svd_rank_method,
        "recovery_degrees": recovery_degrees.copy(),
        "period_plot_window_years": (
            period_plot_lower_bound,
            period_plot_upper_bound,
        ),
        "period_window_is_plot_only": True,
    },
    "by_degree": synthetic_degree_results,
}

if run_output_directory is not None:
    DMD_recovery["metadata"]["run_output_directory"] = (
        run_output_directory
    )

if show_figures_interactively:
    plt.show()
