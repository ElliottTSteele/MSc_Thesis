"""
Run the synthetic DMD recovery pipeline on one input wave at a time.

Each synthetic mode is treated as an independent experiment.  The ideal,
resolution-mapped, and (optionally) uncertainty-perturbed records are fitted
separately, and only the best spatial match to the wave injected into that
record is retained.  The final figures compare recovery across the complete
suite of isolated waves.
"""

# %% FILE SYSTEM AND DEPENDENCY SETUP

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

import h5py
import matplotlib.pyplot as plt
import numpy as np
from pydmd.utils import pseudo_hankel_matrix
from tqdm import tqdm

from src.msc_thesis.paths import FELIX_DIR
from src.msc_thesis.synSetup import (
    dt_years,
    good_record_slice,
    mode_amp_scalings,
)
from src.msc_thesis.synUtils import (
    A_20_dict,
    Component_Load_SV,
    G_Time_Series_Eval,
    H_sv,
    Truncate_Gauss_Coeffs,
)
from src.msc_thesis.synDMD import (
    Long_Period_Taper_Filter,
    Perturbation_Generate,
    apply_hankel_embedding,
    best_spatial_match,
    build_bopdmd,
    build_exact_dmd,
    build_fbdmd,
    extract_optimized_dmd_candidates,
    extract_standard_dmd_candidates,
    filter_candidate_period_range,
    plot_continuous_eigenvalue_recovery,
    plot_quality_factor_recovery,
)


# %% ------------------------------------------------------
# USER SETTINGS
# ------------------------------------------------------

# None discovers every ``mode_<number>`` group in R_splines_arbitrary.h5.
# Replace with, for example, [1, 6, 13] for a quick subset run.
mode_numbers = [1]

DMD_algorithm = "exact"

# Optional time-delay embedding applied independently of the DMD algorithm.
hankel_embedding_flag = False
hankel_d = 3
hankel_reconstruction_method = "first"

svd_rank = 2

# Compare the pre-truncation singular spectra across the isolated runs.
singular_value_plot_flag = True

up_lim_yr_plot = 100

filter_flag = False

# A finite candidate-period window can still be requested.  It is disabled by
# default because the purpose of this script is to test every synthetic wave;
# the available true periods extend well beyond the copied pipeline's former
# 10-year upper limit.  This filter never changes the signal supplied to DMD.
period_limit_flag = False
period_lower_bound = 1.01
period_upper_bound = 10.0


# ---------------------------------------------------------
# PERTURBATION ENSEMBLE
# ---------------------------------------------------------

ensemble_flag = True
# One retains the copied tester's quick-run setting.  Use at least two (and
# preferably enough for stable requested quantiles) to obtain non-zero-width
# uncertainty candlesticks.
n_realisations = 1

noise_temporal_model = "constant"
noise_tau_years = 6

# Perturbation_Generate uses this seed for every isolated mode.  Consequently,
# realisation i is paired across modes, which gives reproducible comparisons.
noise_seed = 42


# ---------------------------------------------------------
# TIME / DEGREE SETTINGS
# ---------------------------------------------------------

# True = use only the high-quality part of the record.
high_q_flag = True
n_skip = 1

# Maximum spherical-harmonic degree used.
Nmax = 8


# ---------------------------------------------------------
# AGGREGATE-PLOT SETTINGS
# ---------------------------------------------------------

# Quantiles used for the thin ensemble whiskers and thick interquantile bars.
ensemble_whisker_quantiles = (0.05, 0.95)
ensemble_box_quantiles = (0.25, 0.75)

# Retain the copied pipeline's growth-rate and quality-factor diagnostics in
# addition to the isolated-experiment summary figures.
eigenvalue_plot_flag = True
quality_factor_plot_flag = True


def _validate_settings():
    """Validate settings that otherwise fail late inside the mode sweep."""

    if DMD_algorithm not in ("exact", "fbdmd", "opdmd"):
        raise ValueError(
            "DMD_algorithm must be one of 'exact', 'fbdmd', or 'opdmd'. "
            f"Received {DMD_algorithm!r}."
        )

    if not isinstance(n_skip, (int, np.integer)) or n_skip < 1:
        raise ValueError("n_skip must be a positive integer.")

    if not isinstance(Nmax, (int, np.integer)) or not 1 <= Nmax <= 20:
        raise ValueError("Nmax must be an integer from 1 to 20.")

    if ensemble_flag and (
        not isinstance(n_realisations, (int, np.integer))
        or n_realisations < 1
    ):
        raise ValueError(
            "n_realisations must be a positive integer when the "
            "uncertainty ensemble is enabled."
        )

    if hankel_embedding_flag and (
        not isinstance(hankel_d, (int, np.integer)) or hankel_d < 1
    ):
        raise ValueError(
            "hankel_d must be a positive integer when Hankel embedding "
            "is enabled."
        )

    if period_limit_flag and not (
        np.isfinite(period_lower_bound)
        and np.isfinite(period_upper_bound)
        and 0.0 < period_lower_bound < period_upper_bound
    ):
        raise ValueError(
            "With period_limit_flag=True, bounds must be finite and "
            "satisfy 0 < period_lower_bound < period_upper_bound."
        )

    quantiles = (
        *ensemble_whisker_quantiles,
        *ensemble_box_quantiles,
    )
    if not all(0.0 <= quantile <= 1.0 for quantile in quantiles):
        raise ValueError("All ensemble quantiles must lie in [0, 1].")

    if not (
        ensemble_whisker_quantiles[0]
        <= ensemble_box_quantiles[0]
        <= 0.5
        <= ensemble_box_quantiles[1]
        <= ensemble_whisker_quantiles[1]
    ):
        raise ValueError(
            "Ensemble quantiles must enclose the median in the order "
            "whisker low <= box low <= 0.5 <= box high <= whisker high."
        )


def _discover_mode_numbers(h5_file):
    """Return numerically sorted synthetic mode identifiers."""

    discovered = []
    for group_name in h5_file.keys():
        if not group_name.startswith("mode_"):
            continue
        suffix = group_name.removeprefix("mode_")
        if suffix.isdigit():
            discovered.append(int(suffix))

    if not discovered:
        raise ValueError("No mode_<number> groups were found in the HDF5 file.")

    return [str(number) for number in sorted(discovered)]


def _resolve_mode_numbers(h5_file):
    """Resolve and validate either the full suite or a requested subset."""

    available = set(_discover_mode_numbers(h5_file))

    if mode_numbers is None:
        selected = sorted(available, key=int)
    else:
        selected = [str(number) for number in mode_numbers]

    if len(selected) != len(set(selected)):
        raise ValueError("mode_numbers contains duplicate mode identifiers.")

    if not selected:
        raise ValueError("mode_numbers must select at least one mode.")

    missing = [
        mode_number
        for mode_number in selected
        if mode_number not in available
    ]
    if missing:
        raise KeyError(
            "Requested synthetic modes are absent from "
            f"R_splines_arbitrary.h5: {missing}"
        )

    return selected


def _build_base_dmd():
    """Construct the requested estimator without changing its conventions."""

    if DMD_algorithm == "exact":
        return build_exact_dmd(svd_rank=svd_rank)
    if DMD_algorithm == "fbdmd":
        return build_fbdmd(svd_rank=svd_rank)
    if DMD_algorithm == "opdmd":
        return build_bopdmd(svd_rank=svd_rank)

    # _validate_settings normally makes this unreachable.
    raise ValueError(f"Unknown DMD algorithm: {DMD_algorithm!r}")


def _prepare_sv_input(gnm_input, A_r_current):
    """Apply the copied temporal, degree, and grid-input preparation."""

    if high_q_flag:
        gnm_input_windowed = gnm_input[good_record_slice, :]
    else:
        gnm_input_windowed = gnm_input

    if filter_flag:
        gnm_input_windowed, _, _ = Long_Period_Taper_Filter(
            gnm_input_windowed,
            dt=dt_years,
            pass_period=10.0,
            stop_period=20.0,
            axis=0,
        )

    gnm_input_band = Truncate_Gauss_Coeffs(
        gnm_input_windowed,
        tmax=Nmax,
    )

    sv_input_all_steps = A_r_current @ gnm_input_band.T
    return sv_input_all_steps[:, ::n_skip]


def _singular_spectrum(sv_input):
    """Return the same pre-rank-selection SVD diagnostics as the pipeline."""

    if hankel_embedding_flag:
        svd_input = pseudo_hankel_matrix(sv_input, d=hankel_d)
    else:
        svd_input = sv_input

    if DMD_algorithm in ("exact", "fbdmd"):
        svd_input = svd_input[:, :-1]

    magnitudes = np.abs(
        np.linalg.svd(
            svd_input,
            compute_uv=False,
        )
    )
    squared_magnitudes = np.square(magnitudes)
    total = np.sum(squared_magnitudes)

    if not np.isfinite(total) or total <= 0.0:
        raise ValueError(
            "Cannot calculate a singular spectrum because its total "
            "squared magnitude is not finite and positive."
        )

    return {
        "magnitude": magnitudes,
        "cumulative_variance": np.cumsum(squared_magnitudes) / total,
    }


def _fit_candidates(sv_input):
    """Fit one DMD model and return its retained physical candidates."""

    n_physical, n_snapshots = sv_input.shape

    if hankel_embedding_flag and hankel_d > n_snapshots:
        raise ValueError(
            f"hankel_d={hankel_d} exceeds the {n_snapshots} available "
            "snapshots."
        )

    base_dmd = _build_base_dmd()
    dmd = apply_hankel_embedding(
        base_dmd,
        enabled=hankel_embedding_flag,
        d=hankel_d,
        reconstruction_method=hankel_reconstruction_method,
    )
    embedding_d = hankel_d if hankel_embedding_flag else None
    dt_snapshot = n_skip * dt_years

    if DMD_algorithm == "opdmd":
        n_fit_snapshots = (
            n_snapshots - hankel_d + 1
            if hankel_embedding_flag
            else n_snapshots
        )
        fit_times = np.arange(n_fit_snapshots) * dt_snapshot
        dmd.fit(sv_input, fit_times)
        candidates = extract_optimized_dmd_candidates(
            dmd=dmd,
            n_physical=n_physical,
            embedding_d=embedding_d,
        )
    else:
        dmd.fit(sv_input)
        candidates = extract_standard_dmd_candidates(
            dmd=dmd,
            dt_snapshot=dt_snapshot,
            n_physical=n_physical,
            embedding_d=embedding_d,
        )

    if period_limit_flag:
        candidates = filter_candidate_period_range(
            *candidates,
            lower_period=period_lower_bound,
            upper_period=period_upper_bound,
        )

    return candidates


def _empty_ensemble_store():
    return {
        "similarity": [],
        "eigenvalue": [],
        "recovered_period": [],
    }


def _append_match(store, match):
    """Append a valid match without silently inventing failed values."""

    if match is None:
        return

    store["similarity"].append(match["similarity"])
    store["eigenvalue"].append(match["eigenvalue"])
    store["recovered_period"].append(match["recovered_period"])


def run_isolated_experiment():
    """
    Fit every requested synthetic wave independently.

    Returns
    -------
    DMD_recovery : dict
        One truth record, ideal match, resolved match, and uncertainty-match
        ensemble per injected wave.  Empty match dictionaries/lists explicitly
        represent recovery failures.
    singular_value_diagnostics : dict
        Pre-truncation spectra grouped by input type and mode.
    """

    _validate_settings()

    file_path = Path(FELIX_DIR) / "R_splines_arbitrary.h5"
    A_r_current = Truncate_Gauss_Coeffs(
        A_20_dict["r"],
        tmax=Nmax,
    )

    DMD_recovery = {}
    singular_value_diagnostics = {
        "ideal": {},
        "resolved": {},
        "perturbed": {},
        "noise_only": {},
    }

    with h5py.File(file_path, "r") as h5_file:
        selected_mode_numbers = _resolve_mode_numbers(h5_file)

        progress = tqdm(
            selected_mode_numbers,
            desc=f"Isolated modes, Nmax={Nmax}",
        )

        for mode_number in progress:
            mode_data = Component_Load_SV(mode_number)
            eigenvalue = mode_data["eigenvalue"]
            true_period = 2.0 * np.pi / np.abs(eigenvalue.imag)

            try:
                amp_scaler = mode_amp_scalings[mode_number]
            except KeyError as exc:
                raise KeyError(
                    f"No amplitude scaling found for mode {mode_number}."
                ) from exc

            # Analytical ideal path: no P, R, or H_sv operator.
            gnm_phasor_degree_20 = Truncate_Gauss_Coeffs(
                mode_data["gnm"],
                tmax=20,
            )
            gnm_mode_ideal = G_Time_Series_Eval(
                amp_scaler * gnm_phasor_degree_20,
                eigenvalue,
            )

            dataset_name = f"mode_{mode_number}/without_decay"
            if dataset_name not in h5_file:
                raise KeyError(f"Missing HDF5 dataset: {dataset_name}")

            gnm_spline = np.asarray(h5_file[dataset_name][()])
            gnm_mode_resolved = H_sv @ (amp_scaler * gnm_spline)

            if gnm_mode_ideal.shape != gnm_mode_resolved.shape:
                raise ValueError(
                    f"Mode {mode_number} ideal and resolved records must "
                    "have matching (time, Gauss coefficient) shapes. "
                    f"Received {gnm_mode_ideal.shape} and "
                    f"{gnm_mode_resolved.shape}."
                )

            gnm_target_band = Truncate_Gauss_Coeffs(
                amp_scaler * mode_data["gnm"],
                tmax=Nmax,
            )
            target_phasor = A_r_current @ gnm_target_band.T

            if target_phasor.shape != (A_r_current.shape[0],):
                raise ValueError(
                    f"Mode {mode_number} target phasor has shape "
                    f"{target_phasor.shape}; expected "
                    f"({A_r_current.shape[0]},)."
                )

            DMD_recovery[mode_number] = {
                "true_period": float(true_period),
                "true_eigenvalue": eigenvalue,
                "DMD_ideal": {},
                "DMD": {},
                "DMD_noised": _empty_ensemble_store(),
                "n_failed_noised_matches": 0,
            }

            if ensemble_flag:
                noise_only = Perturbation_Generate(
                    gnm_mode_resolved,
                    n_realisations=n_realisations,
                    seed=noise_seed,
                    temporal_z=noise_temporal_model,
                    tau=noise_tau_years,
                    dt=dt_years,
                    just_noise=True,
                )
                noised_records = (
                    noise_only + gnm_mode_resolved[None, :, :]
                )
            else:
                noise_only = np.empty(
                    (0, *gnm_mode_resolved.shape),
                    dtype=gnm_mode_resolved.dtype,
                )
                noised_records = noise_only

            queue = [
                ("ideal", gnm_mode_ideal),
                ("resolved", gnm_mode_resolved),
            ]
            queue.extend(
                ("perturbed", record)
                for record in noised_records
            )

            for result_type, gnm_input in queue:
                sv_input = _prepare_sv_input(gnm_input, A_r_current)

                if sv_input.shape[0] != target_phasor.size:
                    raise ValueError(
                        f"Mode {mode_number} {result_type} DMD input and "
                        "target phasor occupy different physical spaces: "
                        f"{sv_input.shape[0]} and {target_phasor.size}."
                    )

                if singular_value_plot_flag:
                    spectrum = _singular_spectrum(sv_input)
                    if result_type == "perturbed":
                        singular_value_diagnostics[
                            result_type
                        ].setdefault(mode_number, []).append(spectrum)
                    else:
                        singular_value_diagnostics[
                            result_type
                        ][mode_number] = spectrum

                (
                    candidate_eigs,
                    candidate_modes,
                    candidate_periods,
                ) = _fit_candidates(sv_input)

                match = best_spatial_match(
                    candidate_modes=candidate_modes,
                    candidate_eigs=candidate_eigs,
                    candidate_periods=candidate_periods,
                    target_phasor=target_phasor,
                )

                if result_type == "ideal":
                    if match is not None:
                        DMD_recovery[mode_number]["DMD_ideal"] = {
                            **match,
                            "degree_max": Nmax,
                        }
                elif result_type == "resolved":
                    if match is not None:
                        DMD_recovery[mode_number]["DMD"] = {
                            **match,
                            "degree_max": Nmax,
                        }
                else:
                    _append_match(
                        DMD_recovery[mode_number]["DMD_noised"],
                        match,
                    )
                    if match is None:
                        DMD_recovery[
                            mode_number
                        ]["n_failed_noised_matches"] += 1

                del candidate_eigs
                del candidate_modes
                del candidate_periods

            if singular_value_plot_flag and ensemble_flag:
                for noise_record in noise_only:
                    noise_sv_input = _prepare_sv_input(
                        noise_record,
                        A_r_current,
                    )
                    spectrum = _singular_spectrum(noise_sv_input)
                    singular_value_diagnostics[
                        "noise_only"
                    ].setdefault(mode_number, []).append(spectrum)

    return DMD_recovery, singular_value_diagnostics


def _sorted_recovery_items(DMD_recovery):
    return sorted(
        DMD_recovery.items(),
        key=lambda item: item[1]["true_period"],
    )


def _plot_ensemble_candlestick(
    ax,
    x,
    samples,
    colour,
    marker,
    label=None,
    zorder=5,
):
    """Plot median, interquantile bar, and outer quantile whiskers."""

    samples = np.asarray(samples, dtype=float)
    samples = samples[np.isfinite(samples)]
    if samples.size == 0:
        return False

    whisker_low, whisker_high = np.quantile(
        samples,
        ensemble_whisker_quantiles,
    )
    box_low, box_high = np.quantile(
        samples,
        ensemble_box_quantiles,
    )
    median = np.median(samples)

    ax.vlines(
        x,
        whisker_low,
        whisker_high,
        color=colour,
        linewidth=1.0,
        alpha=0.8,
        zorder=zorder,
    )
    ax.vlines(
        x,
        box_low,
        box_high,
        color=colour,
        linewidth=4.0,
        alpha=0.9,
        zorder=zorder + 1,
    )
    ax.scatter(
        x,
        median,
        marker=marker,
        s=28,
        color=colour,
        edgecolor="black",
        linewidth=0.35,
        label=label,
        zorder=zorder + 2,
    )
    return True


def plot_true_vs_recovered_period(DMD_recovery):
    """Plot the primary isolated-wave temporal-recovery diagnostic."""

    fig, ax = plt.subplots(figsize=(8.5, 6.5))
    items = _sorted_recovery_items(DMD_recovery)
    true_periods = np.asarray(
        [result["true_period"] for _, result in items]
    )
    plotted_periods = list(true_periods)
    ideal_label_available = True
    resolved_label_available = True
    ensemble_label_available = True

    for _, result in items:
        true_period = result["true_period"]

        ideal = result.get("DMD_ideal", {})
        if "recovered_period" in ideal:
            plotted_periods.append(ideal["recovered_period"])
            ax.scatter(
                true_period,
                ideal["recovered_period"],
                marker="+",
                s=55,
                color="tab:blue",
                linewidth=1.5,
                label="Ideal DMD" if ideal_label_available else None,
                zorder=7,
            )
            ideal_label_available = False

        resolved = result.get("DMD", {})
        if "recovered_period" in resolved:
            plotted_periods.append(resolved["recovered_period"])
            ax.scatter(
                true_period,
                resolved["recovered_period"],
                marker="x",
                s=42,
                color="tab:orange",
                linewidth=1.4,
                label="Resolved DMD" if resolved_label_available else None,
                zorder=7,
            )
            resolved_label_available = False

        ensemble_periods = result["DMD_noised"]["recovered_period"]
        plotted_periods.extend(ensemble_periods)
        ensemble_plotted = _plot_ensemble_candlestick(
            ax,
            true_period,
            ensemble_periods,
            colour="tab:green",
            marker="o",
            label="Uncertainty ensemble"
            if ensemble_label_available
            else None,
        )
        if ensemble_plotted:
            ensemble_label_available = False

    plotted_periods = np.asarray(plotted_periods, dtype=float)
    finite_positive = plotted_periods[
        np.isfinite(plotted_periods) & (plotted_periods > 0.0)
    ]
    lower = np.min(finite_positive) / 1.15
    upper = np.max(finite_positive) * 1.15
    ax.plot(
        [lower, upper],
        [lower, upper],
        color="black",
        linestyle="--",
        linewidth=1.1,
        label="Perfect period recovery",
        zorder=1,
    )
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(lower, upper)
    ax.set_ylim(lower, upper)
    ax.set_xlabel("True input period (years)")
    ax.set_ylabel("Recovered period (years)")
    ax.set_title(
        "Isolated synthetic-wave period recovery\n"
        f"{DMD_algorithm}; Nmax={Nmax}; rank={svd_rank}"
    )
    ax.grid(alpha=0.25, which="both")
    ax.legend(loc="best")
    fig.tight_layout()
    return fig, ax


def plot_similarity_vs_true_period(DMD_recovery):
    """Plot spatial recoverability against the injected wave period."""

    fig, ax = plt.subplots(figsize=(8.5, 6.0))
    items = _sorted_recovery_items(DMD_recovery)
    ideal_label_available = True
    resolved_label_available = True
    ensemble_label_available = True

    for _, result in items:
        true_period = result["true_period"]

        ideal = result.get("DMD_ideal", {})
        if "similarity" in ideal:
            ax.scatter(
                true_period,
                ideal["similarity"],
                marker="+",
                s=55,
                color="tab:blue",
                linewidth=1.5,
                label="Ideal DMD" if ideal_label_available else None,
                zorder=7,
            )
            ideal_label_available = False

        resolved = result.get("DMD", {})
        if "similarity" in resolved:
            ax.scatter(
                true_period,
                resolved["similarity"],
                marker="x",
                s=42,
                color="tab:orange",
                linewidth=1.4,
                label="Resolved DMD" if resolved_label_available else None,
                zorder=7,
            )
            resolved_label_available = False

        ensemble_plotted = _plot_ensemble_candlestick(
            ax,
            true_period,
            result["DMD_noised"]["similarity"],
            colour="tab:green",
            marker="o",
            label="Uncertainty ensemble"
            if ensemble_label_available
            else None,
        )
        if ensemble_plotted:
            ensemble_label_available = False

    ax.axhline(
        1.0,
        color="black",
        linestyle="--",
        linewidth=1.0,
        label="Perfect spatial match",
        zorder=1,
    )
    ax.set_xscale("log")
    ax.set_ylim(0.0, 1.02)
    ax.set_xlabel("True input period (years)")
    ax.set_ylabel("Best recovered spatial similarity")
    ax.set_title(
        "Isolated synthetic-wave spatial recovery\n"
        f"{DMD_algorithm}; Nmax={Nmax}; rank={svd_rank}"
    )
    ax.grid(alpha=0.25, which="both")
    ax.legend(loc="best")
    fig.tight_layout()
    return fig, ax


def _stack_spectrum_category(category_store, value_key):
    """Flatten mode/realisations into one equal-length spectrum matrix."""

    curves = []
    for stored in category_store.values():
        spectra = stored if isinstance(stored, list) else [stored]
        curves.extend(
            np.asarray(spectrum[value_key], dtype=float)
            for spectrum in spectra
        )

    if not curves:
        return None

    lengths = {curve.size for curve in curves}
    if len(lengths) != 1:
        raise ValueError(
            "All isolated singular-spectrum curves must have equal length. "
            f"Received lengths {sorted(lengths)}."
        )

    return np.stack(curves, axis=0)


def _plot_spectrum_summary(
    singular_value_diagnostics,
    value_key,
    ylabel,
    title,
    logarithmic_y=False,
):
    """Summarise spectra across modes and realisations by median and band."""

    fig, ax = plt.subplots(figsize=(8.5, 6.0))
    styles = {
        "ideal": ("tab:blue", "Ideal"),
        "resolved": ("tab:orange", "Resolved"),
        "perturbed": ("tab:green", "Perturbed"),
        "noise_only": ("tab:purple", "Perturbation only"),
    }

    for category, (colour, label) in styles.items():
        curves = _stack_spectrum_category(
            singular_value_diagnostics[category],
            value_key,
        )
        if curves is None:
            continue

        indices = np.arange(1, curves.shape[1] + 1)
        median = np.median(curves, axis=0)
        lower, upper = np.quantile(curves, [0.1, 0.9], axis=0)
        ax.fill_between(
            indices,
            lower,
            upper,
            color=colour,
            alpha=0.12,
            linewidth=0.0,
        )
        ax.plot(
            indices,
            median,
            color=colour,
            linewidth=1.8,
            label=f"{label} median (10–90% across tests)",
        )

    if logarithmic_y:
        ax.set_yscale("log")
    else:
        ax.set_ylim(0.0, 1.0)

    ax.set_xlabel("Singular value index, k")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(alpha=0.25, which="both")
    ax.legend(loc="best")
    fig.tight_layout()
    return fig, ax


def plot_singular_value_summaries(singular_value_diagnostics):
    """Return aggregate cumulative-variance and magnitude figures."""

    embedding_label = (
        f"Hankel d={hankel_d}"
        if hankel_embedding_flag
        else "no embedding"
    )

    cumulative = _plot_spectrum_summary(
        singular_value_diagnostics,
        value_key="cumulative_variance",
        ylabel="Cumulative variance explained (fraction)",
        title=(
            "Pre-truncation cumulative explained variance\n"
            f"isolated tests; {DMD_algorithm}; {embedding_label}; Nmax={Nmax}"
        ),
    )
    magnitude = _plot_spectrum_summary(
        singular_value_diagnostics,
        value_key="magnitude",
        ylabel="Singular value magnitude",
        title=(
            "Pre-truncation singular-value magnitude\n"
            f"isolated tests; {DMD_algorithm}; {embedding_label}; Nmax={Nmax}"
        ),
        logarithmic_y=True,
    )
    return cumulative, magnitude


def print_recovery_summary(DMD_recovery):
    """Report failures explicitly rather than dropping them from figures."""

    n_modes = len(DMD_recovery)
    missing_ideal = [
        mode
        for mode, result in DMD_recovery.items()
        if not result["DMD_ideal"]
    ]
    missing_resolved = [
        mode
        for mode, result in DMD_recovery.items()
        if not result["DMD"]
    ]
    failed_ensemble = sum(
        result["n_failed_noised_matches"]
        for result in DMD_recovery.values()
    )
    requested_ensemble = (
        n_modes * n_realisations
        if ensemble_flag
        else 0
    )

    print("\nISOLATED SYNTHETIC RECOVERY SUMMARY")
    print(f"Modes tested: {n_modes}")
    print(
        "Ideal matches: "
        f"{n_modes - len(missing_ideal)}/{n_modes}; "
        f"failed modes: {missing_ideal or 'none'}"
    )
    print(
        "Resolved matches: "
        f"{n_modes - len(missing_resolved)}/{n_modes}; "
        f"failed modes: {missing_resolved or 'none'}"
    )
    if ensemble_flag:
        print(
            "Perturbed matches: "
            f"{requested_ensemble - failed_ensemble}/"
            f"{requested_ensemble}; failures: {failed_ensemble}"
        )


# %% ------------------------------------------------------
# RUN EXPERIMENT AND PLOT
# ------------------------------------------------------

if __name__ == "__main__":
    DMD_recovery, singular_value_diagnostics = run_isolated_experiment()

    fig_period, ax_period = plot_true_vs_recovered_period(DMD_recovery)
    fig_similarity, ax_similarity = plot_similarity_vs_true_period(
        DMD_recovery
    )

    if eigenvalue_plot_flag:
        fig_eigenvalue, ax_eigenvalue = (
            plot_continuous_eigenvalue_recovery(
                DMD_recovery,
                lower_lim_yr=(
                    period_lower_bound
                    if period_limit_flag
                    else 1.0
                ),
                upper_lim_yr=(
                    period_upper_bound
                    if period_limit_flag
                    else up_lim_yr_plot
                ),
                annotate_modes=False,
            )
        )

    if quality_factor_plot_flag:
        fig_quality, ax_quality = plot_quality_factor_recovery(
            DMD_recovery,
            lower_lim_yr=(
                period_lower_bound
                if period_limit_flag
                else 1.0
            ),
            upper_lim_yr=(
                period_upper_bound
                if period_limit_flag
                else up_lim_yr_plot
            ),
            annotate_modes=False,
        )

    if singular_value_plot_flag:
        (
            (fig_cumulative_variance, ax_cumulative_variance),
            (fig_singular_value_magnitude, ax_singular_value_magnitude),
        ) = plot_singular_value_summaries(
            singular_value_diagnostics
        )

    print_recovery_summary(DMD_recovery)
    plt.show()
