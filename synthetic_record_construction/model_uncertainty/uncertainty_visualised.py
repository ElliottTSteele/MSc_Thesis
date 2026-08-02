"""
Visualise CHAOS-8.6 secular variation and its covariance uncertainty.

The covariance perturbation, CHAOS-8.6 record, and resolved synthetic record
are all constructed on the complete synSetup time axis.  Spectra are computed
only after windowing to the good-record interval.  The perturbation is always
treated as a noise-only Gauss-coefficient record; it is not added to CHAOS.
"""

# %% FILE SYSTEM AND DEPENDENCY SETUP

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

import cartopy.crs as ccrs
import chaosmagpy as cp
import h5py
import matplotlib.pyplot as plt
import numpy as np
from cartopy.util import add_cyclic_point
from matplotlib.colors import BoundaryNorm, ListedColormap, LogNorm

from src.msc_thesis.paths import CHAOS_DIR, FELIX_DIR, FIG_DIR, text_width
from src.msc_thesis.synDMD import Cov_full, Perturbation_Generate
from src.msc_thesis.synSetup import (
    colatitude,
    dt_years,
    good_record_slice,
    longitude,
    mode_amp_scalings,
    r_cmb,
    r_earth,
    times_absolute,
    times_mjd2000,
)
from src.msc_thesis.synUtils import (
    H_sv,
    Lowes_Degree_PSD_All_Degrees,
    Truncate_Gauss_Coeffs,
)


# %% USER SETTINGS

# CHAOS and the covariance product contain complete degree-20 coefficient
# vectors.  Retaining that common degree is required during perturbation.
coefficient_nmax = 20
expected_full_epochs = 148

# Independent latent draws are made at every time step, as in the tester
# scripts.  The seed selects the single displayed noise-only realisation.
noise_seed = 42
noise_temporal_model = "independent"

# Resolved, noise-free synthetic modes included in the summed record.
# These are physical mode numbers, not zero-based indices.
mode_numbers = [
    1, 2, 3, 4, 6, 11, 15, 16, 18, 20, 21, 25, 29, 30, 32, 33,
    34, 36, 37, 38, 39, 41, 43, 45, 48, 53, 54, 57, 59, 60, 61, 62,
]

# Inclusive degree-frequency limits for the three-panel PSD figure.
psd_degree_min = 1
psd_degree_max = 15
psd_frequency_min = 0.0
psd_frequency_max = 0.333
psd_boundary_degree = 10

# Degree truncations used for the two rows of the snapshot figure.
snapshot_low_degree = 10
snapshot_high_degree = 15

# The nearest available good-record epoch is used for all three snapshots.
demo_year = 2020
demo_month = 1

figure_font_size = 11
show_figures_interactively = True

output_directory = Path(FIG_DIR) / "model_uncertainty"
psd_figure_path = output_directory / "chaos_covariance_noise_psd.png"
snapshot_figure_path = (
    output_directory / "chaos_covariance_modes_sv_snapshot.png"
)

chaos_file = Path(CHAOS_DIR) / "CHAOS-8.6.mat"
resolved_wave_file = Path(FELIX_DIR) / "R_splines_arbitrary.h5"


# %% SHARED HELPERS

def save_and_finish_figure(fig, output_path):
    """Save a figure and either display it or close it."""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300)
    if show_figures_interactively:
        plt.show()
    else:
        plt.close(fig)


def load_chaos_sv(model_path, evaluation_times, degree_max):
    """Evaluate CHAOS-8.6 SV Gauss coefficients at supplied epochs."""

    if not model_path.exists():
        raise FileNotFoundError(f"Could not find {model_path}")

    model = cp.load_CHAOS_matfile(str(model_path))
    coefficients = model.synth_coeffs_tdep(
        evaluation_times,
        nmax=degree_max,
        deriv=1,
        extrapolate="off",
    )
    return Truncate_Gauss_Coeffs(
        np.asarray(coefficients, dtype=float),
        tmax=degree_max,
    )


def load_resolved_mode_combination(
    wave_path,
    selected_mode_numbers,
    n_time_points,
):
    """Sum established-amplitude resolved SV records for selected modes."""

    if not wave_path.exists():
        raise FileNotFoundError(f"Could not find {wave_path}")
    if not selected_mode_numbers:
        raise ValueError("mode_numbers must contain at least one mode")
    if len(set(selected_mode_numbers)) != len(selected_mode_numbers):
        raise ValueError("mode_numbers must not contain duplicates")

    resolved_total = None

    with h5py.File(wave_path, "r") as h5_file:
        for mode_number in selected_mode_numbers:
            mode_key = str(mode_number)
            dataset_name = f"mode_{mode_key}/without_decay"

            if dataset_name not in h5_file:
                raise KeyError(f"Missing HDF5 dataset: {dataset_name}")
            if mode_key not in mode_amp_scalings:
                raise KeyError(
                    f"No established amplitude scaling for mode {mode_key}"
                )

            amplitude_scaler = float(mode_amp_scalings[mode_key])
            spline_coefficients = np.asarray(
                h5_file[dataset_name][()],
                dtype=float,
            )
            resolved_mode = H_sv @ (
                amplitude_scaler * spline_coefficients
            )

            if resolved_mode.shape[0] != n_time_points:
                raise ValueError(
                    f"Resolved mode {mode_key} has "
                    f"{resolved_mode.shape[0]} epochs; "
                    f"expected {n_time_points}"
                )
            if resolved_total is None:
                resolved_total = np.zeros_like(
                    resolved_mode,
                    dtype=float,
                )
            if resolved_mode.shape != resolved_total.shape:
                raise ValueError(
                    f"Resolved mode {mode_key} has shape "
                    f"{resolved_mode.shape}; expected "
                    f"{resolved_total.shape}"
                )

            resolved_total += resolved_mode

    return resolved_total


def degree_frequency_spectrum(gnm):
    """Return the Lowes degree-frequency SV spectrum at the CMB."""

    return Lowes_Degree_PSD_All_Degrees(
        gnm,
        a=r_earth,
        r=r_cmb,
        f_sample=1.0 / dt_years,
    )


def discrete_log_ratio_colours(log_ratio):
    """Use the wave-scaling factor-2/factor-3/factor-10 error colours."""

    factor_2 = np.log10(2)
    factor_3 = np.log10(3)
    max_error = np.nanmax(np.abs(log_ratio))
    error_cap = max(1.01, max_error + 1e-6)
    bounds = [
        -error_cap,
        -1,
        -factor_3,
        -factor_2,
        factor_2,
        factor_3,
        1,
        error_cap,
    ]
    colours = [
        "#08306B",
        "#2171B5",
        "#9ECAE1",
        "#FFFFFF",
        "#FCAE91",
        "#FB6A4A",
        "#CB181D",
    ]
    cmap = ListedColormap(colours)
    cmap.set_bad("lightgrey")
    norm = BoundaryNorm(bounds, cmap.N, clip=True)
    ticks = [-1, -factor_3, -factor_2, factor_2, factor_3, 1]
    ticklabels = [
        r"$-1$",
        r"$-\log_{10}(3)$",
        r"$-\log_{10}(2)$",
        r"$\log_{10}(2)$",
        r"$\log_{10}(3)$",
        r"$1$",
    ]
    return cmap, norm, bounds, ticks, ticklabels


def plot_psd_comparison(
    chaos_spectrum,
    perturbation_spectrum,
    degrees,
    frequencies,
    output_path,
):
    """Plot CHAOS, perturbation-only, and signed log-ratio spectra."""

    display_degree_mask = (
        (degrees >= psd_degree_min)
        & (degrees <= psd_degree_max)
    )
    display_frequency_mask = (
        (frequencies >= psd_frequency_min)
        & (frequencies <= psd_frequency_max)
    )

    if not np.any(display_degree_mask):
        raise ValueError("The configured PSD degree range is empty")
    if not np.any(display_frequency_mask):
        raise ValueError("The configured PSD frequency range is empty")

    display_index = np.ix_(
        display_degree_mask,
        display_frequency_mask,
    )
    displayed_degrees = degrees[display_degree_mask]
    displayed_frequencies = frequencies[display_frequency_mask]
    chaos_display = chaos_spectrum[display_index]
    perturbation_display = perturbation_spectrum[display_index]

    positive_power = np.concatenate([
        chaos_display[
            np.isfinite(chaos_display) & (chaos_display > 0)
        ],
        perturbation_display[
            np.isfinite(perturbation_display)
            & (perturbation_display > 0)
        ],
    ])
    if positive_power.size == 0:
        raise ValueError("No positive PSD values are available to plot")

    power_norm = LogNorm(
        vmin=np.min(positive_power),
        vmax=np.max(positive_power),
    )
    power_cmap = plt.get_cmap("viridis").copy()
    power_cmap.set_bad("lightgrey")

    comparison_valid = (
        np.isfinite(chaos_display)
        & np.isfinite(perturbation_display)
        & (chaos_display > 0)
        & (perturbation_display >= 0)
    )
    if not np.any(comparison_valid):
        raise ValueError("No valid PSD cells are available for comparison")

    psd_epsilon = 1e-15 * np.nanmax(
        chaos_display[comparison_valid]
    )
    log_ratio = np.full_like(chaos_display, np.nan, dtype=float)
    log_ratio[comparison_valid] = np.log10(
        (
            perturbation_display[comparison_valid]
            + psd_epsilon
        )
        / (
            chaos_display[comparison_valid]
            + psd_epsilon
        )
    )
    (
        error_cmap,
        error_norm,
        error_bounds,
        error_ticks,
        error_ticklabels,
    ) = discrete_log_ratio_colours(log_ratio)

    if displayed_frequencies.size > 1:
        frequency_spacing = float(
            np.mean(np.diff(displayed_frequencies))
        )
    else:
        frequency_spacing = 1.0 / (
            dt_years * chaos_spectrum.shape[1]
        )
    extent = [
        displayed_frequencies[0] - frequency_spacing / 2,
        displayed_frequencies[-1] + frequency_spacing / 2,
        displayed_degrees[0] - 0.5,
        displayed_degrees[-1] + 0.5,
    ]

    with plt.rc_context({"font.size": figure_font_size}):
        fig, axes = plt.subplots(
            1,
            3,
            figsize=(1.35 * text_width, 0.52 * text_width),
            sharex=True,
            sharey=True,
            constrained_layout=True,
        )

        chaos_image = axes[0].imshow(
            np.ma.masked_invalid(chaos_display),
            origin="lower",
            aspect="auto",
            interpolation="nearest",
            extent=extent,
            cmap=power_cmap,
            norm=power_norm,
        )
        axes[1].imshow(
            np.ma.masked_invalid(perturbation_display),
            origin="lower",
            aspect="auto",
            interpolation="nearest",
            extent=extent,
            cmap=power_cmap,
            norm=power_norm,
        )
        error_image = axes[2].imshow(
            np.ma.masked_invalid(log_ratio),
            origin="lower",
            aspect="auto",
            interpolation="nearest",
            extent=extent,
            cmap=error_cmap,
            norm=error_norm,
        )

        titles = [
            "CHAOS-8.6",
            "Independent perturbation only",
            r"$\log_{10}(P_\mathrm{pert}/P_\mathrm{CHAOS})$",
        ]
        for ax, title in zip(axes, titles):
            ax.set_title(title)
            ax.set_xlim(
                psd_frequency_min,
                psd_frequency_max,
            )
            ax.set_ylim(
                displayed_degrees[0] - 0.5,
                displayed_degrees[-1] + 0.5,
            )
            ax.set_yticks(
                np.arange(
                    displayed_degrees[0],
                    displayed_degrees[-1] + 1,
                    2,
                )
            )
            ax.set_xlabel(r"Frequency / $\mathrm{yr}^{-1}$")
            ax.axhline(
                psd_boundary_degree + 0.5,
                color="black",
                linewidth=1.2,
            )

        axes[0].set_ylabel(r"Spherical harmonic degree $n$")

        power_colourbar = fig.colorbar(
            chaos_image,
            ax=axes[:2],
            location="bottom",
            fraction=0.10,
            pad=0.14,
        )
        power_colourbar.set_label(
            r"Lowes SV power spectrum at the CMB / "
            r"$(\mathrm{nT\,yr^{-1}})^2\,\mathrm{yr}$"
        )
        error_colourbar = fig.colorbar(
            error_image,
            ax=axes[2],
            boundaries=error_bounds,
            ticks=error_ticks,
        )
        error_colourbar.set_ticklabels(error_ticklabels)
        error_colourbar.set_label(
            r"$\log_{10}(P_\mathrm{pert}/P_\mathrm{CHAOS})$"
        )

        save_and_finish_figure(fig, output_path)

    return log_ratio


def radial_sv_grid(gnm_snapshot, degree_max):
    """Evaluate radial SV at the CMB on the shared one-degree grid."""

    radial_sv, _, _ = cp.model_utils.synth_values(
        gnm_snapshot,
        r_cmb,
        colatitude,
        longitude,
        nmax=degree_max,
        source="internal",
        grid=True,
    )
    return np.asarray(radial_sv)


def plot_snapshot_comparison(
    chaos_snapshot,
    perturbation_snapshot,
    resolved_modes_snapshot,
    snapshot_decimal_year,
    output_path,
):
    """Plot two degree truncations with one shared colour bar per row."""

    degree_limits = (
        snapshot_low_degree,
        snapshot_high_degree,
    )
    snapshot_coefficients = (
        ("CHAOS-8.6", chaos_snapshot),
        ("Independent perturbation only", perturbation_snapshot),
        ("Resolved mode combination", resolved_modes_snapshot),
    )

    grids_by_degree = []
    for degree_limit in degree_limits:
        row_grids = [
            radial_sv_grid(coefficients, degree_limit)
            for _, coefficients in snapshot_coefficients
        ]
        grids_by_degree.append(row_grids)

    with plt.rc_context({"font.size": figure_font_size}):
        fig, axes = plt.subplots(
            2,
            3,
            figsize=(1.40 * text_width, 0.75 * text_width),
            subplot_kw={"projection": ccrs.Mollweide()},
            constrained_layout=True,
        )

        for row, (degree_limit, row_grids) in enumerate(
            zip(degree_limits, grids_by_degree)
        ):
            colour_limit = max(
                np.nanmax(np.abs(field))
                for field in row_grids
            )
            if not np.isfinite(colour_limit) or colour_limit <= 0:
                raise ValueError(
                    "Could not determine radial-SV colour limits "
                    f"for n <= {degree_limit}"
                )

            row_image = None
            for column, ((model_title, _), field) in enumerate(
                zip(snapshot_coefficients, row_grids)
            ):
                cyclic_field, cyclic_longitude = add_cyclic_point(
                    field,
                    coord=longitude,
                )
                ax = axes[row, column]
                row_image = ax.pcolormesh(
                    cyclic_longitude,
                    90.0 - colatitude,
                    cyclic_field,
                    transform=ccrs.PlateCarree(),
                    cmap="seismic",
                    vmin=-colour_limit,
                    vmax=colour_limit,
                    shading="auto",
                )
                ax.coastlines(linewidth=0.6, color="black")
                ax.gridlines(
                    linewidth=0.4,
                    linestyle=":",
                    color="grey",
                )
                ax.set_title(
                    f"{model_title}\n"
                    rf"$n \leq {degree_limit}$"
                )

            row_colourbar = fig.colorbar(
                row_image,
                ax=axes[row, :],
                location="right",
                shrink=0.85,
                pad=0.02,
                extend="both",
            )
            row_colourbar.set_label(
                r"Radial SV at the CMB / $\mathrm{nT\,yr^{-1}}$"
            )

        fig.suptitle(
            "Radial SV at the nearest good-record epoch "
            f"{snapshot_decimal_year:.1f}"
        )
        save_and_finish_figure(fig, output_path)


# %% VALIDATE SETTINGS AND SHARED TIME AXIS

full_times_mjd2000 = np.asarray(times_mjd2000, dtype=float)
full_decimal_years = np.asarray(times_absolute, dtype=float)

if full_times_mjd2000.size != expected_full_epochs:
    raise ValueError(
        f"Expected {expected_full_epochs} full epochs, found "
        f"{full_times_mjd2000.size}"
    )
if full_decimal_years.shape != full_times_mjd2000.shape:
    raise ValueError(
        "times_absolute and times_mjd2000 have different shapes"
    )
if not (
    1
    <= psd_degree_min
    <= psd_boundary_degree
    < psd_degree_max
    <= coefficient_nmax
):
    raise ValueError(
        "PSD degrees must satisfy "
        "1 <= minimum <= boundary < maximum <= coefficient_nmax"
    )
if not (
    0.0
    <= psd_frequency_min
    < psd_frequency_max
    <= 0.5 / dt_years
):
    raise ValueError(
        "PSD frequencies must be increasing and not exceed Nyquist"
    )
if not (
    1
    <= snapshot_low_degree
    < snapshot_high_degree
    <= coefficient_nmax
):
    raise ValueError(
        "Snapshot degrees must satisfy "
        "1 <= low < high <= coefficient_nmax"
    )
if not 1 <= int(demo_month) <= 12:
    raise ValueError("demo_month must lie between 1 and 12")

expected_coefficient_count = (coefficient_nmax + 1) ** 2 - 1
expected_full_shape = (
    expected_full_epochs,
    expected_coefficient_count,
)
if Cov_full.shape != (
    expected_full_epochs,
    expected_coefficient_count,
    expected_coefficient_count,
):
    raise ValueError(
        "Covariance shape does not match the configured full record: "
        f"{Cov_full.shape}"
    )

good_times_mjd2000 = full_times_mjd2000[good_record_slice]
good_decimal_years = full_decimal_years[good_record_slice]
if good_times_mjd2000.size < 3:
    raise ValueError("Too few good-record epochs were selected")

print(
    f"Full record: {full_times_mjd2000.size} epochs from "
    f"{full_decimal_years[0]:.1f} to "
    f"{full_decimal_years[-1]:.1f}"
)
print(
    f"Good record: {good_times_mjd2000.size} epochs from "
    f"{good_decimal_years[0]:.1f} to "
    f"{good_decimal_years[-1]:.1f}"
)


# %% BUILD FULL CHAOS, PERTURBATION, AND RESOLVED SYNTHETIC RECORDS

chaos_sv_full = load_chaos_sv(
    chaos_file,
    full_times_mjd2000,
    coefficient_nmax,
)
if chaos_sv_full.shape != expected_full_shape:
    raise ValueError(
        f"Unexpected CHAOS shape {chaos_sv_full.shape}; "
        f"expected {expected_full_shape}"
    )
if not np.all(np.isfinite(chaos_sv_full)):
    raise ValueError("CHAOS SV coefficients contain non-finite values")

perturbation_sv_full = Perturbation_Generate(
    chaos_sv_full,
    n_realisations=1,
    seed=noise_seed,
    temporal_z=noise_temporal_model,
    covariance_used=Cov_full,
    just_noise=True,
)[0]
if perturbation_sv_full.shape != expected_full_shape:
    raise ValueError(
        "Unexpected perturbation shape "
        f"{perturbation_sv_full.shape}; expected {expected_full_shape}"
    )
if not np.all(np.isfinite(perturbation_sv_full)):
    raise ValueError(
        "Covariance perturbation coefficients contain non-finite values"
    )

resolved_modes_sv_full = load_resolved_mode_combination(
    resolved_wave_file,
    mode_numbers,
    expected_full_epochs,
)
if not np.all(np.isfinite(resolved_modes_sv_full)):
    raise ValueError(
        "Resolved mode combination contains non-finite values"
    )


# %% WINDOW TO THE GOOD RECORD AND COMPUTE THE TWO INDEPENDENT PSDs

chaos_sv = chaos_sv_full[good_record_slice]
perturbation_sv = perturbation_sv_full[good_record_slice]
resolved_modes_sv = resolved_modes_sv_full[good_record_slice]

chaos_spectrum, frequencies = degree_frequency_spectrum(chaos_sv)
perturbation_spectrum, perturbation_frequencies = (
    degree_frequency_spectrum(perturbation_sv)
)
if chaos_spectrum.shape != perturbation_spectrum.shape:
    raise ValueError(
        "CHAOS and perturbation PSD arrays have different shapes"
    )
if not np.allclose(frequencies, perturbation_frequencies):
    raise ValueError(
        "CHAOS and perturbation PSD frequency axes differ"
    )

degrees = np.arange(1, chaos_spectrum.shape[0] + 1)
psd_log_ratio = plot_psd_comparison(
    chaos_spectrum,
    perturbation_spectrum,
    degrees,
    frequencies,
    psd_figure_path,
)
print(
    "Displayed PSD log-ratio range:",
    f"{np.nanmin(psd_log_ratio):.3f} to "
    f"{np.nanmax(psd_log_ratio):.3f}",
)


# %% COMMON SNAPSHOT EPOCH AND TWO-ROW CMB MAP COMPARISON

demo_target_mjd2000 = float(
    cp.data_utils.mjd2000(
        int(demo_year),
        int(demo_month),
        1,
    )
)
if not (
    good_times_mjd2000[0]
    <= demo_target_mjd2000
    <= good_times_mjd2000[-1]
):
    raise ValueError(
        "The requested demonstration month lies outside the "
        "good-record interval"
    )

demo_good_index = int(
    np.argmin(
        np.abs(good_times_mjd2000 - demo_target_mjd2000)
    )
)
demo_decimal_year = float(good_decimal_years[demo_good_index])
print(
    f"Requested snapshot {int(demo_year):04d}-"
    f"{int(demo_month):02d}; using nearest good-record epoch "
    f"{demo_decimal_year:.1f}"
)

plot_snapshot_comparison(
    chaos_snapshot=chaos_sv[demo_good_index],
    perturbation_snapshot=perturbation_sv[demo_good_index],
    resolved_modes_snapshot=resolved_modes_sv[demo_good_index],
    snapshot_decimal_year=demo_decimal_year,
    output_path=snapshot_figure_path,
)

print(f"Saved PSD comparison to {psd_figure_path}")
print(f"Saved snapshot comparison to {snapshot_figure_path}")
