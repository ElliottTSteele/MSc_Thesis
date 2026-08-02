# %% CHAOS-8.6 GOOD-RECORD WAVE OPTIMISATION
from IPython import get_ipython

ipython = get_ipython()
if ipython is not None:
    ipython.run_line_magic("load_ext", "autoreload")
    ipython.run_line_magic("autoreload", "2")

import pickle
import sys
from pathlib import Path

import chaosmagpy as cp
import cartopy.crs as ccrs
import h5py
import matplotlib.pyplot as plt
import numpy as np
from cartopy.util import add_cyclic_point
from matplotlib.colors import BoundaryNorm, ListedColormap, LogNorm
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.msc_thesis.paths import *
from src.msc_thesis.synSetup import (
    colatitude,
    dt_years,
    good_record_slice,
    longitude,
    r_cmb,
    r_earth,
    times_absolute,
    times_mjd2000,
)
from src.msc_thesis.synUtils import (
    H_sv,
    Lowes_Degree_PSD_All_Degrees,
    Mean_Instantaneous_Total_Lowes_Power,
    Truncate_Gauss_Coeffs,
)
from src.msc_thesis.synConstruct import Component_Load


# %% SETTINGS
chaos_file = Path(CHAOS_DIR) / "CHAOS-8.6.mat"
wave_file = Path(FELIX_DIR) / "R_splines_arbitrary.h5"

r_choice = r_cmb
nmax = 20
mode_numbers = np.arange(1, 63)

# All domains include their lower and upper frequency limits. Set a lower
# frequency to 0.0 to retain the zero-frequency periodogram bin.
water_level_degree_min = 1
water_level_degree_max = 15
water_level_frequency_min = 0.0
water_level_frequency_max = 0.3333

fit_degree_min = 1
fit_degree_max = 10
fit_frequency_min = 0.0
fit_frequency_max = 0.333

# The final degree-frequency and marginal diagnostics use this domain.
diagnostic_degree_min = 1
diagnostic_degree_max = 15
diagnostic_frequency_min = 0.0
diagnostic_frequency_max = 0.3333

# The requested calendar month is mapped to the nearest available epoch in the
# good-record slice. Both CHAOS-8.6 and the wave combination use that epoch.
demo_year = 2020
demo_month = 1

figure_font_size = 11

# ------------------------------------------------------------------
# MODE-SELECTION SETTINGS
# ------------------------------------------------------------------
# Choose how the optimiser handles the number of selected modes:
#
#   "fixed" -> exactly n_selected_modes modes are retained.
#   "auto"  -> the optimiser may add/remove modes and determines the
#              final number itself by minimising the same RMS log-misfit.
#
mode_count_strategy = "auto"

# Used only when mode_count_strategy == "fixed".
n_selected_modes = 7

# Modes that MUST appear in the final solution.
# These are physical mode numbers, not zero-based array indices.
# Examples:
#   seeded_mode_numbers = []
#   seeded_mode_numbers = [13, 18]
seeded_mode_numbers = []

# Minimum period spacing between selected modes.
#
# With period_exclusion_fraction = 0.25, selecting a mode excludes any other
# mode whose period lies within +/-25% of either mode's period. The pairwise
# check is deliberately symmetric, so the result does not depend on which mode
# happened to be selected first.
period_exclusion_fraction = 0.0 #0.25

# Bounds used only for automatic mode-count selection.
# The minimum is automatically raised if necessary so that all seeded
# modes can be included.
auto_min_modes = 1
auto_max_modes = len(mode_numbers)

random_seed = 2
n_restarts = 3
annealing_steps = 1200
temperature_start = 0.05
temperature_end = 1e-4
max_swap_passes = 4

# In automatic mode, the final deterministic refinement repeatedly tries
# all single-mode additions, removals, and swaps until none improves the
# RMS log-misfit by more than this tolerance.
refinement_tolerance = 1e-6

degree_frequency_figure = (
    Path(FIG_DIR)
    / "final"
    / "chaos8_optimal_wave_degree_frequency.png"
)
degree_marginal_figure = (
    Path(FIG_DIR)
    / "final"
    / "chaos8_optimal_wave_degree_marginal.png"
)
frequency_marginal_figure = (
    Path(FIG_DIR)
    / "final"
    / "chaos8_optimal_wave_frequency_marginal.png"
)
snapshot_figure = (
    Path(FIG_DIR)
    / "final"
    / "chaos8_optimal_wave_radial_sv_snapshot.png"
)
scale_file = (
    Path(FELIX_DIR)
    / "mode_amplitude_scalings_chaos8_good_record_water_level.pkl"
)


def load_sv_gauss(model_path, times, nmax):
    """Evaluate CHAOS SV Gauss coefficients on supplied MJD2000 epochs."""
    if not model_path.exists():
        raise FileNotFoundError(f"Could not find {model_path}")
    model = cp.load_CHAOS_matfile(str(model_path))
    coefficients = model.synth_coeffs_tdep(
        times,
        nmax=nmax,
        deriv=1,
        extrapolate="off",
    )
    return Truncate_Gauss_Coeffs(
        np.asarray(coefficients),
        nmax,
    )


def degree_frequency_spectrum(gnm):
    """Lowes degree-frequency SV spectrum at the selected radius."""
    return Lowes_Degree_PSD_All_Degrees(
        gnm,
        a=r_earth,
        r=r_choice,
        f_sample=1.0 / dt_years,
    )


def rectangular_spectral_mask(
    degrees,
    frequencies,
    degree_min,
    degree_max,
    frequency_min,
    frequency_max,
):
    """Inclusive degree-frequency rectangle."""
    return (
        (degrees[:, None] >= degree_min)
        & (degrees[:, None] <= degree_max)
        & (frequencies[None, :] >= frequency_min)
        & (frequencies[None, :] <= frequency_max)
    )


def log_error_statistics(candidate, target, valid, epsilon):
    """Compute log-ratio field and summary statistics on a supplied mask."""
    comparison_valid = (
        valid
        & np.isfinite(candidate)
        & np.isfinite(target)
        & (candidate >= 0)
        & (target > 0)
    )
    log_error = np.full_like(target, np.nan, dtype=float)
    log_error[comparison_valid] = np.log10(
        (candidate[comparison_valid] + epsilon)
        / (target[comparison_valid] + epsilon)
    )
    values = log_error[comparison_valid]
    return {
        "log_error": log_error,
        "valid": comparison_valid,
        "rms": np.sqrt(np.mean(values**2)),
        "within_factor_2": np.mean(
            np.abs(values) <= np.log10(2)
        ),
        "within_factor_3": np.mean(
            np.abs(values) <= np.log10(3)
        ),
    }


def discrete_log_ratio_colours(log_ratio):
    """Return the discrete factor-2/factor-3/factor-10 colour mapping."""
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


def plot_two_domain_power_comparison(
    reference,
    candidate,
    log_ratio,
    display_valid,
    degrees,
    frequencies,
    reference_title,
    candidate_title,
    ratio_title,
    output_path,
    fitted_degree_max,
):
    """Plot fit-degree and diagnostic-degree PSD comparisons."""
    reference_masked = np.ma.masked_where(~display_valid, reference)
    candidate_masked = np.ma.masked_where(~display_valid, candidate)
    positive_power = np.concatenate([
        reference_masked.compressed(),
        candidate_masked.compressed(),
    ])
    power_norm = LogNorm(
        vmin=np.min(positive_power),
        vmax=np.max(positive_power),
    )
    power_cmap = plt.get_cmap("viridis").copy()
    power_cmap.set_bad("lightgrey")

    (
        error_cmap,
        error_norm,
        error_bounds,
        error_ticks,
        error_ticklabels,
    ) = discrete_log_ratio_colours(log_ratio)

    df = np.mean(np.diff(frequencies))
    fit_degree_mask = degrees <= fitted_degree_max
    if not np.any(fit_degree_mask):
        raise ValueError(
            "No displayed degrees lie inside the fit-degree domain"
        )

    row_specs = [
        (fit_degree_mask, rf"Fit domain: $n \leq {fitted_degree_max}$"),
        (
            np.ones_like(degrees, dtype=bool),
            rf"Diagnostic domain: $n \leq {degrees[-1]}$",
        ),
    ]

    with plt.rc_context({"font.size": figure_font_size}):
        fig, axes = plt.subplots(
            2,
            3,
            figsize=(text_width, 0.92 * text_width),
            sharex=True,
            constrained_layout=True,
        )

        for row, (row_degree_mask, row_label) in enumerate(
            row_specs
        ):
            row_degrees = degrees[row_degree_mask]
            row_extent = [
                frequencies[0] - df / 2,
                frequencies[-1] + df / 2,
                row_degrees[0] - 0.5,
                row_degrees[-1] + 0.5,
            ]

            reference_im = axes[row, 0].imshow(
                reference_masked[row_degree_mask],
                origin="lower",
                aspect="auto",
                interpolation="nearest",
                extent=row_extent,
                cmap=power_cmap,
                norm=power_norm,
            )
            axes[row, 1].imshow(
                candidate_masked[row_degree_mask],
                origin="lower",
                aspect="auto",
                interpolation="nearest",
                extent=row_extent,
                cmap=power_cmap,
                norm=power_norm,
            )
            error_im = axes[row, 2].imshow(
                np.ma.masked_where(
                    ~display_valid[row_degree_mask],
                    log_ratio[row_degree_mask],
                ),
                origin="lower",
                aspect="auto",
                interpolation="nearest",
                extent=row_extent,
                cmap=error_cmap,
                norm=error_norm,
            )

            for ax in axes[row]:
                ax.set_xlim(row_extent[0], row_extent[1])
                ax.set_ylim(row_extent[2], row_extent[3])
                ax.set_yticks(
                    np.arange(
                        2,
                        row_degrees[-1] + 1,
                        2,
                    )
                )

            axes[row, 0].set_ylabel(
                row_label
                + "\n"
                + r"Spherical harmonic degree $n$"
            )

        axes[0, 0].set_title(reference_title)
        axes[0, 1].set_title(candidate_title)
        axes[0, 2].set_title(ratio_title)

        if fitted_degree_max < degrees[-1]:
            for ax in axes[1]:
                ax.axhline(
                    fitted_degree_max + 0.5,
                    color="black",
                    linewidth=1.2,
                    linestyle="--",
                )

        for ax in axes[-1]:
            ax.set_xlabel(r"Frequency / $\mathrm{yr}^{-1}$")

        power_cbar = fig.colorbar(
            reference_im,
            ax=axes[:, :2],
            location="bottom",
            fraction=0.08,
            pad=0.13,
        )
        power_cbar.set_label(
            r"Lowes SV power spectrum at the CMB "
            r"/ $(\mathrm{nT\,yr^{-1}})^2\,\mathrm{yr}$"
        )
        error_cbar = fig.colorbar(
            error_im,
            ax=axes[:, 2],
            boundaries=error_bounds,
            ticks=error_ticks,
        )
        error_cbar.set_ticklabels(error_ticklabels)
        error_cbar.set_label(ratio_title)

        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=300)
        plt.show()


def plot_marginal_comparison(
    coordinates,
    chaos_marginal,
    wave_marginal,
    xlabel,
    title,
    output_path,
    vertical_marker=None,
):
    """Plot one CHAOS/wave marginal spectrum comparison."""
    with plt.rc_context({"font.size": figure_font_size}):
        fig, ax = plt.subplots(
            figsize=(text_width, 0.52 * text_width),
            constrained_layout=True,
        )
        ax.plot(
            coordinates,
            chaos_marginal,
            marker="o",
            linewidth=1.5,
            label="CHAOS-8.6",
        )
        ax.plot(
            coordinates,
            wave_marginal,
            marker="s",
            linewidth=1.5,
            label="Optimal wave combination",
        )
        if vertical_marker is not None:
            ax.axvline(
                vertical_marker,
                color="black",
                linewidth=1.2,
                linestyle="--",
                label="Optimisation boundary",
            )
        ax.set_yscale("log")
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Summed Lowes SV spectrum at the CMB")
        ax.set_title(title)
        ax.grid(True, which="both", alpha=0.25)
        ax.legend()

        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=300)
        plt.show()


def plot_degree_summed_frequency_comparison(
    frequencies,
    chaos_fit_marginal,
    wave_fit_marginal,
    chaos_diagnostic_marginal,
    wave_diagnostic_marginal,
    output_path,
):
    """Compare degree-summed spectra for fit and diagnostic degrees."""
    cases = [
        (
            chaos_fit_marginal,
            wave_fit_marginal,
            rf"Degrees $n \leq {fit_degree_max}$",
        ),
        (
            chaos_diagnostic_marginal,
            wave_diagnostic_marginal,
            rf"Degrees $n \leq {diagnostic_degree_max}$",
        ),
    ]

    with plt.rc_context({"font.size": figure_font_size}):
        fig, axes = plt.subplots(
            2,
            1,
            figsize=(text_width, 0.82 * text_width),
            sharex=True,
            constrained_layout=True,
        )

        for ax, (
            chaos_marginal,
            wave_marginal,
            case_title,
        ) in zip(axes, cases):
            ax.plot(
                frequencies,
                chaos_marginal,
                marker="o",
                linewidth=1.5,
                label="CHAOS-8.6",
            )
            ax.plot(
                frequencies,
                wave_marginal,
                marker="s",
                linewidth=1.5,
                label="Optimal wave combination",
            )
            ax.set_yscale("log")
            ax.set_ylabel("Summed Lowes SV spectrum")
            ax.set_title(case_title)
            ax.grid(True, which="both", alpha=0.25)
            ax.legend()

        axes[-1].set_xlabel(
            r"Frequency / $\mathrm{yr}^{-1}$"
        )
        fig.suptitle("Degree-summed Lowes SV spectrum at the CMB")

        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=300)
        plt.show()


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


def plot_radial_sv_snapshot(
    chaos_snapshot,
    wave_snapshot,
    snapshot_decimal_year,
    output_path,
):
    """Compare radial SV at the configured fit and diagnostic degrees."""
    degree_limits = (
        fit_degree_max,
        diagnostic_degree_max,
    )
    fields_by_degree = []

    for degree_limit in degree_limits:
        chaos_grid = radial_sv_grid(
            chaos_snapshot,
            degree_limit,
        )
        wave_grid = radial_sv_grid(
            wave_snapshot,
            degree_limit,
        )
        fields_by_degree.append((chaos_grid, wave_grid))

    with plt.rc_context({"font.size": figure_font_size}):
        fig, axes = plt.subplots(
            2,
            2,
            figsize=(text_width, 0.78 * text_width),
            subplot_kw={"projection": ccrs.Mollweide()},
            constrained_layout=True,
        )

        for row, (
            degree_limit,
            (chaos_grid, wave_grid),
        ) in enumerate(zip(degree_limits, fields_by_degree)):
            colour_limit = max(
                np.nanmax(np.abs(chaos_grid)),
                np.nanmax(np.abs(wave_grid)),
            )
            if not np.isfinite(colour_limit) or colour_limit <= 0:
                raise ValueError(
                    "Could not determine radial-SV colour limits "
                    f"for n <= {degree_limit}"
                )

            row_image = None
            for column, (field, model_title) in enumerate(
                (
                    (chaos_grid, "CHAOS-8.6"),
                    (wave_grid, "Optimal wave combination"),
                )
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

            colourbar = fig.colorbar(
                row_image,
                ax=axes[row, :],
                location="right",
                shrink=0.85,
                pad=0.03,
                extend="both",
            )
            colourbar.set_label(
                r"Radial SV at the CMB / $\mathrm{nT\,yr^{-1}}$"
            )

        fig.suptitle(
            "Radial SV snapshot at nearest good-record epoch "
            f"{snapshot_decimal_year:.1f}"
        )
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=300)
        plt.show()


# %% VALIDATE SETTINGS AND THE SHARED SYNTHETIC TIME AXIS
spectral_domain_settings = [
    (
        "water-level",
        water_level_degree_min,
        water_level_degree_max,
        water_level_frequency_min,
        water_level_frequency_max,
    ),
    (
        "fit",
        fit_degree_min,
        fit_degree_max,
        fit_frequency_min,
        fit_frequency_max,
    ),
    (
        "diagnostic",
        diagnostic_degree_min,
        diagnostic_degree_max,
        diagnostic_frequency_min,
        diagnostic_frequency_max,
    ),
]

for (
    domain_name,
    domain_degree_min,
    domain_degree_max,
    domain_frequency_min,
    domain_frequency_max,
) in spectral_domain_settings:
    if not (
        1
        <= domain_degree_min
        <= domain_degree_max
        <= nmax
    ):
        raise ValueError(
            f"Invalid {domain_name} degree bounds: "
            f"{domain_degree_min} to {domain_degree_max}"
        )
    if not (
        0.0
        <= domain_frequency_min
        <= domain_frequency_max
    ):
        raise ValueError(
            f"Invalid {domain_name} frequency bounds: "
            f"{domain_frequency_min} to {domain_frequency_max}"
        )

if not 1 <= int(demo_month) <= 12:
    raise ValueError("demo_month must lie between 1 and 12")

if not wave_file.exists():
    raise FileNotFoundError(f"Could not find {wave_file}")

with h5py.File(wave_file, "r") as h5_file:
    example_spline = np.asarray(
        h5_file[f"mode_{mode_numbers[0]}/without_decay"][()]
    )

example_gauss = H_sv @ example_spline
n_synthetic_times = example_gauss.shape[0]
full_times_mjd2000 = np.asarray(times_mjd2000, dtype=float)
full_decimal_years = np.asarray(times_absolute, dtype=float)

if n_synthetic_times != full_times_mjd2000.size:
    raise ValueError(
        "Resolved wave and synSetup time axes differ: "
        f"{n_synthetic_times} wave epochs versus "
        f"{full_times_mjd2000.size} synSetup epochs"
    )
if full_decimal_years.size != n_synthetic_times:
    raise ValueError(
        "times_absolute and the resolved wave records have "
        "different lengths"
    )

good_times_mjd2000 = full_times_mjd2000[good_record_slice]
good_decimal_years = full_decimal_years[good_record_slice]

if good_decimal_years.size < 3:
    raise ValueError("Too few good-record epochs were selected")

print(
    f"Using {n_synthetic_times} full samples from "
    f"{full_decimal_years[0]:.1f} to "
    f"{full_decimal_years[-1]:.1f} at dt={dt_years} yr"
)
print(
    f"Good-record slice contains {good_decimal_years.size} samples "
    f"from {good_decimal_years[0]:.1f} to "
    f"{good_decimal_years[-1]:.1f}"
)


# %% CHAOS-8.6 ON ALL SYNSETUP EPOCHS, THEN WINDOWED TO THE GOOD RECORD
chaos_8_sv_full = load_sv_gauss(
    chaos_file,
    full_times_mjd2000,
    nmax,
)
chaos_8_sv = chaos_8_sv_full[good_record_slice]

chaos_8_spectrum, frequencies = degree_frequency_spectrum(
    chaos_8_sv
)
degrees = np.arange(1, chaos_8_spectrum.shape[0] + 1)

water_level_domain = rectangular_spectral_mask(
    degrees,
    frequencies,
    water_level_degree_min,
    water_level_degree_max,
    water_level_frequency_min,
    water_level_frequency_max,
)
fit_domain = rectangular_spectral_mask(
    degrees,
    frequencies,
    fit_degree_min,
    fit_degree_max,
    fit_frequency_min,
    fit_frequency_max,
)
fit_valid = (
    fit_domain
    & np.isfinite(chaos_8_spectrum)
    & (chaos_8_spectrum > 0)
)

if not np.any(fit_valid):
    raise ValueError("The CHAOS-8.6 fit domain contains no valid cells")

print(
    "Water-level domain:",
    f"n={water_level_degree_min}..{water_level_degree_max}, "
    f"f={water_level_frequency_min:g}.."
    f"{water_level_frequency_max:g} yr^-1, "
    f"{np.sum(water_level_domain)} cells",
)
print(
    "Optimisation domain:",
    f"n={fit_degree_min}..{fit_degree_max}, "
    f"f={fit_frequency_min:g}..{fit_frequency_max:g} yr^-1, "
    f"{np.sum(fit_valid)} valid cells",
)


# %% STRICT WATER-LEVEL SCALING AGAINST CONFIGURED CHAOS-8.6 CELLS
mode_periods = []
water_level_power_scales = []
scaled_mode_mean_powers = []
scaled_mode_series = []
scaled_mode_spectra = []
scalings_dict = {}

with h5py.File(wave_file, "r") as h5_file:
    for mode_number in tqdm(mode_numbers, desc="Scaling modes"):
        mode_data = Component_Load(mode_number)
        eigenvalue = mode_data["eigenvalue"]
        period = 2 * np.pi / np.abs(eigenvalue.imag)

        gnm_spline = np.asarray(
            h5_file[f"mode_{mode_number}/without_decay"][()]
        )
        gnm_mode_full = H_sv @ gnm_spline

        if gnm_mode_full.shape[0] != n_synthetic_times:
            raise ValueError(
                f"Mode {mode_number} has {gnm_mode_full.shape[0]} "
                f"epochs; expected {n_synthetic_times}"
            )

        gnm_mode = gnm_mode_full[good_record_slice]
        mode_spectrum, mode_f = degree_frequency_spectrum(gnm_mode)

        if not np.allclose(mode_f, frequencies):
            raise ValueError(
                f"Frequency mismatch for mode {mode_number}"
            )

        water_level_valid = (
            np.isfinite(chaos_8_spectrum)
            & np.isfinite(mode_spectrum)
            & (chaos_8_spectrum > 0)
            & (mode_spectrum > 0)
            & water_level_domain
        )
        if not np.any(water_level_valid):
            raise ValueError(
                f"No valid water-level cells for mode {mode_number}"
            )

        power_scale = np.min(
            chaos_8_spectrum[water_level_valid]
            / mode_spectrum[water_level_valid]
        )
        amplitude_scale = np.sqrt(power_scale)
        scaled_series = gnm_mode * amplitude_scale

        scaled_mean_power = Mean_Instantaneous_Total_Lowes_Power(
            scaled_series,
            a=r_earth,
            r=r_choice,
        )

        mode_periods.append(period)
        water_level_power_scales.append(power_scale)
        scaled_mode_mean_powers.append(scaled_mean_power)
        scaled_mode_series.append(scaled_series)
        scaled_mode_spectra.append(mode_spectrum * power_scale)
        scalings_dict[str(mode_number)] = amplitude_scale

mode_periods = np.asarray(mode_periods)
water_level_power_scales = np.asarray(water_level_power_scales)
scaled_mode_mean_powers = np.asarray(scaled_mode_mean_powers)
scaled_mode_series = np.asarray(scaled_mode_series)
scaled_mode_spectra = np.asarray(scaled_mode_spectra)

water_level_scale_by_mode = dict(
    zip(mode_numbers, water_level_power_scales)
)

with open(scale_file, "wb") as file:
    pickle.dump(scalings_dict, file)


# %% WATER-LEVEL-SCALED POWER AGAINST MODE PERIOD
chaos_8_mean_power = Mean_Instantaneous_Total_Lowes_Power(
    chaos_8_sv,
    a=r_earth,
    r=r_choice,
)
period_order = np.argsort(mode_periods)

with plt.rc_context({"font.size": figure_font_size}):
    fig, ax = plt.subplots(
        figsize=(text_width, 0.55 * text_width),
        constrained_layout=True,
    )
    ax.plot(
        mode_periods[period_order],
        scaled_mode_mean_powers[period_order],
        "x",
        markersize=6,
    )
    ax.axhline(
        chaos_8_mean_power,
        linestyle="--",
        linewidth=1,
        label="CHAOS-8.6 total mean power",
    )
    ax.set_yscale("log")
    ax.set_xlabel("Wave period / yr")
    ax.set_ylabel(
        r"Mean instantaneous Lowes SV power "
        r"/ $(\mathrm{nT\,yr^{-1}})^2$"
    )
    ax.set_title("Water-level-scaled wave power at the CMB")
    ax.grid(True, which="both", alpha=0.25)
    ax.legend()
    water_level_figure = (
        Path(FIG_DIR)
        / "water_level_scaled_mode_power_chaos8_full_nf.png"
    )
    fig.savefig(water_level_figure, dpi=300)
    plt.show()


# %% LOG-MISFIT OBJECTIVE ON THE CONFIGURED CHAOS-8.6 FIT DOMAIN
psd_epsilon = 1e-15 * np.nanmax(chaos_8_spectrum[fit_valid])


def rms_log_misfit(candidate_spectrum):
    error = np.log10(
        (candidate_spectrum[fit_valid] + psd_epsilon)
        / (chaos_8_spectrum[fit_valid] + psd_epsilon)
    )
    return np.sqrt(np.mean(error**2))


# %% MODE-SELECTION SETUP
if mode_count_strategy not in {"fixed", "auto"}:
    raise ValueError(
        "mode_count_strategy must be either 'fixed' or 'auto'"
    )

mode_number_to_index = {
    int(mode_number): index
    for index, mode_number in enumerate(mode_numbers)
}

unknown_seeded_modes = [
    int(mode_number)
    for mode_number in seeded_mode_numbers
    if int(mode_number) not in mode_number_to_index
]
if unknown_seeded_modes:
    raise ValueError(
        "The following seeded modes are not present in mode_numbers: "
        f"{unknown_seeded_modes}"
    )

seeded_indices = {
    mode_number_to_index[int(mode_number)]
    for mode_number in seeded_mode_numbers
}


def periods_conflict(index_a, index_b):
    """True when either mode lies within the other's exclusion window."""
    period_a = float(mode_periods[index_a])
    period_b = float(mode_periods[index_b])

    return (
        abs(period_a - period_b)
        <= period_exclusion_fraction * period_a
        or abs(period_a - period_b)
        <= period_exclusion_fraction * period_b
    )


def mode_set_respects_period_spacing(mode_set):
    """Check the pairwise period-exclusion constraint for a mode set."""
    indices = sorted(mode_set)

    for position, index_a in enumerate(indices):
        for index_b in indices[position + 1:]:
            if periods_conflict(index_a, index_b):
                return False

    return True


def can_add_mode(mode_set, mode_in):
    """Check whether mode_in can be added without violating spacing."""
    return all(
        not periods_conflict(mode_in, selected_index)
        for selected_index in mode_set
    )


if not mode_set_respects_period_spacing(seeded_indices):
    seeded_periods = {
        int(mode_numbers[index]): float(mode_periods[index])
        for index in sorted(seeded_indices)
    }
    raise ValueError(
        "seeded_mode_numbers violate the period-spacing constraint. "
        f"Seeded mode periods: {seeded_periods}"
    )


if mode_count_strategy == "fixed":
    if n_selected_modes < len(seeded_indices):
        raise ValueError(
            "n_selected_modes cannot be smaller than the number of "
            "seeded modes"
        )
    if n_selected_modes > len(mode_numbers):
        raise ValueError(
            "n_selected_modes cannot exceed the number of available modes"
        )

    min_modes = n_selected_modes
    max_modes = n_selected_modes

else:
    min_modes = max(int(auto_min_modes), len(seeded_indices))
    max_modes = min(int(auto_max_modes), len(mode_numbers))

    if min_modes < 0:
        raise ValueError("auto_min_modes must be non-negative")
    if max_modes < min_modes:
        raise ValueError(
            "auto_max_modes must be >= auto_min_modes and large enough "
            "to contain all seeded modes"
        )

print(
    "\nMode-selection strategy:",
    mode_count_strategy,
)
print(
    "Seeded mandatory modes:",
    np.asarray(sorted(seeded_mode_numbers), dtype=int),
)
print(
    "Pairwise period exclusion:",
    f"+/-{100 * period_exclusion_fraction:.1f}%"
)
if mode_count_strategy == "fixed":
    print("Required number of modes:", n_selected_modes)
else:
    print(
        "Allowed automatic mode-count range:",
        f"{min_modes} to {max_modes}",
    )


# %% GREEDY LINEAR-SPECTRUM INITIAL SET
#
# Start with all mandatory seeded modes, then greedily add modes until:
#   fixed mode -> n_selected_modes is reached
#   auto mode  -> min_modes is reached
#
# The coherent objective below subsequently performs the true optimisation.

initial_indices = list(sorted(seeded_indices))
remaining_indices = [
    index
    for index in range(len(mode_numbers))
    if index not in seeded_indices
]

if initial_indices:
    current_linear_spectrum = np.sum(
        scaled_mode_spectra[initial_indices],
        axis=0,
    )
else:
    current_linear_spectrum = np.zeros_like(chaos_8_spectrum)

initial_target_count = (
    n_selected_modes
    if mode_count_strategy == "fixed"
    else min_modes
)

while len(initial_indices) < initial_target_count:
    current_set_for_spacing = set(initial_indices)
    eligible_positions = [
        position
        for position, index in enumerate(remaining_indices)
        if can_add_mode(current_set_for_spacing, index)
    ]

    if not eligible_positions:
        raise ValueError(
            "Could not construct an initial mode set of the requested size "
            "without violating the period-spacing constraint. Reduce "
            "n_selected_modes / auto_min_modes, reduce "
            "period_exclusion_fraction, or change seeded_mode_numbers."
        )

    trial_losses = [
        rms_log_misfit(
            current_linear_spectrum
            + scaled_mode_spectra[remaining_indices[position]]
        )
        for position in eligible_positions
    ]

    best_eligible_position = int(np.argmin(trial_losses))
    best_position = eligible_positions[best_eligible_position]
    best_index = remaining_indices.pop(best_position)

    initial_indices.append(best_index)
    current_linear_spectrum += scaled_mode_spectra[best_index]

initial_set = set(initial_indices)

if not mode_set_respects_period_spacing(initial_set):
    raise RuntimeError(
        "Internal error: initial mode set violates period spacing"
    )


# %% TRUE COHERENT DEGREE-FREQUENCY OBJECTIVE
loss_cache = {}


def sum_mode_series(mode_set):
    """Return the coherent Gauss-series sum for a set of mode indices."""
    if not mode_set:
        return np.zeros_like(scaled_mode_series[0])

    return np.sum(
        scaled_mode_series[list(sorted(mode_set))],
        axis=0,
    )


def evaluate_mode_set(mode_set, gnm_total=None):
    """Evaluate coherent RMS log-misfit for one selected mode set."""
    key = tuple(sorted(mode_set))

    if key in loss_cache:
        return loss_cache[key]

    if gnm_total is None:
        gnm_total = sum_mode_series(mode_set)

    spectrum, _ = degree_frequency_spectrum(gnm_total)
    loss = rms_log_misfit(spectrum)

    loss_cache[key] = loss
    return loss


initial_gnm = sum_mode_series(initial_set)
initial_loss = evaluate_mode_set(initial_set, initial_gnm)


# %% RANDOM PROPOSAL HELPERS
rng = np.random.default_rng(random_seed)
all_indices = np.arange(len(mode_numbers), dtype=int)


def removable_indices(mode_set):
    """Selected indices that are not locked by seeded_mode_numbers."""
    return np.asarray(
        sorted(set(mode_set) - seeded_indices),
        dtype=int,
    )


def unselected_indices(mode_set):
    """All currently unselected mode indices."""
    return np.setdiff1d(
        all_indices,
        np.asarray(sorted(mode_set), dtype=int),
        assume_unique=False,
    )


def propose_fixed_count_move(current_set, current_gnm):
    """One-for-one valid swap preserving seeds, size, and period spacing."""
    removable = removable_indices(current_set)
    unselected = unselected_indices(current_set)

    if removable.size == 0 or unselected.size == 0:
        return None

    valid_swaps = []

    for mode_out in removable:
        mode_out = int(mode_out)
        retained_set = current_set - {mode_out}

        for mode_in in unselected:
            mode_in = int(mode_in)

            if can_add_mode(retained_set, mode_in):
                valid_swaps.append((mode_out, mode_in))

    if not valid_swaps:
        return None

    mode_out, mode_in = valid_swaps[
        int(rng.integers(len(valid_swaps)))
    ]

    candidate_set = current_set.copy()
    candidate_set.remove(mode_out)
    candidate_set.add(mode_in)

    candidate_gnm = (
        current_gnm
        - scaled_mode_series[mode_out]
        + scaled_mode_series[mode_in]
    )

    return candidate_set, candidate_gnm


def propose_auto_count_move(current_set, current_gnm):
    """
    Randomly propose a valid add, remove, or swap move.

    Seeded modes are never removable. The mode count remains between min_modes
    and max_modes, and every proposed set obeys the period-spacing constraint.
    """
    removable = removable_indices(current_set)
    unselected = unselected_indices(current_set)

    valid_additions = [
        int(mode_in)
        for mode_in in unselected
        if can_add_mode(current_set, int(mode_in))
    ]

    valid_swaps = []
    for mode_out in removable:
        mode_out = int(mode_out)
        retained_set = current_set - {mode_out}

        for mode_in in unselected:
            mode_in = int(mode_in)

            if can_add_mode(retained_set, mode_in):
                valid_swaps.append((mode_out, mode_in))

    allowed_moves = []

    if len(current_set) < max_modes and valid_additions:
        allowed_moves.append("add")

    if len(current_set) > min_modes and removable.size > 0:
        allowed_moves.append("remove")

    if valid_swaps:
        allowed_moves.append("swap")

    if not allowed_moves:
        return None

    move = str(rng.choice(allowed_moves))

    if move == "add":
        mode_in = int(rng.choice(valid_additions))

        candidate_set = current_set.copy()
        candidate_set.add(mode_in)

        candidate_gnm = (
            current_gnm
            + scaled_mode_series[mode_in]
        )

    elif move == "remove":
        mode_out = int(rng.choice(removable))

        candidate_set = current_set.copy()
        candidate_set.remove(mode_out)

        candidate_gnm = (
            current_gnm
            - scaled_mode_series[mode_out]
        )

    else:
        mode_out, mode_in = valid_swaps[
            int(rng.integers(len(valid_swaps)))
        ]

        candidate_set = current_set.copy()
        candidate_set.remove(mode_out)
        candidate_set.add(mode_in)

        candidate_gnm = (
            current_gnm
            - scaled_mode_series[mode_out]
            + scaled_mode_series[mode_in]
        )

    return candidate_set, candidate_gnm


def perturb_starting_set(mode_set):
    """
    Randomly perturb a restart while respecting seeded modes and mode-count
    constraints.
    """
    current_set = mode_set.copy()
    current_gnm = sum_mode_series(current_set)

    reference_count = (
        n_selected_modes
        if mode_count_strategy == "fixed"
        else max(len(current_set), min_modes)
    )
    n_perturb = max(1, int(np.ceil(0.25 * reference_count)))

    for _ in range(n_perturb):
        if mode_count_strategy == "fixed":
            proposal = propose_fixed_count_move(
                current_set,
                current_gnm,
            )
        else:
            proposal = propose_auto_count_move(
                current_set,
                current_gnm,
            )

        if proposal is None:
            break

        current_set, current_gnm = proposal

    return current_set, current_gnm


# %% SIMULATED ANNEALING
best_set = initial_set.copy()
best_gnm = initial_gnm.copy()
best_loss = initial_loss

for restart in range(n_restarts):

    if restart == 0:
        current_set = initial_set.copy()
        current_gnm = initial_gnm.copy()
    else:
        current_set, current_gnm = perturb_starting_set(
            initial_set
        )

    current_loss = evaluate_mode_set(
        current_set,
        current_gnm,
    )

    if current_loss < best_loss:
        best_set = current_set.copy()
        best_gnm = current_gnm.copy()
        best_loss = current_loss

    for step in range(annealing_steps):
        fraction = step / max(annealing_steps - 1, 1)
        temperature = temperature_start * (
            temperature_end / temperature_start
        ) ** fraction

        if mode_count_strategy == "fixed":
            proposal = propose_fixed_count_move(
                current_set,
                current_gnm,
            )
        else:
            proposal = propose_auto_count_move(
                current_set,
                current_gnm,
            )

        if proposal is None:
            break

        candidate_set, candidate_gnm = proposal

        candidate_loss = evaluate_mode_set(
            candidate_set,
            candidate_gnm,
        )
        loss_change = candidate_loss - current_loss

        if (
            loss_change < 0
            or rng.random() < np.exp(-loss_change / temperature)
        ):
            current_set = candidate_set
            current_gnm = candidate_gnm
            current_loss = candidate_loss

        if current_loss < best_loss:
            best_set = current_set.copy()
            best_gnm = current_gnm.copy()
            best_loss = current_loss

    print(
        f"Restart {restart + 1}: "
        f"best RMS log error = {best_loss:.4f}, "
        f"n_modes = {len(best_set)}"
    )


# %% DETERMINISTIC LOCAL REFINEMENT
#
# Fixed-count mode:
#   exhaustively tests every allowed one-for-one swap.
#
# Automatic-count mode:
#   exhaustively tests every allowed single addition, removal, and swap.
#   The best improving move is accepted, then the search repeats.
#
# Seeded modes are never considered for removal.

for refinement_pass in range(max_swap_passes):

    best_candidate_set = None
    best_candidate_loss = best_loss
    best_candidate_gnm = None
    best_move_description = None

    selected = np.asarray(sorted(best_set), dtype=int)
    removable = removable_indices(best_set)
    unselected = unselected_indices(best_set)

    # ----------------------------------------------------------
    # ADDITIONS: automatic-count mode only
    # ----------------------------------------------------------
    if (
        mode_count_strategy == "auto"
        and len(best_set) < max_modes
    ):
        for mode_in in unselected:
            mode_in = int(mode_in)

            if not can_add_mode(best_set, mode_in):
                continue

            candidate_set = best_set.copy()
            candidate_set.add(mode_in)

            candidate_gnm = (
                best_gnm
                + scaled_mode_series[mode_in]
            )

            candidate_loss = evaluate_mode_set(
                candidate_set,
                candidate_gnm,
            )

            if (
                candidate_loss
                < best_candidate_loss - refinement_tolerance
            ):
                best_candidate_set = candidate_set
                best_candidate_loss = candidate_loss
                best_candidate_gnm = candidate_gnm
                best_move_description = (
                    f"add mode {mode_numbers[mode_in]}"
                )

    # ----------------------------------------------------------
    # REMOVALS: automatic-count mode only
    # ----------------------------------------------------------
    if (
        mode_count_strategy == "auto"
        and len(best_set) > min_modes
    ):
        for mode_out in removable:
            mode_out = int(mode_out)

            candidate_set = best_set.copy()
            candidate_set.remove(mode_out)

            candidate_gnm = (
                best_gnm
                - scaled_mode_series[mode_out]
            )

            candidate_loss = evaluate_mode_set(
                candidate_set,
                candidate_gnm,
            )

            if (
                candidate_loss
                < best_candidate_loss - refinement_tolerance
            ):
                best_candidate_set = candidate_set
                best_candidate_loss = candidate_loss
                best_candidate_gnm = candidate_gnm
                best_move_description = (
                    f"remove mode {mode_numbers[mode_out]}"
                )

    # ----------------------------------------------------------
    # SWAPS: both fixed-count and automatic-count modes
    # ----------------------------------------------------------
    for mode_out in removable:
        mode_out = int(mode_out)

        for mode_in in unselected:
            mode_in = int(mode_in)

            retained_set = best_set - {mode_out}
            if not can_add_mode(retained_set, mode_in):
                continue

            candidate_set = best_set.copy()
            candidate_set.remove(mode_out)
            candidate_set.add(mode_in)

            candidate_gnm = (
                best_gnm
                - scaled_mode_series[mode_out]
                + scaled_mode_series[mode_in]
            )

            candidate_loss = evaluate_mode_set(
                candidate_set,
                candidate_gnm,
            )

            if (
                candidate_loss
                < best_candidate_loss - refinement_tolerance
            ):
                best_candidate_set = candidate_set
                best_candidate_loss = candidate_loss
                best_candidate_gnm = candidate_gnm
                best_move_description = (
                    f"swap mode {mode_numbers[mode_out]} "
                    f"-> {mode_numbers[mode_in]}"
                )

    if best_candidate_set is None:
        break

    best_set = best_candidate_set
    best_loss = best_candidate_loss
    best_gnm = best_candidate_gnm

    print(
        f"Refinement pass {refinement_pass + 1}: "
        f"{best_move_description}; "
        f"RMS log error = {best_loss:.4f}, "
        f"n_modes = {len(best_set)}"
    )


# %% FINAL MODE-SELECTION VALIDATION
if not seeded_indices.issubset(best_set):
    raise RuntimeError(
        "Internal error: one or more seeded modes were lost "
        "during optimisation"
    )

if mode_count_strategy == "fixed":
    if len(best_set) != n_selected_modes:
        raise RuntimeError(
            "Internal error: fixed-count optimisation returned the "
            "wrong number of modes"
        )
else:
    if not (min_modes <= len(best_set) <= max_modes):
        raise RuntimeError(
            "Internal error: automatic optimisation returned a mode "
            "count outside the permitted range"
        )

if not mode_set_respects_period_spacing(best_set):
    raise RuntimeError(
        "Internal error: final mode set violates the period-spacing constraint"
    )


# %% FINAL OPTIMAL COHERENT COMBINATION
optimal_indices = np.asarray(sorted(best_set), dtype=int)
optimal_mode_numbers = mode_numbers[optimal_indices]
optimal_mode_periods = mode_periods[optimal_indices]
optimal_gnm_total = best_gnm

optimal_spectrum, optimal_f = degree_frequency_spectrum(
    optimal_gnm_total
)
if not np.allclose(optimal_f, frequencies):
    raise ValueError("Optimal and CHAOS frequency grids differ")

fit_stats = log_error_statistics(
    optimal_spectrum,
    chaos_8_spectrum,
    fit_valid,
    psd_epsilon,
)

print("\nSelected modes:", optimal_mode_numbers)
print("Selected periods:", optimal_mode_periods)
print("Number of selected modes:", len(optimal_mode_numbers))

if len(optimal_mode_periods) > 1:
    sorted_periods = np.sort(optimal_mode_periods)
    adjacent_ratios = sorted_periods[1:] / sorted_periods[:-1]
    print(
        "Minimum adjacent selected-period ratio:",
        f"{np.min(adjacent_ratios):.3f}",
    )

print(
    "Mandatory seeded modes:",
    np.asarray(sorted(seeded_mode_numbers), dtype=int),
)
print(f"Final fit-domain RMS log10 error: {fit_stats['rms']:.4f}")
print(
    "Fit-domain cells within factor 2:",
    f"{fit_stats['within_factor_2']:.1%}",
)
print(
    "Fit-domain cells within factor 3:",
    f"{fit_stats['within_factor_3']:.1%}",
)


# %% DEGREE-FREQUENCY DIAGNOSTIC OVER THE CONFIGURED DISPLAY DOMAIN
degree_display_mask = (
    (degrees >= diagnostic_degree_min)
    & (degrees <= diagnostic_degree_max)
)
frequency_display_mask = (
    (frequencies >= diagnostic_frequency_min)
    & (frequencies <= diagnostic_frequency_max)
)
degree_plot = degrees[degree_display_mask]
frequency_plot = frequencies[frequency_display_mask]
display_ix = np.ix_(
    degree_display_mask,
    frequency_display_mask,
)

chaos_8_plot = chaos_8_spectrum[display_ix]
optimal_plot = optimal_spectrum[display_ix]
diagnostic_valid = (
    np.isfinite(chaos_8_plot)
    & np.isfinite(optimal_plot)
    & (chaos_8_plot > 0)
    & (optimal_plot > 0)
)
diagnostic_stats = log_error_statistics(
    optimal_plot,
    chaos_8_plot,
    diagnostic_valid,
    psd_epsilon,
)

plot_two_domain_power_comparison(
    reference=chaos_8_plot,
    candidate=optimal_plot,
    log_ratio=diagnostic_stats["log_error"],
    display_valid=diagnostic_stats["valid"],
    degrees=degree_plot,
    frequencies=frequency_plot,
    reference_title="CHAOS-8.6",
    candidate_title="Optimal wave\ncombination",
    ratio_title=r"$\log_{10}(P_{\mathrm{wave}}/P_{8.6})$",
    output_path=degree_frequency_figure,
    fitted_degree_max=fit_degree_max,
)

print(
    "Diagnostic-domain RMS log10 error:",
    f"{diagnostic_stats['rms']:.4f}",
)
print(
    "Diagnostic-domain cells within factor 2:",
    f"{diagnostic_stats['within_factor_2']:.1%}",
)
print(
    "Diagnostic-domain cells within factor 3:",
    f"{diagnostic_stats['within_factor_3']:.1%}",
)


# %% DEGREE AND FREQUENCY MARGINAL POWER SPECTRA
chaos_degree_marginal = np.sum(chaos_8_plot, axis=1)
wave_degree_marginal = np.sum(optimal_plot, axis=1)

fit_degree_sum_mask = (
    (degrees >= 1)
    & (degrees <= fit_degree_max)
)
diagnostic_degree_sum_mask = (
    (degrees >= 1)
    & (degrees <= diagnostic_degree_max)
)
fit_frequency_ix = np.ix_(
    fit_degree_sum_mask,
    frequency_display_mask,
)
diagnostic_frequency_ix = np.ix_(
    diagnostic_degree_sum_mask,
    frequency_display_mask,
)

chaos_fit_frequency_marginal = np.sum(
    chaos_8_spectrum[fit_frequency_ix],
    axis=0,
)
wave_fit_frequency_marginal = np.sum(
    optimal_spectrum[fit_frequency_ix],
    axis=0,
)
chaos_diagnostic_frequency_marginal = np.sum(
    chaos_8_spectrum[diagnostic_frequency_ix],
    axis=0,
)
wave_diagnostic_frequency_marginal = np.sum(
    optimal_spectrum[diagnostic_frequency_ix],
    axis=0,
)

plot_marginal_comparison(
    degree_plot,
    chaos_degree_marginal,
    wave_degree_marginal,
    xlabel=r"Spherical harmonic degree $n$",
    title="Frequency-summed Lowes SV spectrum",
    output_path=degree_marginal_figure,
    vertical_marker=fit_degree_max + 0.5,
)

plot_degree_summed_frequency_comparison(
    frequency_plot,
    chaos_fit_frequency_marginal,
    wave_fit_frequency_marginal,
    chaos_diagnostic_frequency_marginal,
    wave_diagnostic_frequency_marginal,
    output_path=frequency_marginal_figure,
)


# %% RADIAL-SV SNAPSHOT AT THE NEAREST GOOD-RECORD EPOCH
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
demo_decimal_year = float(
    good_decimal_years[demo_good_index]
)
print(
    f"Requested snapshot {int(demo_year):04d}-"
    f"{int(demo_month):02d}; using nearest good-record epoch "
    f"{demo_decimal_year:.1f}"
)

plot_radial_sv_snapshot(
    chaos_snapshot=chaos_8_sv[demo_good_index],
    wave_snapshot=optimal_gnm_total[demo_good_index],
    snapshot_decimal_year=demo_decimal_year,
    output_path=snapshot_figure,
)

print(
    "Saved diagnostic figures to",
    Path(FIG_DIR) / "final",
)
print(f"Saved scalings to {scale_file}")
