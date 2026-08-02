"""
Compare one exact synthetic wave with its resolved representation.

Both branches retain the mode's original arbitrary amplitude.  No empirical
amplitude scaling, normalization, or post-resolution amplitude matching is
applied.  The exact and resolved records are constructed on the complete
synSetup time axis before both are cropped to the good-record interval.
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
from matplotlib.colors import LogNorm

from src.msc_thesis.paths import FELIX_DIR, FIG_DIR, text_width
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
    Component_Load_SV,
    G_Time_Series_Eval,
    H_sv,
    Lowes_Degree_PSD_All_Degrees,
    Truncate_Gauss_Coeffs,
)


# %% USER SETTINGS

demo_mode_number = 55

# Exact and resolved records are first constructed at this common degree.
source_degree_max = 20
expected_full_epochs = 148

# Maximum degree retained for all diagnostics, and the lower comparison
# truncation used in the separate marginal and snapshot panels.
diagnostic_degree_max = 15
comparison_degree_max = 10

# Inclusive frequency interval used in the degree-frequency and marginal
# figures.  The underlying periodograms are computed at every FFT frequency.
frequency_display_min = 0.0
frequency_display_max = 0.333

# The nearest good-record epoch is used for both exact and resolved snapshots.
demo_year = 2020
demo_month = 1

figure_font_size = 11
show_figures_interactively = True

resolved_wave_file = Path(FELIX_DIR) / "R_splines_arbitrary.h5"
output_directory = Path(FIG_DIR) / "model_resolution"
degree_frequency_figure_path = (
    output_directory / f"mode_{demo_mode_number}_degree_frequency.png"
)
degree_marginal_n15_figure_path = (
    output_directory / f"mode_{demo_mode_number}_degree_marginal_n15.png"
)
degree_marginal_n10_figure_path = (
    output_directory / f"mode_{demo_mode_number}_degree_marginal_n10.png"
)
frequency_marginal_figure_path = (
    output_directory / f"mode_{demo_mode_number}_frequency_marginal.png"
)
snapshot_figure_path = (
    output_directory / f"mode_{demo_mode_number}_radial_sv_snapshot.png"
)


# %% SHARED HELPERS

def save_and_finish_figure(fig, output_path):
    """Save a figure and either display it or close it."""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300)
    if show_figures_interactively:
        plt.show()
    else:
        plt.close(fig)


def load_unscaled_mode_records(
    mode_number,
    wave_path,
    degree_max,
):
    """
    Construct unscaled exact and resolved SV coefficient records.

    Component_Load_SV returns the no-decay SV Gauss phasor and its purely
    imaginary eigenvalue.  The matching precomputed resolved dataset is also
    the no-decay record.  Neither branch is multiplied by mode_amp_scalings.
    """

    mode_key = str(mode_number)
    mode_data = Component_Load_SV(mode_key)
    eigenvalue = complex(mode_data["eigenvalue"])
    exact_sv_phasor = Truncate_Gauss_Coeffs(
        mode_data["gnm"],
        tmax=degree_max,
    )
    exact_sv_full = G_Time_Series_Eval(
        exact_sv_phasor,
        eigenvalue,
    )

    if not wave_path.exists():
        raise FileNotFoundError(f"Could not find {wave_path}")
    dataset_name = f"mode_{mode_key}/without_decay"
    with h5py.File(wave_path, "r") as h5_file:
        if dataset_name not in h5_file:
            raise KeyError(f"Missing HDF5 dataset: {dataset_name}")
        resolved_spline = np.asarray(
            h5_file[dataset_name][()],
            dtype=float,
        )

    resolved_sv_full = H_sv @ resolved_spline
    return exact_sv_full, resolved_sv_full, eigenvalue


def degree_frequency_spectrum(gnm, degree_max):
    """Return the unnormalised Lowes SV power spectrum at the CMB."""

    degrees = np.arange(1, degree_max + 1)
    spectrum, frequencies = Lowes_Degree_PSD_All_Degrees(
        gnm,
        a=r_earth,
        r=r_cmb,
        f_sample=1.0 / dt_years,
        degrees=degrees,
    )
    return spectrum, frequencies, degrees


def displayed_frequency_mask(frequencies):
    """Inclusive mask for the configured plotting frequency interval."""

    return (
        (frequencies >= frequency_display_min)
        & (frequencies <= frequency_display_max)
    )


def positive_limits(*arrays):
    """Return finite positive limits shared by logarithmic comparisons."""

    positive_values = []
    for values in arrays:
        values = np.asarray(values)
        positive_values.append(
            values[np.isfinite(values) & (values > 0)]
        )
    positive_values = [
        values for values in positive_values if values.size
    ]
    if not positive_values:
        raise ValueError("No positive power values are available to plot")

    combined = np.concatenate(positive_values)
    return float(np.min(combined)), float(np.max(combined))


def plot_degree_frequency_comparison(
    exact_spectrum,
    resolved_spectrum,
    degrees,
    frequencies,
    output_path,
):
    """Plot exact and resolved degree-frequency Lowes power."""

    frequency_mask = displayed_frequency_mask(frequencies)
    if not np.any(frequency_mask):
        raise ValueError("The configured frequency display range is empty")

    displayed_frequencies = frequencies[frequency_mask]
    exact_display = exact_spectrum[:, frequency_mask]
    resolved_display = resolved_spectrum[:, frequency_mask]
    power_min, power_max = positive_limits(
        exact_display,
        resolved_display,
    )
    power_norm = LogNorm(vmin=power_min, vmax=power_max)
    power_cmap = plt.get_cmap("viridis").copy()
    power_cmap.set_bad("lightgrey")

    if displayed_frequencies.size > 1:
        frequency_spacing = float(
            np.mean(np.diff(displayed_frequencies))
        )
    else:
        frequency_spacing = 1.0 / (
            dt_years * exact_spectrum.shape[1]
        )
    extent = [
        displayed_frequencies[0] - frequency_spacing / 2,
        displayed_frequencies[-1] + frequency_spacing / 2,
        degrees[0] - 0.5,
        degrees[-1] + 0.5,
    ]

    with plt.rc_context({"font.size": figure_font_size}):
        fig, axes = plt.subplots(
            1,
            2,
            figsize=(text_width, 0.58 * text_width),
            sharex=True,
            sharey=True,
            constrained_layout=True,
        )

        exact_image = axes[0].imshow(
            np.ma.masked_where(exact_display <= 0, exact_display),
            origin="lower",
            aspect="auto",
            interpolation="nearest",
            extent=extent,
            cmap=power_cmap,
            norm=power_norm,
        )
        axes[1].imshow(
            np.ma.masked_where(resolved_display <= 0, resolved_display),
            origin="lower",
            aspect="auto",
            interpolation="nearest",
            extent=extent,
            cmap=power_cmap,
            norm=power_norm,
        )

        for ax, title in zip(
            axes,
            ("Exact (pre-resolution)", "Resolved"),
        ):
            ax.set_title(title)
            ax.set_xlim(
                frequency_display_min,
                frequency_display_max,
            )
            ax.set_ylim(degrees[0] - 0.5, degrees[-1] + 0.5)
            ax.set_yticks(
                np.arange(
                    degrees[0],
                    degrees[-1] + 1,
                    2,
                )
            )
            ax.set_xlabel(r"Frequency / $\mathrm{yr}^{-1}$")
            ax.axhline(
                comparison_degree_max + 0.5,
                color="black",
                linewidth=1.2,
            )

        axes[0].set_ylabel(r"Spherical harmonic degree $n$")

        colourbar = fig.colorbar(
            exact_image,
            ax=axes,
            location="bottom",
            fraction=0.10,
            pad=0.13,
        )
        colourbar.set_label(
            r"Arbitrary Lowes SV power at the CMB / "
            r"$(\mathrm{nT\,yr^{-1}})^2$"
        )
        fig.suptitle(
            f"Unscaled mode {demo_mode_number}: "
            "degree-frequency power"
        )
        save_and_finish_figure(fig, output_path)


def degree_marginal(
    spectrum,
    degrees,
    frequencies,
    degree_max,
):
    """Sum displayed frequency-bin power independently at each degree."""

    degree_mask = degrees <= degree_max
    frequency_mask = displayed_frequency_mask(frequencies)
    marginal = np.sum(
        spectrum[np.ix_(degree_mask, frequency_mask)],
        axis=1,
    )
    return degrees[degree_mask], marginal


def plot_degree_marginal_comparison(
    exact_spectrum,
    resolved_spectrum,
    degrees,
    frequencies,
    degree_max,
    output_path,
):
    """Plot frequency-summed power against degree at one truncation."""

    degree_values, exact_marginal = degree_marginal(
        exact_spectrum,
        degrees,
        frequencies,
        degree_max,
    )
    resolved_degrees, resolved_marginal = degree_marginal(
        resolved_spectrum,
        degrees,
        frequencies,
        degree_max,
    )
    if not np.array_equal(degree_values, resolved_degrees):
        raise ValueError("Exact and resolved degree axes differ")

    power_min, power_max = positive_limits(
        exact_marginal,
        resolved_marginal,
    )

    with plt.rc_context({"font.size": figure_font_size}):
        fig, ax = plt.subplots(
            figsize=(text_width, 0.55 * text_width),
            constrained_layout=True,
        )
        ax.plot(
            degree_values,
            exact_marginal,
            marker="o",
            linewidth=1.5,
            label="Exact (pre-resolution)",
        )
        ax.plot(
            degree_values,
            resolved_marginal,
            marker="s",
            linewidth=1.5,
            label="Resolved",
        )
        ax.set_yscale("log")
        ax.set_ylim(power_min / 1.5, power_max * 1.5)
        ax.set_xlim(0.75, degree_max + 0.25)
        ax.set_xticks(degree_values)
        ax.set_xlabel(r"Spherical harmonic degree $n$")
        ax.set_ylabel(
            r"Arbitrary Lowes SV power / "
            r"$(\mathrm{nT\,yr^{-1}})^2$"
        )
        ax.set_title(
            f"Unscaled mode {demo_mode_number}: "
            rf"degree marginal, $n \leq {degree_max}$"
        )
        ax.grid(True, which="both", alpha=0.25)
        ax.legend()
        save_and_finish_figure(fig, output_path)


def frequency_marginal(
    spectrum,
    degrees,
    frequencies,
    degree_max,
):
    """Sum Lowes power through degree_max at each displayed frequency."""

    degree_mask = degrees <= degree_max
    frequency_mask = displayed_frequency_mask(frequencies)
    marginal = np.sum(
        spectrum[np.ix_(degree_mask, frequency_mask)],
        axis=0,
    )
    return frequencies[frequency_mask], marginal


def plot_frequency_marginal_comparison(
    exact_spectrum,
    resolved_spectrum,
    degrees,
    frequencies,
    output_path,
):
    """Compare frequency marginals through degrees 10 and 15."""

    degree_limits = (
        comparison_degree_max,
        diagnostic_degree_max,
    )

    with plt.rc_context({"font.size": figure_font_size}):
        fig, axes = plt.subplots(
            2,
            1,
            figsize=(text_width, 0.82 * text_width),
            sharex=True,
            constrained_layout=True,
        )

        for ax, degree_max in zip(axes, degree_limits):
            frequency_values, exact_marginal = frequency_marginal(
                exact_spectrum,
                degrees,
                frequencies,
                degree_max,
            )
            (
                resolved_frequencies,
                resolved_marginal,
            ) = frequency_marginal(
                resolved_spectrum,
                degrees,
                frequencies,
                degree_max,
            )
            if not np.allclose(
                frequency_values,
                resolved_frequencies,
            ):
                raise ValueError(
                    "Exact and resolved marginal frequency axes differ"
                )

            power_min, power_max = positive_limits(
                exact_marginal,
                resolved_marginal,
            )
            ax.plot(
                frequency_values,
                exact_marginal,
                marker="o",
                linewidth=1.5,
                label="Exact (pre-resolution)",
            )
            ax.plot(
                frequency_values,
                resolved_marginal,
                marker="s",
                linewidth=1.5,
                label="Resolved",
            )
            ax.set_yscale("log")
            ax.set_ylim(power_min / 1.5, power_max * 1.5)
            ax.set_xlim(
                frequency_display_min,
                frequency_display_max,
            )
            ax.set_ylabel(
                "Degree-summed arbitrary\n"
                r"Lowes SV power / $(\mathrm{nT\,yr^{-1}})^2$"
            )
            ax.set_title(rf"$n \leq {degree_max}$")
            ax.grid(True, which="both", alpha=0.25)
            ax.legend()

        axes[-1].set_xlabel(r"Frequency / $\mathrm{yr}^{-1}$")
        fig.suptitle(
            f"Unscaled mode {demo_mode_number}: frequency marginal"
        )
        save_and_finish_figure(fig, output_path)


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
    exact_snapshot,
    resolved_snapshot,
    snapshot_decimal_year,
    output_path,
):
    """Plot exact and resolved CMB radial SV with shared row scales."""

    degree_limits = (
        comparison_degree_max,
        diagnostic_degree_max,
    )
    snapshot_coefficients = (
        ("Exact (pre-resolution)", exact_snapshot),
        ("Resolved", resolved_snapshot),
    )

    grids_by_degree = []
    for degree_max in degree_limits:
        grids_by_degree.append([
            radial_sv_grid(coefficients, degree_max)
            for _, coefficients in snapshot_coefficients
        ])

    with plt.rc_context({"font.size": figure_font_size}):
        fig, axes = plt.subplots(
            2,
            2,
            figsize=(1.08 * text_width, 0.92 * text_width),
            subplot_kw={"projection": ccrs.Mollweide()},
            constrained_layout=True,
        )

        for row, (degree_max, row_grids) in enumerate(
            zip(degree_limits, grids_by_degree)
        ):
            colour_limit = max(
                np.nanmax(np.abs(field))
                for field in row_grids
            )
            if not np.isfinite(colour_limit) or colour_limit <= 0:
                raise ValueError(
                    "Could not determine radial-SV colour limits "
                    f"for n <= {degree_max}"
                )

            row_image = None
            for column, ((title, _), field) in enumerate(
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
                    f"{title}\n"
                    rf"$n \leq {degree_max}$"
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
                "Arbitrary radial SV\n"
                r"/ $\mathrm{nT\,yr^{-1}}$"
            )

        fig.suptitle(
            f"Unscaled mode {demo_mode_number}: "
            f"radial SV at {snapshot_decimal_year:.1f}"
        )
        save_and_finish_figure(fig, output_path)


# %% VALIDATE SETTINGS AND TIME AXES

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
    <= comparison_degree_max
    < diagnostic_degree_max
    <= source_degree_max
):
    raise ValueError(
        "Degrees must satisfy "
        "1 <= comparison < diagnostic <= source"
    )
if not (
    0.0
    <= frequency_display_min
    < frequency_display_max
    <= 0.5 / dt_years
):
    raise ValueError(
        "Display frequencies must be increasing and not exceed Nyquist"
    )
if not 1 <= int(demo_month) <= 12:
    raise ValueError("demo_month must lie between 1 and 12")

good_times_mjd2000 = full_times_mjd2000[good_record_slice]
good_decimal_years = full_decimal_years[good_record_slice]
if good_times_mjd2000.size < 3:
    raise ValueError("Too few good-record epochs were selected")

print(
    f"Mode {demo_mode_number}; no amplitude scaling or normalization"
)
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


# %% CONSTRUCT FULL EXACT AND RESOLVED RECORDS, THEN WINDOW AND TRUNCATE

(
    exact_sv_full,
    resolved_sv_full,
    mode_eigenvalue,
) = load_unscaled_mode_records(
    demo_mode_number,
    resolved_wave_file,
    source_degree_max,
)

expected_source_coefficients = (source_degree_max + 1) ** 2 - 1
expected_full_shape = (
    expected_full_epochs,
    expected_source_coefficients,
)
if exact_sv_full.shape != expected_full_shape:
    raise ValueError(
        f"Unexpected exact record shape {exact_sv_full.shape}; "
        f"expected {expected_full_shape}"
    )
if resolved_sv_full.shape != expected_full_shape:
    raise ValueError(
        f"Unexpected resolved record shape {resolved_sv_full.shape}; "
        f"expected {expected_full_shape}"
    )
if not np.all(np.isfinite(exact_sv_full)):
    raise ValueError("Exact SV record contains non-finite values")
if not np.all(np.isfinite(resolved_sv_full)):
    raise ValueError("Resolved SV record contains non-finite values")

exact_sv = Truncate_Gauss_Coeffs(
    exact_sv_full[good_record_slice],
    tmax=diagnostic_degree_max,
)
resolved_sv = Truncate_Gauss_Coeffs(
    resolved_sv_full[good_record_slice],
    tmax=diagnostic_degree_max,
)
expected_diagnostic_coefficients = (
    (diagnostic_degree_max + 1) ** 2 - 1
)
expected_good_shape = (
    good_times_mjd2000.size,
    expected_diagnostic_coefficients,
)
if exact_sv.shape != expected_good_shape:
    raise ValueError(
        f"Unexpected truncated exact shape {exact_sv.shape}; "
        f"expected {expected_good_shape}"
    )
if resolved_sv.shape != expected_good_shape:
    raise ValueError(
        f"Unexpected truncated resolved shape {resolved_sv.shape}; "
        f"expected {expected_good_shape}"
    )

true_period = 2.0 * np.pi / np.abs(mode_eigenvalue.imag)
print(f"Mode period: {true_period:.3f} yr")


# %% LOWES DEGREE-FREQUENCY POWER AND MARGINALS

(
    exact_spectrum,
    frequencies,
    degrees,
) = degree_frequency_spectrum(
    exact_sv,
    diagnostic_degree_max,
)
(
    resolved_spectrum,
    resolved_frequencies,
    resolved_degrees,
) = degree_frequency_spectrum(
    resolved_sv,
    diagnostic_degree_max,
)

if exact_spectrum.shape != resolved_spectrum.shape:
    raise ValueError(
        "Exact and resolved degree-frequency spectra have "
        "different shapes"
    )
if not np.allclose(frequencies, resolved_frequencies):
    raise ValueError("Exact and resolved frequency axes differ")
if not np.array_equal(degrees, resolved_degrees):
    raise ValueError("Exact and resolved degree axes differ")

plot_degree_frequency_comparison(
    exact_spectrum,
    resolved_spectrum,
    degrees,
    frequencies,
    degree_frequency_figure_path,
)
plot_degree_marginal_comparison(
    exact_spectrum,
    resolved_spectrum,
    degrees,
    frequencies,
    diagnostic_degree_max,
    degree_marginal_n15_figure_path,
)
plot_degree_marginal_comparison(
    exact_spectrum,
    resolved_spectrum,
    degrees,
    frequencies,
    comparison_degree_max,
    degree_marginal_n10_figure_path,
)
plot_frequency_marginal_comparison(
    exact_spectrum,
    resolved_spectrum,
    degrees,
    frequencies,
    frequency_marginal_figure_path,
)


# %% SHARED DEMONSTRATION EPOCH AND CMB RADIAL-SV SNAPSHOTS

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
    exact_snapshot=exact_sv[demo_good_index],
    resolved_snapshot=resolved_sv[demo_good_index],
    snapshot_decimal_year=demo_decimal_year,
    output_path=snapshot_figure_path,
)

print(f"Saved model-resolution figures to {output_directory}")
