# %% CHAOS-7.18 VS CHAOS-8.6 DEGREE-FREQUENCY SV POWER
from IPython import get_ipython

ipython = get_ipython()
if ipython is not None:
    ipython.run_line_magic("load_ext", "autoreload")
    ipython.run_line_magic("autoreload", "2")

import sys
from pathlib import Path

import chaosmagpy as cp
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import BoundaryNorm, ListedColormap, LogNorm

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.msc_thesis.paths import *
from pyscripts.R_test_synthetics.syn_pipeline import *


# %% SETTINGS
chaos_7_file = Path(CHAOS_DIR) / "CHAOS-7.18.mat"
chaos_8_file = Path(CHAOS_DIR) / "CHAOS-8.6.mat"

sample_dt = 0.2
start_decimal_year = 2000.0 + 7.0 / 12.0  # approximately 2000/08/01
end_decimal_year = 2024.0

nmax = 20
r_choice = r_cmb

degree_min, degree_max = 1, 20
frequency_min, frequency_max = 0.0, 0.3

output_file = (
    Path(FIG_DIR)
    / "final"
    / "chaos_7_18_vs_8_6_degree_frequency_psd_cmb.png"
)


def decimal_year_to_mjd2000(decimal_years):
    """Convert decimal years to MJD2000 using each calendar year's length."""
    decimal_years = np.asarray(decimal_years, dtype=float)
    output = np.empty_like(decimal_years)
    epoch = np.datetime64("2000-01-01T00:00:00", "ns")

    for index, decimal_year in np.ndenumerate(decimal_years):
        year = int(np.floor(decimal_year))
        fraction = decimal_year - year

        year_start = np.datetime64(
            f"{year:04d}-01-01T00:00:00",
            "ns",
        )
        next_year = np.datetime64(
            f"{year + 1:04d}-01-01T00:00:00",
            "ns",
        )

        timestamp = year_start + np.timedelta64(
            int(np.rint(
                fraction
                * (next_year - year_start)
                / np.timedelta64(1, "ns")
            )),
            "ns",
        )

        output[index] = (
            (timestamp - epoch)
            / np.timedelta64(1, "D")
        )

    return output


def load_sv_gauss(model_path, times, nmax):
    """Load a CHAOS model and evaluate SV coefficients on shared epochs."""
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


# %% SHARED 0.2-YEAR SAMPLING GRID
decimal_years = np.arange(
    start_decimal_year,
    end_decimal_year + 0.5 * sample_dt,
    sample_dt,
)
decimal_years = decimal_years[
    decimal_years <= end_decimal_year + 1e-12
]
times_mjd2000_comparison = decimal_year_to_mjd2000(decimal_years)

print(
    f"Using {decimal_years.size} samples from "
    f"{decimal_years[0]:.4f} to {decimal_years[-1]:.4f} "
    f"at dt={sample_dt} yr"
)


# %% EVALUATE BOTH MODELS ON IDENTICAL EPOCHS
chaos_7_sv = load_sv_gauss(
    chaos_7_file,
    times_mjd2000_comparison,
    nmax,
)
chaos_8_sv = load_sv_gauss(
    chaos_8_file,
    times_mjd2000_comparison,
    nmax,
)

if chaos_7_sv.shape != chaos_8_sv.shape:
    raise ValueError(
        "CHAOS-7.18 and CHAOS-8.6 coefficient arrays have "
        f"different shapes: {chaos_7_sv.shape} and {chaos_8_sv.shape}"
    )


# %% DEGREE-FREQUENCY LOWES POWER AT THE CMB
chaos_7_spectrum, chaos_7_f = Lowes_Degree_PSD_All_Degrees(
    chaos_7_sv,
    a=r_earth,
    r=r_choice,
)
chaos_8_spectrum, chaos_8_f = Lowes_Degree_PSD_All_Degrees(
    chaos_8_sv,
    a=r_earth,
    r=r_choice,
)

if not np.allclose(chaos_7_f, chaos_8_f):
    raise ValueError("CHAOS-7.18 and CHAOS-8.6 frequencies differ")

frequencies = chaos_7_f
degrees = np.arange(1, chaos_7_spectrum.shape[0] + 1)

degree_mask = (
    (degrees >= degree_min)
    & (degrees <= degree_max)
)
frequency_mask = (
    (frequencies >= max(frequency_min, 0.0))
    & (frequencies < frequency_max)
)

degree_plot = degrees[degree_mask]
frequency_plot = frequencies[frequency_mask]

chaos_7_plot = chaos_7_spectrum[
    np.ix_(degree_mask, frequency_mask)
]
chaos_8_plot = chaos_8_spectrum[
    np.ix_(degree_mask, frequency_mask)
]

valid = (
    np.isfinite(chaos_7_plot)
    & np.isfinite(chaos_8_plot)
    & (chaos_7_plot > 0)
    & (chaos_8_plot > 0)
)

if not np.any(valid):
    raise ValueError("No valid positive PSD cells in the display interval")

chaos_7_masked = np.ma.masked_where(~valid, chaos_7_plot)
chaos_8_masked = np.ma.masked_where(~valid, chaos_8_plot)


# %% LOG10 DIFFERENCE: POSITIVE MEANS MORE POWER IN CHAOS-8.6
log_power_ratio = np.full_like(chaos_7_plot, np.nan, dtype=float)
log_power_ratio[valid] = np.log10(
    chaos_8_plot[valid] / chaos_7_plot[valid]
)

ratio_values = log_power_ratio[valid]

print(
    "RMS log10 power ratio:",
    f"{np.sqrt(np.mean(ratio_values**2)):.4f}",
)
print(
    "Cells agreeing within factor 2:",
    f"{np.mean(np.abs(ratio_values) <= np.log10(2)):.1%}",
)
print(
    "Cells agreeing within factor 3:",
    f"{np.mean(np.abs(ratio_values) <= np.log10(3)):.1%}",
)


# %% SHARED POWER NORMALIZATION
positive_power = np.concatenate([
    chaos_7_masked.compressed(),
    chaos_8_masked.compressed(),
])

power_norm = LogNorm(
    vmin=np.min(positive_power),
    vmax=np.max(positive_power),
)
power_cmap = plt.get_cmap("viridis").copy()
power_cmap.set_bad("lightgrey")


# %% DISCRETE LOG-RATIO NORMALIZATION
factor_2 = np.log10(2)
factor_3 = np.log10(3)
max_error = np.nanmax(np.abs(log_power_ratio))
error_cap = max(1.01, max_error + 1e-6)

error_bounds = [
    -error_cap,
    -1,
    -factor_3,
    -factor_2,
    factor_2,
    factor_3,
    1,
    error_cap,
]
error_colors = [
    "#08306B",
    "#2171B5",
    "#9ECAE1",
    "#FFFFFF",
    "#FCAE91",
    "#FB6A4A",
    "#CB181D",
]

error_cmap = ListedColormap(error_colors)
error_cmap.set_bad("lightgrey")
error_norm = BoundaryNorm(
    error_bounds,
    error_cmap.N,
    clip=True,
)


# %% THREE-PANEL COMPARISON
df = np.mean(np.diff(frequency_plot))
extent = [
    frequency_plot[0] - df / 2,
    frequency_plot[-1] + df / 2,
    degree_plot[0] - 0.5,
    degree_plot[-1] + 0.5,
]

with plt.rc_context({"font.size": 12}):
    fig, axes = plt.subplots(
        1,
        3,
        figsize=(text_width, 4.8),
        sharex=True,
        sharey=True,
        constrained_layout=True,
    )

    chaos_7_im = axes[0].imshow(
        chaos_7_masked,
        origin="lower",
        aspect="auto",
        interpolation="nearest",
        extent=extent,
        cmap=power_cmap,
        norm=power_norm,
    )
    axes[1].imshow(
        chaos_8_masked,
        origin="lower",
        aspect="auto",
        interpolation="nearest",
        extent=extent,
        cmap=power_cmap,
        norm=power_norm,
    )
    ratio_im = axes[2].imshow(
        np.ma.masked_invalid(log_power_ratio),
        origin="lower",
        aspect="auto",
        interpolation="nearest",
        extent=extent,
        cmap=error_cmap,
        norm=error_norm,
    )

    axes[0].set_title("CHAOS-7.18")
    axes[1].set_title("CHAOS-8.6")
    axes[2].set_title(
        r"$\log_{10}(P_{8.6}/P_{7.18})$"
    )

    for ax in axes:
        ax.set_xlabel(r"Frequency / $\mathrm{yr}^{-1}$")
        ax.set_xlim(extent[0], extent[1])
        ax.set_ylim(extent[2], extent[3])
        ax.set_yticks(np.arange(2, degree_max + 1, 2))

    axes[0].set_ylabel(r"Spherical harmonic degree $n$")

    power_cbar = fig.colorbar(
        chaos_7_im,
        ax=axes[:2],
        location="bottom",
        fraction=0.08,
        pad=0.13,
    )
    power_cbar.set_label(
        r"Lowes SV power spectrum at the CMB "
        r"/ $(\mathrm{nT\,yr^{-1}})^2\,\mathrm{yr}$"
    )

    error_cbar = fig.colorbar(
        ratio_im,
        ax=axes[2],
        boundaries=error_bounds,
        ticks=[
            -1,
            -factor_3,
            -factor_2,
            factor_2,
            factor_3,
            1,
        ],
    )
    error_cbar.set_ticklabels([
        r"$-1$",
        r"$-\log_{10}(3)$",
        r"$-\log_{10}(2)$",
        r"$\log_{10}(2)$",
        r"$\log_{10}(3)$",
        r"$1$",
    ])
    error_cbar.set_label(
        r"$\log_{10}(P_{8.6}/P_{7.18})$"
    )

    output_file.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_file, dpi=300, bbox_inches="tight")
    plt.show()

print(f"Saved figure to {output_file}")