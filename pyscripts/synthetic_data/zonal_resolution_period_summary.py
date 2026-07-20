"""
Longitude-invariant summary of how CHAOS resolution affects synthetic waves.

The script compares input and post-resolution radial secular-variation time
series for all Felix modes, averages power around each latitude circle, splits
the modes by period, and produces one four-panel figure:

1. input-wave power concentration;
2. resolved-wave power concentration;
3. zonal signal distortion ratio (SDR);
4. resolved/input power retention.

All unrelated regional maps, longitude-dependent statistics, covariance
calculations, and DMD calculations are intentionally omitted.
"""

# %% SETTING UP AUTOUPDATES

from IPython import get_ipython

ipython = get_ipython()
if ipython is not None:
    ipython.run_line_magic("load_ext", "autoreload")
    ipython.run_line_magic("autoreload", "2")


# %% FILE SYSTEM AND DEPENDENCY SETUP

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

import h5py
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

from src.msc_thesis.paths import *
from src.msc_thesis.synSetup import *
from src.msc_thesis.synUtils import *


# %% ------------------------------------------------------
# USER SETTINGS
# ---------------------------------------------------------

MODE_NUMBERS = [str(number) for number in range(1, 63)]

NMIN = 1
NMAX = 20
PERIOD_SPLIT = 10.0

# Use only the higher-quality satellite-era part of both time series.
HIGH_Q_FLAG = True

# Temporal subsampling after the optional high-quality selection.
N_SKIP = 1

# Exclude zonal ratios where a mode has almost no input signal.
SIGNAL_MASK_FRACTION = 0.01

# Latitude boundaries shown for the equatorial +/-30 degree band.
REGION_BOUNDARY = 30.0

# Interquartile range plotted around each group median.
LOWER_PERCENTILE = 25
UPPER_PERCENTILE = 75

# Optional figure output.
SAVE_FIGURE = False
FIGURE_PATH = (
    PROJECT_ROOT
    / "outputs"
    / "zonal_resolution_period_summary.png"
)


# %% ------------------------------------------------------
# ZONAL STATISTICS
# ---------------------------------------------------------

def select_time_window(array):
    """Apply identical temporal selection to input and resolved arrays."""

    if HIGH_Q_FLAG:
        array = array[good_record_slice]

    return array[::N_SKIP]


def zonal_wave_statistics(
    signal_input,
    signal_resolved,
    latitudes,
    signal_mask_fraction=0.01,
):
    """
    Calculate longitude-invariant statistics as functions of latitude.

    A zonal mean averages around all longitudes at a fixed latitude. A shift
    of a wave east or west therefore leaves these statistics unchanged.

    Parameters
    ----------
    signal_input, signal_resolved : ndarray
        Arrays with shape (nt, nlat, nlon).
    latitudes : ndarray
        Latitudes in degrees.
    signal_mask_fraction : float
        Zonal input-power values below this fraction of the maximum zonal
        input power are excluded from SDR and retention ratios.

    Returns
    -------
    dict
        Input- and resolved-power concentration, zonal SDR, and power
        retention.
    """

    signal_input = np.asarray(signal_input)
    signal_resolved = np.asarray(signal_resolved)
    latitudes = np.asarray(latitudes)

    if signal_input.shape != signal_resolved.shape:
        raise ValueError(
            "Input and resolved signals must have identical shapes: "
            f"{signal_input.shape} != {signal_resolved.shape}"
        )

    if signal_input.ndim != 3:
        raise ValueError(
            "Signals must have shape (nt, nlat, nlon)"
        )

    if signal_input.shape[1] != len(latitudes):
        raise ValueError(
            "Latitude dimension does not match the latitude array"
        )

    error = signal_resolved - signal_input

    # Mean over time and longitude; latitude is retained.
    zonal_input_power = np.mean(
        np.abs(signal_input) ** 2,
        axis=(0, 2),
    )

    zonal_resolved_power = np.mean(
        np.abs(signal_resolved) ** 2,
        axis=(0, 2),
    )

    zonal_error_power = np.mean(
        np.abs(error) ** 2,
        axis=(0, 2),
    )

    # Area weighting is required only when different latitudes are combined.
    latitude_weights = np.cos(np.deg2rad(latitudes))

    global_input_mean = np.average(
        zonal_input_power,
        weights=latitude_weights,
    )

    global_resolved_mean = np.average(
        zonal_resolved_power,
        weights=latitude_weights,
    )

    # Independent per-mode normalisation removes arbitrary mode amplitude.
    input_power_concentration = (
        zonal_input_power / global_input_mean
    )

    resolved_power_concentration = (
        zonal_resolved_power / global_resolved_mean
    )

    signal_threshold = (
        signal_mask_fraction
        * np.nanmax(zonal_input_power)
    )

    valid_signal = zonal_input_power >= signal_threshold

    zonal_sdr_db = np.full(
        len(latitudes),
        np.nan,
        dtype=np.float64,
    )

    valid_sdr = valid_signal & (zonal_error_power > 0)

    zonal_sdr_db[valid_sdr] = 10.0 * np.log10(
        zonal_input_power[valid_sdr]
        / zonal_error_power[valid_sdr]
    )

    perfect_recovery = (
        valid_signal
        & (zonal_error_power == 0)
        & (zonal_input_power > 0)
    )
    zonal_sdr_db[perfect_recovery] = np.inf

    zonal_power_retention = np.full(
        len(latitudes),
        np.nan,
        dtype=np.float64,
    )

    zonal_power_retention[valid_signal] = (
        zonal_resolved_power[valid_signal]
        / zonal_input_power[valid_signal]
    )

    return {
        "input_power_concentration": input_power_concentration,
        "resolved_power_concentration": resolved_power_concentration,
        "zonal_sdr_db": zonal_sdr_db,
        "zonal_power_retention": zonal_power_retention,
    }


def median_and_interval(
    values,
    lower_percentile=25,
    upper_percentile=75,
):
    """Median and percentile interval across modes."""

    values = np.asarray(values)

    with np.errstate(invalid="ignore"):
        median = np.nanmedian(values, axis=0)
        lower = np.nanpercentile(
            values,
            lower_percentile,
            axis=0,
        )
        upper = np.nanpercentile(
            values,
            upper_percentile,
            axis=0,
        )

    return median, lower, upper


def plot_period_group(
    ax,
    latitudes,
    values,
    mode_mask,
    color,
    label,
):
    """Plot the median and interquartile interval for one period group."""

    if not np.any(mode_mask):
        raise ValueError(f"Period group '{label}' contains no modes")

    median, lower, upper = median_and_interval(
        values[mode_mask],
        lower_percentile=LOWER_PERCENTILE,
        upper_percentile=UPPER_PERCENTILE,
    )

    ax.plot(
        latitudes,
        median,
        color=color,
        linewidth=1.8,
        label=label,
    )

    ax.fill_between(
        latitudes,
        lower,
        upper,
        color=color,
        alpha=0.18,
        linewidth=0,
    )


# %% ------------------------------------------------------
# LOAD AND COMPARE ALL MODES
# ---------------------------------------------------------

A_r_20 = A_20_dict["r"]
A_r = Truncate_Gauss_Coeffs(
    A_r_20,
    tmax=NMAX,
    tmin=NMIN,
)

expected_coefficients = (
    NMAX * (NMAX + 2)
    - (NMIN - 1) * (NMIN + 1)
)
if A_r.shape[1] != expected_coefficients:
    raise ValueError(
        f"Expected {expected_coefficients} Gauss coefficients, "
        f"but the synthesis matrix contains {A_r.shape[1]}"
    )

file_path = Path(FELIX_DIR) / "R_splines_arbitrary.h5"

periods = []
input_concentrations = []
resolved_concentrations = []
zonal_sdr_values = []
power_retentions = []

with h5py.File(file_path, "r") as h5_file:
    for mode_number in tqdm(
        MODE_NUMBERS,
        desc="Calculating zonal resolution statistics",
    ):

        mode_data = Component_Load_SV(mode_number)
        eigenvalue = mode_data["eigenvalue"]
        period = 2.0 * np.pi / np.abs(eigenvalue.imag)

        # Input radial SV grid time series.
        gnm_phasor = Truncate_Gauss_Coeffs(
            mode_data["gnm"],
            tmax=NMAX,
            tmin=NMIN,
        )

        gnm_input = G_Time_Series_Eval(
            gnm_phasor,
            eigenvalue,
        )
        gnm_input = select_time_window(gnm_input)

        sv_input_flat = A_r @ gnm_input.T
        sv_input = Series_To_Cube(sv_input_flat)

        # Post-resolution radial SV grid time series.
        gnm_splines = np.asarray(
            h5_file[f"mode_{mode_number}/without_decay"][()]
        )

        gnm_sv_resolved = H_sv @ gnm_splines
        gnm_sv_resolved = Truncate_Gauss_Coeffs(
            gnm_sv_resolved,
            tmax=NMAX,
            tmin=NMIN,
        )
        gnm_sv_resolved = select_time_window(gnm_sv_resolved)

        sv_resolved_flat = A_r @ gnm_sv_resolved.T
        sv_resolved = Series_To_Cube(sv_resolved_flat)

        results = zonal_wave_statistics(
            signal_input=sv_input,
            signal_resolved=sv_resolved,
            latitudes=latitude,
            signal_mask_fraction=SIGNAL_MASK_FRACTION,
        )

        periods.append(period)
        input_concentrations.append(
            results["input_power_concentration"]
        )
        resolved_concentrations.append(
            results["resolved_power_concentration"]
        )
        zonal_sdr_values.append(
            results["zonal_sdr_db"]
        )
        power_retentions.append(
            results["zonal_power_retention"]
        )

periods = np.asarray(periods)
input_concentrations = np.asarray(input_concentrations)
resolved_concentrations = np.asarray(resolved_concentrations)
zonal_sdr_values = np.asarray(zonal_sdr_values)
power_retentions = np.asarray(power_retentions)


# %% ------------------------------------------------------
# PERIOD-SEPARATED SUMMARY FIGURE
# ---------------------------------------------------------

short_period_mask = periods <= PERIOD_SPLIT
long_period_mask = periods > PERIOD_SPLIT

print(f"Short-period modes (T <= {PERIOD_SPLIT:g} yr): "
      f"{np.sum(short_period_mask)}")
print(f"Long-period modes (T > {PERIOD_SPLIT:g} yr): "
      f"{np.sum(long_period_mask)}")

groups = [
    (
        short_period_mask,
        "tab:blue",
        rf"Short, $T\leq{PERIOD_SPLIT:g}$ yr",
    ),
    (
        long_period_mask,
        "tab:red",
        rf"Long, $T>{PERIOD_SPLIT:g}$ yr",
    ),
]

fig, axes = plt.subplots(
    1,
    4,
    figsize=(17, 4.5),
    sharex=True,
)

for mode_mask, color, label in groups:
    plot_period_group(
        axes[0],
        latitude,
        input_concentrations,
        mode_mask,
        color,
        label,
    )

    plot_period_group(
        axes[1],
        latitude,
        resolved_concentrations,
        mode_mask,
        color,
        label,
    )

    plot_period_group(
        axes[2],
        latitude,
        zonal_sdr_values,
        mode_mask,
        color,
        label,
    )

    plot_period_group(
        axes[3],
        latitude,
        power_retentions,
        mode_mask,
        color,
        label,
    )

axes[0].axhline(
    1.0,
    color="grey",
    linestyle="--",
    linewidth=1,
)
axes[1].axhline(
    1.0,
    color="grey",
    linestyle="--",
    linewidth=1,
)
axes[2].axhline(
    0.0,
    color="grey",
    linestyle="--",
    linewidth=1,
)
axes[3].axhline(
    1.0,
    color="grey",
    linestyle="--",
    linewidth=1,
)

axes[0].set_title("Input wave distribution")
axes[1].set_title("Resolved wave distribution")
axes[2].set_title("Resolution fidelity")
axes[3].set_title("Power retention")

axes[0].set_ylabel("Input-power concentration")
axes[1].set_ylabel("Resolved-power concentration")
axes[2].set_ylabel("Zonal SDR [dB]")
axes[3].set_ylabel("Resolved/input power")

for ax in axes:
    ax.set_xlabel("Latitude [degrees]")
    ax.axvline(
        -REGION_BOUNDARY,
        color="grey",
        linestyle=":",
        linewidth=1,
    )
    ax.axvline(
        REGION_BOUNDARY,
        color="grey",
        linestyle=":",
        linewidth=1,
    )
    ax.grid(alpha=0.15)

axes[0].legend()

fig.suptitle(
    "Short- and long-period wave behaviour by latitude"
)
fig.tight_layout()

if SAVE_FIGURE:
    FIGURE_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(
        FIGURE_PATH,
        dpi=250,
        bbox_inches="tight",
    )
    print(f"Saved figure to: {FIGURE_PATH}")

plt.show()
