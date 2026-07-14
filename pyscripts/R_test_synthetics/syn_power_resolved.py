# %% SETTING UP AUTOUPDATES

from IPython import get_ipython

ipython = get_ipython()
if ipython is not None:
    ipython.run_line_magic("load_ext", "autoreload")
    ipython.run_line_magic("autoreload", "2")

# %% FILE SYSTEM AND DEPENDENCY SETUP

# forcing root location such that the notebook can access everything 
import sys
from pathlib import Path
print(sys.path)
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import matplotlib.pyplot as plt
import h5py
import gc
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import chaosmagpy as cp
from tqdm import tqdm
from matplotlib.colors import LogNorm
from scipy.signal import periodogram
import chaosmagpy as cp

# import paths.py to establish file structure and directories
from src.msc_thesis.paths import *
# import library made for these synthetic tests
from pyscripts.R_test_synthetics.syn_pipeline import *

# Chaos data handling and psd creation
r_choice = r_cmb
chaos_file = Path(CHAOS_DIR) / "CHAOS-8.6.mat"

if not chaos_file.exists():
    raise FileNotFoundError(f"CHAOS model not found: {chaos_file}")

chaos_model = cp.load_CHAOS_matfile(str(chaos_file))

chaos_sv_gnm = chaos_model.synth_coeffs_tdep(
    times_mjd2000,
    nmax=20,
    deriv=1,
    extrapolate="off",
)

chaos_sv_gnm = np.asarray(chaos_sv_gnm)
chaos_sv_gnm = Truncate_Gauss_Coeffs(chaos_sv_gnm, 20)

chaos_good = chaos_sv_gnm[good_record_slice]

chaos_psd, chaos_f = Lowes_Degree_PSD_All_Degrees(chaos_good, a=r_earth, r=r_choice)

# getting masks (will be applied to both PSDs)
f_mask = (chaos_f > 0) #& (chaos_f < 0.2)
g_mask = (degrees > 0) & (degrees <= 20)

# getting windowed chaos psd
chaos_psd_plot = chaos_psd[:, f_mask]
chaos_psd_plot = chaos_psd_plot[g_mask, :]

# getting windowed axes
frequencies_plot = chaos_f[f_mask]
degrees_plot = degrees[g_mask]

mean_chaos_p_total = Mean_Instantaneous_Total_Lowes_Power(chaos_good, a=r_earth, r=r_choice)
mean_chaos_p_total_es = Mean_Instantaneous_Total_Lowes_Power(chaos_good, a=r_earth, r=r_earth)


# %%
mode_numbers = np.arange(1, 63, 1)
periods = []
P_scalers = []

# function to compute metrics for given derivative, stage

# function to run through baseline scaling (per mode):
for mode_number in tqdm(mode_numbers):

    # loading in mode data
    # loading in MF gauss phasor
    mode_data = Component_Load(mode_number)
    eigenvalue = mode_data["eigenvalue"]
    period = 2 * np.pi / (eigenvalue.imag)
    periods.append(period)

    file_path = Path(f"{FELIX_DIR}/R_splines_arbitrary.h5")

    with h5py.File(file_path, "r") as h5_file:

        mode_group = h5_file[f"mode_{mode_number}"]

        gnm_spl = mode_group["without_decay"][()]

    gnm_spl = np.asarray(gnm_spl)
    gnm_mode = H_sv @ gnm_spl
    
    # windowing to good satelllite record period
    gnm_mode_good = gnm_mode[good_record_slice]

    # getting mode psd at r_cmb
    mode_psd, mode_f = Lowes_Degree_PSD_All_Degrees(gnm_mode_good, a=r_earth, r=r_choice)

    mode_psd_plot = mode_psd[:, f_mask]
    mode_psd_plot = mode_psd_plot[g_mask, :]

    # finding water level height factor (metric at CMB)
    contacts = chaos_psd_plot / mode_psd_plot
    P_scale = np.min(contacts)

    P_scalers.append(P_scale)


power_lists = {
    "20": [],
    "R20": []
}

power_peaks = {
    "20": [],
    "R20": []
}

degree_peaks = {
    "20": [],
    "R20": []
}

rel_peak_powers = {
    "20": [],
    "R20": []
}

frequency_peaks = {
    "20": [],
    "R20": []
}

degree_peaks = {
    "20": [],
    "R20": []
}

peak_f_band_powers = {
    "20": [],
    "R20": []
}
def coherent_signal_metrics(input_signal, resolved_signal):
    """
    Compare a true input signal x with resolved output y.

    Both arrays must have the same shape, typically:
        (n_times, n_gauss)

    Returns:
        coherent_gain
        coherent_power_retained
        structural_fidelity
        distortion_power_relative
        output_power_relative
    """

    x = np.asarray(input_signal)
    y = np.asarray(resolved_signal)

    if x.shape != y.shape:
        raise ValueError(
            f"Shape mismatch: input {x.shape}, resolved {y.shape}"
        )

    x = x.ravel()
    y = y.ravel()

    x_power = np.vdot(x, x).real
    y_power = np.vdot(y, y).real

    if x_power == 0:
        raise ValueError("Input signal has zero power.")

    inner = np.vdot(x, y)

    # Best-fitting coherent copy of the input within the output
    alpha = inner / x_power
    y_coherent = alpha * x
    y_distortion = y - y_coherent

    coherent_power = np.vdot(
        y_coherent,
        y_coherent,
    ).real

    distortion_power = np.vdot(
        y_distortion,
        y_distortion,
    ).real

    coherent_gain = np.abs(alpha)

    coherent_power_retained = (
        coherent_power / x_power
    )

    structural_fidelity = (
        np.abs(inner) ** 2
        / (x_power * y_power)
        if y_power > 0
        else np.nan
    )

    return {
        "coherent_gain": coherent_gain,
        "coherent_power_retained": coherent_power_retained,
        "structural_fidelity": structural_fidelity,
        "distortion_power_relative": distortion_power / x_power,
        "output_power_relative": y_power / x_power,
    }

def Lowes_Coefficient_Weights(lmax, a, r):
    weights = []

    for n in range(1, lmax + 1):
        degree_weight = (
            (n + 1)
            * (a / r) ** (2 * n + 4)
        )

        # There are 2n + 1 Gauss coefficients at degree n
        weights.extend(
            [degree_weight] * (2 * n + 1)
        )

    return np.asarray(weights)
weights = Lowes_Coefficient_Weights(
    lmax=20,
    a=r_earth,
    r=r_cmb,
)

sqrt_weights = np.sqrt(weights)[None, :]

metric_list = []

for i, mode_number in tqdm(enumerate(mode_numbers)):

    # loading in MF gauss phasor
    mode_data = Component_Load(mode_number)
    
    eigenvalue = mode_data["eigenvalue"]
    period = 2 * np.pi / (eigenvalue.imag)

    # gnm handling:
    g_phasor_full_mf = mode_data["gnm"]
    g_phasor_full_sv = eigenvalue * g_phasor_full_mf.copy()

    # getting 60 and 20 degree pre-R truncations
    gnm_mode_60_sv = G_Time_Series_Eval(g_phasor_full_sv, eigenvalue)
    gnm_mode_20_sv = Truncate_Gauss_Coeffs(gnm_mode_60_sv, tmax=20)

    file_path = Path(f"{FELIX_DIR}/R_splines_arbitrary.h5")

    with h5py.File(file_path, "r") as h5_file:

        mode_group = h5_file[f"mode_{mode_number}"]

        gnm_spl = mode_group["without_decay"][()]

    gnm_spl = np.asarray(gnm_spl)
    gnm_mode_R20_sv = H_sv @ gnm_spl

    # windowing to good satelllite record period
    gnm_mode_60_sv_good = gnm_mode_60_sv[good_record_slice]
    gnm_mode_20_sv_good = gnm_mode_20_sv[good_record_slice]
    gnm_mode_R20_sv_good = gnm_mode_R20_sv[good_record_slice]

    gnm_dict = {
        "20": gnm_mode_20_sv_good,
        "R20": gnm_mode_R20_sv_good
    }

    metrics = coherent_signal_metrics(
    gnm_mode_20_sv_good * sqrt_weights,
    gnm_mode_R20_sv_good * sqrt_weights,
)
    
    metric_list.append(metrics)

    for key in gnm_dict:
        gnm_mode_good = gnm_dict[key] * np.sqrt(P_scalers[i])
        # also compute psd, find frequency and power value of peak

        mode_psd, mode_f = \
            Lowes_Degree_PSD_All_Degrees(\
                gnm_mode_good, a=r_earth, r=r_choice)
        
        
        # getting total mean instantaneous lowes power at surface and cmb
        P_total = np.sum(mode_psd)
        
        power_lists[key].append(P_total)

        
    
        # derive from truncated, apply to both
        if key == "20":

            peak_flat_idx = np.nanargmax(mode_psd)

            peak_degree_idx, peak_frequency_idx = np.unravel_index(
                peak_flat_idx,
                mode_psd.shape,
            )

            peak_degree = degrees[peak_degree_idx]
            peak_frequency = mode_f[peak_frequency_idx]


        peak_power = mode_psd[peak_degree_idx, peak_frequency_idx]

        # get power integrated along peak band
        peak_f_band_power = np.sum(mode_psd[:, peak_frequency_idx])
        
        power_peaks[key].append(np.max(mode_psd))
        frequency_peaks[key].append(peak_frequency)

        peak_power = mode_psd[peak_degree_idx, peak_frequency_idx]
        # relative peak power
        rel_peak_power = peak_power / P_total
        rel_peak_powers[key].append(rel_peak_power/ peak_f_band_power)

        degree_peaks[key].append(peak_degree)
        peak_f_band_powers[key].append(peak_f_band_power/P_total)


# %% 
for key in power_lists:
    plt.plot(periods, rel_peak_powers[key], 'x', label=key)

plt.yscale("log")
plt.legend()
plt.show()

# %% 
for key in power_lists:
    plt.plot(periods, peak_f_band_powers[key], 'x', label=key)


plt.legend()
plt.show()

# %%

c_gains = [item["coherent_power_retained"]/ \
           item["distortion_power_relative"] for item in\
           metric_list]
plt.scatter(periods, 10*np.log10(c_gains))
# %%
c_gains = [item["structural_fidelity"]for item in\
           metric_list]
plt.scatter(periods, c_gains)
# %%
c_gains = [item["coherent_gain"]for item in\
           metric_list]
plt.scatter(periods, c_gains)
# %%
# %% LOWES-WEIGHTED SIGNAL-TO-DISTORTION RATIO

mode_numbers = np.arange(1, 63)

derivative_orders = {
    "MF": 0,
    "SV": 1,
    "SA": 2,
}

periods = []
SDR = {
    "MF": [],
    "SV": [],
    "SA": [],
}


def Lowes_Coefficient_Weights(lmax, a, r):

    weights = []

    for n in range(1, lmax + 1):

        degree_weight = (
            (n + 1)
            * (a / r) ** (2 * n + 4)
        )

        weights.extend(
            [degree_weight] * (2 * n + 1)
        )

    return np.asarray(weights)


def Signal_To_Distortion_Ratio(input_signal, resolved_signal, weights):
    """
    Decompose the resolved signal as

        y = y_coherent + e,

    where y_coherent is the projection of y onto x, then calculate

        SDR = <y_coherent, y_coherent> / <e, e>.
    """

    sqrt_weights = np.sqrt(weights)[None, :]

    x = (input_signal * sqrt_weights).ravel()
    y = (resolved_signal * sqrt_weights).ravel()

    alpha = np.vdot(x, y) / np.vdot(x, x)

    y_coherent = alpha * x
    distortion = y - y_coherent

    coherent_power = np.vdot(
        y_coherent,
        y_coherent,
    ).real

    distortion_power = np.vdot(
        distortion,
        distortion,
    ).real

    return (
        coherent_power / distortion_power
        if distortion_power > 0
        else np.inf
    )


weights = Lowes_Coefficient_Weights(
    lmax=20,
    a=r_earth,
    r=r_cmb,
)

file_path = Path(
    f"{FELIX_DIR}/R_splines_arbitrary.h5"
)

with h5py.File(file_path, "r") as h5_file:

    for mode_number in tqdm(
        mode_numbers,
        desc="Computing SDR",
    ):

        mode_data = Component_Load(mode_number)

        eigenvalue = mode_data["eigenvalue"]
        g_phasor_mf = mode_data["gnm"]

        periods.append(
            2 * np.pi / np.abs(eigenvalue.imag)
        )

        gnm_spl = np.asarray(
            h5_file[
                f"mode_{mode_number}/without_decay"
            ][()]
        )

        for derivative, order in derivative_orders.items():

            # True input signal truncated from nmax=60 to nmax=20
            g_phasor = (
                eigenvalue ** order
            ) * g_phasor_mf

            gnm_60 = G_Time_Series_Eval(
                g_phasor,
                eigenvalue,
            )

            gnm_20 = Truncate_Gauss_Coeffs(
                gnm_60,
                tmax=20,
            )

            # Resolution-matrix output at nmax=20
            gnm_R20 = (
                H_dict[derivative]
                @ gnm_spl
            )

            gnm_20_good = gnm_20[
                good_record_slice
            ]

            gnm_R20_good = gnm_R20[
                good_record_slice
            ]

            SDR[derivative].append(
                Signal_To_Distortion_Ratio(
                    gnm_20_good,
                    gnm_R20_good,
                    weights,
                )
            )


periods = np.asarray(periods)

for derivative in SDR:
    SDR[derivative] = np.asarray(
        SDR[derivative]
    )

# %% PLOT SDR FOR MF, SV AND SA

period_order = np.argsort(periods)

plot_settings = {
    "MF": {
        "marker": "o",
        "label": "MF",
    },
    "SV": {
        "marker": "x",
        "label": "SV",
    },
    "SA": {
        "marker": "s",
        "label": "SA",
    },
}

with plt.rc_context({
    "font.size": 12,
    "axes.titlesize": 12,
    "axes.labelsize": 12,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "legend.fontsize": 12,
}):

    fig, ax = plt.subplots(
        figsize=(
            text_width,
            0.55 * text_width,
        ),
        constrained_layout=True,
    )

    for derivative, settings in plot_settings.items():

        ax.plot(
            periods[period_order],
            SDR[derivative][period_order],
            linestyle="none",
            marker=settings["marker"],
            markersize=6,
            markerfacecolor=(
                "none"
                if settings["marker"] != "x"
                else None
            ),
            label=settings["label"],
        )

    ax.axhline(
        1,
        color="black",
        linestyle=":",
        linewidth=1,
        label="Equal coherent and distortion power",
    )

    ax.set_xscale("linear")
    ax.set_yscale("log")

    ax.set_xlabel("Wave period [yr]")
    ax.set_ylabel(
        r"Signal-to-distortion ratio (SDR) "
    )

    ax.set_title(
        "Lowes-weighted signal preservation after resolution"
    )

    ax.grid(
        True,
        which="both",
        alpha=0.25,
    )

    ax.legend()

    fig.savefig(
        f"{FIG_DIR}/final/MF_SV_SA_signal_to_distortion_ratio.png",
        dpi=300,
        bbox_inches="tight",
    )

    plt.show()
# %%
