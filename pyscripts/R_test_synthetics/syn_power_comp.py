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
    "60":[],
    "20": [],
    "R20": []
}

power_peaks = {
    "60":[],
    "20": [],
    "R20": []
}

frequency_peaks = {
    "60":[],
    "20": [],
    "R20": []
}

degree_peaks = {
    "60":[],
    "20": [],
    "R20": []
}

peak_f_band_powers = {
    "60":[],
    "20": [],
    "R20": []
}

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
        "60":gnm_mode_60_sv_good,
        "20": gnm_mode_20_sv_good,
        "R20": gnm_mode_R20_sv_good
    }

    for key in gnm_dict:
        gnm_mode_good = gnm_dict[key] * np.sqrt(P_scalers[i])

        # getting total mean instantaneous lowes power at surface and cmb
        P_total = Mean_Instantaneous_Total_Lowes_Power(\
            gnm_mode_good, a=r_earth, r=r_choice)
        
        power_lists[key].append(P_total)

        # also compute psd, find frequency and power value of peak

        mode_psd, mode_f = \
            Lowes_Degree_PSD_All_Degrees(\
                gnm_mode_good, a=r_earth, r=r_choice)
        
    

        peak_flat_idx = np.nanargmax(mode_psd)

        peak_degree_idx, peak_frequency_idx = np.unravel_index(
            peak_flat_idx,
            mode_psd.shape,
        )
        

        if key == "60":

            peak_degree = np.arange(1, 61)[peak_degree_idx]
        else:
            peak_degree = degrees[peak_degree_idx]

        peak_frequency = mode_f[peak_frequency_idx]
        peak_power = mode_psd[peak_degree_idx, peak_frequency_idx]

        # get power integrated along peak band
        peak_f_band_power = np.sum(mode_psd[:, peak_frequency_idx])
        
        power_peaks[key].append(np.max(mode_psd))
        frequency_peaks[key].append(peak_frequency)
        peak_f_band_powers[key].append(peak_f_band_power/P_total)
        degree_peaks[key].append(peak_degree)
# %%

for key in power_lists:
    plt.plot(periods, degree_peaks[key], 'x', label=key)


plt.legend()
plt.show()

# %% 
for key in power_lists:
    plt.plot(periods, power_lists[key], 'x', label=key)

plt.yscale("log")
plt.legend()
plt.show()

# %%
for key in power_lists:
    plt.plot(periods, power_peaks[key], 'x', label=key)
    
plt.yscale("log")
plt.legend()
plt.show()

# %%
for key in power_lists:
    plt.plot(periods, peak_f_band_powers[key], 'x', label=key)
    
plt.legend()
plt.show()
# %%
for key in power_lists:
    plt.plot(periods, np.asarray(degree_peaks[key]), 'x', label=key)
    
plt.legend()
plt.show()
# %%
# %% MF TRUNCATION POWER AND PEAK-DEGREE DIAGNOSTICS

mode_numbers = np.arange(1, 63)

periods = []
truncation_power_fraction = []

degree_peaks = {
    "60": [],
    "20": [],
    "R20": [],
}

r_choice = r_cmb
file_path = Path(FELIX_DIR) / "R_splines_arbitrary.h5"

with h5py.File(file_path, "r") as h5_file:

    for mode_number in tqdm(
        mode_numbers,
        desc="Computing MF diagnostics",
    ):

        mode_data = Component_Load(mode_number)

        eigenvalue = mode_data["eigenvalue"]
        g_phasor_mf = mode_data["gnm"]

        periods.append(
            2 * np.pi / np.abs(eigenvalue.imag)
        )

        # Original nmax=60 MF time series
        gnm_mode_60 = G_Time_Series_Eval(
            g_phasor_mf,
            eigenvalue,
        )

        # MF truncated to nmax=20 before applying resolution
        gnm_mode_20 = Truncate_Gauss_Coeffs(
            gnm_mode_60,
            tmax=20,
        )

        # Resolved nmax=20 MF time series
        gnm_spl = np.asarray(
            h5_file[
                f"mode_{mode_number}/without_decay"
            ][()]
        )

        gnm_mode_R20 = H_mf @ gnm_spl

        # Use the reliable satellite-record interval
        gnm_mode_60 = gnm_mode_60[good_record_slice]
        gnm_mode_20 = gnm_mode_20[good_record_slice]
        gnm_mode_R20 = gnm_mode_R20[good_record_slice]

        # Fraction of original MF power retained by truncation
        power_60 = Mean_Instantaneous_Total_Lowes_Power(
            gnm_mode_60,
            a=r_earth,
            r=r_choice,
        )

        power_20 = Mean_Instantaneous_Total_Lowes_Power(
            gnm_mode_20,
            a=r_earth,
            r=r_choice,
        )

        truncation_power_fraction.append(
            power_20 / power_60
        )

        # Degree containing the largest power-spectrum value
        gnm_stages = {
            "60": gnm_mode_60,
            "20": gnm_mode_20,
            "R20": gnm_mode_R20,
        }

        for stage, gnm_mode in gnm_stages.items():

            mode_spectrum, mode_f = Lowes_Degree_PSD_All_Degrees(
                gnm_mode,
                a=r_earth,
                r=r_choice,
            )

            peak_degree_idx, peak_frequency_idx = np.unravel_index(
                np.nanargmax(mode_spectrum),
                mode_spectrum.shape,
            )

            degree_peaks[stage].append(
                peak_degree_idx + 1
            )


periods = np.asarray(periods)
truncation_power_fraction = np.asarray(
    truncation_power_fraction
)

for stage in degree_peaks:
    degree_peaks[stage] = np.asarray(
        degree_peaks[stage]
    )

period_order = np.argsort(periods)

# %% COMBINED FIGURE

with plt.rc_context({
    "font.size": 12,
    "axes.titlesize": 12,
    "axes.labelsize": 12,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "legend.fontsize": 10,
}):

    fig, axes = plt.subplots(
        2,
        1,
        figsize=(text_width, text_width),
        constrained_layout=True,
    )

    # (a) MF power retained after truncation
    axes[0].plot(
        periods[period_order],
        truncation_power_fraction[period_order],
        linestyle="none",
        marker="o",
        markerfacecolor="none",
        markersize=6,
        label=r"$P_{20}/P_{60}$",
    )

    axes[0].axhline(
        1,
        color="black",
        linestyle=":",
        linewidth=1.2,
        label="No power removed",
    )

    axes[0].set_xscale("linear")
    axes[0].set_xlabel("Signal period [yr]")
    axes[0].set_ylabel("MF power fraction retained")
    axes[0].set_title("Power retained after truncation to $n=20$")
    axes[0].grid(True, which="both", alpha=0.25)
    axes[0].legend()

    axes[0].text(
        0.02,
        0.96,
        "(a)",
        transform=axes[0].transAxes,
        ha="left",
        va="top",
        fontweight="bold",
    )

    # (b) Degree of global power-spectrum maximum
    axes[1].plot(
        periods[period_order],
        degree_peaks["60"][period_order],
        linestyle="none",
        marker="o",
        markerfacecolor="none",
        markersize=6,
        label=r"Original, $n_{\max}=60$",
    )

    axes[1].plot(
        periods[period_order],
        degree_peaks["20"][period_order],
        linestyle="none",
        marker="s",
        markerfacecolor="none",
        markersize=6,
        label=r"Truncated, $n_{\max}=20$",
    )

    axes[1].plot(
        periods[period_order],
        degree_peaks["R20"][period_order],
        linestyle="none",
        marker="x",
        markersize=6,
        label=r"Resolved, $n_{\max}=20$",
    )

    axes[1].set_xscale("linear")
    axes[1].set_xlabel("Signal period [yr]")
    axes[1].set_ylabel("Peak power-spectrum degree")
    axes[1].set_title("Degree of maximum MF spectral power")
    axes[1].set_yticks(np.arange(0, 61, 10))
    axes[1].grid(True, which="both", alpha=0.25)
    axes[1].legend()

    axes[1].text(
        0.02,
        0.96,
        "(b)",
        transform=axes[1].transAxes,
        ha="left",
        va="top",
        fontweight="bold",
    )

    fig.savefig(
        f"{FIG_DIR}/final/MF_truncation_power_and_peak_degree.png",
        dpi=300,
        bbox_inches="tight",
    )

    plt.show()
# %%
