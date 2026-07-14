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
# mode_numbers = [ 4, 18, 23, 29, 45, 48, 49, 52, 54, 56]
mode_numbers = np.arange(1,63,1)
scaled_p = []
scaled_p_es = []
periods = []
gnm_mode_tot = np.zeros(np.shape(chaos_good))
for mode_number in mode_numbers:
    # getting real values from original felix data
    mode_data = Component_Load(mode_number)

    g_phasor_full = mode_data["gnm"]
    eigenvalue = mode_data["eigenvalue"]
    period = 2 * np.pi / (eigenvalue.imag)
    periods.append(period)

    file_path = Path(f"{FELIX_DIR}/R_splines_arbitrary.h5")

    with h5py.File(file_path, "r") as h5_file:

        mode_group = h5_file[f"mode_{mode_number}"]

        gnm_spl = mode_group["without_decay"][()]

    gnm_spl = np.asarray(gnm_spl)

    gnm_mode = H_sv @ gnm_spl

    # getting good record period
    gnm_mode_good = gnm_mode[good_record_slice]
    mode_P_total = Mean_Instantaneous_Total_Lowes_Power(gnm_mode_good, a=r_earth, r=r_choice)
    mode_P_total_es = Mean_Instantaneous_Total_Lowes_Power(gnm_mode_good, a=r_earth, r=r_earth)

    # getting mode psd
    mode_psd, mode_f = Lowes_Degree_PSD_All_Degrees(gnm_mode_good, a=r_earth, r=r_choice)

    # getting windowed mode psd
    mode_psd_plot = mode_psd[:, f_mask]
    mode_psd_plot = mode_psd_plot[g_mask, :]

    # finding water level height factor
    contacts = chaos_psd_plot / mode_psd_plot

    P_scale = np.min(contacts)

    mode_P_scaled = mode_P_total * P_scale
    scaled_p.append(mode_P_scaled)
    scaled_p_es.append(mode_P_total_es * P_scale)

    gnm_mode_tot += gnm_mode_good* np.sqrt(P_scale)

mode_psd, mode_f = Lowes_Degree_PSD_All_Degrees(gnm_mode_tot, a=r_earth, r=r_choice)
# getting windowed mode psd
mode_psd_plot = mode_psd[:, f_mask]
mode_psd_plot = mode_psd_plot[g_mask, :]
# %%

# %%
with plt.rc_context({
    "font.size": 10,
    "axes.titlesize": 10,
    "axes.labelsize": 10,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
}):

    fig, axes = plt.subplots(
        2,
        1,
        figsize=(text_width, text_width),
        constrained_layout=True,
    )

    # CMB-referenced power
    axes[0].plot(
        periods,
        scaled_p,
        linestyle="none",
        marker="x",
        color="red",
        markersize=6,
        label="Scaled wave modes",
    )

    axes[0].axhline(
        mean_chaos_p_total,
        color="black",
        linestyle=":",
        linewidth=1.2,
        label="CHAOS-8.6",
    )

    axes[0].text(
        0.03,
        mean_chaos_p_total / 1.35,
        "CHAOS-8.6",
        transform=axes[0].get_yaxis_transform(),
        ha="left",
        va="top",
        color="black",
    )

    axes[0].set_title("CMB evaluation")
    axes[0].set_xlabel("Signal period [yr]")
    axes[0].set_ylabel(
        r"Scaled mean instantaneous "+"\n"+
        r"total power [$\mathrm{nT^2\,yr^{-2}}$]"
    )
    axes[0].set_yscale("log")
    axes[0].grid(True, which="both", alpha=0.25)
    axes[0].legend()

    # Surface-referenced power
    axes[1].plot(
        periods,
        scaled_p_es,
        linestyle="none",
        marker="o",
        color="green",
        markerfacecolor="none",
        markersize=6,
        label="Scaled wave modes",
    )

    axes[1].axhline(
        mean_chaos_p_total_es,
        color="black",
        linestyle=":",
        linewidth=1.2,
        label="CHAOS-8.6",
    )

    axes[1].text(
        0.03,
        mean_chaos_p_total_es / 1.35,
        "CHAOS-8.6",
        transform=axes[1].get_yaxis_transform(),
        ha="left",
        va="top",
        color="black",
    )

    axes[1].set_title("Earth-surface evaluation")
    axes[1].set_xlabel("Signal period [yr]")
    axes[1].set_ylabel(
        r"Scaled mean instantaneous "+"\n"+
        r"total power [$\mathrm{nT^2\,yr^{-2}}$]"
    )
    axes[1].set_yscale("log")
    axes[1].grid(True, which="both", alpha=0.25)
    axes[1].legend()

    plt.savefig(
        f"{FIG_DIR}/final/mean_total_power.png",
        dpi=300,
        bbox_inches="tight",
    )

    plt.show()

# %%
f_mask_plot = (chaos_f > 0) & (chaos_f < 0.2)
g_mask_plot = (degrees > 0) & (degrees <= 9)

PSD_3D_Bar_Visualise(
    degree_psds=chaos_psd_plot,
    frequencies=frequencies_plot,
    degrees=degrees_plot,
    title=(
        "CHAOS-8.6 degree-wise Lowes-weighted secular-variation PSD\n"
        f"{times_dyear[0]:.1f}–{times_dyear[-1]:.1f}, "
        f"$n_{{max}}={20}$"
    ),
    cmap="rainbow",
    vmin=1e-2,
    vmax=1e8,
)
# %%
PSD_3D_Bar_Visualise(
    degree_psds=mode_psd_plot,
    frequencies=frequencies_plot,
    degrees=degrees_plot,
    title=(
        "All 63 modes (scaled) combined degree-wise Lowes-weighted secular-variation PSD\n"
        f"{times_dyear[0]:.1f}–{times_dyear[-1]:.1f}, "
        f"$n_{{max}}={20}$"
    ),
    cmap="rainbow",
    vmin=1e-2,
    vmax=1e8,
)

# %%
