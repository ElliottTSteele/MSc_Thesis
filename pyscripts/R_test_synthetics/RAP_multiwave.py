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
r_choice = r_cmb
chaos_good = chaos_sv_gnm[good_record_slice]

chaos_psd, chaos_f = Lowes_Degree_PSD_All_Degrees(chaos_good, a=r_earth, r=r_choice)

# getting masks (will be applied to both PSDs)
f_mask = (chaos_f > 0) & (chaos_f < 0.5)
g_mask = (degrees > 0) & (degrees <= 21)

# getting windowed chaos psd
chaos_psd_plot = chaos_psd[:, f_mask]
chaos_psd_plot = chaos_psd_plot[g_mask, :]

# getting windowed axes
frequencies_plot = chaos_f[f_mask]
degrees_plot = degrees[g_mask]

mean_chaos_p_total = Mean_Instantaneous_Total_Lowes_Power(chaos_good, a=r_earth, r=r_choice)

# %%
mode_numbers = np.arange(1,63, 1)
scaled_p = []
contact_idx = []
periods = []
accept_counter = 0
gnm_input_total = np.zeros(np.shape(chaos_good))
linear_psd_sum = np.zeros(np.shape(chaos_psd))

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

    # getting mode psd
    mode_psd, mode_f = Lowes_Degree_PSD_All_Degrees(gnm_mode_good, a=r_earth, r=r_choice)

    # getting windowed mode psd
    mode_psd_plot = mode_psd[:, f_mask]
    mode_psd_plot = mode_psd_plot[g_mask, :]

    # finding water level height factor
    contacts = chaos_psd_plot / mode_psd_plot

    P_scale = np.min(contacts)
    P_loc = np.argmin(contacts)

    contact_idx.append(P_loc)
    accept_counter += 1

    mode_P_scaled = mode_P_total * P_scale
    scaled_p.append(mode_P_scaled)

    gnm_input_total += gnm_mode_good * np.sqrt(P_scale)
    linear_psd_sum += Lowes_Degree_PSD_All_Degrees(np.sqrt(P_scale)*gnm_mode_good, a=r_earth, r=r_choice)[0]

# %%
plt.scatter(periods, contact_idx)

# %%

cumulative_mode_psd, _ = Lowes_Degree_PSD_All_Degrees(gnm_input_total, a=r_earth, r=r_choice)

# %%
cumulative_mode_psd_plot = linear_psd_sum 

eps = 1e-20

log_ratio = np.log10(
    (cumulative_mode_psd_plot[:,f_mask] + eps)
    / (chaos_psd_plot + eps)
)


fig, ax = plt.subplots(figsize=(10, 6))

limit = np.nanmax(np.abs(log_ratio))

mesh = ax.pcolormesh(
    frequencies_plot,
    degrees_plot,
    log_ratio,
    shading="auto",
    cmap="seismic",
    vmin=-1,
    vmax=1,
)
for ridx in contact_idx:
    i, j = np.unravel_index(ridx, chaos_psd_plot.shape)
    ax.scatter(frequencies_plot[i], degrees_plot[j], alpha=0.1, color="green")

ax.set_xlabel("Frequency [yr$^{-1}$]")
ax.set_ylabel("Spherical-harmonic degree $n$")

cbar = fig.colorbar(mesh, ax=ax)
cbar.set_label(
    r"$\log_{10}(S_{\mathrm{synthetic}}/S_{\mathrm{CHAOS}})$"
)

plt.tight_layout()
plt.show()

# %%

plt.plot(periods, scaled_p, 'x')
plt.axhline(mean_chaos_p_total)
plt.yscale("log")
plt.show()  

# %%
# establishing a tighter plotting scale to demonstrate good matching 
# in reliable R recovery zone

d_clip_mask = (degrees_plot > 0) & (degrees_plot < 21)
f_clip_mask = (frequencies_plot > 0) & (frequencies_plot < 0.5)

def PSD_Clipper(psd, d=d_clip_mask, f=f_clip_mask):
    psd = psd[:, f]
    psd = psd[d, :]

    return(psd)


mode_clip = PSD_Clipper(cumulative_mode_psd_plot)
chaos_clip = PSD_Clipper(chaos_psd_plot)
f_clip = frequencies_plot[f_clip_mask]
d_clip = degrees_plot[d_clip_mask]

# %%

PSD_3D_Bar_Visualise(
    degree_psds=chaos_clip,
    frequencies=f_clip,
    degrees=d_clip,
    title=(
        "CHAOS-8.6 degree-wise Lowes-weighted secular-variation PSD\n"
        f"{times_dyear[0]:.1f}–{times_dyear[-1]:.1f}, "
        f"$n_{{max}}={20}$"
    ),
    cmap="rainbow",
    vmin=1e-2,
    vmax=1e3,
)
# %%
PSD_3D_Bar_Visualise(
    degree_psds=mode_clip,
    frequencies=f_clip,
    degrees=d_clip,
    title=(
        "CHAOS-8.6 degree-wise Lowes-weighted secular-variation PSD\n"
        f"{times_dyear[0]:.1f}–{times_dyear[-1]:.1f}, "
        f"$n_{{max}}={20}$"
    ),
    cmap="rainbow",
    vmin=1e-2,
    vmax=1e3,
)

# %%
# %%
eps = 1e-20

log_ratio = np.log10(
    (cumulative_mode_psd_plot + eps)
    / (chaos_psd_plot + eps)
)

log_ratio_clip = log_ratio[
    np.ix_(d_clip_mask, f_clip_mask)
]
fig, ax = plt.subplots(figsize=(10, 6))

limit = np.nanmax(np.abs(log_ratio_clip))

mesh = ax.pcolormesh(
    frequencies_plot[f_clip_mask],
    degrees_plot[d_clip_mask],
    log_ratio_clip,
    shading="auto",
    cmap="seismic",
    vmin=-1,
    vmax=1,
)

ax.set_xlabel("Frequency [yr$^{-1}$]")
ax.set_ylabel("Spherical-harmonic degree $n$")

cbar = fig.colorbar(mesh, ax=ax)
cbar.set_label(
    r"$\log_{10}(S_{\mathrm{synthetic}}/S_{\mathrm{CHAOS}})$"
)

plt.tight_layout()
plt.show()
# %%
PSD_3D_Bar_Visualise(
    degree_psds=chaos_clip,
    frequencies=f_clip,
    degrees=d_clip,
    title=(
        "CHAOS-8.6 degree-wise Lowes-weighted secular-variation PSD\n"
        f"{times_dyear[0]:.1f}–{times_dyear[-1]:.1f}, "
        f"$n_{{max}}={20}$"
    ),
    cmap="rainbow",
    vmin=1e-2,
    vmax=1e3,
)
# %%
