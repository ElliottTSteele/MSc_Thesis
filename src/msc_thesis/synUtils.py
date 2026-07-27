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
from matplotlib import cm
from matplotlib.animation import FuncAnimation
import os


# import paths.py to establish file structure and directories
from src.msc_thesis.paths import *

# import simulation setup
from src.msc_thesis.synSetup import *

# ---------------------------------------------------------
# RESOLUTION MATRIX
# ---------------------------------------------------------

# code to read in non-R (i.e. P, Hc) parts
def R_Read_In(R_part, which):

    # safe guard against R
    if which == "R":
        return "R should not be read in naively! Use RAM safe alternative."

    f = h5py.File((f'{CHAOS_RESOL_DIR}/CHAOS_Resol_1997_2026_0806_{which}.h5'),"r")

    R_sub_matrix = np.asarray(f[R_part])

    f.close()
    return R_sub_matrix

# reading in the 'cheap' matrices P and H (over all components)
derivative_keys = ["MF", "SV", "SA"]

P = R_Read_In("P", "MF")
H_mf = R_Read_In("H", "MF")
H_sv = R_Read_In("H", "SV")
H_sa = R_Read_In("H", "SA")

H_dict = {
    "MF":H_mf,
    "SV":H_sv,
    "SA":H_sa
}

# applies R in 2 halves, fine for RAM
def R_Apply_Two_Halves(gnm_spl, verbose=True):
    """
    Apply the CHAOS resolution matrix in two row halves.

    Equivalent to:

        gnm_spl_vec = gnm_spl.ravel(order="F")
        gnm_filt_spl_vec = R @ gnm_spl_vec
        gnm_filt_spl = gnm_filt_spl_vec.reshape(gnm_spl.shape, order="F")

    but only half of the rows of R are loaded into RAM at once.
    """

    gnm_spl_vec = gnm_spl.ravel(order="F")

    file_path = Path(
        f"{CHAOS_RESOL_DIR}/CHAOS_Resol_1997_2026_0806_MF.h5"
    )

    with h5py.File(file_path, "r") as f:
        R = f["R"]

        n_rows, n_cols = R.shape

        if n_cols != gnm_spl_vec.size:
            raise ValueError(
                f"Shape mismatch: R has {n_cols} columns, "
                f"but gnm_spl_vec has length {gnm_spl_vec.size}"
            )

        out_dtype = np.result_type(R.dtype, gnm_spl_vec.dtype)
        gnm_filt_spl_vec = np.empty(n_rows, dtype=out_dtype)

        midpoint = n_rows // 2

        if verbose:
            print(f"Processing rows 0:{midpoint} / {n_rows}")

        R_first = R[:midpoint, :]
        gnm_filt_spl_vec[:midpoint] = R_first @ gnm_spl_vec
        del R_first

        if verbose:
            print(f"Processing rows {midpoint}:{n_rows} / {n_rows}")

        R_second = R[midpoint:, :]
        gnm_filt_spl_vec[midpoint:] = R_second @ gnm_spl_vec
        del R_second

    return gnm_filt_spl_vec.reshape(gnm_spl.shape, order="F")
# GAUSS COEFFICIENT HANDLING CODE:
from scipy.signal import periodogram


# ---------------------------------------------------------
# GAUSS COEFFICIENT HANDLING
# ---------------------------------------------------------

# number of gauss coefficients for SH degree lmax
def n_Gauss_Coeffs(lmax):

    return lmax * (lmax + 2)


def Truncate_Gauss_Coeffs(gauss_data, tmax, tmin=1):

    gauss_data = np.asarray(gauss_data)

    # Start of degree tmin in a vector beginning at n=1
    start_index = tmin**2 - 1

    # Exclusive endpoint immediately after degree tmax
    stop_index = (tmax + 1)**2 - 1

    if gauss_data.shape[-1] < stop_index:
        raise ValueError(
            f"Input only has {gauss_data.shape[-1]} coefficients, "
            f"but tmax={tmax} requires at least {stop_index}."
        )

    return gauss_data[..., start_index:stop_index]


# infer maximum spherical harmonic degree from number of coefficients
def Gauss_Lmax(gauss_data):

    n_coeffs = np.asarray(gauss_data).shape[-1]
    lmax = int(np.sqrt(n_coeffs + 1) - 1)

    if n_Gauss_Coeffs(lmax) != n_coeffs:
        raise ValueError(
            f"{n_coeffs} is not a complete set of Gauss coefficients."
        )

    return lmax


# return the coefficient slice corresponding to spherical harmonic degree n
def Gauss_Degree_Slice(n):

    if n < 1:
        raise ValueError("Spherical harmonic degree must be at least 1.")

    return slice(n**2 - 1, (n + 1)**2 - 1)


# calculate the Lowes weighting for spherical harmonic degree n
def Lowes_Degree_Weight(n, a=r_earth, r=r_cmb):

    return (n + 1) * (a / r)**(2*n + 4)


# code to calculate instantaneous Lowes power for a degree from g(t_step=k)
def Instantaneous_Lowes_Degree_Power(
    gauss_step,
    n,
    a=r_earth,
    r=r_cmb,
):

    gauss_step = np.asarray(gauss_step)

    if gauss_step.ndim != 1:
        raise ValueError("gauss_step must have shape (ng,).")

    degree_slice = Gauss_Degree_Slice(n)

    if degree_slice.stop > gauss_step.size:
        raise ValueError(
            f"Input does not contain coefficients up to degree n={n}."
        )

    degree_coeffs = gauss_step[degree_slice]
    degree_weight = Lowes_Degree_Weight(n, a=a, r=r)

    return degree_weight * np.sum(np.abs(degree_coeffs)**2)


# code to calculate instantaneous Lowes spectrum from g(t_step=k)
def Instantaneous_Lowes_Spectrum(
    gauss_step,
    a=r_earth,
    r=r_cmb,
):

    gauss_step = np.asarray(gauss_step)
    lmax = Gauss_Lmax(gauss_step)

    return np.array([
        Instantaneous_Lowes_Degree_Power(
            gauss_step,
            n,
            a=a,
            r=r,
        )
        for n in range(1, lmax + 1)
    ])


# code to calculate instantaneous total Lowes power from g(t_step=k)
def Instantaneous_Total_Lowes_Power(
    gauss_step,
    a=r_earth,
    r=r_cmb,
):

    return np.sum(
        Instantaneous_Lowes_Spectrum(
            gauss_step,
            a=a,
            r=r,
        )
    )


# intakes a time series g = [g(0), ..., g(t_step=k), ..., g(t_step=K)]^T of gauss
# coefficients (nt, ng) and returns mean instantaneous total Lowes power
def Mean_Instantaneous_Total_Lowes_Power(
    gauss_data,
    a=r_earth,
    r=r_cmb,
):

    gauss_data = np.asarray(gauss_data)

    if gauss_data.ndim != 2:
        raise ValueError("gauss_data must have shape (nt, ng).")

    total_power = np.array([
        Instantaneous_Total_Lowes_Power(
            gauss_step,
            a=a,
            r=r,
        )
        for gauss_step in gauss_data
    ])

    return np.mean(total_power)


# intakes g_i(t_step) time series and computes PSD s_i(f)
def Gauss_Coeff_PSD(
    gauss_coeff_series,
    fs,
    detrend="linear",
    window="hann",
    scaling="spectrum",
):

    gauss_coeff_series = np.asarray(gauss_coeff_series)

    frequencies, psd = periodogram(
        gauss_coeff_series,
        fs=fs,
        detrend=detrend,
        window=window,
        scaling=scaling,
        axis=0,
    )

    return frequencies, psd


# computes Lowes-weighted PSD at degree n for g(t_step) (nt, ng)
def Lowes_Degree_PSD(
    gauss_data,
    n,
    fs,
    a=r_earth,
    r=r_cmb,
    detrend="linear",
    window="hann",
    scaling="density",
):

    gauss_data = np.asarray(gauss_data)

    if gauss_data.ndim != 2:
        raise ValueError("gauss_data must have shape (nt, ng).")

    degree_slice = Gauss_Degree_Slice(n)

    if degree_slice.stop > gauss_data.shape[1]:
        raise ValueError(
            f"Input does not contain coefficients up to degree n={n}."
        )

    frequencies, coeff_psds = Gauss_Coeff_PSD(
        gauss_data[:, degree_slice],
        fs=fs,
        detrend=detrend,
        window=window,
        scaling=scaling,
    )

    degree_weight = Lowes_Degree_Weight(n, a=a, r=r)
    degree_psd = degree_weight * np.sum(coeff_psds, axis=1)

    return frequencies, degree_psd

degrees = np.arange(1, 21)
def Lowes_Degree_PSD_All_Degrees(gnm, a, r, f_sample=1/dt_years, degrees=degrees):

    degree_psds = []

    if np.shape(gnm)[1] == 3720:
        degrees=np.arange(1, 61)

    for n in degrees:

        frequencies, degree_psd = Lowes_Degree_PSD(
            gnm,
            n=n,
            fs=f_sample,
            a=a,
            r=r,
            detrend="constant",
            window="hann",
            scaling="spectrum",
        )

        degree_psds.append(degree_psd)

    # Shape: (n_degrees, n_frequencies)
    degree_psds = np.stack(
        degree_psds,
        axis=0,
    )

    return degree_psds, frequencies

# ---------------------------------------------------------
# GAUSS COEFFICIENT -> GRID CONVERSION
# ---------------------------------------------------------


# for projecting gauss coefficients onto gridded data form
# these need to be in degrees for chaosmagpy
theta_v = np.rad2deg(theta_grid.copy().ravel())
phi_v = np.rad2deg(phi_grid.copy().ravel())
radius_v = np.full(len(phi_v), r_cmb)

# gauss -> physical forward operators
# using nmax = 20
A_r, A_t, A_p = cp.model_utils.design_gauss(
    radius_v, theta_v, phi_v, nmax=20, source="internal"
)

A_20_dict = {"r": A_r, "theta": A_t, "phi": A_p}


# ---------------------------------------------------------
# LOADING SYNTHETIC DATA
# ---------------------------------------------------------

# basic mode i loading code - assumes SV wanted
def Component_Load_SV(mode_number, directory=FELIX_DIR):
    # select a mode number and corresponding file
    file = h5py.File(f'{directory}/mode_surface_including_gnm_{mode_number}.h5',"r")

    # Converts V_alfven (arbitrary) to nT (arbitrary)
    va_to_nt_arbitrary = (v_alfven * np.sqrt(mu_0 * rho) * 1e9) 

    # load all components, transpose to lat, long

    # gauss coefficient (magnetic scalar potential) phasors
    gnm = va_to_nt_arbitrary * (np.asarray(file["gnmr"]) +\
         1j*np.asarray(file["gnmi"])) # [nT, arbitrary]

    # Br
    br = va_to_nt_arbitrary * (np.asarray(file["brr"]).T +\
         1j*np.asarray(file["bri"]).T) # [nT, arbitrary]

    # u_theta
    utheta = np.asarray(file["uthetar"]).T +\
        1j*np.asarray(file["uthetai"]).T # [V_alfven, arbitrary]
    # u_phi
    uphi = np.asarray(file["uphir"]).T +\
        1j*np.asarray(file["uphii"]).T # [V_alfven, arbitrary]

    # loading in true period and decay rate
    omega = file["omega"][()] # angular frequency (rad/year)
    sigma = file["sigma"][()] # annual decay rate (fractional decay/year)

    eigenvalue_decaying = sigma + 1j * omega # with decay

    eigenvalue = 0 + 1j * omega # storing as eigenvalue - NO DECAY

    file.close() # close file

    mode_i_info = {
        "mode_number": mode_number,
        "gnm": eigenvalue * gnm,
        "sv": eigenvalue * br,
        "utheta": utheta,
        "uphi": uphi,
        "eigenvalue": eigenvalue,
        "eigenvalue_decaying": eigenvalue_decaying
    }

    # returns all basic components for the mode
    return mode_i_info

# function to create time series from gauss coefficient phasor
def G_Time_Series_Eval(G_mode_i, eigenvalue):

    mode_i_contribution_list = []
    
    for t in times_dyear:
        # computing at time t
        G_t_mode_i = np.real(np.exp(eigenvalue * t) * G_mode_i)
        G_t_mode_i = np.ravel(G_t_mode_i) # state vector
        mode_i_contribution_list.append(G_t_mode_i)
        
    mode_i_contribution_array = np.vstack(mode_i_contribution_list)

    return mode_i_contribution_array

# ---------------------------------------------------------
# GENERIC VISUALISATION CODE
# ---------------------------------------------------------



# converts a state vector series array ((ntheta*nphi), nt) into a cube (nt, ntheta, nphi)
def Series_To_Cube(series, state_shape=state_shape):

    return series.T.reshape((series.shape[1], *state_shape))

# basic movie for data cube
def Cube_Movie(series, name="data_cube", fig_dir=FIG_DIR, fps=5, cmap="seismic"):
    """
    data_cube shape: (nt, nlat, nlon)
    """
    if series.ndim == 2: # convet to cube if ravelled state series
        cube = Series_To_Cube(series)
    else:
        cube = series

    os.makedirs(fig_dir, exist_ok=True)

    nt = cube.shape[0]

    vmax = np.nanmax(np.abs(cube))
    vmin = -vmax

    fig, ax = plt.subplots(figsize=(10, 5), dpi=150)

    im = ax.imshow(
        cube[0],
        cmap=cmap,
        origin="upper",
        extent=[0, 360, 180, 0],
        aspect="auto",
        vmin=vmin,
        vmax=vmax
    )

    ax.set_xlabel("Longitude [°]")
    ax.set_ylabel("Colatitude [°]")

    title = ax.set_title(f"{name} | frame 0/{nt-1}")

    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label(name)

    def update(frame):
        im.set_data(cube[frame])
        title.set_text(f"{name} | frame {frame}/{nt-1}")
        return im, title

    anim = FuncAnimation(
        fig,
        update,
        frames=nt,
        interval=1000 / fps,
        blit=True
    )

    filename = os.path.join(fig_dir, f"{name}.mp4")
    anim.save(filename, fps=fps, dpi=150)

    plt.close(fig)

    return filename