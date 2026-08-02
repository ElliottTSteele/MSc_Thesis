'''
synthetic_loss.py

Purpose: To examine the effects of various transforms on signal loss

notes:
- aims to cover:
    - physical to gauss coefficient N-truncated
    - gauss coefficient to B-spline, i.e. is P lossless
'''

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
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import chaosmagpy as cp

# import paths.py to establish file structure and directories
from src.msc_thesis.paths import *

# %% code to read in a synthetic data set

# preamble code that allows conversion from v_a to physical units
seconds_in_year = 365.25 * 24 * 60 * 60
seconds_in_day = 24 * 60 * 60

# known quantities
rho = 1e4 # [kg/m3] mass density of fluid outer core
mu_0 = 4 * np.pi * 1e-7 # [N/A2] magnetic permeability of free space
Le=2e-4 # [unitless] Lehnert number (ration of rotation time over Alfvén time) used in calculations
L=3845 # [km] radius of earth's core
t_omega=1/(2*np.pi) # [days] earth's 'rotation time' 

# function to compute value of Va
def Alfven_Velocity_Compute(
    Le, #Lehnert number (ration of rotation time over Alfvén time) used in calculations
    L, # radius of earth's core (km)
    t_omega # earth's 'rotation time' (days/ rad)
):
    # want output in [metres/seconds]
    # therefore must convert all units to SI
    L *= 1000 # km -> m
    t_omega *= seconds_in_day # days -> hours -> minutes -> seconds
    
    return((Le * L) / t_omega) # units of [m/s]

# compute Alfven Velocity
v_alfven = Alfven_Velocity_Compute(Le, L, t_omega) # [m/s]

# %% Defining the spatial parameters

radius = 3485 # km radius at CMB

state_shape = (181, 360) # snapshot shape used globally
n_points_globally = state_shape[0]*state_shape[1] # number of points in grid

# all data uses 1^o evenly spaced grid, incl poles
# latitude, colatitude, longitude (degrees):
colatitude = np.arange(0, 181, 1)
latitude = colatitude - 90
longitude = np.arange(0, 360, 1)
# theta, phi (radians):
theta = np.deg2rad(colatitude)
phi = np.deg2rad(longitude)

# latitude weighting variables
phi_grid, theta_grid = np.meshgrid(phi, theta)
W_theta = np.sin(theta)
W2D = np.sin(theta_grid)
W2D_norm = W2D / np.sum(W2D)

# max spherical harmonic degree provided by Felix - 60 as of 03/07/2026
lmax_felix_provided = 60


# %% defining time parameters

# note: times relative to start of synthetic time series (i.e. t=0)
t_start = 0.0 # start time
t_end = 29.5 # end time
t_record = t_end - t_start # record length
f_bin = 1 / t_record # associatied fourier resolution

dt_years = 0.2   # 6-month sampling (same as CHAOS independent information)
f_sample = 1 / dt_years    # samples per year

# getting decimal year and julian date formats
times_dyear = np.arange(t_start, t_end, dt_years)
n_times = int(len(times_dyear)) # number of total samples
print(n_times) # needs to be 148 for resolution matrix compatibility 
print(times_dyear[-1])
# choosing if decay on or not (globally)
decay_flag = True


# %% function to load in all components for given mode
def Component_Load(mode_number, directory=FELIX_DIR):
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
    eigenval = sigma + 1j * omega # storing as eigenvalue

    file.close() # close file

    mode_i_info = {
        "mode_number": mode_number,
        "gnm": gnm,
        "br": br,
        "utheta": utheta,
        "uphi": uphi,
        "eigenvalue": eigenval
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
    
# %% Code to read in the resolution matrix data

def R_Read_In(R_part, which):
    f = h5py.File((f'{CHAOS_RESOL_DIR}/CHAOS_Resol_1997_2026_0806_{which}.h5'),"r")

    R_sub_matrix = np.asarray(f[R_part])

    f.close()
    return R_sub_matrix

# %% Defining functions to transform between physical and gauss

# these need to be in degrees for chaosmagpy
theta_v = np.rad2deg(theta_grid.copy().ravel())
phi_v = np.rad2deg(phi_grid.copy().ravel())
radius_v = np.full(len(phi_v), radius)

# gauss -> physical forward operators
# using nmax = 20
A_r, A_t, A_p = cp.model_utils.design_gauss(
    radius_v, theta_v, phi_v, nmax=20, source="internal"
)

A_20_dict = {"r": A_r, "theta": A_t, "phi": A_p}

# given a maximum spherical harmonic degree lmax
# this function returns the number of gauss coefficients
def n_Gauss_Coeffs(lmax):

    return lmax * (lmax + 2)

# provided an array containing gauss coefficient data
# where shape(array)[-1] = number of gauss coefficients
# this function truncates to degree tmax
def Truncate_Gauss_Coeffs(gauss_data, tmax):

    gauss_data = np.asarray(gauss_data)

    n_keep = n_Gauss_Coeffs(tmax)

    if gauss_data.shape[-1] < n_keep:
        raise ValueError(
            f"Input only has {gauss_data.shape[-1]} coefficients, "
            f"but tmax={tmax} requires {n_keep}."
        )

    return gauss_data[..., :n_keep]

# %% comparing simple loss between 2 arrays
def Loss_Compare(data1, data2, name):
    fig, axes = plt.subplots(1,
                            3,
                            figsize=(10, 4),
                            subplot_kw={'projection': ccrs.EqualEarth()},
                            constrained_layout=True)


    # Use common colour limits so the colourbar is meaningful for both plots
    vmin = min(data1.min(), data2.min())
    vmax = max(data1.max(), data2.max())

    im0 = axes[0].imshow(data1, origin="lower", aspect="auto",
                        vmin=vmin, vmax=vmax,
                        transform=ccrs.PlateCarree())
    axes[0].set_title("Data 1")
    axes[0].set_xlabel("x")
    axes[0].set_ylabel("y")

    im1 = axes[1].imshow(data2, origin="lower", aspect="auto",
                        vmin=vmin, vmax=vmax,
                        transform=ccrs.PlateCarree())
    axes[1].set_title("Data 2")
    axes[1].set_xlabel("x")
    axes[1].set_ylabel("y")

    im1 = axes[2].imshow(data1-data2, origin="lower", aspect="auto",
                        vmin=vmin, vmax=vmax,
                        transform=ccrs.PlateCarree())
    axes[2].set_title("Data 2")
    axes[2].set_xlabel("x")
    axes[2].set_ylabel("y")

    # Shared colourbar for both subplots
    cbar = fig.colorbar(im1, ax=axes, shrink=0.9)
    cbar.set_label("Value")

    fig.savefig(f"{FIG_DIR}/{name}.png",
                dpi=300,
                bbox_inches="tight")
    plt.show()

# %% Examining the shapes of the provided wave data
mode_numbers = [1, 20, 40, 62]

# pulling in the small (i.e. non R) matrices from resolution data
P_res = R_Read_In("P", "MF")
H_res_mf = R_Read_In("H", "MF")
H_res_sv = R_Read_In("H", "SV")
H_res_sa = R_Read_In("H", "SA")

for mode_number in mode_numbers:
    mode_i_info = Component_Load(mode_number, FELIX_DIR)
    eigenvalue_i = mode_i_info["eigenvalue"]

    # testing physical coefficient vs truncated gauss

    gnm_t60 = mode_i_info["gnm"] # [nT, arbitrary]
    gnm_t20 = Truncate_Gauss_Coeffs(gnm_t60, tmax=20)
    A_t20 = A_20_dict["r"] # forward operator for nmax=20

    br_t20 = A_t20 @ gnm_t20 # [nT/yr, arbitrary]

    # testing truncated gauss vs truncated gauss after B-spline
    # have N=20 phasor for wave (gnm_t20), need time series
    G_i_time_series = G_Time_Series_Eval(gnm_t20, eigenvalue_i)
    Gsv_i_time_series = G_Time_Series_Eval(eigenvalue_i * gnm_t20, eigenvalue_i)
    Gsa_i_time_series = G_Time_Series_Eval(eigenvalue_i**2 * gnm_t20, eigenvalue_i)
    

    # now projecting onto B-spline
    B_spline_i = P_res @ G_i_time_series

    # now retreiving back (no derivative applied)
    G_i_times_series_post_spline = H_res_mf @ B_spline_i
    Gsv_i_times_series_post_spline = H_res_sv @ B_spline_i
    Gsa_i_times_series_post_spline = H_res_sa @ B_spline_i

    # getting the first time point of both as real evaluated field
    Br_G_no_spline = (A_t20 @ G_i_time_series[140,:]).reshape(state_shape)
    SV_G_no_spline = (A_t20 @ Gsv_i_time_series[140,:]).reshape(state_shape)
    SA_G_no_spline = (A_t20 @ Gsa_i_time_series[140,:]).reshape(state_shape)
    Br_G_yes_spline = (A_t20 @ G_i_times_series_post_spline[140,:]).reshape(state_shape)
    SV_G_yes_spline = (A_t20 @ Gsv_i_times_series_post_spline[140,:]).reshape(state_shape)
    SA_G_yes_spline = (A_t20 @ Gsa_i_times_series_post_spline[140,:]).reshape(state_shape)


    Loss_Compare(np.real(mode_i_info["br"]), 
                np.real(br_t20.reshape(state_shape)), 
                f"mode_{mode_number}_comparison_phys_v_gauss")
    
    # comparing the MF loss
    Loss_Compare(Br_G_no_spline, 
                Br_G_yes_spline, 
                f"mode_{mode_number}_comparison_gauss_ba_spline")
    
    # comparing the SV loss
    Loss_Compare(SV_G_no_spline, 
                SV_G_yes_spline, 
                f"mode_{mode_number}_comparison_gauss_sv_spline")
    
    # comparing the SA loss
    Loss_Compare(SA_G_no_spline, 
                SA_G_yes_spline, 
                f"mode_{mode_number}_comparison_gauss_sa_spline")
    


# %%
np.set_printoptions(
    precision=2,        # 2 digits after decimal, not exactly 2 s.f.
    suppress=True,      # avoid scientific notation for small-ish numbers
    linewidth=200,      # wider lines before wrapping
    threshold=np.inf    # print whole array, no ellipsis
)
print((P_res @ H_res_mf)[:10,:10])
print((P_res @ H_res_sv)[:10,:10])
print((P_res @ H_res_sa)[:10,:10])
#print(np.allclose(np.eye(64), P_res @ H_res, rtol=50))

# %%
