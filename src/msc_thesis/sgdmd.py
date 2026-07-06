# -----------------------------------------------------------------------

# dependencies

import numpy as np
import matplotlib.pyplot as plt
import chaosmagpy as cp

import os
import sys
import pydmd
import cmath
import copy
import h5py
import math
import pickle
import gc
import pydmd

from pathlib import Path
from scipy.signal import periodogram
from matplotlib.animation import FuncAnimation
from IPython.display import Video
from tqdm import tqdm
from scipy.interpolate import make_interp_spline
from chaosmagpy.chaos import BaseModel

# importing relative directory paths
from src.msc_thesis.paths import *

# defining global parameters

# ---------------------------------------------------------------
# Spatial
# ---------------------------------------------------------------

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

# ---------------------------------------------------------------
# V alfven/ unit related
# ---------------------------------------------------------------

seconds_in_year = 365.25 * 24 * 60 * 60
seconds_in_day = 24 * 60 * 60

# known quantities
rho = 1e4 # [kg/m3] mass density of fluid outer core
mu_0 = 4 * np.pi * 1e-7 # [N/A2] magnetic permeability of free space
Le=2e-4 # [unitless] Lehnert number (ration of rotation time over Alfvén time) used in calculations
L=radius # [km] radius of earth's core
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

# ---------------------------------------------------------------
# Power Scaling related
# ---------------------------------------------------------------

# try and eliminate this over time - cluttered
data = np.load(f"{DATA_DIR}/power_scaling_branched_global.npz")
POWER_SCALING = {
    "A": data["A"].item(),
    "k": data["k"].item(),
    "A_branch": data["A_branch"].item(),
    "k_branch": data["k_branch"].item(),
    "branch_period": data["branch_period"].item(),
    "f_branch": data["f_branch"].item(),
    "f_bin": data["f_bin"].item(),
    "T_min": data["T_min"].item(),
    "T_max": data["T_max"].item(),
    "f_bin_limits": data["f_bin_limits"],
    "bin_powers": data["bin_powers"],
}

# retrieving power scaling fit globally
A2 = POWER_SCALING["A_branch"]
k2 = POWER_SCALING["k_branch"]
A1 = POWER_SCALING["A"]
k1 = POWER_SCALING["k"]
f_branch= POWER_SCALING["f_branch"]

with open(f"{DATA_DIR}/power_law_psd_dict.pkl", "rb") as f:
    power_law_psd_dict = pickle.load(f)


# ---------------------------------------------------------------
# Gauss Coefficient Related
# ---------------------------------------------------------------

# defining inversion weighting factor
root_W = np.sqrt(W2D.ravel())

# these need to be in degrees for chaosmagpy
theta_v = np.rad2deg(theta_grid.copy().ravel())
phi_v = np.rad2deg(phi_grid.copy().ravel())
radius_v = np.full(len(phi_v), radius)

# construction forward operators to map gauss coefficients
# onto respective magnetic field components
A_r, A_t, A_p = cp.model_utils.design_gauss(
    radius_v, theta_v, phi_v, nmax=20, source="internal"
)

# setting up A dictionary
A_dict = {
    "r":A_r,
    "theta":A_t,
    "phi":A_p
}


def n_Gauss_Coeffs(lmax):
    """
    Number of internal-field Gauss coefficients up to spherical harmonic degree lmax,
    using standard geomagnetic ordering with g_l^m and h_l^m.

    Count = sum_{l=1}^{lmax} (2l + 1) = lmax(lmax + 2)
    """
    return lmax * (lmax + 2)


def Truncate_Gauss_Coeffs(gnm, lmax):
    """
    Truncate a Gauss coefficient vector or array of vectors to spherical harmonic degree lmax.

    Assumes standard ordering:
        g10,
        g11, h11,
        g20, g21, h21, g22, h22,
        ...

    Assumes coefficient axis is the final axis.

    Parameters
    ----------
    gnm : array-like
        Gauss coefficient vector or array of vectors.
        Examples:
            (ncoeff,)
            (nt, ncoeff)
            (nmodes, nt, ncoeff)

    lmax : int
        Maximum spherical harmonic degree to retain.

    Returns
    -------
    gnm_trunc : np.ndarray
        Input array truncated along the final axis.
    """
    gnm = np.asarray(gnm)

    n_keep = n_Gauss_Coeffs(lmax)

    if gnm.shape[-1] < n_keep:
        raise ValueError(
            f"Input only has {gnm.shape[-1]} coefficients, "
            f"but lmax={lmax} requires {n_keep}."
        )

    return gnm[..., :n_keep]


# A_c = forward operator to convert gauss coefficients to physical grid
# evaluated with nmax=20, to used on l truncated fields, apply function below
def A_Truncated(direction, nmax=20, A_dict=A_dict):
    """
    Truncate the forward operator for a given direction to a specified maximum spherical harmonic degree.

    Parameters
    ----------
    direction : str
        The direction for which to retrieve the forward operator. 
        Must be one of 'r', 'theta', or 'phi'.
    nmax : int, optional
        The maximum spherical harmonic degree to retain. Default is 20.
    A_dict : dict, optional
        A dictionary containing the forward operators for each direction.

    Returns
    -------
    A_truncated : np.ndarray
        The truncated forward operator for the specified direction.
    """
    if direction not in A_dict:
        raise ValueError(f"Invalid direction '{direction}'. Must be one of 'r', 'theta', or 'phi'.")

    A = A_dict[direction]
    
    # Truncate the forward operator to the specified maximum degree
    A_truncated = Truncate_Gauss_Coeffs(A.copy(), nmax)

    return A_truncated

# function to project any lat-lon grid data to degree 20 SH:
# given linear system: b_c = A_c @ coeffs 
def SH_L_Project(field, direction, nmax=14):
    
    # ravel the field vector
    field_v = np.ravel(field)

    # select appropriate forward operator
    A = A_Truncated(direction, nmax=nmax, A_dict=A_dict)

    # embedding weighting into least squares
    A_lstsq = root_W[:, None] * A.copy()

    # getting the corresponding gauss coefficients
    b_lstsq = root_W * field_v.copy()
    
    # can invert, with radial forward operator and radial 
    # field in spatial coordinates
    coeffs = np.linalg.lstsq(A_lstsq, b_lstsq, rcond=None)[0]

    # truncating gauss coefficients to desired degree
    coeffs_truncated = Truncate_Gauss_Coeffs(coeffs, nmax) 
    
    # convert back to a spatial grid field form
    field_v_truncate = A @ coeffs_truncated
    return field_v_truncate.reshape(state_shape)

# ---------------------------------------------------------------
# Felix Wave Loading related
# ---------------------------------------------------------------

# function to load in all data components
def Felix_Component_Load(mode_number, directory=FELIX_DIR):
    # select a mode number and corresponding file
    file = h5py.File(f'{directory}/mode_surface_including_gnm_{mode_number}.h5',"r")

    # Converts V_alfven (arbitrary) to nT (arbitrary)
    va_to_nt_arbitrary = (v_alfven * np.sqrt(mu_0 * rho) * 1e9) 

    # load all components, transpose to lat, long

    # gauss coefficient (magnetic scalar potential) phasors
    gnm = va_to_nt_arbitrary * (np.asarray(file["gnmr"]) + 1j*np.asarray(file["gnmi"])) # [nT, arbitrary]

    # Br
    br = va_to_nt_arbitrary * (np.asarray(file["brr"]).T + 1j*np.asarray(file["bri"]).T) # [nT, arbitrary]

    # u_theta
    utheta = np.asarray(file["uthetar"]).T + 1j*np.asarray(file["uthetai"]).T # [V_alfven, arbitrary]
    # u_phi
    uphi = np.asarray(file["uphir"]).T + 1j*np.asarray(file["uphii"]).T # [V_alfven, arbitrary]

    # loading in true period and decay rate
    omega = file["omega"][()] # angular frequency (rad/year)
    sigma = file["sigma"][()] # annual decay rate (fractional decay/year)
    eigenval = sigma + 1j * omega # storing as eigenvalue

    file.close() # close file

    return(gnm, br, utheta, uphi, eigenval)
# %% dsfg

# generates eigenvalue for wave based on desired period and decay
def Felix_Wave_Eigenvalue(eigenval, period, sigma):
    
    # if changing period from default
    if period != "default":
        frequency = 1/period # get standard frequency
        omega = 2 * np.pi * frequency # get angular frequency
        eigenval = eigenval.real + 1j*omega # redefine eigenvalue

    if sigma != "default":
        eigenval = sigma + 1j*eigenval.imag # redefine eigenvalue

    return(eigenval)

# defaults to assuming this is being applied to the SV, as this is most common
def Felix_Analytic_Power(field):
    
    # global-mean squared secular variation 
    mean_abs_field_squared_global = np.sum(W2D_norm * np.abs(field)**2)

    analytic_power = 0.5 * mean_abs_field_squared_global
    
    return(analytic_power)

def Felix_Target_Power(eigenval, bin_width, bin_choice='wave-centred'):
    # getting oscillation parameters
    omega = eigenval.imag
    frequency = omega / (2*np.pi)

    if bin_choice == 'pre-set':
        # find power-law bin
        f_bin_limits = POWER_SCALING["f_bin_limits"]
        bin_powers = POWER_SCALING["bin_powers"]
    
        bin_idx = np.searchsorted(f_bin_limits, frequency, side="right") - 1
    
        if bin_idx < 0 or bin_idx >= len(bin_powers):
            raise ValueError("Wave frequency is outside the defined power-scaling bins.")
    
        target_power = bin_powers[bin_idx]
        
    elif bin_choice == 'wave-centred':
        # detects which power law branch to use
        if frequency < POWER_SCALING["f_branch"]:
            A_use = POWER_SCALING["A_branch"]
            k_use = POWER_SCALING["k_branch"]
            branch_used = "COV-OBS branch"
        else:
            A_use = POWER_SCALING["A"]
            k_use = POWER_SCALING["k"]
            branch_used = "CHAOS branch"
    
        # PSD at wave frequency
        psd_wave = A_use * frequency **k_use
        # Convert PSD to approximate finite-band power
        target_power = psd_wave * bin_width

    return(target_power)

def Felix_Amplitude_Scaler(
    sv,
    eigenval,
    bin_choice,
    bin_width,
):

    target_power = Felix_Target_Power(eigenval, bin_width, bin_choice)
        
    analytic_power = Felix_Analytic_Power(sv)
    
    if analytic_power <= 0:
        raise ValueError("Analytic power is non-positive.")

    # amplitude scaling based on power of wave relative to target
    power_scaler = target_power / analytic_power
    amplitude_scaler = np.sqrt(power_scaler)

    check_ratio = amplitude_scaler**2 * analytic_power / (target_power)

    return (amplitude_scaler)

# construct the generic forward operator for converting to gauss coefficients
# IMPORTANT - HIGH DEGREE L PROJECTION INCLUDES ALL LOWER DEGREES
# DONT COMPUTE SEPARATELY JUST USE COLS NEEDED



def Felix_Wave_Obtain(times, # times, starting t=0.0
                      mode_number, # mode number
                      bin_choice='wave-centred', # wave-centred, i.e. P = PSD * df
                      period='default', # period
                      sigma='default', # annual decay rate [MAKE DECAY PER CYCLE]
                      directory=FELIX_DIR, # directory for wave files
                        ):

    # get record time from data
    T_record = int(times[-1])

    # getting file data
    gnm, br, utheta, uphi, eigenval = Felix_Component_Load(mode_number, directory)
    
    # getting time evolution information
    eigenval = Felix_Wave_Eigenvalue(eigenval, period, sigma)
    sv = br * eigenval # computing SVr phasor [nT/yr unscaled]
    # truncating sv to spherical harmonic degree 14
    # sv = SH_L_Project(sv.real, "r") + 1j * SH_L_Project(sv.imag, "r")

    # IMPORTANT - bin width calculation, must be from video to be valid with 
    # PSD power law
    bin_width = (times[-1]+ times[1]) # assumes times = linspace(start, end, dt)

    # getting the amplitude scaling coefficient for power law scaling
    amplitude_scaler = Felix_Amplitude_Scaler(sv, # (complex) sv_r of wave
                                              eigenval, # eigenvalue of wave
                                              bin_choice=bin_choice,
                                              bin_width=1/T_record # bin width for power
                                                                   )

    # correctly scaling all of the data
    # need a way to get SV as well (should just be phase shifted Br for waves alone?)
    gnm_mf_scaled = gnm.copy() * amplitude_scaler # [nT scaled]
    gnm_sv_scaled = eigenval*(gnm.copy() * amplitude_scaler) # [nT/yr scaled]
    br_scaled = br.copy() * amplitude_scaler # [nT scaled]
    sv_scaled = sv.copy() * amplitude_scaler # [nT/yr scaled]
    utheta_scaled = utheta.copy() * amplitude_scaler * v_alfven * 31557.6 # [km/yr, scaled]
    uphi_scaled = uphi.copy() * amplitude_scaler * v_alfven * 31557.6 # [km/yr, scaled]
    
    return([gnm_mf_scaled, gnm_sv_scaled, br_scaled, sv_scaled, utheta_scaled, uphi_scaled, eigenval]) # order: Gnm, Br, SVr, Utheta, Uphi, Lambda

# function that takes input_wave_data
# and outputs the state vector time evolution array for DMD
def Total_Input_Evaluate(input_wave_data, 
                                   times, 
                                   c # which field do you want to retrieve [br, sv, utheta, uphi]
                                  ):

    n_times = len(times)
    total_synthetic_input = np.zeros((n_points_globally, n_times))
    
    for mode_i_data in input_wave_data:

        # getting mode metadata
        eigenval = mode_i_data["eigenvalue"]

        c_mode_i = mode_i_data[c]

        mode_i_contribution_list = []
        
        for t in times:
            # computing at time t
            c_t_mode_i = np.real(np.exp(eigenval * t) * c_mode_i)
            c_t_mode_i = np.ravel(c_t_mode_i) # state vector
            mode_i_contribution_list.append(c_t_mode_i)
            
        mode_i_contribution_array = np.column_stack(mode_i_contribution_list)
        total_synthetic_input += mode_i_contribution_array

    return(total_synthetic_input)

# function that intakes wave data and computes analytic power in each component
def Felix_Power_Suite(input_wave_data):

    power_suite = []
    
    for wave_data in input_wave_data:

        power_row = {
            "mode_number":None,
            "frequency":None,
            "field_power":{
                "sv":None,
                "utheta":None,
                "uphi":None
            }
        }
        
        eigenval = wave_data["eigenvalue"]
        wave_frequency = eigenval.imag / (2*np.pi)
        power_row["frequency"] = wave_frequency
        mode_number = wave_data["mode_number"]
        power_row["mode_number"] = mode_number

        for c in power_row["field_power"]:

            power_row["field_power"][c] = Felix_Analytic_Power(wave_data[c])

        # each row should be [sv_power, utheta_power, uphi_power]
        power_suite.append(power_row)

    # adding cumulative power row:
    power_row = {
            "mode_number":None,
            "frequency":None,
            "field_power":{
                "sv":None,
                "utheta":None,
                "uphi":None
        }
    }

    for c in power_row["field_power"]:
        total_power_c = [power_row["field_power"][c] for power_row in power_suite] # getting list of just powers from one component
        total_power_c_mag = np.sum(total_power_c)
        power_row["field_power"][c] = total_power_c_mag
        power_row["mode_number"] = "total"
        power_row["frequency"] = np.nan
        
    power_suite.append(power_row)
    return(np.asarray(power_suite))

# now have all wave data, all separate Gnm
# make code that:
#   - evaluates each mode Gnm through time
#   - sums all these together to get total Gnm
#   - converts this into chaosmagpy model

# code returns Gnm total for synthetic input, shape (nt, n_coeffs)
def Wave_Gnm_Total_Obtain(input_wave_data, times, component):

    for i, input_wave_i in enumerate(input_wave_data):

        # retrieving wave data
        eigenvalue = input_wave_i["eigenvalue"]

        if component == "mf":
            Gnm_i_phasor = input_wave_i["gnm_mf"]
        elif component == "sv":
            Gnm_i_phasor = input_wave_i["gnm_sv"]

        if i == 0:
            # initialise total Gnm storage, store first mode
            Gnm_total = np.array([np.real(np.exp(eigenvalue*t) * Gnm_i_phasor) for t in times])
        else:
            # creating full state series for Gnm for this wave
            Gnm_i = np.array([np.real(np.exp(eigenvalue*t) * Gnm_i_phasor) for t in times])
            # adding to total
            Gnm_total += Gnm_i

    return(Gnm_total) # return the total gauss coefficient representation for all summed waves

def Total_Input_Model_Make_From_IWD(input_wave_data, times, component):

    # getting total gauss coefficients of whole input signal
    Gnm_total = Wave_Gnm_Total_Obtain(input_wave_data, times, component=component)
    
    # fitting spline form for the gauss coefficient time series
    spl = make_interp_spline(times, Gnm_total, k=3)                  # cubic B-spline through the snapshots
    model = BaseModel.from_bspline("mode", knots=spl.t, coeffs=spl.c, order=spl.k + 1)

    return(model)

def Total_Input_Field_Make_From_IWD(input_wave_data, times):

    # get chaos model from input data
    model_mf = Total_Input_Model_Make_From_IWD(input_wave_data, times, component="mf")
    model_sv = Total_Input_Model_Make_From_IWD(input_wave_data, times, component="sv")


    # making grid manually - want [nt, ntheta, nphi]
    t_eval = times.copy()[:, None, None]
    colat_eval = colatitude.copy()[None, :, None]
    lon_eval = longitude.copy()[None, None, :]

    print("done! just need to convert into real field on grid")

    # set nmax of total input for converting back to gridded field
    # note: Felix data provided at nmax = 60, but this takes a very long time to evaluate to
    # for now use 20
    nmax_total_input = 60
    
    # takes magnetic field potential (model) in gauss coeffs and grid geometry to compute component fields
    # note both fields use their own gauss coeff models, therefore deriv MUST BE 0
    Br_total, Btheta_total, Bphi_total = \
    model_mf.synth_values(t_eval, radius, colat_eval, lon_eval, nmax=nmax_total_input, grid=False, deriv=0)

    SVr_total, SVtheta_total, SVphi_total = \
    model_sv.synth_values(t_eval, radius, colat_eval, lon_eval, nmax=nmax_total_input, grid=False, deriv=0)

    # storing these total input sums, in dictionary
    # just storing radial components for now
    total_input_wave_fields = {
        "br":Br_total, # radial magnetic field strength
        "sv":SVr_total # radial secular variation
    }

    return(total_input_wave_fields)

def Total_Input_Model_Make_From_Gnm(Gnm, times, component):

    # fitting spline form for the gauss coefficient time series
    spl = make_interp_spline(times, Gnm, k=3)                  # cubic B-spline through the snapshots
    model = BaseModel.from_bspline("mode", knots=spl.t, coeffs=spl.c, order=spl.k + 1)

    return(model)

# right now assumes SV Gnm input (from resolution matrix)
def Total_Input_Field_Make_From_Gnm(Gnm, times):

    # get chaos model from input data
    # model_mf = Total_Input_Model_Make_From_Gnm(Gnm, times, component="mf")
    model_sv = Total_Input_Model_Make_From_Gnm(Gnm, times, component="sv")


    # making grid manually - want [nt, ntheta, nphi]
    t_eval = times.copy()[:, None, None]
    colat_eval = colatitude.copy()[None, :, None]
    lon_eval = longitude.copy()[None, None, :]

    print("done! just need to convert into real field on grid")

    # set nmax of total input for converting back to gridded field
    # note: Felix data provided at nmax = 60, but this takes a very long time to evaluate to
    # for now use 20
    nmax_total_input = 20
    
    # takes magnetic field potential (model) in gauss coeffs and grid geometry to compute component fields
    # note both fields use their own gauss coeff models, therefore deriv MUST BE 0
    #Br_total, Btheta_total, Bphi_total = \
    #model_mf.synth_values(t_eval, radius, colat_eval, lon_eval, nmax=nmax_total_input, grid=False, deriv=0)

    SVr_total, SVtheta_total, SVphi_total = \
    model_sv.synth_values(t_eval, radius, colat_eval, lon_eval, nmax=nmax_total_input, grid=False, deriv=0)

    # storing these total input sums, in dictionary
    # just storing radial components for now
    total_input_wave_fields = {
    #    "br":Br_total, # radial magnetic field strength
        "sv":SVr_total # radial secular variation
    }

    return(total_input_wave_fields)
# ---------------------------------------------------------------
# Power metric quality control related
# ---------------------------------------------------------------

# converts a state vector series array ((ntheta*nphi), nt) into a cube (nt, ntheta, nphi)
def Series_To_Cube(series, state_shape=state_shape):

    return series.T.reshape((series.shape[1], *state_shape))

# converts a cube (nt, ntheta, nphi) into a state vector series array ((ntheta*nphi), nt)
def Cube_To_Series(cube):
    nt = cube.shape[0]
    return cube.reshape(nt, -1).T

# deprecated really, try and not use power scaling directly anymore
def PSD_Evaluate(f, POWER_SCALING=POWER_SCALING):
    """
    Evaluate the branched PSD.

    For f >= f_branch, use the original k1 law.
    For f < f_branch, use the branched k2 law.
    """
    # retrieving power scaling fit globally
    A2 = POWER_SCALING["A_branch"]
    k2 = POWER_SCALING["k_branch"]
    A1 = POWER_SCALING["A"]
    k1 = POWER_SCALING["k"]
    f_branch= POWER_SCALING["f_branch"]

    f = np.asarray(f)

    return np.where(
        f < f_branch,
        A2 * f**k2,
        A1 * f**k1
    )

# by component
def Component_PSD_Evaluate(f, c):
    """
    Evaluate the branched PSD.

    For f >= f_branch, use the original k1 law.
    For f < f_branch, use the branched k2 law.
    """

    fits = power_law_psd_dict[c]
    # retrieving power scaling fit globally
    A2c = fits["A2"]
    k2c = fits["k2"]
    A1c = fits["A1"]
    k1c = fits["k1"]
    f_branch= power_law_psd_dict["f_branch"]

    f = np.asarray(f)

    return np.where(
        f < f_branch,
        A1c * f**k1c,
        A2c * f**k2c
    )

# inputs a state vector series 'cube' into a PSD
def Global_Weighted_P_Metric(series, 
                            scaling, # spectrum or density (PSD)
                            fs, # sampling frequency
                            window='hann' # hann for most cases, boxcar for demos
                            ):
    """
    Br_sv shape: (nt, ntheta, nphi)
    Returns global sin(theta)-weighted PSD.
    """
    if series.ndim == 2: # convet to cube if ravelled state series
        cube = Series_To_Cube(series)
    else:
        cube = series

    freqs, p_metric = periodogram(
        cube,
        fs=fs,
        window=window, # BOXCAR OR HANN
        detrend=False,
        scaling=scaling,
        axis=0
    )

    mean_p_metric = np.sum(p_metric * W2D_norm[None, :, :], axis=(1, 2))

    return freqs, mean_p_metric

def PSD_Evaluate_Branched(f, A1, k1, A2, k2, f_branch):
    """
    Evaluate the branched PSD.

    For f >= f_branch, use the original k1 law.
    For f < f_branch, use the branched k2 law.
    """

    f = np.asarray(f)

    return np.where(
        f < f_branch,
        A2 * f**k2,
        A1 * f**k1
    )

# plot for power QC comparison - SV ONLY RIGHT NOW, PROBABLY SHOULD REWRITE TO HANDLE ALL COMPONENTS
def Power_QC(tot_input_SV, times, input_wave_freqs, scaling='spectrum', window='hann', POWER_SCALING=POWER_SCALING, limits=True):

    f_law_limit = 1/72 # covobs finite support limit
    
    # this evaluates the derived PSD power law at a set of frequencies
    def PSD_Evaluate_Branched(f, A1, k1, A2, k2, f_branch):

        f = np.asarray(f)

        return np.where(
            f < f_branch,
            A2 * f**k2,
            A1 * f**k1
        )

    # intialise figure
    plt.figure(figsize=(8, 6))

    # sampling parameters from times array
    dt = times[1]
    t_span = times[-1] + dt
    fs = 1/dt
    # limits of frequency resolution for data
    f_low = 3/t_span
    f_high = 0.25 * fs # half nyquist rate

    # retrieving power scaling fit
    A2 = POWER_SCALING["A_branch"]
    k2 = POWER_SCALING["k_branch"]
    A1 = POWER_SCALING["A"]
    k1 = POWER_SCALING["k"]
    f_branch= POWER_SCALING["f_branch"]

    # evaluate chosen power metric (density or spectrum)
    freq_fft, p_fft = Global_Weighted_P_Metric(tot_input_SV, fs=fs, scaling=scaling, window=window)

    # defining the reliably sampled frequency range
    # criteria here: atleast 3 full cycles imaged, atleast half nyquist frequency
    reliable = np.where((freq_fft >= f_low) & (freq_fft <= f_high))
    if not limits:
        reliable= np.where(freq_fft >= 0)

    # getting reliable fourier frequencies
    freq_fft_r = freq_fft.copy()[reliable]
    p_fft_r = p_fft.copy()[reliable]

    # intialise plotting frequencies
    law_plot_f = np.logspace(np.log10(f_law_limit), np.log10(f_high), num=1000)

    if scaling=='density':
        plt.plot(freq_fft_r, p_fft_r, label='Actual FFT PSD', linewidth=1,  color='orange', marker='x', markersize=11)
        
        law_plot_p = PSD_Evaluate_Branched(
            law_plot_f,
            A1, k1,
            A2, k2,
            f_branch
            )

    elif scaling == 'spectrum':
        
        law_plot_p = PSD_Evaluate_Branched(
            law_plot_f,
            A1, k1,
            A2, k2,
            f_branch
            )

        law_plot_p = law_plot_p * 0.04
    
        plt.plot(freq_fft_r, p_fft_r, label='Actual FFT Power', linewidth=1,  color='orange', marker='x', markersize=11)

    # plots either power metric, just label it power law predicted in either case
    plt.plot(law_plot_f, law_plot_p, label='Power Law Predicted', linewidth=1,color='black')

    # plotting input periods for reference
    for i, freq in enumerate(input_wave_freqs):
        plt.axvline(freq, label='input frequency' if i ==0 else '', color='red', linestyle='--')

    # putting lines at reliable frequency limits
    plt.axvline(f_high, linestyle=':', color='blue', label='FFT reliability limit')
    plt.axvline(f_low, linestyle=':', color='blue')
    plt.axvline(f_law_limit, color='grey', linestyle=':', label='Power law limit (72 year period)')


    
    plt.xlabel("frequency [yr$-1$]")

    if scaling == 'density':
        plt.ylabel("Power Spectral Density [(nT$^2$yr$^{-2}$)(yr$^{-1}$)]")
    elif scaling == 'spectrum':
        plt.ylabel("Power [nT$^2$yr$^{-2}$]")


    plt.yscale('log')
    plt.xscale('log')
    plt.grid(True, which="both", ls="-")
    plt.title("Power Metric Plot")
    
    plt.legend()

# print this code to look at input energy, vs output measure.
'''for wave_data_i in input_wave_data:
    br = wave_data_i[1]
    eigenval = wave_data_i[-1]
    print(Felix_Analytic_Power(br, eigenval))
    print(Felix_Target_Power(eigenval, bin_choice='wave-centred', bin_width=(1/25)))
    f_input, P_input = Global_Weighted_P_Metric(tot_input_sv, 'spectrum',f_sample)
    print(np.sum(P_input))'''

import numpy as np
import matplotlib.pyplot as plt


def evaluate_broken_power_law(f, A1, A2, k1, k2, f_branch):
    """
    Evaluate broken power law.

    Assumes:
      branch 1: f <= f_branch uses A1 * f**k1
      branch 2: f >  f_branch uses A2 * f**k2
    """

    f = np.asarray(f, dtype=float)

    psd = np.full_like(f, np.nan, dtype=float)

    low_mask = f <= f_branch
    high_mask = f > f_branch

    psd[low_mask] = A1 * f[low_mask]**k1
    psd[high_mask] = A2 * f[high_mask]**k2

    return psd


def plot_power_law_with_analytic_wave_psd(
    power_law_psd_dict,
    psd_by_component,
    save_path=None,
    show=True
):
    """
    Plot power-law PSD curves and analytic wave PSD estimates for sv, utheta, uphi.
    """

    components = ["sv", "utheta", "uphi"]

    component_labels = {
        "sv": r"$\dot{B}_r$",
        "utheta": r"$u_\theta$",
        "uphi": r"$u_\phi$"
    }

    component_colours = {
        "sv": "blue",
        "utheta": "green",
        "uphi": "red"
    }

    f_branch = power_law_psd_dict["f_branch"]

    f_wave = np.asarray(psd_by_component["frequency"], dtype=float)

    # frequency range for smooth PSD curves
    f_min = np.nanmin(f_wave[f_wave > 0])
    f_max = np.nanmax(f_wave)

    # include branch frequency in range if relevant
    f_min = min(f_min, f_branch) / 1.3
    f_max = max(f_max, f_branch) * 1.3

    f_plot = np.logspace(
        np.log10(f_min),
        np.log10(f_max),
        500
    )

    fig, axes = plt.subplots(
        3, 1,
        figsize=(8, 12),
        dpi=200,
        sharex=True
    )

    for ax, c in zip(axes, components):

        A1 = power_law_psd_dict[c]["A1"]
        A2 = power_law_psd_dict[c]["A2"]
        k1 = power_law_psd_dict[c]["k1"]
        k2 = power_law_psd_dict[c]["k2"]

        psd_fit = evaluate_broken_power_law(
            f_plot,
            A1=A1,
            A2=A2,
            k1=k1,
            k2=k2,
            f_branch=f_branch
        )

        psd_wave = np.asarray(psd_by_component["psd"][c], dtype=float)

        wave_mask = (
            np.isfinite(f_wave)
            & np.isfinite(psd_wave)
            & (f_wave > 0)
            & (psd_wave > 0)
        )

        ax.loglog(
            f_plot,
            psd_fit,
            color=component_colours[c],
            linewidth=2,
            label="Power-law PSD"
        )

        ax.scatter(
            f_wave[wave_mask],
            psd_wave[wave_mask],
            color=component_colours[c],
            edgecolor="black",
            s=70,
            zorder=3,
            label="Analytic wave PSD"
        )

        ax.axvline(
            f_branch,
            color="black",
            linestyle="--",
            linewidth=1,
            alpha=0.7,
            label=r"$f_{\mathrm{branch}}$" if c == "sv" else None
        )

        ax.set_ylabel("PSD")
        ax.set_title(component_labels[c])
        ax.grid(True, which="both", alpha=0.35)
        ax.legend()

    axes[-1].set_xlabel(r"Frequency [yr$^{-1}$]")

    fig.tight_layout()

    if save_path is not None:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")

    if show:
        plt.show()
    else:
        plt.close(fig)




# ---------------------------------------------------------------
# noise related
# ---------------------------------------------------------------
# need code to do the above, i.e. fit FFT etc, then get a power law from that with break at 12 years

# generating gaussian white noise with power of 1
# assumes number of steps = 50, i.e. chaos record
def Normalised_Gaussian_White_Noise(rng):
    norm_gaussian_white_noise = rng.normal(
        loc=0.0, # mean of 0
        scale=1, # standard deviation of 1
        size=(50, len(theta), len(phi))
    )
    return(norm_gaussian_white_noise)


# Naive power law noise - only generate what is used
def Frugal_Power_Law_Coloured_Noise(
    times,
    spatial_shape=state_shape,
    period_band=(1.0, 100.0),
    noise_power_fraction=1.0,
    rng=None,
):

    if rng is None:
        rng = np.random.default_rng()
        
    nt = len(times)
    dt = times[1] - times[0]
    T_record = nt * dt

    freqs = np.fft.rfftfreq(nt, d=dt)
    df = 1.0 / T_record

    T_min, T_max = period_band
    f_min = 1.0 / T_max
    f_max = 1.0 / T_min

    valid = (freqs >= f_min) & (freqs <= f_max)

    coeffs = np.zeros((len(freqs), *spatial_shape), dtype=complex)

    powers = np.zeros_like(freqs)
    powers[valid] = noise_power_fraction * PSD_Evaluate(freqs[valid]) * df

    phases = rng.uniform(0, 2*np.pi, size=(valid.sum(), *spatial_shape))

    # Fourier coefficient amplitude, not time-domain amplitude
    amps = (nt / 2.0) * np.sqrt(2.0 * powers[valid])

    coeffs[valid] = amps[:, None, None] * np.exp(1j * phases)

    coeffs[0] = 0.0

    if nt % 2 == 0:
        coeffs[-1] = coeffs[-1].real + 0j

    noise = np.fft.irfft(coeffs, n=nt, axis=0)

    return noise


# power law noise that generates over longer cycle, then crops
# to avoid fourier artefacts
def Power_Law_Coloured_Noise(
    times,
    c,
    spatial_shape=state_shape,
    period_band=(1.0, 100.0),
    noise_power_fraction=0.1,
    rng=None,
    long_factor=10,
    crop="random",   # "random", "start", or "end"
):

    if rng is None:
        rng = np.random.default_rng()

    n_real = len(times)
    dt = times[1] - times[0]

    # generate longer record
    nt = long_factor * n_real
    T_record = nt * dt

    freqs = np.fft.rfftfreq(nt, d=dt)
    df = 1.0 / T_record

    T_min, T_max = period_band
    f_min = 1.0 / T_max
    f_max = 1.0 / T_min

    valid = (freqs >= f_min) & (freqs <= f_max)

    coeffs = np.zeros((len(freqs), *spatial_shape), dtype=complex)

    powers = np.zeros_like(freqs)
    powers[valid] = noise_power_fraction * Component_PSD_Evaluate(freqs[valid], c) * df

    phases = rng.uniform(0, 2 * np.pi, size=(valid.sum(), *spatial_shape))

    # Fourier coefficient amplitude for irfft convention
    amps = (nt / 2.0) * np.sqrt(2.0 * powers[valid])

    coeffs[valid] = amps[:, None, None] * np.exp(1j * phases)

    # remove mean
    coeffs[0] = 0.0

    # ensure Nyquist coefficient is real if present
    if nt % 2 == 0:
        coeffs[-1] = coeffs[-1].real + 0j

    noise_long = np.fft.irfft(coeffs, n=nt, axis=0)

    # crop back to actual signal length
    if crop == "random":
        start = rng.integers(0, nt - n_real + 1)
    elif crop == "start":
        start = 0
    elif crop == "end":
        start = nt - n_real
    else:
        raise ValueError("crop must be 'random', 'start', or 'end'")

    noise = noise_long[start:start + n_real]

    return noise

# noise smoothing:
from scipy.ndimage import gaussian_filter
import numpy as np

def Spatially_Smooth_Noise(noise, noise_fraction=1, sigma_theta=5, sigma_phi=5):
    """
    Smooth noise spatially at each time step.

    noise shape: (nt, n_theta, n_phi)
    sigma_theta, sigma_phi: smoothing length in grid cells
    """

    # note observed that psd actually preserved post smoothing
    # so just rescale total power before and after (boxcar)
    f, p = Global_Weighted_P_Metric(noise, 'spectrum', 0.5, window='boxcar')
    pre_smoothing_power = np.sum(p)

    noise_smooth = np.empty_like(noise)

    for i in range(noise.shape[0]):
        noise_smooth[i] = gaussian_filter(
            noise[i],
            sigma=(sigma_theta, sigma_phi),
            mode=("reflect", "wrap")
        )

    f, p = Global_Weighted_P_Metric(noise_smooth, 'spectrum', 0.5, window='boxcar')
    post_smoothing_power = np.sum(p)

    amplitude_scaler = np.sqrt(noise_fraction * pre_smoothing_power / post_smoothing_power)

    return amplitude_scaler * noise_smooth

# ---------------------------------------------------------------
# Decomposition related
# ---------------------------------------------------------------
# function to compute economical SVD and return its component matrices
def Econ_SVD(data):

    # compute economical SVD of data matrix F
    U_econ, s_econ, Vh_econ = np.linalg.svd(data, full_matrices=False)

    econ_svd = [U_econ, s_econ, Vh_econ]

    return(econ_svd) # return final truncated SVD


# function to truncate (k/ variance) SVD and return output
def Truncate_SVD(svd, 
                 truncate_by = "variance", # truncate by variance by default
                 var_target = 0.99, # want 99% of variance by default
                 k=0 # use all non zero singular values by default
                    ):

    U, s, Vh = svd

    # code that truncates based on energy, defined k etc.

    # directly define number of singular values to truncate to
    if truncate_by == 'k':
        U_truncated = U[:, :k]
        s_truncated = s[:k]
        Vh_truncated = Vh[:k, :]

    # define by variance, using variance threshhold
    elif truncate_by == 'variance':

        # getting variance (squared singular values)
        energy = s**2
        # getting the fraction relative to total for each
        variance_fraction = energy / energy.sum()
        # get cumulative sum array to identify where
        # target variance is met
        cumulative_variance = np.cumsum(variance_fraction)
        
        # r-1 = lowest rank sum that meets variance target
        r = np.searchsorted(cumulative_variance, 0.99) + 1

        # truncate to r
        U_truncated = U[:, :r]
        s_truncated = s[:r]
        Vh_truncated = Vh[:r, :]

    truncated_svd = [U_truncated, s_truncated, Vh_truncated]
    
    return(truncated_svd) # return final truncated SVD

def Exact_DMD(data,
              # below are for internal SVD
              truncate_by = None, # None, variance, k
              var_target = 0.99, # want 99% of variance by default
              k=0): # use all non zero singular values by default):

    # establishing number of time steps from data
    n_times = np.shape(data)[1]

    # first split data into X and X'
    X = data[:, :-1] # exclude final time step
    X_prime = data[:, 1:] # exclude first time step

    # from this have U = modes, s = coefficients, Vt = dynamics
    if truncate_by == None:
        data_svd = Econ_SVD(X)
    else:
        data_svd = Truncate_SVD(Econ_SVD(X), 
                                truncate_by=truncate_by, 
                                var_target=var_target, 
                                k=k)
    
    U, s, Vh = data_svd
    
    # constructing A_tilde
    Uh = U.conj().T
    V = Vh.conj().T
    s_inv = 1/s
    A_tilde = Uh @ X_prime @ V * s_inv[None, :]
    
    # getting eigen solutions (L = values, W = vectors) of
    # reduced order system (k-truncated earlier) 
    # L = full order coefficients
    L, W = np.linalg.eig(A_tilde)
    
    # getting PHI = high dimensional modes
    PHI = X_prime @ V * s_inv[None, :] @ W
    
    # dynamics and coefficients
    
    # obtaining dynamics:
    '''
    dynamics:
    - each time step is advanced by multiplying by lambda
    - initial time state = initial amplitude (b_i)
    - each additional time state = previous * lambda_i
    Therefore can can make a vector v_i for each mode, where 
    each element is the next time step
    '''
    
    # getting initial amplitudes
    # X = PHI @ b
    # therefore b = np.inv(PHI, X)
    x0 = X[:, 0]
    b0, *_ = np.linalg.lstsq(PHI, x0, rcond=None)
    PHI = b0 * PHI # all modes are scaled to initial amplitudes

    # storing DMD time evolution coefficients for each mode
    # DONT NEED TO STORE, EASY TO RECONSTRUCT IN LOOP WITH IDX I AND L
    #n_times = new_snap_shape[1] # how many time steps
    #time_indices = np.arange(n_times+1) # array listing each time step
    #dynamics = b[:, None] * L[:, None]**time_indices[None, :]

    exact_DMD = [PHI,  L]

    return(exact_DMD)

def DMD_Mode_Pair(dmd_out, tol=1e-8):

    modes, eigenvalues = dmd_out
    eigenvalues = np.asarray(eigenvalues)

    used = set()
    mode_indices = []

    # for each eigenvalue returned in decomposition
    for idx, eig in enumerate(eigenvalues):

        # if index already examined, skip
        if idx in used:
            continue

        # real/static/non-oscillatory mode
        # i.e. if no imaginary part, skip
        if abs(eig.imag) < tol:
            mode_indices.append(idx)
            used.add(idx)
            continue

        # look for unused conjugate partner
        conj_eig = np.conjugate(eig)

        candidates = []
        for j, eig_j in enumerate(eigenvalues):
            if j == idx or j in used:
                continue

            if np.isclose(eig_j, conj_eig, atol=tol, rtol=tol):
                candidates.append(j)

        if len(candidates) > 0:
            idx_pair = candidates[0]
            mode_indices.append((idx, idx_pair))
            used.add(idx)
            used.add(idx_pair)
        else:
            # no pair found
            mode_indices.append(idx)
            used.add(idx)

    return mode_indices

# evaluates total recomposition from DMD
def Total_Output_Evaluate(decomposition, 
                          times
                                  ):

    modes, eigenvalues = decomposition

    n_times = len(times) # how many time steps
    powers = np.arange(n_times) # array of time step number

    # essentially computing how each mode amplitude evolves through time
    dynamics = eigenvalues[:, None]**powers[None, :]

    # vectorised construction of modes through time with dynamics
    total_reconstructed_output = modes @ dynamics

    # make real as numerical reality likely leaves imaginary residuals
    return np.real(total_reconstructed_output)

# converts a discrete (i.e. DMD output eigenvalue) into a continuous type form
# see notes for explanation
def D_To_C_Eigenvalue_Converter(eigenvalue_d, dt, tol=1e-10):
    eigenvalue_d = np.asarray(eigenvalue_d, dtype=complex)

    sigma_c = np.log(np.abs(eigenvalue_d)) / dt
    omega_c = np.angle(eigenvalue_d) / dt

    # getting rid of floating point error:
    omega_c = np.where(np.abs(omega_c) < tol, 0.0, omega_c)
    sigma_c = np.where(np.abs(sigma_c) < tol, 0.0, sigma_c)

    return sigma_c + 1j * omega_c

# converts a continuous eigenvalue into a discrete type form
def C_To_D_Eigenvalue_Converter(eigenvalue_c, dt, tol=1e-10):
    eigenvalue_c = np.asarray(eigenvalue_c, dtype=complex)

    # first getting r and theta
    r = np.exp(eigenvalue_c.real * dt)
    theta = eigenvalue_c.imag * dt
    # then converting to x + iy
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    
    # getting rid of floating point error:
    x = np.where(np.abs(x) < tol, 0.0, x)
    y = np.where(np.abs(y) < tol, 0.0, y)

    return x + 1j * y

# ---------------------------------------------------------------
# success metric related
# ---------------------------------------------------------------


def Spatial_Relative_Error(input_phasor, output_phasor):
    return np.linalg.norm(output_phasor - input_phasor) / np.linalg.norm(input_phasor)

'''def period_relative_error(input_eigenvalue, output_eigenvalue):
    T_input = 2 * np.pi / np.abs(input_eigenvalue.imag)
    T_output = 2 * np.pi / np.abs(output_eigenvalue.imag)

    return abs(T_output - T_input) / T_input'''

def Complex_Corr(a, b):
    a = a.ravel()
    b = b.ravel()

    den = np.linalg.norm(a) * np.linalg.norm(b)
    if den == 0:
        return np.nan

    return np.abs(np.vdot(a, b)) / den

def period_relative_error(input_eigenvalue, output_eigenvalues, tol=1e-12):
    omega_input = np.abs(input_eigenvalue.imag)
    omega_output = np.abs(np.asarray(output_eigenvalues).imag)

    # initialise all outputs as invalid
    errors = np.full_like(omega_output, np.inf, dtype=float)

    # if input frequency is zero, all period errors are undefined
    if np.isclose(omega_input, 0.0, atol=tol):
        return errors

    # valid output modes are non-static
    valid = ~np.isclose(omega_output, 0.0, atol=tol)

    errors[valid] = np.abs((omega_input / omega_output[valid]) - 1)

    return errors

# ---------------------------------------------------------------
# summary related
# ---------------------------------------------------------------


def plot_performance_summary(performance_summary, mode_labels=None, save_dir=FIG_DIR, show=True):
    """
    For each input mode in performance_summary, create one row figure with:
      1) eigenvalue plot in the complex plane
      2) spatial correlation vs normalised spatial error scatter plot

    Individual noise realisations are plotted in component colours.
    Component-wise averages across noise realisations are shown as black versions
    of the same component markers.
    """

    component_styles = {
        "sv": {
            "color": "blue",
            "marker": "s",
            "label": "SV"
        },
        "gnm_sv": {
            "color": "brown",
            "marker": "s",
            "label": "SV (gnm)"
        },
        "utheta": {
            "color": "green",
            "marker": "^",
            "label": r"$u_\theta$"
        },
        "uphi": {
            "color": "red",
            "marker": "o",
            "label": r"$u_\phi$"
        },
    }

    components = ["sv", "gnm_sv", "utheta", "uphi"]

    if mode_labels is None:
        mode_labels = [f"{i+1}" for i in range(len(performance_summary))]

    for i, wave_summary in enumerate(performance_summary):

        mode_label = mode_labels[i]
        true_eig = wave_summary["true_eigenvalue"]

        fig, axes = plt.subplots(
            1, 2,
            figsize=(14, 6),
            dpi=200
        )

        ax_eig, ax_perf = axes

        # ==========================================================
        # LEFT PLOT: EIGENVALUES
        # ==========================================================

        ax_eig.scatter(
            np.real(true_eig),
            np.imag(true_eig),
            color="black",
            marker="x",
            s=140,
            linewidths=2.5,
            label="Truth"
        )

        for comp in components:

            style = component_styles[comp]
            eig_list = wave_summary[comp]["recovered_eigenvalues"]

            eig_real = []
            eig_imag = []

            for eig in eig_list:

                if eig is None:
                    continue

                r = np.real(eig)
                im = np.imag(eig)

                if np.isfinite(r) and np.isfinite(im):
                    eig_real.append(r)
                    eig_imag.append(im)

            eig_real = np.asarray(eig_real)
            eig_imag = np.asarray(eig_imag)

            if eig_real.size > 0:

                # individual realisations
                ax_eig.scatter(
                    eig_real,
                    eig_imag,
                    color=style["color"],
                    marker=style["marker"],
                    s=65,
                    alpha=0.65,
                    label=style["label"]
                )

                # average recovery
                ax_eig.scatter(
                    np.nanmean(eig_real),
                    np.nanmean(eig_imag),
                    color="black",
                    marker=style["marker"],
                    s=130,
                    edgecolors="black",
                    linewidths=1.5,
                    label=f"{style['label']} mean"
                )

        ax_eig.set_xlabel(r"Re($\omega$) [yr$^{-1}$]")
        ax_eig.set_ylabel(r"Im($\omega$) [rad yr$^{-1}$]")
        ax_eig.set_title("Recovered eigenvalues")
        ax_eig.grid(True)
        ax_eig.legend(fontsize=8)

        # ==========================================================
        # RIGHT PLOT: CORRELATION VS NORMALISED SPATIAL ERROR
        # ==========================================================

        # perfect spatial recovery
        ax_perf.scatter(
            1.0,
            0.0,
            color="black",
            marker="x",
            s=140,
            linewidths=2.5,
            label="Perfect match"
        )

        for comp in components:

            style = component_styles[comp]

            corr = np.asarray(
                wave_summary[comp]["spatial_correlations"],
                dtype=float
            )

            err = np.asarray(
                wave_summary[comp]["norm_spatial_errors"],
                dtype=float
            )

            mask = np.isfinite(corr) & np.isfinite(err)

            if np.any(mask):

                # individual realisations
                ax_perf.scatter(
                    corr[mask],
                    err[mask],
                    color=style["color"],
                    marker=style["marker"],
                    s=65,
                    alpha=0.65,
                    label=style["label"]
                )

                # average recovery
                ax_perf.scatter(
                    np.nanmean(corr[mask]),
                    np.nanmean(err[mask]),
                    color="black",
                    marker=style["marker"],
                    s=130,
                    edgecolors="black",
                    linewidths=1.5,
                    label=f"{style['label']} mean"
                )

        ax_perf.set_xlabel("Spatial correlation")
        ax_perf.set_ylabel("Normalised spatial error")
        ax_perf.set_title("Spatial recovery")
        ax_perf.grid(True)
        ax_perf.legend(fontsize=8)

        # sensible default limits
        ax_perf.set_xlim(0, 1.05)

        y_max = 1.0
        for comp in components:
            err = np.asarray(
                wave_summary[comp]["norm_spatial_errors"],
                dtype=float
            )
            if np.any(np.isfinite(err)):
                y_max = max(y_max, np.nanmax(err))

        ax_perf.set_ylim(0, 1.1 * y_max)

        fig.suptitle(f"Input mode {mode_label}", fontsize=14)
        fig.tight_layout()

        if save_dir is not None:
            fig.savefig(
                os.path.join(save_dir, f"mode_{mode_label}_performance_summary.png"),
                dpi=300,
                bbox_inches="tight"
            )

        if show:
            plt.show()
        else:
            plt.close(fig)
# ---------------------------------------------------------------
# Video related
# ---------------------------------------------------------------

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
# assuming that this is dealing with inputs and outputs from same space (i.e. all Br, all SV etc.)

def Frame_Image(field, ax, vlimits):
    vmin, vmax = vlimits
    im = ax.imshow(
        (field),
        cmap="seismic",
        vmin=vmin,
        vmax=vmax,
        origin="lower"
        )
    return(im)

units = {
    "sv":"[nT/year]",
    "utheta":"[km/year]",
    "uphi":"[km/year]"
}

# previous plotting skeleton
def Video_Compare(input_data, output_data, total_input, total_output, times, c='sv'):

    input_eigenvalues = []
    input_phasors = []
    output_eigenvalues = []
    output_phasors = []

    # retrieving all the data for video creation
    for input_data_l in input_data: # for each input index
        input_eigenvalues.append(input_data_l["eigenvalue"])
        input_phasors.append(input_data_l[c])
        match_idx_l = input_data_l["best_match_idx"][c][-1] # takes last index (i.e. last saved run in ensemble)
        output_data_l = output_data[c][match_idx_l]
        output_eigenvalues.append(output_data_l["eigenvalue_continuous"])
        output_phasors.append(output_data_l["phasor_continuous"])

    # time spacing
    dt = times[1]

    # number of final plot rows
    # as dmd modes can repeat:
    num_rows = len(input_data) + 1 
    # *note: 1 extra is for the total summed superposition
    num_cols = 3 # for 3 columns: input, output, difference

    fig, axes = plt.subplots(
        num_rows,
        num_cols,
        figsize=(3 * num_cols + 1, 1.8 * num_rows + 3),
        squeeze=False,
        constrained_layout=True
    )
    
    title = fig.suptitle(
        f"Pattern comparison | frame 1 / {len(times)}",
        fontsize=14,
        y=0.995
    )

    # intialise images for updating
    images = [[None for j in range(num_cols)] for i in range(num_rows)]

    # ----------------- START OF MATCHED INPUT SIGNALS -------------------------
    
    # loop over each wave
    for i, input_i_phasor in enumerate(input_phasors):

        # pull out relevant input and matched output
        # remember = real(phasor at t=0)
        input_i_initial = np.real(input_i_phasor)
        output_i_initial = np.real(output_phasors[i])
        diff_i_initial = np.sqrt((np.abs(input_i_initial - output_i_initial)**2))

        # getting temporal properties for plotting titles
        input_i_eigenvalue = input_eigenvalues[i]
        output_i_eigenvalue = output_eigenvalues[i]
        
        # defining colour bar scales for this row - used for all plots
        # get the largest absolute value across all plots, use as maximum on common scale
        vmax = np.max(np.abs([input_i_initial, output_i_initial, diff_i_initial]))
        vmin = -vmax
        vlimits = (vmin, vmax)
    
        # set axes for the row, ordered: input, output, difference
        ax_in, ax_out, ax_diff = axes[i, :]
    
        im_out = Frame_Image(output_i_initial, ax_out, vlimits)
        im_diff = Frame_Image(diff_i_initial, ax_diff, vlimits)
        im_in = Frame_Image(input_i_initial, ax_in, vlimits)

        # setting colour bar for the whole row
        cbar = fig.colorbar(
            im_in,
            ax=axes[i, :],
            location="right",
            shrink=0.8
        )
        
        cbar.set_label(units[c])

        # getting periods of input and output
        input_period = (2*np.pi)/input_i_eigenvalue.imag
        output_period = (2*np.pi)/output_i_eigenvalue.imag
    
        # displaying the input signal period
        if isinstance(input_period, str):
            ax_in.set_title(f"Input period = {input_period}")
        else:
            ax_in.set_title(f"Input period = {input_period:.2f}")

        # displaying the output signal period
        if isinstance(output_period, str):
            ax_out.set_title(f"Output period = {output_period}")
        else:
            ax_out.set_title(f"Output period = {output_period:.2f}")

        # adding difference title
        ax_diff.set_title(f"Difference between")

        # setting row of images accordingly
        images[i] = im_in, im_out, im_diff
    
    # ----------------- END OF MATCHED INPUT SIGNALS -------------------------        

    # ----------------- START OF TOTAL SIGNALS -------------------------

    # getting the total signal data for input, output and difference
    tot_input_v, tot_output_v = total_input[:,0], total_output[:,0]
    tot_diff_v = np.abs(tot_output_v - tot_input_v)
    
    # plot the total reconstruction from both
    vmax = np.max(np.abs([tot_input_v, tot_output_v, tot_diff_v]))
    vmin = -vmax
    v_limits = (vmin, vmax)

    # input
    ax_tot_in = axes[-1, 0]
    im_tot_in = Frame_Image(tot_input_v.reshape(state_shape), ax_tot_in, v_limits)
    images[-1][0] = im_tot_in

    # output
    ax_tot_out = axes[-1, 1]
    im_tot_out = Frame_Image(tot_output_v.reshape(state_shape), ax_tot_out, v_limits)  
    images[-1][1] = im_tot_out

    # difference
    ax_tot_diff = axes[-1, 2]
    im_tot_diff = Frame_Image(tot_diff_v.reshape(state_shape), ax_tot_diff, v_limits)  
    images[-1][2] = im_tot_diff

    cbar = fig.colorbar(
        im_tot_in,
        ax=axes[-1, :],
        location="right",
        shrink=0.8
    )
    cbar.set_label("nT")

    # setting titles:
    ax_tot_in.set_title(f"Total input signal")
    ax_tot_out.set_title(f"Total output signal")
    ax_tot_diff.set_title(f"Difference between")

    # ----------------- END OF TOTAL SIGNALS -------------------------
        

    # hide blank axes and remove them from layout calculation
    for r in range(num_rows):
        for c in range(num_cols):
            if images[r][c] is None:
                axes[r, c].set_visible(False)
                axes[r, c].set_in_layout(False)
    # define update (essentially same as above but with flexible axes)

    # updating each time step 
    # this now requires actually computing the time step before updating
    def update(k): # k = timestep

        time = k * dt

        # matched inputs
        for i, _ in enumerate(input_data):
            
            input_phasor_i = input_phasors[i]
            input_eigenvalue_i = input_eigenvalues[i]

            output_phasor_i = output_phasors[i]
            output_eigenvalue_i = output_eigenvalues[i]
            
            input_i_k = np.real(np.exp(input_eigenvalue_i * time) * input_phasor_i)
            output_i_k = np.real(np.exp(output_eigenvalue_i * time) * output_phasor_i)

            diff_i_k = np.abs(input_i_k - output_i_k)
    
            images[i][0].set_data(input_i_k)
            images[i][1].set_data(output_i_k)
            images[i][2].set_data(diff_i_k)
            
        # total input and reconstruction
        k_tot_input = total_input[:,k].reshape(state_shape)
        k_tot_output = total_output[:,k].reshape(state_shape)
        k_tot_diff = np.abs(k_tot_output - k_tot_input)

        images[-1][0].set_data(k_tot_input)
        images[-1][1].set_data(k_tot_output)
        images[-1][2].set_data(k_tot_diff)

        # updating title
        title.set_text(
            f"Pattern comparison | frame {k + 1} / {len(times)}"
        )

        # returning all used image artist objects
        return [im for row in images for im in row if im is not None]
    
    # hide any axes that never received an image
    for r in range(num_rows):
        for c in range(num_cols):
            if images[r][c] is None:
                axes[r, c].set_visible(False)
    
    anim = FuncAnimation(
            fig,
            update,
            frames=len(times),
            interval=50,
            blit=True
        )

    from datetime import datetime

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    anim.save(
        f"{FIG_DIR}/video_test_{timestamp}.mp4",
        writer="ffmpeg",
        fps=2,
        dpi=100
    )
    
    plt.close(fig)
    return Video(f"{FIG_DIR}/video_test_{timestamp}.mp4", embed=True)


def Synthetic_Video_Comparison(input_data, total_input, decomp_data, total_output, mode_indices, times):

    # time spacing
    dt = times[1]

    # unpacking decomposition (assume br at the moment)
    modes, mode_eigenvalues = decomp_data

    # felix wave primitives
    num_inputs = len(input_data)
    input_wave_start_c = [data_row[1] for data_row in input_data] # i.e. Re[Br]
    input_wave_start = [data_row.real for data_row in input_wave_start_c] # i.e. Re[Br]
    wave_eigenvalues = [data_row[-1] for data_row in input_data]
    input_periods = [(2*np.pi/eig.imag) for eig in wave_eigenvalues]
    
    # constructing first frame of each output mode (need all to be real)
    num_outputs = len(mode_indices)
    output_mode_start = []
    output_periods = []
    
    for idx in mode_indices:
        
        if type(idx)==tuple:
            i_1, i_2 = idx
            output_period = np.abs(0.5/(cmath.phase(mode_eigenvalues[i_1])/(2*np.pi)))
            output_mode = np.real(modes[:, i_1] + modes[:, i_2])
            
        else:
            output_period = np.abs(0.5/(cmath.phase(mode_eigenvalues[idx])/(2*np.pi)))
            output_mode = np.real(modes[:, idx])

        output_periods.append(output_period)
        output_mode_start.append(output_mode)

        
    # intialising correlation matrix
    corr_matrix = np.zeros((num_outputs, num_inputs))

    # computing and storing correlation matrix
    for i, out in enumerate(output_mode_start):
        for j, inp in enumerate(input_wave_start):
            corr_matrix[i, j] = np.corrcoef(inp.ravel(), out.ravel())[0,1]

     # finding, for each wave, which mode has the highest correlation
    best_match_mode_idx = np.argmax(corr_matrix, axis=0)
    
    # make lists, each element representing an input/ output pattern
    input_idx = np.arange(num_inputs)
    output_idx = np.arange(num_outputs)

    # defining indices for the modes that were not an optimal match to the true inputs
    unmatched_mode_idx = [x for x in output_idx if x not in best_match_mode_idx]

    # number of final plot rows
    # as dmd modes can repeat:
    num_rows = len(unmatched_mode_idx) + len(best_match_mode_idx) + 1 
    # *note: 1 extra is for the total summed superposition

    num_cols = 3 # for 3 columns: input, output, difference
    snapshot_shape = (181, 360) # this is the snapshot shape used for synthetics
    
    fig, axes = plt.subplots(
        num_rows,
        num_cols,
        figsize=(3 * num_cols + 1, 1.8 * num_rows + 3),
        squeeze=False,
        constrained_layout=True
    )
    
    title = fig.suptitle(
        f"Pattern comparison | frame 1 / {len(times)}",
        fontsize=14,
        y=0.995
    )

    #fig.tight_layout()
    
    # intialise images for updating
    images = [[None for j in range(num_cols)] for i in range(num_rows)]

    # ----------------- START OF MATCHED INPUT SIGNALS -------------------------
    
    # loop over each wave
    for i in input_idx:

        # have set of initial conditions already
        input_state_v = input_wave_start[i]

        # get matched mode index
        match = best_match_mode_idx[i]

        output_state_v = output_mode_start[match].reshape(state_shape)
        diff_state_v = np.sqrt((np.abs(input_state_v - output_state_v)**2))

        # defining colour bar scales for this row - used for all plots
        # get the largest absolute value across all plots, use as maximum on common scale
        vmax = np.max(np.abs([input_state_v, output_state_v, diff_state_v]))
        vmin = -vmax
        vlimits = (vmin, vmax)
    
        # set axes for the row, ordered: input, output, difference
        ax_in, ax_out, ax_diff = axes[i, :]
    
        im_out = Frame_Image(output_state_v, ax_out, vlimits)
        im_diff = Frame_Image(diff_state_v, ax_diff, vlimits)
        im_in = Frame_Image(input_state_v, ax_in, vlimits)

        # setting colour bar for the whole row
        cbar = fig.colorbar(
            im_in,
            ax=axes[i, :],
            location="right",
            shrink=0.8
        )
        
        cbar.set_label("nT")

        # getting periods of input and output
        input_period = input_periods[i]
        output_period = output_periods[match]
    
        # displaying the input signal period
        if isinstance(input_period, str):
            ax_in.set_title(f"Input period = {input_period}")
        else:
            ax_in.set_title(f"Input period = {input_period:.2f}")

        # displaying the output signal period
        if isinstance(output_period, str):
            ax_out.set_title(f"Output period = {output_period}")
        else:
            ax_out.set_title(f"Output period = {output_period:.2f}")

        # adding difference title
        ax_diff.set_title(f"Difference between")

        # setting row of images accordingly
        images[i] = im_in, im_out, im_diff
    
    # ----------------- END OF MATCHED INPUT SIGNALS -------------------------        

    # ----------------- START OF UNMATCHED OUTPUT SIGNALS -------------------------
    
    # if there are unmatched modes
    for j, idx in enumerate(unmatched_mode_idx):
        output_state_v = output_mode_start[idx].reshape(state_shape)
        output_period = output_periods[idx]
        j += len(input_idx)
    
        vmin = np.min(output_state_v)
        vmax = np.max(output_state_v)
        vlimits = (vmin, vmax)
    
        ax_out = axes[j, 1]
        im_out = Frame_Image(output_state_v, ax_out, vlimits)
    
        images[j][1] = im_out

        # setting colour bar for the whole row
        # in this case just an unmatched mode
        cbar = fig.colorbar(
            im_out,
            ax=axes[i, :],
            location="right",
            shrink=0.8
        )
        cbar.set_label("nT")

        ax_out.set_title(f"Output_period = {output_period}")

    # ----------------- END OF UNMATCHED OUTPUT SIGNALS -------------------------

    # ----------------- START OF TOTAL SIGNALS -------------------------

    # getting the total signal data for input, output and difference
    tot_input_v, tot_output_v = total_input[:,0], total_output[:,0]
    tot_diff_v = np.abs(tot_output_v - tot_input_v)
    
    # plot the total reconstruction from both
    vmax = np.max(np.abs([tot_input_v, tot_output_v, tot_diff_v]))
    vmin = -vmax

    # input
    ax_tot_in = axes[-1, 0]
    im_tot_in = ax_tot_in.imshow(
        (tot_input_v.reshape(state_shape)),
        cmap="seismic",
        vmin=vmin,
        vmax=vmax,
        origin="lower")    
    images[-1][0] = im_tot_in

    # output
    ax_tot_out = axes[-1, 1]
    initial_tot_output = total_output[:,0].reshape(state_shape)
    im_tot_out = ax_tot_out.imshow(
        (tot_output_v.reshape(state_shape)),
        cmap="seismic",
        vmin=vmin,
        vmax=vmax,
        origin="lower")   
    images[-1][1] = im_tot_out

    # difference
    ax_tot_diff = axes[-1, 2]
    im_tot_diff = ax_tot_diff.imshow(
        (tot_diff_v.reshape(state_shape)),
        cmap="seismic",
        vmin=vmin,
        vmax=vmax,
        origin="lower")  
    images[-1][2] = im_tot_diff

    cbar = fig.colorbar(
        im_tot_in,
        ax=axes[-1, :],
        location="right",
        shrink=0.8
    )
    cbar.set_label("nT")

    # setting titles:
    ax_tot_in.set_title(f"Total input signal")
    ax_tot_out.set_title(f"Total output signal")
    ax_tot_diff.set_title(f"Difference between")

    # ----------------- END OF TOTAL SIGNALS -------------------------
        

    # hide blank axes and remove them from layout calculation
    for r in range(num_rows):
        for c in range(num_cols):
            if images[r][c] is None:
                axes[r, c].set_visible(False)
                axes[r, c].set_in_layout(False)
    # define update (essentially same as above but with flexible axes)

    # updating each time step 
    # this now requires actually computing the time step before updating
    def update(time_step):

        # each update need:
        # input: eigenvalue, time, complex mode field
        # output: initial field, time (step!), eigenvalue
        # already have: mode_eigenvalues, wave_eigenvalues
        # k (=timestep), t (=times[k]), complex mode field = [data[1] for data in input_wave_data]
        # initial output fields - have, but complicated as have best match idx, which corresponds to
        # mode_pairing = tuple or int corresponding to indices of the paired dmd modes

        time = time_step * dt

        # matched inputs
        for i in input_idx:
            input_state_v = np.real(np.exp(wave_eigenvalues[i] * time) * input_wave_start_c[i])
            match = best_match_mode_idx[i]
            idx = mode_indices[match]
            if type(idx)==tuple:
                i_1, i_2 = idx
                part_1 = modes[:, i_1] * (mode_eigenvalues[i_1]**time_step)
                part_2 = modes[:, i_2] * (mode_eigenvalues[i_2]**time_step)
                output_state_v = (np.real(part_1 + part_2)).reshape(state_shape)    
            else:
                idx = mode_indices[match]
                output_state_v = (np.real(modes[:, idx] * mode_eigenvalues[idx])).reshape(state_shape)

            diff_state_v = np.abs(input_state_v - output_state_v)
    
            images[i][0].set_data(input_state_v.reshape(snapshot_shape))
            images[i][1].set_data(output_state_v.reshape(snapshot_shape))
            images[i][2].set_data(diff_state_v.reshape(snapshot_shape))

        # unmatched output modes
        for j, idx in enumerate(unmatched_mode_idx):

            if type(idx)==tuple:
                i_1, i_2 = idx
                part_1 = modes[:, i_1] * (mode_eigenvalues[i_1]**time_step)
                part_2 = modes[:, i_2] * (mode_eigenvalues[i_2]**time_step)
                output_state_v = (np.real(part_1 + part_2)).reshape(state_shape)    
            else:
                output_state_v = (np.real(modes[:, idx] * mode_eigenvalues[idx])).reshape(state_shape)
            row = j + len(input_idx)
            images[row][1].set_data(output_state_v.reshape(state_shape))

        # total input and reconstruction
        k_tot_input = total_input[:,time_step].reshape(state_shape)
        k_tot_output = total_output[:,time_step].reshape(state_shape)
        k_tot_diff = np.abs(k_tot_output - k_tot_input)

        images[-1][0].set_data(k_tot_input)
        images[-1][1].set_data(k_tot_output)
        images[-1][2].set_data(k_tot_diff)

        # updating title
        title.set_text(
            f"Pattern comparison | frame {time_step + 1} / {len(times)}"
        )

        # returning all used image artist objects
        return [im for row in images for im in row if im is not None]
    
    # hide any axes that never received an image
    for r in range(num_rows):
        for c in range(num_cols):
            if images[r][c] is None:
                axes[r, c].set_visible(False)
    
    anim = FuncAnimation(
            fig,
            update,
            frames=len(times),
            interval=50,
            blit=True
        )

    from datetime import datetime

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    anim.save(
        f"{FIG_DIR}/video_test_{timestamp}.mp4",
        writer="ffmpeg",
        fps=2,
        dpi=100
    )
    
    plt.close(fig)
    return Video(f"{FIG_DIR}/video_test_{timestamp}.mp4", embed=True)



def Synthetic_Video_Comparison_v2(component, input_data, total_input, decomp_data, total_output, mode_indices, times):

    comp_idx_dict = {
      "br": 1,
      "utheta": 2,
      "uphi": 3
    }

    component_idx = comp_idx_dict[component]

    # time spacing
    dt = times[1]

    # unpacking decomposition (assume br at the moment)
    modes, mode_eigenvalues = decomp_data

    # felix wave primitives
    num_inputs = len(input_data)
    input_wave_start_c = [data_row[component_idx] for data_row in input_data] # i.e. Re[Br]
    input_wave_start = [data_row.real for data_row in input_wave_start_c] # i.e. Re[Br]
    wave_eigenvalues = [data_row[-1] for data_row in input_data]
    input_periods = [(2*np.pi/eig.imag) for eig in wave_eigenvalues]
    
    # constructing first frame of each output mode (need all to be real)
    num_outputs = len(mode_indices)
    output_mode_start = []
    output_periods = []
    
    for idx in mode_indices:
        
        if type(idx)==tuple:
            i_1, i_2 = idx
            output_period = np.abs(0.5/(cmath.phase(mode_eigenvalues[i_1])/(2*np.pi)))
            output_mode = np.real(modes[:, i_1] + modes[:, i_2])
            
        else:
            output_period = np.abs(0.5/(cmath.phase(mode_eigenvalues[idx])/(2*np.pi)))
            output_mode = np.real(modes[:, idx])

        output_periods.append(output_period)
        output_mode_start.append(output_mode)

        
    # intialising correlation matrix
    corr_matrix = np.zeros((num_outputs, num_inputs))

    # computing and storing correlation matrix
    for i, out in enumerate(output_mode_start):
        for j, inp in enumerate(input_wave_start):
            corr_matrix[i, j] = np.corrcoef(inp.ravel(), out.ravel())[0,1]

     # finding, for each wave, which mode has the highest correlation
    best_match_mode_idx = np.argmax(corr_matrix, axis=0)
    
    # make lists, each element representing an input/ output pattern
    input_idx = np.arange(num_inputs)
    output_idx = np.arange(num_outputs)

    # defining indices for the modes that were not an optimal match to the true inputs
    unmatched_mode_idx = [x for x in output_idx if x not in best_match_mode_idx]

    # number of final plot rows
    # as dmd modes can repeat:
    num_rows = len(unmatched_mode_idx) + len(best_match_mode_idx) + 1 
    # *note: 1 extra is for the total summed superposition

    num_cols = 3 # for 3 columns: input, output, difference
    snapshot_shape = (181, 360) # this is the snapshot shape used for synthetics
    
    fig, axes = plt.subplots(
        num_rows,
        num_cols,
        figsize=(3 * num_cols + 1, 1.8 * num_rows + 3),
        squeeze=False,
        constrained_layout=True
    )
    
    title = fig.suptitle(
        f"Pattern comparison | frame 1 / {len(times)}",
        fontsize=14,
        y=0.995
    )

    #fig.tight_layout()
    
    # intialise images for updating
    images = [[None for j in range(num_cols)] for i in range(num_rows)]

    # ----------------- START OF MATCHED INPUT SIGNALS -------------------------
    
    # loop over each wave
    for i in input_idx:

        # have set of initial conditions already
        input_state_v = input_wave_start[i]

        # get matched mode index
        match = best_match_mode_idx[i]

        output_state_v = output_mode_start[match].reshape(state_shape)
        diff_state_v = np.sqrt((np.abs(input_state_v - output_state_v)**2))

        # defining colour bar scales for this row - used for all plots
        # get the largest absolute value across all plots, use as maximum on common scale
        vmax = np.max(np.abs([input_state_v, output_state_v, diff_state_v]))
        vmin = -vmax
        vlimits = (vmin, vmax)
    
        # set axes for the row, ordered: input, output, difference
        ax_in, ax_out, ax_diff = axes[i, :]
    
        im_out = Frame_Image(output_state_v, ax_out, vlimits)
        im_diff = Frame_Image(diff_state_v, ax_diff, vlimits)
        im_in = Frame_Image(input_state_v, ax_in, vlimits)

        # setting colour bar for the whole row
        cbar = fig.colorbar(
            im_in,
            ax=axes[i, :],
            location="right",
            shrink=0.8
        )
        
        cbar.set_label("nT")

        # getting periods of input and output
        input_period = input_periods[i]
        output_period = output_periods[match]
    
        # displaying the input signal period
        if isinstance(input_period, str):
            ax_in.set_title(f"Input period = {input_period}")
        else:
            ax_in.set_title(f"Input period = {input_period:.2f}")

        # displaying the output signal period
        if isinstance(output_period, str):
            ax_out.set_title(f"Output period = {output_period}")
        else:
            ax_out.set_title(f"Output period = {output_period:.2f}")

        # adding difference title
        ax_diff.set_title(f"Difference between")

        # setting row of images accordingly
        images[i] = im_in, im_out, im_diff
    
    # ----------------- END OF MATCHED INPUT SIGNALS -------------------------        

    # ----------------- START OF UNMATCHED OUTPUT SIGNALS -------------------------
    
    # if there are unmatched modes
    for j, idx in enumerate(unmatched_mode_idx):
        output_state_v = output_mode_start[idx].reshape(state_shape)
        output_period = output_periods[idx]
        j += len(input_idx)
    
        vmin = np.min(output_state_v)
        vmax = np.max(output_state_v)
        vlimits = (vmin, vmax)
    
        ax_out = axes[j, 1]
        im_out = Frame_Image(output_state_v, ax_out, vlimits)
    
        images[j][1] = im_out

        # setting colour bar for the whole row
        # in this case just an unmatched mode
        cbar = fig.colorbar(
            im_out,
            ax=axes[i, :],
            location="right",
            shrink=0.8
        )
        cbar.set_label("nT")

        ax_out.set_title(f"Output_period = {output_period}")

    # ----------------- END OF UNMATCHED OUTPUT SIGNALS -------------------------

    # ----------------- START OF TOTAL SIGNALS -------------------------

    # getting the total signal data for input, output and difference
    tot_input_v, tot_output_v = total_input[:,0], total_output[:,0]
    tot_diff_v = np.abs(tot_output_v - tot_input_v)
    
    # plot the total reconstruction from both
    vmax = np.max(np.abs([tot_input_v, tot_output_v, tot_diff_v]))
    vmin = -vmax

    # input
    ax_tot_in = axes[-1, 0]
    im_tot_in = ax_tot_in.imshow(
        (tot_input_v.reshape(state_shape)),
        cmap="seismic",
        vmin=vmin,
        vmax=vmax,
        origin="lower")    
    images[-1][0] = im_tot_in

    # output
    ax_tot_out = axes[-1, 1]
    initial_tot_output = total_output[:,0].reshape(state_shape)
    im_tot_out = ax_tot_out.imshow(
        (tot_output_v.reshape(state_shape)),
        cmap="seismic",
        vmin=vmin,
        vmax=vmax,
        origin="lower")   
    images[-1][1] = im_tot_out

    # difference
    ax_tot_diff = axes[-1, 2]
    im_tot_diff = ax_tot_diff.imshow(
        (tot_diff_v.reshape(state_shape)),
        cmap="seismic",
        vmin=vmin,
        vmax=vmax,
        origin="lower")  
    images[-1][2] = im_tot_diff

    cbar = fig.colorbar(
        im_tot_in,
        ax=axes[-1, :],
        location="right",
        shrink=0.8
    )
    cbar.set_label("nT")

    # setting titles:
    ax_tot_in.set_title(f"Total input signal")
    ax_tot_out.set_title(f"Total output signal")
    ax_tot_diff.set_title(f"Difference between")

    # ----------------- END OF TOTAL SIGNALS -------------------------
        

    # hide blank axes and remove them from layout calculation
    for r in range(num_rows):
        for c in range(num_cols):
            if images[r][c] is None:
                axes[r, c].set_visible(False)
                axes[r, c].set_in_layout(False)
    # define update (essentially same as above but with flexible axes)

    # updating each time step 
    # this now requires actually computing the time step before updating
    def update(time_step):

        # each update need:
        # input: eigenvalue, time, complex mode field
        # output: initial field, time (step!), eigenvalue
        # already have: mode_eigenvalues, wave_eigenvalues
        # k (=timestep), t (=times[k]), complex mode field = [data[1] for data in input_wave_data]
        # initial output fields - have, but complicated as have best match idx, which corresponds to
        # mode_pairing = tuple or int corresponding to indices of the paired dmd modes

        time = time_step * dt

        # matched inputs
        for i in input_idx:
            input_state_v = np.real(np.exp(wave_eigenvalues[i] * time) * input_wave_start_c[i])
            match = best_match_mode_idx[i]
            idx = mode_indices[match]
            if type(idx)==tuple:
                i_1, i_2 = idx
                part_1 = modes[:, i_1] * (mode_eigenvalues[i_1]**time_step)
                part_2 = modes[:, i_2] * (mode_eigenvalues[i_2]**time_step)
                output_state_v = (np.real(part_1 + part_2)).reshape(state_shape)    
            else:
                idx = mode_indices[match]
                output_state_v = (np.real(modes[:, idx] * mode_eigenvalues[idx])).reshape(state_shape)

            diff_state_v = np.abs(input_state_v - output_state_v)
    
            images[i][0].set_data(input_state_v.reshape(snapshot_shape))
            images[i][1].set_data(output_state_v.reshape(snapshot_shape))
            images[i][2].set_data(diff_state_v.reshape(snapshot_shape))

        # unmatched output modes
        for j, idx in enumerate(unmatched_mode_idx):

            if type(idx)==tuple:
                i_1, i_2 = idx
                part_1 = modes[:, i_1] * (mode_eigenvalues[i_1]**time_step)
                part_2 = modes[:, i_2] * (mode_eigenvalues[i_2]**time_step)
                output_state_v = (np.real(part_1 + part_2)).reshape(state_shape)    
            else:
                output_state_v = (np.real(modes[:, idx] * mode_eigenvalues[idx])).reshape(state_shape)
            row = j + len(input_idx)
            images[row][1].set_data(output_state_v.reshape(state_shape))

        # total input and reconstruction
        k_tot_input = total_input[:,time_step].reshape(state_shape)
        k_tot_output = total_output[:,time_step].reshape(state_shape)
        k_tot_diff = np.abs(k_tot_output - k_tot_input)

        images[-1][0].set_data(k_tot_input)
        images[-1][1].set_data(k_tot_output)
        images[-1][2].set_data(k_tot_diff)

        # updating title
        title.set_text(
            f"Pattern comparison | frame {time_step + 1} / {len(times)}"
        )

        # returning all used image artist objects
        return [im for row in images for im in row if im is not None]
    
    # hide any axes that never received an image
    for r in range(num_rows):
        for c in range(num_cols):
            if images[r][c] is None:
                axes[r, c].set_visible(False)
    
    anim = FuncAnimation(
            fig,
            update,
            frames=len(times),
            interval=50,
            blit=True
        )

    from datetime import datetime

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    anim.save(
        f"{FIG_DIR}/video_test_{timestamp}.mp4",
        writer="ffmpeg",
        fps=2,
        dpi=100
    )
    
    plt.close(fig)
    return Video(f"{FIG_DIR}/video_test_{timestamp}.mp4", embed=True)



