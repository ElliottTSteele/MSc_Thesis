'''
syn_pipeline.py

purpose:
be a function repository inside R test synthetics to store static helper
code. This code will not be updated (to prevent breakage of code that
utilises it) inside this file once implemented, but it may be extracted and
places into sdm py after completion
'''

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
import pickle
from pathlib import Path

# import paths.py to establish file structure and directories
from src.msc_thesis.paths import *

# ---------------------------------------------------------
# NON-PHYSICAL -> PHYSICAL UNIT CONVERSION
# ---------------------------------------------------------

# PREAMBLE FOR ARBITRARY ALFVEN TO ARBITRARY PHYSICAL UNITS
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

# max spherical harmonic degree provided by Felix - 60 as of 03/07/2026
lmax_felix_provided = 60

# ---------------------------------------------------------
# SPATIAL PARAMETERS
# ---------------------------------------------------------

r_cmb = 3485 # km radius at CMB
r_earth = 6371.2 # km radius at earth surface

# latitude/ longitude parameters
d_degree = 5 # cell width in degrees
colatitude = np.arange(5, 176, d_degree)
latitude = 90 - colatitude 
longitude = np.arange(0, 360, d_degree)
# theta, phi (radians):
theta = np.deg2rad(colatitude)
phi = np.deg2rad(longitude)

# latitude weighting variables
phi_grid, theta_grid = np.meshgrid(phi, theta)
W_theta = np.sin(theta)
W2D = np.sin(theta_grid)
W2D_norm = W2D / np.sum(W2D)

# getting dimensionality
state_shape = (len(theta), len(phi)) # snapshot shape used globally
n_points_globally = len(theta) * len(phi) # number of points in grid

# ---------------------------------------------------------
# TEMPORAL PARAMETERS
# ---------------------------------------------------------

# Time information consistent with resolution output
f = h5py.File((f'{CHAOS_RESOL_DIR}/CHAOS_Resol_1997_2026_0806_SV.h5'),"r")

# getting time parameters
# array of time steps (decimal year)
times_res = np.asarray(f['tp'])
# value of temporal sampling spacing
dt_res = np.asarray(f["dt"])
# number of time points in total
Nt_res = np.asarray(f["n_tp"])

'''print(f"The resolved output format spans {np.min(times_res)}-{np.max(times_res)} \n\
With a gauss coefficient knot spacing of {dt_res} years \n\
Resulting in {Nt_res} time sample points per time series")
'''
f.close()

# converting to mjd (to withdraw from ChaosMagPy)
times_res_mjd = cp.dyear_to_mjd(times_res)

# getting relative times (for synthetic consistency)
times_res_relative = times_res - times_res[0]

# record length and fourier resolution
t_start = times_res[0] # start time
t_end = times_res[-1] # end time
t_record = t_end - t_start # record length
f_bin = 1 / t_record # associatied fourier resolution

# ---------------------------------------------------------
# DMD INPUT RECORD DEFINITION
# ---------------------------------------------------------

# real start of data input for DMD
t_dmd_start = 2000.0
t_dmd_end = 2024.0
dt_sample = 0.5
f_sample = 1/dt_sample
times_used = np.arange(t_dmd_start, t_dmd_end+0.5, dt_sample)
times_used_relative = times_used - t_dmd_start

times_evaluate_ideal_phasors = times_used - times_res[0]
times_used_mjd = cp.dyear_to_mjd(times_used)

fourier_bin_width = 1/( t_dmd_end - t_dmd_start)



# ---------------------------------------------------------
# MODE SCALINGS
# ---------------------------------------------------------


scale_file = Path(FELIX_DIR) / "amplitude_scalings.pkl"

with open(scale_file, "rb") as file:
    mode_amp_scalings = pickle.load(file)

# ---------------------------------------------------------
# MATCH SIMILARITY THRESHOLDS
# ---------------------------------------------------------

period_frac_threshold = 0.25
frequency_error_threshold = fourier_bin_width/2

match_file = Path(FELIX_DIR) / "match_metrics.pkl"

with open(match_file, "rb") as file:
    similarity_thresholds = pickle.load(file)