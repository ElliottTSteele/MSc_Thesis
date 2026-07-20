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

# ---------------------------------------------------------
# TEMPORAL PARAMETERS
# ---------------------------------------------------------

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

# ---------------------------------------------------------
# HIGH QUALITY RECORD DEFINITION
# ---------------------------------------------------------

# important events and reliable times recorded
t_r_start = 1997.1
times_mjd2000 = cp.data_utils.dyear_to_mjd(times_dyear+t_r_start)
dt_years = 0.2

notable_times_raw = {
    "CHAMP Start (2000/08)": 2000 + 8/12,
    "CHAMP End (2010/09)": 2010 + 9/12,
    "Swarm Start (2013/11)": 2013 + 11/12,
    "(2026/01)": 2026 
}

notable_times_relative = {}
notable_times_aspline = {}

for event in notable_times_raw:
    notable_times_relative[event] = \
        (notable_times_raw[event] - t_r_start)
    
# 'good' record mask
# absolute decimal-year time associated with each Gauss time step
times_absolute = t_r_start + times_dyear

# reliable record limits
good_record_start = notable_times_raw["CHAMP Start (2000/08)"]
good_record_end = notable_times_raw["(2026/01)"]

# indices lying within the good record
good_record_idx = np.where(
    (times_absolute >= good_record_start)
    & (times_absolute <= good_record_end)
)[0]

# start/end indices and equivalent slice
good_record_start_idx = good_record_idx[0]
good_record_end_idx = good_record_idx[-1]

good_record_slice = slice(
    good_record_start_idx,
    good_record_end_idx + 1
)


# ---------------------------------------------------------
# MODE SCALINGS
# ---------------------------------------------------------


scale_file = Path(FELIX_DIR) / "mode_amplitude_scalings_v1.pkl"

with open(scale_file, "rb") as file:
    mode_amp_scalings = pickle.load(file)