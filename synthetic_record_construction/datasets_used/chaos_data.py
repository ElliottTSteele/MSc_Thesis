# load in chaos 8.6 data - present SV at demo time step on mollweide projection

# Copyright (C) 2025 Clemens Kloss
#
# This script is free software: you can redistribute it and/or modify it under
# the terms of the GNU Lesser General Public License as published by the Free
# Software Foundation, either version 3 of the License, or (at your option) any
# later version.
#
# This script is distributed in the hope that it will be useful, but WITHOUT
# ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
# FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public License for more
# details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with this script. If not, see <https://www.gnu.org/licenses/>.

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.msc_thesis.paths import *
import chaosmagpy as cp
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
from src.msc_thesis.synSetup import *

model = cp.CHAOS.from_mat(f'{CHAOS_DIR}/CHAOS-8.6.mat')  # load the mat-file of CHAOS-7

time = cp.data_utils.mjd2000(2020, 1, 1)  # convert date to mjd2000
radius = r_cmb  # radius of the core surface in km
theta = np.linspace(1., 179., 181)  # colatitude in degrees
phi = np.linspace(-180., 180, 361)  # longitude in degrees

# compute radial SV up to degree 16 using CHAOS
B, _, _ = model.synth_values_tdep(time, radius, theta, phi,
                                  nmax=15, deriv=1, grid=True)

limit = 30e3  # nT/yr colorbar limit

fig = plt.figure(figsize=(text_width, text_width), layout="constrained")
gs = fig.add_gridspec(3, 2, height_ratios=[1, 0.72, 0.06])
axes = [
    fig.add_subplot(gs[0, :], projection=ccrs.Mollweide()),
    fig.add_subplot(
        gs[1, 0], projection=ccrs.NearsidePerspective(central_latitude=90.)
    ),
    fig.add_subplot(
        gs[1, 1], projection=ccrs.NearsidePerspective(central_latitude=-90.)
    ),
]

for ax in axes:
    pc = ax.pcolormesh(phi, 90. - theta, B, cmap='seismic', vmin=-limit,
                       vmax=limit, transform=ccrs.PlateCarree())
    ax.gridlines(linewidth=0.5, linestyle='dashed', color='grey',
                 ylocs=np.linspace(-90, 90, num=7),  # parallels
                 xlocs=np.linspace(-180, 180, num=13))  # meridians
    ax.coastlines(linewidth=0.8, color='k')

for ax, title in zip(axes, ["World view", "North pole view", "South pole view"]):
    ax.set_title(title)

cax = fig.add_subplot(gs[2, :])
clb = fig.colorbar(pc, cax=cax, extend='both', orientation='horizontal')
clb.set_label('Secular Variation (nT/yr)', fontsize=12)
fig.suptitle(r"CHAOS-8.6 Secular Variation $n \leq 15$ (2020/01/01)")

plt.show()
