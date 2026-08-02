"""
Construction-specific compatibility helpers.

The former synthetic construction pipeline combined shared setup, general
utilities, and a small amount of construction-only functionality. Shared
definitions are re-exported here from the same ``synSetup`` and ``synUtils``
modules used by the testing programs. The construction-only helpers below
preserve the behaviour of their former counterparts.
"""

import h5py
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm
from matplotlib.colors import LogNorm

from src.msc_thesis.paths import FELIX_DIR
from src.msc_thesis.synSetup import *
from src.msc_thesis.synUtils import *


def PSD_3D_Bar_Visualise(
    degree_psds,
    frequencies,
    degrees=None,
    title=None,
    cmap="rainbow",
    vmin=None,
    vmax=None,
):
    """
    Plot degree-wise PSD as contiguous 3D bars.

    Each bar fills its complete frequency-degree cell, with cell boundaries
    halfway between neighbouring frequency and degree coordinates.

    Parameters
    ----------
    degree_psds : (n_degree, n_freq) array
        PSD values.
    frequencies : (n_freq,) array
        Uniformly spaced frequency coordinates.
    degrees : (n_degree,) array-like, optional
        Degree coordinates. Defaults to 1, ..., n_degree.
    title : str, optional
        Figure title.
    cmap : str, optional
        Colormap name.
    vmin, vmax : float, optional
        Colour-normalisation limits.
    """

    degree_psds = np.asarray(degree_psds, dtype=float)
    frequencies = np.asarray(frequencies, dtype=float)

    if degree_psds.ndim != 2:
        raise ValueError("degree_psds must be a 2D array.")

    if frequencies.ndim != 1:
        raise ValueError("frequencies must be a 1D array.")

    if degree_psds.shape[1] != frequencies.size:
        raise ValueError(
            "frequencies length must match the second dimension of degree_psds."
        )

    if frequencies.size < 2:
        raise ValueError("At least two frequency points are required.")

    if degrees is None:
        degrees = np.arange(1, degree_psds.shape[0] + 1, dtype=float)
    else:
        degrees = np.asarray(degrees, dtype=float)

    if degrees.ndim != 1 or degrees.size != degree_psds.shape[0]:
        raise ValueError(
            "degrees length must match the first dimension of degree_psds."
        )

    if degrees.size < 2:
        raise ValueError("At least two spherical harmonic degrees are required.")

    # Constant cell widths
    df = frequencies[1] - frequencies[0]
    dn = degrees[1] - degrees[0]

    if df <= 0:
        raise ValueError("frequencies must be strictly increasing.")

    if dn <= 0:
        raise ValueError("degrees must be strictly increasing.")

    if not np.allclose(np.diff(frequencies), df):
        raise ValueError("frequencies must be uniformly spaced.")

    if not np.allclose(np.diff(degrees), dn):
        raise ValueError("degrees must be uniformly spaced.")

    # Lower-left corner of every cell
    frequency_left = frequencies - df / 2
    degree_left = degrees - dn / 2

    X, Y = np.meshgrid(
        frequency_left,
        degree_left,
        indexing="xy",
    )

    valid = np.isfinite(degree_psds) & (degree_psds > 0)

    positive_psd = degree_psds[valid]

    if positive_psd.size == 0:
        raise ValueError("No positive finite PSD values found.")

    if vmin is None:
        vmin = positive_psd.min()

    if vmax is None:
        vmax = positive_psd.max()

    if vmin <= 0 or vmax <= vmin:
        raise ValueError("Require 0 < vmin < vmax.")

    # Start bars at the lower positive limit so the logarithmic z-axis works
    z_base = vmin

    X = X[valid].ravel()
    Y = Y[valid].ravel()
    CVAL = degree_psds[valid].ravel()

    Z0 = np.full_like(X, z_base)
    DX = np.full_like(X, df)
    DY = np.full_like(X, dn)
    DZ = np.maximum(CVAL - z_base, np.finfo(float).eps * z_base)

    # Colour bars by PSD value
    norm = LogNorm(vmin=vmin, vmax=vmax)
    cmap_obj = plt.get_cmap(cmap)
    colors = cmap_obj(norm(CVAL))

    # Plot
    fig = plt.figure(figsize=(13, 9))
    ax = fig.add_subplot(111, projection="3d")

    ax.bar3d(
        X,
        Y,
        Z0,
        DX,
        DY,
        DZ,
        color=colors,
        shade=True,
        zsort="average",
        linewidth=0,
        edgecolor="none",
    )

    ax.set_xlabel("Frequency [cycles yr$^{-1}$]", labelpad=12)
    ax.set_ylabel("Spherical harmonic degree $n$", labelpad=12)
    ax.set_zlabel(
        r"Lowes-weighted SV PSD "
        r"$[(\mathrm{nT\,yr^{-1}})^2/(\mathrm{cycles\,yr^{-1}})]$",
        labelpad=14,
    )

    ax.set_xlim(
        frequencies[0] - df / 2,
        frequencies[-1] + df / 2,
    )
    ax.set_ylim(
        degrees[0] - dn / 2,
        degrees[-1] + dn / 2,
    )

    ax.set_zscale("log")
    ax.set_yticks(degrees)

    if title is not None:
        ax.set_title(title)

    ax.view_init(elev=28, azim=45)

    sm = cm.ScalarMappable(norm=norm, cmap=cmap_obj)
    sm.set_array([])

    cbar = fig.colorbar(
        sm,
        ax=ax,
        pad=0.10,
        shrink=0.65,
        aspect=22,
    )
    cbar.set_label("Lowes-weighted SV PSD")

    plt.tight_layout()
    plt.show()


def Component_Load(mode_number, directory=FELIX_DIR):
    """
    Load the original complex wave components and decaying eigenvalue.

    This intentionally preserves the former ``syn_pipeline.Component_Load``
    convention.  It is distinct from ``synUtils.Component_Load_SV``, which
    converts the magnetic phasors to secular variation and removes decay from
    its main eigenvalue.
    """

    # Select a mode number and corresponding file.
    file = h5py.File(
        f"{directory}/mode_surface_including_gnm_{mode_number}.h5",
        "r",
    )

    # Converts V_alfven (arbitrary) to nT (arbitrary).
    va_to_nt_arbitrary = v_alfven * np.sqrt(mu_0 * rho) * 1e9

    # Gauss coefficient (magnetic scalar potential) phasors.
    gnm = va_to_nt_arbitrary * (
        np.asarray(file["gnmr"]) + 1j * np.asarray(file["gnmi"])
    )

    # Br.
    br = va_to_nt_arbitrary * (
        np.asarray(file["brr"]).T + 1j * np.asarray(file["bri"]).T
    )

    # u_theta.
    utheta = np.asarray(file["uthetar"]).T + 1j * np.asarray(file["uthetai"]).T

    # u_phi.
    uphi = np.asarray(file["uphir"]).T + 1j * np.asarray(file["uphii"]).T

    # True period and decay rate.
    omega = file["omega"][()]
    sigma = file["sigma"][()]
    eigenvalue = sigma + 1j * omega

    file.close()

    return {
        "mode_number": mode_number,
        "gnm": gnm,
        "br": br,
        "utheta": utheta,
        "uphi": uphi,
        "eigenvalue": eigenvalue,
    }
