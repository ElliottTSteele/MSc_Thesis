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
from matplotlib import cm
from matplotlib.colors import BoundaryNorm
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Rectangle
from scipy.signal import (
    butter,
    buttord,
    filtfilt,
    freqz,
)

import warnings

# import paths.py to establish file structure and directories
from src.msc_thesis.paths import *

# import simulation setup
from src.msc_thesis.synSetup import *
from src.msc_thesis.synUtils import *
from src.msc_thesis.synFigures import *

# importing from pyDMD
from pydmd import HankelDMD, DMD, FbDMD, BOPDMD
from pydmd.preprocessing import hankel_preprocessing


# ---------------------------------------------------------------
# Universal limit choices applied across all
# ---------------------------------------------------------------

# By default, retain static modes and oscillatory modes of every period.
# Optional finite bounds remain supported by finalise_candidate_suite for
# experiments that explicitly request them.
candidate_period_min = None
candidate_period_max = None

# ---------------------------------------------------------
# covariance derived perturbation generation
# ---------------------------------------------------------

file_path = Path(f"{CHAOS_COV_DIR}/CHAOS_Cov_1997_2026_0806_SV.h5")

with h5py.File(file_path, "r") as cov_file:

    Cov_full = np.asarray(cov_file["Cnm"])
    print(f"C has shape dimensions: {np.shape(Cov_full)}")

def factor_time_covariance(covariance):

    covariance = np.asarray(covariance, dtype=np.float64)

    # NumPy performs this over all time steps.
    L = np.linalg.cholesky(covariance)

    # checks if needed:
    '''print(np.shape(L))
    print(np.allclose(Cov_full, L @ L.swapaxes(-1, -2)))'''

    return L

def Perturbation_Generate(gnm_input,
                          dt=dt_sample,
                          n_realisations=100,
                          seed=42, 
                          covariance_used=Cov_full, 
                          temporal_z = "independent", 
                          tau=5,
                          just_noise=False):
    
    Nt, Ng = np.shape(gnm_input)
    
    L = factor_time_covariance(covariance_used)

    rng = np.random.default_rng(seed)

    # Store the covariance-correlated perturbation realisations.
    perturbed_gauss = np.empty(
        (n_realisations, Nt, Ng),
        dtype=np.float64,
    )

    if temporal_z == "independent":
        # New independent z for every realisation and every time step.
        z = rng.standard_normal(
            (n_realisations, Nt, Ng)
        )

        for t in range(Nt):
            perturbed_gauss[:, t, :] = z[:, t, :] @ L[t].T

    elif temporal_z == "constant":
        # One latent z per realisation, reused at every time step.
        z_constant = rng.standard_normal(
            (n_realisations, Ng)
        )

        for t in range(Nt):
            perturbed_gauss[:, t, :] = z_constant @ L[t].T

    elif temporal_z == "ar1":
        if tau is None:
            raise ValueError(
                "tau must be provided when temporal_z='ar1'"
            )

        if tau <= 0:
            raise ValueError("tau must be positive")

        # AR(1) coefficient corresponding to the specified
        # e-folding correlation time tau.
        rho = np.exp(-dt / tau)

        # Innovation scaling ensures that every z[t] has stationary
        # marginal variance equal to one.
        innovation_scale = np.sqrt(1.0 - rho**2)

        z_ar1 = np.empty(
            (n_realisations, Nt, Ng),
            dtype=np.float64,
        )

        # Draw the initial state from the stationary N(0, I)
        # distribution.
        z_ar1[:, 0, :] = rng.standard_normal(
            (n_realisations, Ng)
        )

        # Generate temporally correlated latent vectors.
        for t in range(1, Nt):
            innovation = rng.standard_normal(
                (n_realisations, Ng)
            )

            z_ar1[:, t, :] = (
                rho * z_ar1[:, t - 1, :]
                + innovation_scale * innovation
            )

        # Apply the time-dependent Gauss covariance factors.
        for t in range(Nt):
            perturbed_gauss[:, t, :] = (
                z_ar1[:, t, :] @ L[t].T
            )

    else:
        raise ValueError(
            "temporal_z must be one of "
            "'independent', 'constant', or 'ar1'"
        )

    noise_gauss = perturbed_gauss.copy()

    if just_noise:
        return(perturbed_gauss)

    # Add the mean CHAOS model.
    perturbed_gauss += gnm_input[None, :, :]

    # return ensemble of original models + noise realisations
    return(perturbed_gauss)



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

# ---------------------------------------------------------
# BUIDLING DMD CLASS OBJECT 
# ---------------------------------------------------------

def Build_Exact_DMD(svd_rank, fbdmd=False):

    return DMD(
        svd_rank=svd_rank,
        exact=True,
        forward_backward=fbdmd,
    )

def Build_BOPDMD(svd_rank, bopdmd=False, bopdmd_params=None):

    if bopdmd:
        return BOPDMD(
                svd_rank=svd_rank,
                num_trials=bopdmd_params["num_trials"],
                trial_size=bopdmd_params["trial_size"],
                remove_bad_bags=bopdmd_params["remove_bad_bags"],
                eig_constraints={"conjugate_pairs", "imag"},
                varpro_opts_dict={
                    "maxiter": 100,
                    "tol": 1e-1,
                    "verbose": False,
                },
            )
    else:
        return BOPDMD(
                svd_rank=svd_rank,
                varpro_opts_dict={
                    "maxiter": 500,
                    "tol": 0.00001,
                    "verbose": False,
                },
            )

# ---------------------------------------------------------
# RUN DMD
# ---------------------------------------------------------

# Area weights for the flattened spatial grid
area_weights_2d = np.sin(theta_grid)
sqrt_area_weights_flat = np.sqrt(area_weights_2d.ravel())

assert sqrt_area_weights_flat.size == n_points_globally

def Weight_Snapshot_Series(X):
    return(sqrt_area_weights_flat[:, None] * X)

def Unweight_Phasor(
    weighted_phasor,
    pole_value=np.nan,
    sqrt_area_weights_flat=sqrt_area_weights_flat
):
    weighted_phasor = np.asarray(weighted_phasor)
    sqrt_area_weights_flat = np.asarray(sqrt_area_weights_flat)

    if weighted_phasor.size != sqrt_area_weights_flat.size:
        raise ValueError(
            f"Phasor length {weighted_phasor.size} does not match "
            f"weight length {sqrt_area_weights_flat.size}."
        )

    physical_phasor = np.full(
        weighted_phasor.shape,
        pole_value,
        dtype=complex,
    )

    nonzero = sqrt_area_weights_flat > 0

    physical_phasor[nonzero] = (
        weighted_phasor[nonzero]
        / sqrt_area_weights_flat[nonzero]
    )

    return physical_phasor


def Run_DMD(X, times_used, svd_rank, dmd_settings):

    # unpacking dmd settings
    dmd_type = dmd_settings["dmd_type"]
    hankel_flag = dmd_settings["hankel_flag"]
    embedding_d = dmd_settings["embedding_d"]
    weight_flag=dmd_settings["weight_flag"]

    # weight snapshot series input data
    if weight_flag:
        X = Weight_Snapshot_Series(X)

    # build desired dmd object
    if dmd_type == "exact":
        dmd_base = Build_Exact_DMD(svd_rank)
    elif dmd_type == "opdmd":
        dmd_base = Build_BOPDMD(svd_rank)

    # apply hankel preprocessing if enabled
    if hankel_flag:

        dmd = hankel_preprocessing(dmd_base, 
                                   d=embedding_d, 
                                   reconstruction_method="first")
        
        times_fit = times_used[: -(embedding_d-1)]

    else:

        dmd = dmd_base
        times_fit = times_used

    if dmd_type == 'exact':
        dmd.fit(X)

    elif dmd_type == 'opdmd':
        dmd.fit(X, times_fit)

    recovered_modes = DMD_Mode_Process(dmd, dmd_type, dt_sample, hankel_flag, embedding_d)

    # getting integer effective svd rank
    effective_rank = dmd.operator.shape[0]

    if weight_flag:
        recovered_modes["phasors"] = np.asarray([Unweight_Phasor(phasor) for phasor in recovered_modes["phasors"]])

    recovered_modes["effective_rank"] = effective_rank

    return(recovered_modes)
# ---------------------------------------------------------
# CONVERT DMD OUTPUT -> PHYSICAL MODES
# ---------------------------------------------------------

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

# physicalise phasor by taking first embedded state vector
def Physicalise_Phasor(embedded_phasor, embedding_d):

    embedded_length = len(embedded_phasor)
    physical_length = int(embedded_length / embedding_d)
    physical_phasor = embedded_phasor[:physical_length]

    return(physical_phasor)

# takes in dmd object, and time spacing
def DMD_Mode_Process(
    dmd,
    dmd_type_used,
    dt,
    hankel,
    embedding_d,
    atol=1e-4,
    rtol=1e-4
    ):

    # gets eigenvalues, modes and amplitues (raw output)
    eigenvalues = np.asarray(dmd.eigs, dtype=complex)
    modes = np.asarray(dmd.modes, dtype=complex)
    amplitudes = np.asarray(dmd.amplitudes, dtype=complex)

    # Complete dynamically scaled modal coefficients b_j phi_j.
    scaled_modes = modes * amplitudes[np.newaxis, :]

    # initialise objects to handle pairing algorithm
    used = set()
    output_phasors = []
    output_eigenvalues = []
    mode_information = []

    # for each eigenvalue in modes
    for i, eig_i in enumerate(eigenvalues):

        # if eigenvalue already 'ticked off' from examination, skip
        if i in used:
            continue

        # ---------------------------------------------------------
        # Real/non-oscillatory eigenvalue - i.e. if eigenvalue has no frequency
        # ---------------------------------------------------------
        if np.isclose(eig_i.imag, 0.0, atol=atol, rtol=rtol):

            output_phasors.append(scaled_modes[:, i])
            output_eigenvalues.append(
                complex(eig_i.real, 0.0)
            )

            mode_information.append({
                "type": "real",
                "indices": (i,),
                "pair_error": np.nan,
            })

            used.add(i)
            continue

        # ---------------------------------------------------------
        # Find the closest unused conjugate eigenvalue - if mode not static
        # ---------------------------------------------------------

        # conjugate candidates = not the current mode and not those already examined
        candidate_indices = [
            j
            for j in range(eigenvalues.size)
            if j != i and j not in used
        ]

        # if viable candidates exist
        if candidate_indices:

            # the desired conjugate value
            conjugate_target = np.conj(eig_i)

            errors = np.array([
                np.abs(eigenvalues[j] - conjugate_target)
                for j in candidate_indices
            ])

            best_position = np.argmin(errors)
            j = candidate_indices[best_position]
            pair_error = errors[best_position]

            # if eigenvalue conjugate found, then there is a match
            is_pair = np.isclose(
                eigenvalues[j],
                conjugate_target,
                atol=atol,
                rtol=rtol,
            )

        else:
            j = None
            pair_error = np.inf
            is_pair = False

        # ---------------------------------------------------------
        # Combine a conjugate pair
        # ---------------------------------------------------------

        # if conjugate match found
        if is_pair:

            # Positive frequency corresponds to positive discrete angle.
            # i.e. which one of pair has positive frequency
            if np.angle(eig_i) > 0:
                positive_idx = i
                negative_idx = j
            else:
                positive_idx = j
                negative_idx = i

            # taking positive frequency eigenvalue
            eig_positive = eigenvalues[positive_idx]

            # If the pair were exact:
            #
            # scaled_negative = conj(scaled_positive)
            #
            # and therefore this equals 2 * scaled_positive.
            phasor = (
                scaled_modes[:, positive_idx]
                + np.conj(scaled_modes[:, negative_idx])
            )

            output_phasors.append(phasor)
            output_eigenvalues.append(eig_positive)

            mode_information.append({
                "type": "conjugate_pair",
                "indices": (positive_idx, negative_idx),
                "pair_error": pair_error,
            })

            used.add(i)
            used.add(j)

        # ---------------------------------------------------------
        # Complex mode without a conjugate partner
        # ---------------------------------------------------------
        else:

            used.add(i)

    # physicalise hankel embedded state vectors
    if hankel:
        output_phasors = [Physicalise_Phasor(phasor, embedding_d) for phasor in output_phasors]

    # storing continuous vs discrete eigenvalues appropriately
    # (DMD output type varies with method)

    if dmd_type_used == 'exact':

        discrete_eigenvalues = output_eigenvalues

        continuous_eigenvalues = np.asarray(
            D_To_C_Eigenvalue_Converter(
                output_eigenvalues,
                dt=dt,
            )
        )

    elif dmd_type_used == 'opdmd':

        continuous_eigenvalues = output_eigenvalues

        discrete_eigenvalues = np.asarray(
            C_To_D_Eigenvalue_Converter(
                output_eigenvalues,
                dt=dt,
            )
        )

    # getting periods from continuous eigenvalues
    periods = np.zeros(len(continuous_eigenvalues))
    for i, eig in enumerate(continuous_eigenvalues):
        try:
            period = 2.0 * np.pi / np.abs(eig.imag)
        except ZeroDivisionError:
            period = np.inf
        periods[i] = period

    return {
        "phasors": output_phasors,
        "discrete_eigenvalues": discrete_eigenvalues,
        "continuous_eigenvalues": np.asarray(continuous_eigenvalues),
        "periods": periods,
        "mode_information": mode_information
    }


# ---------------------------------------------------------------
# success metric related
# ---------------------------------------------------------------



# returns quantiles of data set
def finite_quantiles(
    values,
    quantiles=(0.05, 0.5, 0.95),
):
    """Return finite-data quantiles, or None if no finite values exist."""

    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]

    if values.size == 0:
        return None

    return np.quantile(values, quantiles)

def SV_Grid_Phasor_Record_Power(
    phasor,
    eigenvalue,
    evaluation_times,
    spatial_weights=None,
):
    """
    Return mean-square SV power over a specified finite record.

    For

        x(t) = Re[phasor * exp(lambda*t)],

    the metric is

        mean_t[sum(w * x(t)^2) / sum(w)].

    ``evaluation_times`` must be expressed relative to the reference time of
    the supplied phasor. The calculation is exact at those discrete times and
    includes initial amplitude, phase, oscillation, and exponential
    growth/decay. Static modes are handled by the same expression.

    The implementation evaluates the weighted spatial quadratic form
    analytically, avoiding construction of an ``(n_time, n_space)`` array.
    Supplying spherical cell-area weights such as sin(colatitude) gives an
    area-weighted global mean square. Equal weights are used if omitted.
    """

    phasor = np.asarray(
        phasor,
        dtype=complex,
    ).ravel()

    if phasor.size == 0:
        raise ValueError(
            "phasor must contain at least one spatial value."
        )

    if not np.all(
        np.isfinite(phasor.real)
        & np.isfinite(phasor.imag)
    ):
        return np.nan

    evaluation_times = np.asarray(
        evaluation_times,
        dtype=float,
    ).ravel()

    if evaluation_times.size == 0:
        raise ValueError(
            "evaluation_times must contain at least one time."
        )

    if not np.all(
        np.isfinite(evaluation_times)
    ):
        raise ValueError(
            "evaluation_times must be finite."
        )

    if spatial_weights is None:
        weights = np.ones(
            phasor.size,
            dtype=float,
        )
    else:
        weights = np.asarray(
            spatial_weights,
            dtype=float,
        ).ravel()

        if weights.size != phasor.size:
            raise ValueError(
                "spatial_weights must contain one weight per phasor "
                f"value: {weights.size} vs {phasor.size}."
            )

    if (
        not np.all(np.isfinite(weights))
        or np.any(weights < 0.0)
    ):
        raise ValueError(
            "spatial_weights must be finite and non-negative."
        )

    weight_sum = np.sum(weights)

    if weight_sum <= 0.0:
        raise ValueError(
            "spatial_weights must have a positive sum."
        )

    eigenvalue = complex(
        eigenvalue
    )

    if not (
        np.isfinite(eigenvalue.real)
        and np.isfinite(eigenvalue.imag)
    ):
        return np.nan

    weighted_abs_square = (
        np.sum(
            weights
            * np.abs(phasor)**2
        )
        / weight_sum
    )

    weighted_complex_square = (
        np.sum(
            weights
            * phasor**2
        )
        / weight_sum
    )

    with np.errstate(
        over="ignore",
        invalid="ignore",
    ):
        temporal_factor = np.exp(
            2.0
            * eigenvalue
            * evaluation_times
        )

        record_power = (
            0.5
            * np.exp(
                2.0
                * eigenvalue.real
                * evaluation_times
            )
            * weighted_abs_square
            + 0.5
            * np.real(
                weighted_complex_square
                * temporal_factor
            )
        )

    if np.any(
        np.isposinf(record_power)
    ):
        return np.inf

    if not np.all(
        np.isfinite(record_power)
    ):
        return np.nan

    mean_power = float(
        np.mean(record_power)
    )

    # Roundoff can produce an extremely small negative value when the true
    # mean square is zero.
    if (
        mean_power < 0.0
        and np.isclose(
            mean_power,
            0.0,
            atol=1e-14,
            rtol=0.0,
        )
    ):
        mean_power = 0.0

    return mean_power

def Mode_Quality_Factor(
    eigenvalues,
    static_tol=1e-12,
):
    """
    Return Q = |omega| / (2|sigma|) for continuous eigenvalues.

    Static modes return NaN. Undamped oscillatory modes return infinity.
    """

    eigenvalues = np.asarray(
        eigenvalues,
        dtype=complex,
    )

    omega = np.abs(eigenvalues.imag)
    sigma = np.abs(eigenvalues.real)

    quality = np.full(
        eigenvalues.shape,
        np.nan,
        dtype=float,
    )

    oscillatory = omega > static_tol
    damped = oscillatory & (sigma > 0.0)
    undamped = oscillatory & (sigma == 0.0)

    quality[damped] = (
        omega[damped]
        / (2.0 * sigma[damped])
    )
    quality[undamped] = np.inf

    return quality

# comparing the complex phasor similarity of candidates vs true phasor
def best_spatial_match(
    candidate_modes,
    candidate_eigs,
    candidate_periods,
    target_phasor,
):
    """
    Return the candidate with maximum finite complex-phasor similarity.

    Returns None if no valid candidate produces a finite similarity.
    """

    if candidate_eigs.size == 0:
        return None

    best = None
    best_similarity = -np.inf

    for idx in range(candidate_eigs.size):

        similarity = Complex_Phasor_Compare(
            candidate_modes[:, idx],
            target_phasor,
        )

        if not np.isfinite(similarity):
            continue

        if similarity > best_similarity:
            best_similarity = float(similarity)
            best = {
                "candidate_index": int(idx),
                "similarity": float(similarity),
                "eigenvalue": candidate_eigs[idx],
                "recovered_period": float(
                    candidate_periods[idx]
                ),
            }

    return best

# function for comparing 2 complex phasors of same shape
def Complex_Phasor_Compare(x, y):
    x = np.asarray(x).ravel()
    y = np.asarray(y).ravel()

    denominator = np.linalg.norm(x) * np.linalg.norm(y)

    if denominator == 0:
        return np.nan

    return np.abs(np.vdot(x, y)) / denominator

def Spatial_Relative_Error(input_phasor, output_phasor):
    return np.linalg.norm(output_phasor - input_phasor) / np.linalg.norm(input_phasor)

# period error
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


def Grid_Phasor_To_Gauss(
    grid_phasor,
    design_matrix,
    spatial_weights=None,
    rcond=None,
):
    """
    Estimate complex Gauss coefficients from a gridded radial-field phasor.

    Parameters
    ----------
    grid_phasor : array, shape (n_points,)
        Complex physical-space SV phasor.
    design_matrix : array, shape (n_points, n_gauss)
        Radial-field spherical-harmonic design matrix.
    spatial_weights : array, shape (n_points,), optional
        Area weights, normally sin(colatitude).
    rcond : float, optional
        Singular-value cutoff passed to np.linalg.lstsq.

    Returns
    -------
    gauss_phasor : array, shape (n_gauss,)
    """
    grid_phasor = np.asarray(grid_phasor, dtype=complex).ravel()
    design_matrix = np.asarray(design_matrix)

    if design_matrix.shape[0] != grid_phasor.size:
        raise ValueError(
            f"Incompatible shapes: A={design_matrix.shape}, "
            f"phasor={grid_phasor.shape}"
        )

    # Exclude NaNs, including NaN pole values introduced by unweighting.
    finite = (
        np.isfinite(grid_phasor.real)
        & np.isfinite(grid_phasor.imag)
        & np.all(np.isfinite(design_matrix), axis=1)
    )

    A = design_matrix[finite, :]
    b = grid_phasor[finite]

    if spatial_weights is not None:
        weights = np.asarray(spatial_weights, dtype=float).ravel()[finite]

        if np.any(weights < 0):
            raise ValueError("Spatial weights must be non-negative.")

        sqrt_weights = np.sqrt(weights)

        A = sqrt_weights[:, None] * A
        b = sqrt_weights * b

    gauss_phasor, residuals, rank, singular_values = np.linalg.lstsq(
        A,
        b,
        rcond=rcond,
    )

    return gauss_phasor

# Matching DMD recovered modes to several input modes
def Match_Input_Output_Modes(syn_suite_info, recovered_modes, nmax, syn_record=True, record_key="resolved"):

    nmax_key = str(nmax)

    # have inputs phasors and eigenvalues 
    input_phasors = []
    input_periods = []
    input_mode_numbers = []

    # define projection
    A_r = Truncate_Gauss_Coeffs(A_15_r, tmax=nmax)


    # CONVERTING INPUT SUITE TO SET OF INPUT PHASORS AND PERIODS
    for mode_key in syn_suite_info: 

        if syn_record:

            input_gnm_phasor = Truncate_Gauss_Coeffs(syn_suite_info[mode_key]["gnm_phasor"], tmax=nmax)

            input_phasor = A_r @ input_gnm_phasor.T
            input_period = syn_suite_info[mode_key]["true_period"]

        else:

            # this is in SV
            input_phasor_sv = syn_suite_info[mode_key]["phasor"]

            A_target = Truncate_Gauss_Coeffs(
                A_15_r,
                tmax=nmax,
            )

            gauss_phasor_target = Grid_Phasor_To_Gauss(
                input_phasor_sv,
                design_matrix=A_target,
                spatial_weights= W2D.ravel(),
            )

            grid_phasor_target = A_target @ gauss_phasor_target

            input_phasor = grid_phasor_target
            input_period = syn_suite_info[mode_key]["period"]

        input_periods.append(input_period)
        input_phasors.append(input_phasor)
        input_mode_numbers.append(mode_key)

    input_periods = np.asarray(input_periods)
    input_phasors = np.asarray(input_phasors)
    number_of_inputs = len(input_periods)
    input_frequencies = 1/input_periods


    # STORING THE PHASOR AND PERIOD OF EACH INPUT
    output_periods = np.asarray(recovered_modes["periods"])
    output_frequencies = 1/output_periods
    output_phasors = np.asarray(recovered_modes["phasors"])
    number_of_outputs = len(output_periods)
    output_indices =(np.arange(0, number_of_outputs, 1)).tolist()


    # MATCHING INPUTS AND OUTPUTS
    unmatched_output_indices = output_indices.copy()

    # making correlation grid
    correlation_grid = np.zeros((number_of_outputs, number_of_inputs))

    # 1) compute correlation grid
    for i, output_phasor in enumerate(output_phasors):
        for j, input_phasor in enumerate(input_phasors):
            correlation_grid[i, j] = Complex_Phasor_Compare(input_phasor, output_phasor)

    # An undefined similarity cannot be an eligible match. Without this,
    # NaNs survive the threshold comparison and np.argmax may select them.
    correlation_grid[~np.isfinite(correlation_grid)] = 0.0

    # 2) zero out ineligebile matches based on period
    for j, input_period in enumerate(input_periods):

        period_error_threshold = 0.3 * input_period
        low_t = max(input_period - period_error_threshold, 0.01)
        high_t = input_period + period_error_threshold

        ineligible_output_mask = (output_periods < low_t) | (output_periods > high_t)

        correlation_grid[ineligible_output_mask, j] = 0

    if record_key == "background":
        # CHANGE TO BACKGROUND LATER
        sim_threshold = similarity_thresholds["background"][nmax_key]
    else:
        sim_threshold = similarity_thresholds["full_match"][nmax_key] 


    correlation_grid[correlation_grid<sim_threshold] = 0

    # storing matches:
    match_dict = {}
    leftover_output_indices = []

    guest_merged=False
    candidate_mode = False
    for i, input_mode_key in enumerate(input_mode_numbers):
        match_dict[input_mode_key] = False

        # before altering correlation matrix - see which is best guest match if applicable
        mode_num_info = input_mode_key.split("_")
        first_part = mode_num_info[0]
        guest_flag = (first_part[0] == 'g')

        if guest_flag:
            guest_number = input_mode_key
            candidate_mode = first_part[1:]
            guest_correlation_col = correlation_grid[:, i]
            if np.any(guest_correlation_col > 0):
                guest_best_match_idx = int(
                    np.argmax(guest_correlation_col)
                )
            else:
                guest_best_match_idx = None




    # 3) Pick the best match for each output. The list is an explicit queue:
    # rejected or displaced outputs are appended for reconsideration. Do not
    # remove from it while iterating, because that skips the following output.
    pending_position = 0
    while pending_position < len(unmatched_output_indices):
        unmatched_output_idx = unmatched_output_indices[pending_position]
        pending_position += 1

        output_correlation_row = correlation_grid[unmatched_output_idx, :]
        

        # if output has no eligible matches, then stop here
        if np.all(output_correlation_row == 0):
            leftover_output_indices.append(unmatched_output_idx)
            continue

        # get largest correlation for this output
        match_input_idx = np.argmax(output_correlation_row)
        input_mode_key = input_mode_numbers[match_input_idx]
        match_correlation = (output_correlation_row[match_input_idx])

  
        # if input has already been matched before
        if match_dict[input_mode_key]:

            # recover previous match
            competitor_correlation = match_dict[input_mode_key]["match_correlation"]
            competitor_idx = match_dict[input_mode_key]["match_output_idx"]


            if competitor_correlation >= match_correlation:

                # if previous match better, remove this input from current output pool
                correlation_grid[unmatched_output_idx, match_input_idx] = 0
                unmatched_output_indices.append(unmatched_output_idx)
            else:

                # if this match better, for previous output to find new input
                match_dict[input_mode_key] = {
                                "match_output_idx":unmatched_output_idx, 
                                "match_correlation":match_correlation}
                
                correlation_grid[competitor_idx, match_input_idx] = 0
                
                unmatched_output_indices.append(competitor_idx)

        else:
            
            match_dict[input_mode_key] = {
                "match_output_idx":unmatched_output_idx, 
                "match_correlation":match_correlation}

    # 4) now storing everything into match results
    match_results = {}
    match_count = 0

    for input_idx, mode_key in enumerate(input_mode_numbers):

        # if there is a stored match
        if match_dict[mode_key]:


            input_period = input_periods[input_idx]

            match_idx = match_dict[mode_key]["match_output_idx"]

            if mode_key == candidate_mode:
                if (
                    guest_best_match_idx is not None
                    and match_idx == guest_best_match_idx
                ):
                    guest_merged=True

            
            match_period = output_periods[match_idx]
            match_similarity = correlation_grid[match_idx, input_idx]
            match_period_error = np.abs(match_period - input_period)/(input_period)


            # store the match
            match_results[mode_key] = {
                        "match_period_error":match_period_error, 
                        "match_similarity":match_similarity,
                        "match_period":match_period
                    }
            match_count += 1

        else:
            match_results[mode_key] = False

    if guest_merged:
        if isinstance(match_results.get(guest_number), dict):
            discarded_guest_output = match_dict[
                guest_number
            ]["match_output_idx"]
            if discarded_guest_output not in leftover_output_indices:
                leftover_output_indices.append(discarded_guest_output)
            match_count -= 1
        match_results[guest_number] = 0

    # FINALISING STORAGE IN MATCH RESULTS
    # unmatched indices tracked result indices that are 'unmatched'
    # return the periods of these if desired
    unmatched_periods = (output_periods[leftover_output_indices]).tolist()

    # saving match count
    match_results["match_count"] = match_count

    # get effective svd rank
    effective_rank = recovered_modes["effective_rank"]
    match_results["effective_rank"] = effective_rank

    return(match_results, unmatched_periods)



# Matching DMD recovered modes to several input modes
def Match_Input_Output_Modes_old(syn_suite_info, recovered_modes, nmax, syn_record=True, record_key="resolved"):


    # get effective svd rank
    effective_rank = recovered_modes["effective_rank"]

    # have input and output phasors + periods

    # define projection
    A_r = Truncate_Gauss_Coeffs(A_15_r, tmax=nmax)

    # defining nmax based threshold
    nmax_key = str(nmax)

    if record_key == "background":
        # CHANGE TO BACKGROUND LATER
        sim_threshold = similarity_thresholds["background"][nmax_key]
    else:
        sim_threshold = similarity_thresholds["partial_match"][nmax_key]

    # match all inputs to an output

    output_periods = np.asarray(recovered_modes["periods"])
    output_phasors = recovered_modes["phasors"]
    number_of_outputs = len(output_periods)
    output_indices =(np.arange(0, number_of_outputs, 1)).tolist()
    unmatched_output_indices = output_indices.copy()

    match_results = {}

    # record how many matches made
    match_count = 0

    for mode_key in syn_suite_info: 

        if syn_record:

            input_gnm_phasor = Truncate_Gauss_Coeffs(syn_suite_info[mode_key]["gnm_phasor"], tmax=nmax)

            input_phasor = A_r @ input_gnm_phasor.T

            input_period = syn_suite_info[mode_key]["true_period"]
        else:
            # this is in SV
            input_phasor_sv = syn_suite_info[mode_key]["phasor"]

            A_target = Truncate_Gauss_Coeffs(
                A_15_r,
                tmax=nmax,
            )

            gauss_phasor_target = Grid_Phasor_To_Gauss(
                input_phasor_sv,
                design_matrix=A_target,
                spatial_weights= W2D.ravel(),
            )

            grid_phasor_target = A_target @ gauss_phasor_target
            input_phasor = grid_phasor_target

            input_period = syn_suite_info[mode_key]["period"]

        match_made = False

        best_similarity = 0



        for output_idx, output_period in enumerate(output_periods):

            # finding absolute error in frequency space
            frequency_error = np.abs(1/input_period - 1/output_period)
            period_error = np.abs(output_period - input_period) / (input_period)


            if frequency_error < frequency_error_threshold:

                output_phasor = output_phasors[output_idx]

                similarity = Complex_Phasor_Compare(input_phasor, output_phasor)

                if similarity >= sim_threshold and similarity > best_similarity:

                    match_made = True
                    match_output_idx = output_idx # index of matched mode
                    match_period = output_period
                    match_period_error = period_error
                    match_similarity = similarity
                    best_similarity = similarity

        # if there is a match, save its properties
        if match_made:
            
            if match_output_idx in unmatched_output_indices:
                unmatched_output_indices.remove(match_output_idx)

            match_results[mode_key] = {
                "match_period_error":match_period_error, 
                "match_similarity":match_similarity,
                "match_period":match_period
            }
            match_count += 1
        else:
            match_results[mode_key] = False

    # unmatched indices tracked result indices that are 'unmatched'
    # return the periods of these if desired
    unmatched_periods = (output_periods[unmatched_output_indices]).tolist()

    # saving match count
    match_results["match_count"] = match_count

    match_results["effective_rank"] = effective_rank


    return(match_results, unmatched_periods)


# -------------------------------------------- PIPELINE WRAPPER ------------------------------------------------------



# -------------------------------------
# LOADING IN DATA
# -------------------------------------
def Record_Dict_Construct(mode_numbers, nmax=15):

    # CHAOS gauss coeffs
    gnm_chaos = CHAOS_Full_SV_Record_Obtain(nmax=nmax)

    # Synthetic gauss coeffs
    gnm_syn_ideal, gnm_syn, syn_suite_info = Synthetic_Full_SV_Record_Obtain(mode_numbers, times_used_relative, nmax=nmax)

    # applying just to ideal synthetic record - equivalent to others
    gnm_syn_non_wave = Non_Wave_Spectral_Infill(gnm_chaos, gnm_syn, dt_sample=dt_sample)
    gnm_syn_background = gnm_syn + gnm_syn_non_wave

    # store full versions in record dictionary
    record_dict = {
    "ideal":gnm_syn_ideal,
    "resolved":gnm_syn,
    "background":gnm_syn_background,
    }

    return(record_dict, syn_suite_info)

# -------------------------------------
# RUNNING DMD AND RECORDING RESULTS
# -------------------------------------
def Perform_DMD_on_Record(syn_suite_info, record_physical, nmax, svd_rank, dmd_settings, syn_record, record_key):

    # running DMD
    try:
        # recovering physical modes from dmd
        recovered_modes = Run_DMD(record_physical, times_used_relative, svd_rank, dmd_settings)

        # find how many modes are 'feasibly real'
        recovered_eigenvalues = recovered_modes["continuous_eigenvalues"]

        recovered_periods = np.zeros(len(recovered_eigenvalues))
        for i, eig in enumerate(recovered_eigenvalues):
            try:
                period = 2.0 * np.pi / np.abs(eig.imag)
            except ZeroDivisionError:
                period = np.inf
            recovered_periods[i] = period

        feasible_count = 0

        # making feasible mask
        feasible_mask = (recovered_periods > 3) & (recovered_periods < 100)
        feasible_periods = recovered_periods[feasible_mask]
        feasible_periods = feasible_periods.tolist()

        feasible_count = len(feasible_periods)

        # matching input and output modes
        match_results, unmatched_periods = Match_Input_Output_Modes(syn_suite_info, recovered_modes, nmax=nmax, 
                                                    syn_record=syn_record, record_key=record_key)

        match_results["feasible_count"] = feasible_count


    except np.linalg.LinAlgError:

        recovered_modes = False
        match_results = None
        unmatched_periods = False
        print("fail")


    return(match_results, unmatched_periods, recovered_modes)


# -------------------------------------
# GIVEN SET OF RECORDS, APPLY DMD RETURN RESULTS
# -------------------------------------


# function to run pipeline on whole record set, attempting to retrieve the modes in the suite
def Pipeline_Run(record_dict_global, syn_suite_info_global, dmd_settings, 
                 nmax, svd_rank, ensemble_settings, syn_record=True,
                 total_unmatched_period_return=False):
    """Run DMD on the supplied records and optional perturbation ensemble.

    For an enabled ensemble with a fractional SVD energy threshold, the
    selected unperturbed record is deferred until after the ensemble. Its
    explicit integer rank is the ceiling of the median effective rank from
    every successful ensemble DMD run, including runs with zero mode matches.
    DMD failures are excluded. The selected record's results store the rank
    choice under ``rank_selection``.
    """

    # getting the relevant degree projection operator
    A_r = Truncate_Gauss_Coeffs(A_15_r, tmax=nmax)

    # truncting input records to relevant degree
    record_dict = {}
    for record_key in record_dict_global:
        record_dict[record_key] = Truncate_Gauss_Coeffs(record_dict_global[record_key],
                                                        tmax=nmax)

    # Pre-population preserves the input record ordering when the selected
    # unperturbed record is deferred until after the ensemble.
    results_dict = {record_key: None for record_key in record_dict}
    record_for_ensemble = ensemble_settings["record_for_ensemble"]
    ensemble_flag = ensemble_settings["ensemble_flag"]
    use_ensemble_median_rank = (
        ensemble_flag
        and isinstance(svd_rank, (float, np.floating))
        and 0 < svd_rank < 1
    )

    # for each record stored
    for record_key in record_dict:

        # A fractional rank is an energy threshold. For an ensemble run, the
        # perturbed realisations must be fitted first so that the selected
        # unperturbed record can use their median effective integer rank.
        if use_ensemble_median_rank and record_key == record_for_ensemble:
            continue

        # pulling out record specified
        record_used = record_dict[record_key]
        # projecting to physical grid SV space
        record_physical = A_r @ record_used.T
        # performing DMD and returning match metrics
        record_match_results, _, rec_modes_temp = Perform_DMD_on_Record(syn_suite_info_global, record_physical, 
                                                     nmax, svd_rank, dmd_settings, syn_record, record_key=record_key)
        # storing these results
        if record_key == record_for_ensemble:
            rec_modes_returned = rec_modes_temp

        results_dict[record_key] = record_match_results

    
    # Running perturbation ensemble
    if ensemble_flag:

        total_unmatched_periods = []


        n_ensemble = ensemble_settings["n_ensemble"]

        results_dict["ensemble"] = {}

        perturbation_path = (f"{CHAOS_COV_DIR}/CHAOS_0806_nmax15_BSpl_nT_perturbations_Hleft_N1000.h5")

        with h5py.File(perturbation_path, "r") as f:

            for ensemble_idx in range(n_ensemble):

                # loading in i-th perturbation gauss time series
                spline_perturbation_i = f["perturbations"][ensemble_idx]
                gnm_perturbation_i = H_sv @ spline_perturbation_i
                # truncation perturbation gauss time series to nmax
                gnm_perturbation_i = Truncate_Gauss_Coeffs(gnm_perturbation_i, tmax=nmax)
                # forming realised perturbed time series
                gnm_perturbed_i = record_dict[record_for_ensemble] + gnm_perturbation_i
                # projecting to physical space
                record_physical = A_r @ gnm_perturbed_i.T

                perturbation_match_results, unmatched_periods, _ = \
                    Perform_DMD_on_Record(syn_suite_info_global, record_physical, 
                                        nmax, svd_rank, dmd_settings, 
                                        syn_record, record_key=record_for_ensemble)
                
                results_dict["ensemble"][str(ensemble_idx)] = perturbation_match_results


                if type(unmatched_periods) == list:

                    total_unmatched_periods += unmatched_periods

        if use_ensemble_median_rank:
            ensemble_effective_ranks = []
            for ensemble_result in results_dict["ensemble"].values():
                # Match failures still produce a result dictionary and are
                # included. Only DMD failures, stored as None, are excluded.
                if not ensemble_result:
                    continue
                effective_rank = ensemble_result.get("effective_rank")
                if (
                    effective_rank is None
                    or not np.isfinite(effective_rank)
                    or effective_rank <= 0
                    or not float(effective_rank).is_integer()
                ):
                    raise ValueError(
                        "Each successful ensemble DMD run must provide a "
                        "positive integer effective_rank."
                    )
                ensemble_effective_ranks.append(int(effective_rank))

            if not ensemble_effective_ranks:
                raise RuntimeError(
                    "Cannot select an unperturbed DMD rank because every "
                    "ensemble DMD run failed."
                )

            ensemble_effective_rank_median = float(np.median(
                ensemble_effective_ranks
            ))
            unperturbed_rank_used = int(np.ceil(
                ensemble_effective_rank_median
            ))

            record_used = record_dict[record_for_ensemble]
            record_physical = A_r @ record_used.T
            record_match_results, _, rec_modes_returned = (
                Perform_DMD_on_Record(
                    syn_suite_info_global,
                    record_physical,
                    nmax,
                    unperturbed_rank_used,
                    dmd_settings,
                    syn_record,
                    record_key=record_for_ensemble,
                )
            )
            if record_match_results is not None:
                record_match_results["rank_selection"] = {
                    "ensemble_svd_rank": float(svd_rank),
                    "ensemble_effective_rank_median": (
                        ensemble_effective_rank_median
                    ),
                    "unperturbed_rank_used": unperturbed_rank_used,
                }
            results_dict[record_for_ensemble] = record_match_results
    else:
        
        results_dict["ensemble"] = False

    if total_unmatched_period_return:
        
        return(results_dict, total_unmatched_periods, rec_modes_returned)
    else:
        
        return(results_dict)

# converting ensemble into median results for plotting

def Percentile_Summary(array, axis=0, p=5):

    array = np.asarray(array)

    median = np.nanmedian(array, axis=axis)
    pLow = np.nanpercentile(array, p, axis=axis)
    pHigh = np.nanpercentile(array, 100-p, axis=axis)

    return median, pLow, pHigh

# define a function to process ensemble case into median
def Ensemble_To_Median(results_dict, syn_suite_info):
    """Add ensemble percentile summaries to a pipeline results dictionary.

    The effective-rank ``value`` remains the raw ensemble median. When the
    unperturbed record used an ensemble-derived rank, the ceiling rank that
    was actually used is stored separately as ``unperturbed_rank_used``.
    """
    
    if results_dict["ensemble"]:
        suite_ensemble_result = {}

        for mode_key in syn_suite_info:
            suite_ensemble_result[mode_key] = {
                "runs_with_mode_match":0,
                "match_periods":[],
                "match_similarities":[],
                "match_period_errors":[],
            }

        suite_ensemble_result["match_count"] = []
        suite_ensemble_result["effective_rank"] = []

        ensemble_convergence_count=0
        ensemble_feasible_count_list=[]

        ensemble_dict = results_dict["ensemble"]

        for ensemble_idx in ensemble_dict:
   
            ensemble_i_dict = ensemble_dict[ensemble_idx]

            if ensemble_i_dict:

                ensemble_convergence_count +=1

                for mode_key in ensemble_i_dict:
                    

                    if mode_key == "match_count" or mode_key == "effective_rank":
                        
                        suite_ensemble_result[mode_key].append(ensemble_i_dict[mode_key])
                    elif mode_key == "feasible_count":
                        ensemble_feasible_count_list.append(ensemble_i_dict[mode_key])
                    else:

                        if ensemble_i_dict[mode_key]:

                            suite_ensemble_result[mode_key]["runs_with_mode_match"] += 1

                            suite_ensemble_result[mode_key]["match_periods"].append(
                                ensemble_i_dict[mode_key]["match_period"])
                            
                            suite_ensemble_result[mode_key]["match_similarities"].append(
                                ensemble_i_dict[mode_key]["match_similarity"])

                            suite_ensemble_result[mode_key]["match_period_errors"].append(
                                ensemble_i_dict[mode_key]["match_period_error"])



        results_dict["median"] = {}
        results_dict["median"]["convergence_count"] = ensemble_convergence_count

        # getting per run suite match count stats
        feasible_count_median, feasible_count_p05, feasible_count_p95 = Percentile_Summary(
            ensemble_feasible_count_list, p=5
        )
        results_dict["median"]["feasible_count"] = {}
        results_dict["median"]["feasible_count"]["value"] = feasible_count_median
        results_dict["median"]["feasible_count"]["p05"] = feasible_count_p05
        results_dict["median"]["feasible_count"]["p95"] = feasible_count_p95

        # getting per run suite match count stats
        match_count_median, match_count_p05, match_count_p95 = Percentile_Summary(
            suite_ensemble_result["match_count"], p=5
        )
        results_dict["median"]["match_count"] = {}
        results_dict["median"]["match_count"]["value"] = match_count_median
        results_dict["median"]["match_count"]["p05"] = match_count_p05
        results_dict["median"]["match_count"]["p95"] = match_count_p95

        # getting per run suite match count stats
        effective_rank_median, effective_rank_p05, effective_rank_p95 = Percentile_Summary(
            suite_ensemble_result["effective_rank"], p=5
        )
        results_dict["median"]["effective_rank"] = {}
        results_dict["median"]["effective_rank"]["value"] = effective_rank_median
        results_dict["median"]["effective_rank"]["p05"] = effective_rank_p05
        results_dict["median"]["effective_rank"]["p95"] = effective_rank_p95

        rank_selection = None
        for record_result in results_dict.values():
            if isinstance(record_result, dict) and isinstance(
                record_result.get("rank_selection"),
                dict,
            ):
                if rank_selection is not None:
                    raise ValueError(
                        "Multiple unperturbed ensemble rank selections were "
                        "found in results_dict."
                    )
                rank_selection = record_result["rank_selection"]

        if rank_selection is not None:
            stored_median = rank_selection[
                "ensemble_effective_rank_median"
            ]
            if not np.isclose(stored_median, effective_rank_median):
                raise ValueError(
                    "Stored ensemble effective-rank median is inconsistent "
                    "with the ensemble results."
                )
            results_dict["median"]["effective_rank"][
                "unperturbed_rank_used"
            ] = rank_selection["unperturbed_rank_used"]
    
        # for each mode_key in synthetic suite
        for mode_key in syn_suite_info:

            results_dict["median"][mode_key] = {}

            mode_ensemble_result = suite_ensemble_result[mode_key]
            for data_key in mode_ensemble_result:

                ensemble_data = mode_ensemble_result[data_key]

                if type(ensemble_data) == int:
                    results_dict["median"][mode_key][data_key] = ensemble_data

                else:


                    results_dict["median"][mode_key][data_key] = {}

                    median, p05, p95 = Percentile_Summary(ensemble_data, p=5)
                    results_dict["median"][mode_key][data_key]["value"] = median
                    results_dict["median"][mode_key][data_key]["p05"] = p05
                    results_dict["median"][mode_key][data_key]["p95"] = p95



    else:
        print("Cannot convert ensemble to median, no ensemble results")
        return()

    return(results_dict)


#
# PLOTTING RESULTS
#

# HEATMAP
def Rank_Degree_Heatmap(record_dict_global, syn_suite_info_global,
                        nmax_list, svd_rank_list,
                        dmd_settings, ensemble_settings, syn_record=True,
                        chosen_rank_degree=None,
                        similarity_threshold_key="full_match"):
    """
    Run the rank-degree grid and plot ensemble recovery and quality metrics.

    ``chosen_rank_degree`` may be ``(svd_rank_setting, nmax)``. The
    corresponding full cell is outlined with a red dashed line.
    ``similarity_threshold_key`` selects the threshold family used to
    normalise the spatial-quality annotation.
    The matched fraction pools matched and total recovered output modes in
    the 3–100 year band across converged ensemble realisations only.
    """

    mode_numbers = syn_suite_info_global.keys()
    if similarity_threshold_key not in similarity_thresholds:
        raise KeyError(
            f"Unknown similarity threshold family: "
            f"'{similarity_threshold_key}'."
        )

    # One value per rank-degree configuration for the cell colour and labels.
    match_count_heatmap_low = np.zeros((len(nmax_list), len(svd_rank_list)))
    effective_rank_array = np.zeros((len(nmax_list), len(svd_rank_list)))
    spatial_quality_array = np.zeros((len(nmax_list), len(svd_rank_list)))
    temporal_error_array = np.zeros((len(nmax_list), len(svd_rank_list)))
    fraction_matched_array = np.full(
        (len(nmax_list), len(svd_rank_list)),
        np.nan,
    )

    # for each nmax to be tested
    for n_idx, nmax in enumerate(nmax_list):

        # for each svd rank to be tested
        for r_idx, svd_rank in enumerate(svd_rank_list):

            # getting output of dmd for a run with a specific nmax, svd rank
            results_dict = Pipeline_Run(record_dict_global, syn_suite_info_global,
                dmd_settings, nmax=nmax, svd_rank=svd_rank,
                ensemble_settings=ensemble_settings, syn_record=syn_record)

            # Pool recovered output modes across converged ensemble members.
            # ``feasible_count`` is already restricted to 3 < period < 100
            # years. Count matched outputs in that same band explicitly so
            # that neither the deterministic run nor out-of-band matches enter
            # the fraction.
            ensemble_feasible_count = 0
            ensemble_matched_count = 0
            ensemble_results = results_dict["ensemble"]
            if not isinstance(ensemble_results, dict):
                raise ValueError(
                    "Rank_Degree_Heatmap requires an enabled ensemble."
                )
            for ensemble_result in ensemble_results.values():
                if not ensemble_result:
                    continue

                feasible_count = ensemble_result["feasible_count"]
                if not np.isfinite(feasible_count) or feasible_count < 0:
                    raise ValueError(
                        "Each converged ensemble result must have a finite, "
                        "non-negative feasible_count."
                    )
                ensemble_feasible_count += int(feasible_count)

                for mode_key in syn_suite_info_global:
                    mode_match = ensemble_result[mode_key]
                    if not isinstance(mode_match, dict):
                        continue
                    match_period = mode_match["match_period"]
                    if np.isfinite(match_period) and 3 < match_period < 100:
                        ensemble_matched_count += 1

            if ensemble_matched_count > ensemble_feasible_count:
                raise ValueError(
                    "Matched recovered-mode count exceeds the total recovered "
                    "mode count in the 3–100 year band."
                )
            if ensemble_feasible_count > 0:
                fraction_matched_array[n_idx, r_idx] = (
                    ensemble_matched_count / ensemble_feasible_count
                )

            results_dict = Ensemble_To_Median(results_dict, syn_suite_info_global)

            # matching statistics
            match_count = results_dict["median"]["match_count"]["value"]

            similarity_threshold = similarity_thresholds[
                similarity_threshold_key
            ][str(nmax)]
            mode_spatial_quality = []
            mode_temporal_error = []

            for mode_key in syn_suite_info_global:
                mode_results = results_dict["median"][mode_key]
                mode_was_recovered = (
                    mode_results["runs_with_mode_match"] > 0
                )

                if mode_was_recovered:
                    median_similarity = mode_results[
                        "match_similarities"
                    ]["value"]
                    median_period_error = mode_results[
                        "match_period_errors"
                    ]["value"]

                    spatial_quality = (
                        (median_similarity - similarity_threshold)
                        / (1 - similarity_threshold)
                    )
                    quality_tolerance = 1e-12
                    if not (
                        -quality_tolerance
                        <= spatial_quality
                        <= 1 + quality_tolerance
                    ):
                        raise ValueError(
                            f"Threshold-relative spatial quality for mode "
                            f"'{mode_key}' is {spatial_quality:.6f} at "
                            f"nmax={nmax}, rank={svd_rank}. The matching and "
                            "normalisation thresholds are inconsistent."
                        )
                    spatial_quality = float(np.clip(
                        spatial_quality,
                        0.0,
                        1.0,
                    ))
                    temporal_error = median_period_error
                else:
                    spatial_quality = 0.0
                    temporal_error = 1.0

                mode_spatial_quality.append(spatial_quality)
                mode_temporal_error.append(temporal_error)

            # effective rank
            n_r_effective_rank = results_dict["median"]["effective_rank"]["value"]
            match_count_heatmap_low[n_idx, r_idx] = match_count
            spatial_quality_array[n_idx, r_idx] = np.mean(
                mode_spatial_quality
            )
            temporal_error_array[n_idx, r_idx] = np.mean(
                mode_temporal_error
            )
            effective_rank_array[n_idx, r_idx] = n_r_effective_rank


    svd_rank_labels = Make_SVD_Labels(svd_rank_list)

    heatmap_figure, heatmap_axis, heatmap_colourbar = (
        Plot_Rank_Degree_Quality_Heatmap(
            match_count_heatmap_low,
            spatial_quality_array,
            temporal_error_array,
            fraction_matched_array,
            effective_rank_array,
            nmax_list,
            svd_rank_list,
            svd_rank_labels,
            n_modes=len(mode_numbers),
            figure_width=text_width,
            chosen_rank_degree=chosen_rank_degree,
        )
    )

    return heatmap_figure, heatmap_axis, heatmap_colourbar



# constructing visual labels
def Make_SVD_Labels(svd_rank_list):
    svd_rank_labels = []
    for rank_value in svd_rank_list:
        if rank_value == -1:
            label = "None"
        elif rank_value == 0:
            label = "SVHT"
        elif rank_value > 1:
            label = f"r $\leq$ {rank_value}"
        elif rank_value < 1:
            label = f"Var. $\geq$ {rank_value*100}%"
        svd_rank_labels.append(label)

    return(svd_rank_labels)

def Plot_Rank_Degree_Quality_Heatmap(
    match_count,
    spatial_quality,
    temporal_error,
    fraction_matched,
    effective_rank,
    nmax_values,
    svd_rank_values,
    svd_rank_labels,
    n_modes,
    figure_width,
    chosen_rank_degree=None,
):
    """
    Plot median recovery with rank, spatial, temporal, and matched-fraction
    cell annotations.

    ``fraction_matched`` is the pooled fraction of recovered output modes in
    the 3–100 year band that were matched, using ensemble realisations only.

    The magma colour scale always spans zero to the number of input modes.
    ``chosen_rank_degree`` may identify one configuration as
    ``(svd_rank_setting, nmax)`` for a red dashed outline.
    """

    match_count = np.asarray(match_count, dtype=float)
    spatial_quality = np.asarray(spatial_quality, dtype=float)
    temporal_error = np.asarray(temporal_error, dtype=float)
    fraction_matched = np.asarray(fraction_matched, dtype=float)
    effective_rank = np.asarray(effective_rank, dtype=float)

    expected_shape = (len(nmax_values), len(svd_rank_labels))
    metric_arrays = {
        "match_count": match_count,
        "spatial_quality": spatial_quality,
        "temporal_error": temporal_error,
        "fraction_matched": fraction_matched,
        "effective_rank": effective_rank,
    }
    for array_name, array in metric_arrays.items():
        if array.shape != expected_shape:
            raise ValueError(
                f"{array_name} has shape {array.shape}; "
                f"expected {expected_shape}."
            )

    if len(svd_rank_values) != len(svd_rank_labels):
        raise ValueError(
            "svd_rank_values and svd_rank_labels must have equal length."
        )

    count_boundaries = np.arange(-0.5, n_modes + 1.5, 1)
    count_cmap = plt.get_cmap("magma", n_modes + 1)
    count_norm = BoundaryNorm(
        count_boundaries,
        count_cmap.N,
        clip=True,
    )

    fig, ax = plt.subplots(
        figsize=(figure_width, 0.6 * figure_width),
        constrained_layout=True,
    )
    image = ax.imshow(
        match_count,
        cmap=count_cmap,
        norm=count_norm,
        aspect="auto",
        interpolation="nearest",
    )

    for row in range(expected_shape[0]):
        for col in range(expected_shape[1]):
            rank_value = effective_rank[row, col]
            spatial_value = spatial_quality[row, col]
            temporal_value = temporal_error[row, col]
            fraction_matched_value = fraction_matched[row, col]

            rank_text = f"{rank_value:.0f}" if np.isfinite(rank_value) else "--"
            spatial_text = (
                f"{spatial_value:.2f}"
                if np.isfinite(spatial_value)
                else "--"
            )
            temporal_text = (
                f"{temporal_value:.2f}"
                if np.isfinite(temporal_value)
                else "--"
            )
            fraction_matched_text = (
                f"{fraction_matched_value:.2f}"
                if np.isfinite(fraction_matched_value)
                else "--"
            )

            ax.text(
                col,
                row,
                f"r = {rank_text}\n"
                rf"$Q_S$ = {spatial_text}" "\n"
                rf"$E_T$ = {temporal_text}" "\n"
                rf"$F_M$ = {fraction_matched_text}",
                ha="center",
                va="center",
                color="black",
                fontsize=8,
                linespacing=1.1,
                bbox={
                    "boxstyle": "square,pad=0.20",
                    "facecolor": "white",
                    "edgecolor": "black",
                    "linewidth": 0.5,
                },
                zorder=3,
            )

    chosen_outline = None
    if chosen_rank_degree is not None:
        if len(chosen_rank_degree) != 2:
            raise ValueError(
                "chosen_rank_degree must be (svd_rank_setting, nmax)."
            )

        chosen_rank, chosen_degree = chosen_rank_degree
        rank_matches = [
            index
            for index, rank in enumerate(svd_rank_values)
            if np.isclose(rank, chosen_rank, rtol=0, atol=1e-12)
        ]
        degree_matches = [
            index
            for index, degree in enumerate(nmax_values)
            if np.isclose(degree, chosen_degree, rtol=0, atol=1e-12)
        ]

        if not rank_matches or not degree_matches:
            raise ValueError(
                "chosen_rank_degree must identify values present in "
                "svd_rank_list and nmax_list."
            )

        chosen_col = rank_matches[0]
        chosen_row = degree_matches[0]
        chosen_outline = Rectangle(
            (chosen_col - 0.5, chosen_row - 0.5),
            1,
            1,
            facecolor="none",
            edgecolor="red",
            linewidth=2,
            linestyle="--",
            clip_on=False,
            zorder=4,
            label="Chosen cell",
        )
        ax.add_patch(chosen_outline)

    ax.set_xticks(np.arange(len(svd_rank_labels)))
    ax.set_xticklabels(svd_rank_labels, rotation=35, ha="right")
    ax.set_yticks(np.arange(len(nmax_values)))
    ax.set_yticklabels(nmax_values)
    ax.set_xlabel("SVD rank truncation")
    ax.set_ylabel(r"Maximum spherical harmonic degree, $n_{\max}$")
    ax.set_title("Rank-degree DMD performance")

    if chosen_outline is not None:
        fig.legend(
            handles=[chosen_outline],
            labels=["Chosen cell"],
            loc="upper right",
            frameon=False,
        )

    colourbar = fig.colorbar(
        image,
        ax=ax,
        boundaries=count_boundaries,
        ticks=np.arange(n_modes + 1),
    )
    colourbar.set_label("median modes recovered")

    return fig, ax, colourbar

def Plot_Match_Count_Heatmap(
    match_count_heatmap,
    effective_rank_heatmap,
    nmax_values,
    svd_rank_labels,
    n_modes,
    figure_width,
    name="heatmap"
):
    """
    Plot the number of recovered modes for each DMD configuration.

    Each cell is annotated with the effective integer SVD rank used.
    """

    match_count_heatmap = np.asarray(match_count_heatmap)
    effective_rank_heatmap = np.asarray(effective_rank_heatmap)

    expected_shape = (len(nmax_values), len(svd_rank_labels))

    if match_count_heatmap.shape != expected_shape:
        raise ValueError(
            f"match_count_heatmap has shape {match_count_heatmap.shape}; "
            f"expected {expected_shape}."
        )

    if effective_rank_heatmap.shape != expected_shape:
        raise ValueError(
            f"effective_rank_heatmap has shape {effective_rank_heatmap.shape}; "
            f"expected {expected_shape}."
        )

    # Discrete colours centred on integers 0, 1, ..., n_modes
    boundaries = np.arange(-0.5, n_modes + 1.5, 1)
    cmap = plt.get_cmap("magma", n_modes + 1)
    norm = BoundaryNorm(boundaries, cmap.N)

    fig, ax = plt.subplots(
        figsize=(figure_width, 0.6 * figure_width),
        constrained_layout=True,
    )

    fig.suptitle(name)

    image = ax.imshow(
        match_count_heatmap,
        cmap=cmap,
        norm=norm,
        aspect="auto",
        interpolation="nearest",
    )

    # Print effective SVD rank at the centre of each cell
    for row in range(match_count_heatmap.shape[0]):
        for col in range(match_count_heatmap.shape[1]):

            effective_rank = effective_rank_heatmap[row, col]
            match_count = match_count_heatmap[row, col]

            # Choose contrasting text colour
            text_colour = (
                "white"
                if match_count < 0.55 * n_modes
                else "black"
            )

            ax.text(
                col,
                row,
                f"{effective_rank:.0f}",
                ha="center",
                va="center",
                color=text_colour,
                fontsize=9,
            )

    ax.set_xticks(np.arange(len(svd_rank_labels)))
    ax.set_xticklabels(svd_rank_labels, rotation=35, ha="right")

    ax.set_yticks(np.arange(len(nmax_values)))
    ax.set_yticklabels(nmax_values)

    ax.set_xlabel("SVD rank truncation")
    ax.set_ylabel(r"Maximum spherical harmonic degree, $n_{\max}$")

    colourbar = fig.colorbar(
        image,
        ax=ax,
        boundaries=boundaries,
        ticks=np.arange(n_modes + 1),
    )
    colourbar.set_label("Number of modes recovered")

    return fig, ax, colourbar

def Plot_Heatmap(
    heatmap_data,
    annotation_data,
    nmax_values,
    svd_rank_labels,
    figure_width,
    colourbar_title,
    name="heatmap"
):
    """
    Plot a heatmap for each DMD configuration.

    Parameters
    ----------
    heatmap_data : array-like
        Values represented by the cell colours.

    annotation_data : array-like
        Values printed at the centre of each cell.

    nmax_values : array-like
        Labels for the heatmap rows.

    svd_rank_labels : array-like
        Labels for the heatmap columns.

    figure_width : float
        Figure width in inches.

    colourbar_title : str
        Label displayed beside the colourbar.
    """

    heatmap_data = np.asarray(heatmap_data)
    annotation_data = np.asarray(annotation_data)

    expected_shape = (len(nmax_values), len(svd_rank_labels))

    if heatmap_data.shape != expected_shape:
        raise ValueError(
            f"heatmap_data has shape {heatmap_data.shape}; "
            f"expected {expected_shape}."
        )

    if annotation_data.shape != expected_shape:
        raise ValueError(
            f"annotation_data has shape {annotation_data.shape}; "
            f"expected {expected_shape}."
        )

    fig, ax = plt.subplots(
        figsize=(figure_width, 0.6 * figure_width),
        constrained_layout=True,
    )

    fig.suptitle(name)

    vmax = np.max(heatmap_data)
    vmin = 0

    # Colour scale is inferred automatically from the supplied data
    image = ax.imshow(
        heatmap_data,
        cmap="viridis",
        aspect="auto",
        interpolation="nearest",
        vmin=vmin, 
        vmax=vmax
    )

    # Print annotation values at the centre of each cell
    for row in range(heatmap_data.shape[0]):
        for col in range(heatmap_data.shape[1]):

            annotation = annotation_data[row, col]
            cell_value = heatmap_data[row, col]

            # Find the displayed colour and choose contrasting text
            rgba = image.cmap(image.norm(cell_value))
            luminance = (
                0.2126 * rgba[0]
                + 0.7152 * rgba[1]
                + 0.0722 * rgba[2]
            )

            text_colour = "black" if luminance > 0.5 else "white"

            ax.text(
                col,
                row,
                f"{annotation:.0f}",
                ha="center",
                va="center",
                color=text_colour,
                fontsize=9,
            )

    ax.set_xticks(np.arange(len(svd_rank_labels)))
    ax.set_xticklabels(
        svd_rank_labels,
        rotation=35,
        ha="right",
    )

    ax.set_yticks(np.arange(len(nmax_values)))
    ax.set_yticklabels(nmax_values)

    ax.set_xlabel("SVD rank truncation")
    ax.set_ylabel(
        r"Maximum spherical harmonic degree, $n_{\max}$"
    )

    colourbar = fig.colorbar(
        image,
        ax=ax,
    )
    colourbar.set_label(colourbar_title)

    return fig, ax, colourbar

def Plot_Match_Count_Heatmap_old(
    match_count_heatmap,
    nmax_values,
    svd_rank_labels,
    n_modes,
    figure_width,
):
    """Plot the number of recovered modes for each DMD configuration."""

    match_count_heatmap = np.asarray(match_count_heatmap)
    expected_shape = (len(nmax_values), len(svd_rank_labels))
    if match_count_heatmap.shape != expected_shape:
        raise ValueError(
            f"match_count_heatmap has shape {match_count_heatmap.shape}; "
            f"expected {expected_shape}."
        )

    # Discrete colours centred on integers 0, 1, ..., n_modes
    boundaries = np.arange(-0.5, n_modes + 1.5, 1)
    cmap = plt.get_cmap("magma", n_modes + 1)
    norm = BoundaryNorm(boundaries, cmap.N)

    fig, ax = plt.subplots(
        figsize=(figure_width, 0.6 * figure_width),
        constrained_layout=True,
    )

    image = ax.imshow(
        match_count_heatmap,
        cmap=cmap,
        norm=norm,
        aspect="auto",
        interpolation="nearest",
    )

    ax.set_xticks(np.arange(len(svd_rank_labels)))
    ax.set_xticklabels(svd_rank_labels, rotation=35, ha="right")

    ax.set_yticks(np.arange(len(nmax_values)))
    ax.set_yticklabels(nmax_values)

    ax.set_xlabel("SVD rank truncation")
    ax.set_ylabel(r"Maximum spherical harmonic degree, $n_{\max}$")

    colourbar = fig.colorbar(
        image,
        ax=ax,
        boundaries=boundaries,
        ticks=np.arange(n_modes + 1),
    )
    colourbar.set_label("Number of modes recovered")

    return fig, ax, colourbar

from matplotlib.lines import Line2D

def Plot_Period_Similarity_Results(
    results_dict,
    syn_suite_info,
    similarity_threshold,
    record_markers,
    record_plotting_params,
    figure_width,
    records_to_plot=("resolved", "median", "background"),
    period_windows=((1, 26), (26, 100)),
    similarity_margin=0.02,
):
    """Plot recovered periods and similarities, including ensemble uncertainty."""

    records_to_plot = list(records_to_plot)
    period_windows = list(period_windows)
    if not period_windows:
        raise ValueError("period_windows must contain at least one window.")

    record_labels = {
        "ideal": "Ideal",
        "resolved": "Resolved",
        "background": "Background",
        "median": "Ensemble median (5–95%)",
    }
    unknown_records = set(records_to_plot) - set(record_labels)
    if unknown_records:
        raise ValueError(f"Unknown record types: {unknown_records}")

    mode_keys = list(syn_suite_info.keys())
    similarity_limits = (
        0,
        1.01,
    )

    fig, axes = plt.subplots(
        ncols=len(period_windows),
        nrows=1,
        figsize=(figure_width, 0.6 * figure_width),
        sharey=True,
        squeeze=False,
    )
    axes = axes[0]

    for ax, period_limits in zip(axes, period_windows):
        ax.set_xscale("linear")
        ax.set_xlim(period_limits)
        ax.set_ylim(similarity_limits)
        ax.set_xlabel("Recovered period (years)")
        ax.set_title(
            f"Periods from {period_limits[0]:g}–"
            f"{period_limits[1]:g} years"
        )
        ax.grid(
            True,
            which="both",
            linestyle=":",
            linewidth=0.7,
            alpha=0.5,
        )

        ax.axhline(
            similarity_threshold,
            color="black",
            linestyle="--",
            linewidth=1.2,
            zorder=1,
        )

        for mode_key in mode_keys:
            mode_info = syn_suite_info[mode_key]

            try:
                true_period = mode_info["true_period"]
            except:
                true_period = mode_info["period"]

            mode_colour = mode_info["colour"]

            ax.axvline(
                true_period,
                color=mode_colour,
                linestyle="--",
                linewidth=1.0,
                alpha=0.7,
                zorder=1,
            )

            for record_key in records_to_plot:
                record_result = results_dict.get(record_key, False)
                if not record_result:
                    continue

                mode_result = record_result.get(mode_key, False)
                if not mode_result:
                    continue

                marker = record_markers[record_key]
                if record_key == "median":
                    if mode_result["runs_with_mode_match"] == 0:
                        continue

                    period_result = mode_result["match_periods"]
                    similarity_result = mode_result["match_similarities"]
                    period_median = period_result["value"]
                    similarity_median = similarity_result["value"]

                    if not (
                        np.isfinite(period_median)
                        and np.isfinite(similarity_median)
                    ):
                        continue

                    period_errorbars = np.array([
                        [period_median - period_result["p05"]],
                        [period_result["p95"] - period_median],
                    ])
                    similarity_errorbars = np.array([
                        [similarity_median - similarity_result["p05"]],
                        [similarity_result["p95"] - similarity_median],
                    ])

                    ax.errorbar(
                        [period_median],
                        [similarity_median],
                        xerr=period_errorbars,
                        yerr=similarity_errorbars,
                        color=mode_colour,
                        ecolor=mode_colour,
                        marker=marker,
                        markersize=record_plotting_params["marker_size"],
                        markeredgecolor="black",
                        markeredgewidth=0.8,
                        linestyle="none",
                        linewidth=1.2,
                        capsize=3,
                        alpha=record_plotting_params["transparency"],
                        zorder=3,
                    )
                else:
                    ax.plot(
                        mode_result["match_period"],
                        mode_result["match_similarity"],
                        color=mode_colour,
                        marker=marker,
                        markersize=record_plotting_params["marker_size"],
                        markeredgecolor="black",
                        markeredgewidth=0.8,
                        linestyle="none",
                        alpha=record_plotting_params["transparency"],
                        zorder=2,
                    )

    axes[0].set_ylabel("Recovered-mode similarity")

    record_handles = [
        Line2D(
            [],
            [],
            color="none",
            marker=record_markers[record_key],
            markerfacecolor="0.6",
            markeredgecolor="black",
            markersize=record_plotting_params["marker_size"],
            label=record_labels[record_key],
        )
        for record_key in records_to_plot
    ]

    mode_handles = [
        Line2D(
            [],
            [],
            color=syn_suite_info[mode_key]["colour"],
            linestyle="--",
            linewidth=1.2,
            label=(
                f"Mode {mode_key}: "
                f"{syn_suite_info[mode_key]['true_period']:.2f} yr"
            ),
        )
        for mode_key in mode_keys
    ]

    threshold_handle = Line2D(
        [],
        [],
        color="black",
        linestyle="--",
        linewidth=1.2,
        label=f"Similarity threshold: {similarity_threshold:.3f}",
    )

    fig.legend(
        handles=record_handles + [threshold_handle] + mode_handles,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.0),
        ncol=4,
        frameon=False,
    )
    fig.tight_layout(rect=(0, 0.22, 1, 1))

    return fig, axes


def Plot_Hankel_Embedding_Performance(
    hankel_d_values,
    mode_recovery_performance,
    syn_suite_info,
    ensemble_match_counts,
    figure_width,
    chosen_embedding_d=None,
    x_margin_fraction=0.05,
):
    """Plot per-mode recovery metrics against Hankel embedding dimension.

    The first two panels show the conditional ensemble median and p05--p95
    range for threshold-relative spatial quality and fractional period error.
    The third shows per-mode ensemble recovery count. The fourth shows the
    ensemble median and p05--p95 range for the total number of modes
    recovered. Embedding dimension is the shared vertical axis and increases
    downwards from zero.
    """

    hankel_d_values = np.asarray(hankel_d_values, dtype=float)
    if hankel_d_values.ndim != 1 or hankel_d_values.size == 0:
        raise ValueError("hankel_d_values must be a non-empty 1D sequence.")
    if not np.all(np.isfinite(hankel_d_values)):
        raise ValueError("hankel_d_values must contain only finite values.")
    if x_margin_fraction < 0:
        raise ValueError("x_margin_fraction must be non-negative.")
    if chosen_embedding_d is not None:
        if not np.isfinite(chosen_embedding_d):
            raise ValueError("chosen_embedding_d must be finite or None.")
        if not 0 <= chosen_embedding_d <= np.max(hankel_d_values):
            raise ValueError(
                "chosen_embedding_d must lie between 0 and the maximum "
                "tested embedding dimension."
            )

    metric_settings = (
        (
            "spatial_quality",
            "Threshold-relative\nspatial similarity",
            True,
        ),
        (
            "match_period_errors",
            "Fractional\nperiod error",
            True,
        ),
        (
            "runs_with_mode_match",
            "Ensemble recovery\ncount",
            False,
        ),
    )

    mode_keys = list(syn_suite_info.keys())
    required_percentiles = ("p05", "median", "p95")

    if not isinstance(ensemble_match_counts, dict):
        raise TypeError(
            "ensemble_match_counts must contain p05, median, and p95 arrays."
        )
    missing_match_percentiles = (
        set(required_percentiles) - set(ensemble_match_counts)
    )
    if missing_match_percentiles:
        raise KeyError(
            "ensemble_match_counts is missing: "
            f"{sorted(missing_match_percentiles)}."
        )
    match_count_summary = {
        percentile: np.asarray(
            ensemble_match_counts[percentile],
            dtype=float,
        )
        for percentile in required_percentiles
    }
    for percentile, values in match_count_summary.items():
        if values.shape != hankel_d_values.shape:
            raise ValueError(
                f"ensemble_match_counts['{percentile}'] has shape "
                f"{values.shape}; expected {hankel_d_values.shape}."
            )
        if not np.all(np.isfinite(values)):
            raise ValueError(
                f"ensemble_match_counts['{percentile}'] must contain "
                "only finite values."
            )
        if np.any((values < 0) | (values > len(mode_keys))):
            raise ValueError(
                "Ensemble match-count percentiles must lie between zero "
                "and the number of input modes."
            )
    if np.any(
        (match_count_summary["p05"] > match_count_summary["median"])
        | (match_count_summary["median"] > match_count_summary["p95"])
    ):
        raise ValueError(
            "Ensemble match-count percentiles must satisfy "
            "p05 <= median <= p95."
        )

    unknown_modes = set(mode_recovery_performance) - set(mode_keys)
    if unknown_modes:
        raise ValueError(
            f"mode_recovery_performance contains unknown modes: "
            f"{unknown_modes}"
        )

    fig, axes = plt.subplots(
        nrows=1,
        ncols=4,
        figsize=(figure_width, 0.4 * figure_width),
        sharey=True,
        squeeze=False,
    )
    axes = axes[0]

    for axis, (metric_key, x_label, show_spread) in zip(
        axes,
        metric_settings,
    ):
        finite_metric_values = []

        for mode_key in mode_keys:
            mode_performance = mode_recovery_performance.get(mode_key)
            if mode_performance is None:
                raise ValueError(
                    f"No embedding results found for mode '{mode_key}'."
                )
            if metric_key not in mode_performance:
                raise KeyError(
                    f"Mode '{mode_key}' has no '{metric_key}' results."
                )

            mode_colour = syn_suite_info[mode_key]["colour"]
            if show_spread:
                metric_summary = mode_performance[metric_key]
                if not isinstance(metric_summary, dict):
                    raise TypeError(
                        f"Mode '{mode_key}' metric '{metric_key}' must "
                        "contain p05, median, and p95 arrays."
                    )
                missing_percentiles = (
                    set(required_percentiles) - set(metric_summary)
                )
                if missing_percentiles:
                    raise KeyError(
                        f"Mode '{mode_key}' metric '{metric_key}' is "
                        f"missing {sorted(missing_percentiles)}."
                    )
                percentile_values = {
                    percentile: np.asarray(
                        metric_summary[percentile],
                        dtype=float,
                    )
                    for percentile in required_percentiles
                }
                for percentile, values in percentile_values.items():
                    if values.shape != hankel_d_values.shape:
                        raise ValueError(
                            f"Mode '{mode_key}' metric '{metric_key}' "
                            f"percentile '{percentile}' has shape "
                            f"{values.shape}; expected "
                            f"{hankel_d_values.shape}."
                        )
                    finite_metric_values.extend(
                        values[np.isfinite(values)]
                    )
                if np.any(
                    (percentile_values["p05"] > percentile_values["median"])
                    | (percentile_values["median"] > percentile_values["p95"])
                ):
                    raise ValueError(
                        f"Mode '{mode_key}' metric '{metric_key}' must "
                        "satisfy p05 <= median <= p95."
                    )
                axis.fill_betweenx(
                    hankel_d_values,
                    percentile_values["p05"],
                    percentile_values["p95"],
                    color=mode_colour,
                    alpha=0.15,
                    linewidth=0,
                    zorder=1,
                )
                metric_values = percentile_values["median"]
            else:
                metric_values = np.asarray(
                    mode_performance[metric_key],
                    dtype=float,
                )
                if metric_values.shape != hankel_d_values.shape:
                    raise ValueError(
                        f"Mode '{mode_key}' metric '{metric_key}' has "
                        f"shape {metric_values.shape}; expected "
                        f"{hankel_d_values.shape}."
                    )
                finite_metric_values.extend(
                    metric_values[np.isfinite(metric_values)]
                )

            axis.plot(
                metric_values,
                hankel_d_values,
                color=mode_colour,
                linewidth=1.5,
                zorder=2,
            )

        if not finite_metric_values:
            raise ValueError(
                f"No finite values found for metric '{metric_key}'."
            )

        metric_min = float(np.min(finite_metric_values))
        metric_max = float(np.max(finite_metric_values))
        metric_range = metric_max - metric_min
        if metric_range == 0:
            margin = x_margin_fraction * max(abs(metric_min), 1.0)
        else:
            margin = x_margin_fraction * metric_range
        if metric_key == "match_period_errors":
            axis.set_xlim(0, 0.2)
        else:
            axis.set_xlim(metric_min - margin, metric_max + margin)
        axis.set_xlabel(x_label)
        axis.grid(
            True,
            which="major",
            linestyle=":",
            linewidth=0.7,
            alpha=0.6,
        )

        if chosen_embedding_d is not None:
            axis.axhline(
                chosen_embedding_d,
                color="0.5",
                linestyle="--",
                linewidth=1.2,
                zorder=3,
            )

    match_count_axis = axes[3]
    match_count_axis.fill_betweenx(
        hankel_d_values,
        match_count_summary["p05"],
        match_count_summary["p95"],
        color="black",
        alpha=0.15,
        linewidth=0,
        zorder=1,
    )
    match_count_axis.plot(
        match_count_summary["median"],
        hankel_d_values,
        color="black",
        linewidth=1.5,
        zorder=2,
    )
    match_count_axis.set_xlim(0, len(mode_keys))
    match_count_axis.set_xticks(np.arange(len(mode_keys) + 1))
    match_count_axis.set_xlabel("Median modes\nrecovered")
    match_count_axis.grid(
        True,
        which="major",
        linestyle=":",
        linewidth=0.7,
        alpha=0.6,
    )
    if chosen_embedding_d is not None:
        match_count_axis.axhline(
            chosen_embedding_d,
            color="0.5",
            linestyle="--",
            linewidth=1.2,
            zorder=3,
        )

    axes[0].set_ylabel(r"Hankel embedding"+"\n" + r"dimension, $d$")
    axes[0].set_yticks(hankel_d_values)
    axes[0].set_ylim(np.max(hankel_d_values), 1)
    median_handle = Line2D(
        [],
        [],
        color="0.25",
        linewidth=1.5,
        label="Solid line: median",
    )
    spread_handle = Patch(
        facecolor="0.4",
        edgecolor="none",
        alpha=0.15,
        label="Shaded area: p05–p95 spread",
    )
    fig.legend(
        handles=[median_handle, spread_handle],
        loc="lower center",
        bbox_to_anchor=(0.5, 0.01),
        ncol=2,
        frameon=False,
        handlelength=2.2,
        columnspacing=1.8,
    )
    fig.subplots_adjust(
        left=0.12,
        right=0.98,
        top=0.96,
        bottom=0.44,
        wspace=0.18,
    )

    return fig, axes


def Plot_Targeted_Period_Recovery(
    targeted_test_results_total,
    candidate_mode_numbers,
    guest_type_list,
    tested_period_separations,
    n_ensemble,
    mode_colours,
    figure_width,
    background_type_order=("pair", "base_set", "background"),
):
    """Plot targeted-period recovery fractions for each record context.

    Pair and base-set results use the full-match recovery criterion and are
    drawn with solid lines and filled circles. Background results use the
    background-distinction matching threshold and are drawn with dashed lines
    and open circles.
    """

    candidate_mode_numbers = [str(mode) for mode in candidate_mode_numbers]
    guest_type_list = list(guest_type_list)
    tested_period_separations = np.asarray(
        tested_period_separations,
        dtype=float,
    )
    background_type_order = list(background_type_order)

    if tested_period_separations.ndim != 1:
        raise ValueError(
            "tested_period_separations must be a one-dimensional sequence."
        )
    if tested_period_separations.size == 0:
        raise ValueError("tested_period_separations cannot be empty.")
    if not np.all(np.isfinite(tested_period_separations)):
        raise ValueError(
            "tested_period_separations must contain only finite values."
        )
    if n_ensemble <= 0:
        raise ValueError("n_ensemble must be positive.")

    missing_background_types = (
        set(background_type_order) - set(targeted_test_results_total)
    )
    if missing_background_types:
        raise KeyError(
            f"Missing targeted-period results for: "
            f"{missing_background_types}"
        )

    missing_colours = set(candidate_mode_numbers) - set(mode_colours)
    if missing_colours:
        raise KeyError(f"Missing candidate-mode colours for: {missing_colours}")

    row_labels = {
        "pair": "Isolated pair",
        "base_set": "Embedded in base set",
        "background": "Background record",
    }

    fig, axes = plt.subplots(
        nrows=len(background_type_order),
        ncols=len(guest_type_list),
        figsize=(figure_width, 0.95 * figure_width),
        sharex=True,
        sharey=True,
        squeeze=False,
    )

    for row_index, background_type in enumerate(background_type_order):
        background_results = targeted_test_results_total[background_type]
        is_background = background_type == "background"

        for column_index, guest_type in enumerate(guest_type_list):
            axis = axes[row_index, column_index]

            for separation in tested_period_separations:
                axis.axvline(
                    separation,
                    color="0.75",
                    linestyle="--",
                    linewidth=0.7,
                    alpha=0.7,
                    zorder=0,
                )

            if 0 not in tested_period_separations:
                axis.axvline(
                    0,
                    color="0.5",
                    linestyle="--",
                    linewidth=1.0,
                    zorder=0,
                )

            for candidate_mode in candidate_mode_numbers:
                try:
                    mode_results = background_results[
                        candidate_mode
                    ][guest_type]
                except KeyError as exc:
                    raise KeyError(
                        f"Missing results for record '{background_type}', "
                        f"candidate mode '{candidate_mode}', guest type "
                        f"'{guest_type}'."
                    ) from exc

                separations = np.asarray(
                    [float(value) for value in mode_results],
                    dtype=float,
                )
                success_fractions = np.asarray([
                    mode_results[value] / n_ensemble
                    for value in mode_results
                ])
                sort_indices = np.argsort(separations)
                mode_colour = mode_colours[candidate_mode]

                axis.plot(
                    separations[sort_indices],
                    success_fractions[sort_indices],
                    color=mode_colour,
                    linestyle="--" if is_background else "-",
                    marker="o",
                    markerfacecolor="none" if is_background else mode_colour,
                    markeredgecolor=mode_colour,
                    markeredgewidth=1.2,
                    linewidth=1.5,
                    label=f"Mode {candidate_mode}",
                    zorder=2,
                )

            axis.set_ylim(0, 1)

            if row_index == 0:
                axis.set_title(
                    guest_type.replace("_", " ").title()
                )

            if column_index == 0:
                axis.set_ylabel(
                    row_labels.get(background_type, background_type)
                )

            if row_index == len(background_type_order) - 1:
                axis.set_xlabel("Guest-period perturbation (%)")

    candidate_handles = [
        Line2D(
            [],
            [],
            color=mode_colours[candidate_mode],
            marker="o",
            markerfacecolor=mode_colours[candidate_mode],
            markeredgecolor=mode_colours[candidate_mode],
            linewidth=1.5,
            label=f"Mode {candidate_mode}",
        )
        for candidate_mode in candidate_mode_numbers
    ]
    fig.legend(
        handles=candidate_handles,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.0),
        ncol=len(candidate_mode_numbers),
        frameon=False,
    )
    fig.supylabel(
        "Fraction with both modes retrieved",
        x=0.01,
    )
    fig.tight_layout(rect=(0.05, 0.08, 1, 1))

    return fig, axes


def Plot_Single_Run_Ensemble_Results(
    results_dict,
    syn_suite_info,
    total_unmatched_periods,
    similarity_threshold,
    record_markers,
    record_plotting_params,
    figure_width,
    non_perturbed_record="resolved",
    period_windows=((2, 25), (25, 100)),
):
    """Plot one non-perturbed run and its perturbation ensemble.

    Each period window is a column containing a threshold-relative spatial
    quality panel above a stacked histogram of the ensemble's matched and
    unmatched recovered periods. Matched histogram entries retain their
    reference-mode colour; unmatched recovered modes are grey. Histogram
    limits use the total number of attempted ensemble realisations.

    The ensemble summaries in ``results_dict["median"]`` are assumed to have
    been added with :func:`Ensemble_To_Median`. The percentile error bars are
    therefore the existing 5th--95th percentile intervals.
    """

    period_windows = [tuple(window) for window in period_windows]
    if len(period_windows) != 2:
        raise ValueError("period_windows must contain exactly two windows.")
    for period_limits in period_windows:
        if len(period_limits) != 2 or period_limits[0] >= period_limits[1]:
            raise ValueError(
                "Each period window must contain increasing (lower, upper) limits."
            )
    non_perturbed_results = results_dict.get(non_perturbed_record, False)
    if not non_perturbed_results:
        raise ValueError(
            f"No successful non-perturbed results found for "
            f"'{non_perturbed_record}'."
        )

    median_results = results_dict.get("median", False)
    if not median_results:
        raise ValueError(
            "No ensemble median results found. Run Ensemble_To_Median first."
        )

    ensemble_results = results_dict.get("ensemble", False)
    if not isinstance(ensemble_results, dict):
        raise ValueError("No ensemble results found in results_dict.")
    n_ensemble = len(ensemble_results)
    if n_ensemble == 0:
        raise ValueError("No ensemble realisations found in results_dict.")

    similarity_threshold = float(similarity_threshold)
    if not np.isfinite(similarity_threshold):
        raise ValueError("similarity_threshold must be finite.")
    if not 0 <= similarity_threshold < 1:
        raise ValueError(
            "similarity_threshold must lie between zero (inclusive) and "
            "one (exclusive)."
        )

    def threshold_relative_quality(similarity):
        quality = (
            (np.asarray(similarity, dtype=float) - similarity_threshold)
            / (1 - similarity_threshold)
        )
        quality_tolerance = 1e-12
        finite_quality = quality[np.isfinite(quality)]
        if np.any(
            (finite_quality < -quality_tolerance)
            | (finite_quality > 1 + quality_tolerance)
        ):
            raise ValueError(
                "Recovered-mode similarity is inconsistent with the "
                "matching threshold used for threshold-relative spatial "
                "quality."
            )
        quality = np.clip(quality, 0.0, 1.0)
        return float(quality) if quality.ndim == 0 else quality

    mode_keys = sorted(
        syn_suite_info,
        key=lambda key: syn_suite_info[key].get(
            "true_period",
            syn_suite_info[key].get("period", np.inf),
        ),
    )
    mode_period_recoveries = {mode_key: [] for mode_key in mode_keys}

    for ensemble_result in ensemble_results.values():
        if not isinstance(ensemble_result, dict):
            continue
        for mode_key in mode_keys:
            mode_result = ensemble_result.get(mode_key, False)
            if isinstance(mode_result, dict):
                match_period = mode_result.get("match_period", np.nan)
                if np.isfinite(match_period):
                    mode_period_recoveries[mode_key].append(match_period)

    if total_unmatched_periods is None:
        total_unmatched_periods = []
    unmatched_periods = np.asarray(total_unmatched_periods, dtype=float)
    unmatched_periods = unmatched_periods[np.isfinite(unmatched_periods)].tolist()

    # Preserve the compact panels while reserving enough vertical space for
    # the single-row legend below the shared x labels.
    figure_height = 0.50 * figure_width
    fig, axes = plt.subplots(
        nrows=2,
        ncols=2,
        figsize=(figure_width, figure_height),
        sharex="col",
        sharey="row",
        squeeze=False,
        gridspec_kw={"height_ratios": (1, 1), "hspace": 0.08},
    )

    non_perturbed_marker = record_markers.get("resolved", "s")
    median_marker = record_markers.get("median", "P")
    marker_size = record_plotting_params["marker_size"]
    transparency = record_plotting_params["transparency"]

    histogram_values = [
        mode_period_recoveries[mode_key]
        for mode_key in mode_keys
    ] + [unmatched_periods]
    histogram_colours = [
        syn_suite_info[mode_key]["colour"]
        for mode_key in mode_keys
    ] + ["0.65"]

    for column, period_limits in enumerate(period_windows):
        similarity_ax = axes[0, column]
        histogram_ax = axes[1, column]

        similarity_ax.set_xlim(period_limits)
        similarity_ax.set_ylim(0, 1)
        similarity_ax.set_title(
            f"{period_limits[0]:g}–{period_limits[1]:g} year periods"
        )
        similarity_ax.grid(
            True,
            linestyle=":",
            linewidth=0.7,
            alpha=0.5,
        )
        histogram_ax.hist(
            histogram_values,
            bins=100,
            range=period_limits,
            stacked=True,
            color=histogram_colours,
            edgecolor="none",
            zorder=1,
        )
        histogram_ax.set_xlim(period_limits)
        histogram_ax.set_ylim(0, n_ensemble)
        histogram_ax.set_xlabel(
            "Recovered period (years)",
            labelpad=3,
        )
        histogram_ax.grid(
            True,
            axis="y",
            linestyle=":",
            linewidth=0.7,
            alpha=0.5,
        )

        for mode_key in mode_keys:
            mode_info = syn_suite_info[mode_key]
            true_period = mode_info.get("true_period", mode_info.get("period"))
            mode_colour = mode_info["colour"]

            for ax in (similarity_ax, histogram_ax):
                ax.axvline(
                    true_period,
                    color=mode_colour,
                    linestyle="--",
                    linewidth=1.2,
                    alpha=0.85,
                    zorder=4,
                )

            mode_result = non_perturbed_results.get(mode_key, False)
            if isinstance(mode_result, dict):
                spatial_quality = threshold_relative_quality(
                    mode_result["match_similarity"]
                )
                similarity_ax.plot(
                    mode_result["match_period"],
                    spatial_quality,
                    color=mode_colour,
                    marker=non_perturbed_marker,
                    markersize=marker_size,
                    markeredgecolor="black",
                    markeredgewidth=0.8,
                    linestyle="none",
                    alpha=transparency,
                    zorder=2,
                )

            median_mode_result = median_results.get(mode_key, False)
            if not isinstance(median_mode_result, dict):
                continue
            if median_mode_result.get("runs_with_mode_match", 0) == 0:
                continue

            period_result = median_mode_result["match_periods"]
            similarity_result = median_mode_result["match_similarities"]
            period_median = period_result["value"]
            similarity_summary = threshold_relative_quality(np.array([
                similarity_result["p05"],
                similarity_result["value"],
                similarity_result["p95"],
            ]))
            similarity_low, similarity_median, similarity_high = (
                similarity_summary
            )

            if not (
                np.isfinite(period_median)
                and np.isfinite(similarity_median)
            ):
                continue

            period_errorbars = np.array([
                [period_median - period_result["p05"]],
                [period_result["p95"] - period_median],
            ])
            similarity_errorbars = np.array([
                [similarity_median - similarity_low],
                [similarity_high - similarity_median],
            ])

            similarity_ax.errorbar(
                [period_median],
                [similarity_median],
                xerr=period_errorbars,
                yerr=similarity_errorbars,
                color=mode_colour,
                ecolor=mode_colour,
                marker=median_marker,
                markersize=marker_size,
                markeredgecolor="black",
                markeredgewidth=0.8,
                linestyle="none",
                linewidth=1.2,
                capsize=3,
                alpha=transparency,
                zorder=3,
            )

    axes[0, 0].set_ylabel(
        "Threshold-relative\nspatial quality, " r"$Q_S$"
    )
    axes[1, 0].set_ylabel("Ensemble\nrecovery count")

    legend_handles = [
        Line2D(
            [],
            [],
            color="none",
            marker=non_perturbed_marker,
            markerfacecolor="0.6",
            markeredgecolor="black",
            markersize=marker_size,
            label="No perturbation",
        ),
        Line2D(
            [],
            [],
            color="none",
            marker=median_marker,
            markerfacecolor="0.6",
            markeredgecolor="black",
            markersize=marker_size,
            label="Median",
        ),
        Line2D(
            [],
            [],
            color="0.45",
            linestyle="--",
            linewidth=1.2,
            label="Reference period",
        ),
    ]
    fig.legend(
        handles=legend_handles,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.01),
        ncol=3,
        frameon=False,
    )
    fig.subplots_adjust(
        left=0.10,
        right=0.98,
        top=0.94,
        bottom=0.22,
        wspace=0.12,
        hspace=0.08,
    )

    return fig, axes

from matplotlib.colors import to_rgb


def Single_Run_Ensemble_Results_Table(
    results_dict,
    syn_suite_info,
    non_perturbed_record="resolved",
    ensemble_percentage_basis="attempted",
    figure_width=None,
    font_size=None,
):
    """Tabulate one non-perturbed run and its ensemble median by mode.

    Reference modes are period-ordered columns. Results are rows, and median
    period and similarity entries retain the existing 5th--95th percentile
    summaries.
    """

    if figure_width is None:
        figure_width = text_width
    if font_size is None:
        font_size = plt.rcParams["font.size"]

    non_perturbed_results = results_dict.get(non_perturbed_record, False)
    if not non_perturbed_results:
        raise ValueError(
            f"No successful non-perturbed results found for "
            f"'{non_perturbed_record}'."
        )

    median_results = results_dict.get("median", False)
    if not median_results:
        raise ValueError(
            "No ensemble median results found. Run Ensemble_To_Median first."
        )

    ensemble_results = results_dict.get("ensemble", {})
    n_attempted = (
        len(ensemble_results)
        if isinstance(ensemble_results, dict)
        else 0
    )
    n_converged = median_results.get("convergence_count", 0)

    if ensemble_percentage_basis == "attempted":
        ensemble_denominator = n_attempted
    elif ensemble_percentage_basis == "converged":
        ensemble_denominator = n_converged
    else:
        raise ValueError(
            "ensemble_percentage_basis must be 'attempted' or 'converged'."
        )

    mode_keys = sorted(
        syn_suite_info,
        key=lambda key: syn_suite_info[key].get(
            "true_period",
            syn_suite_info[key].get("period", np.inf),
        ),
    )
    record_labels = {
        "ideal": "Ideal",
        "resolved": "Resolved",
        "background": "Background",
    }
    non_perturbed_label = record_labels.get(
        non_perturbed_record,
        non_perturbed_record.replace("_", " ").title(),
    )

    row_labels = [
        "Reference period\n(yr)",
        f"{non_perturbed_label} period\n(yr)",
        f"{non_perturbed_label}\nsimilarity",
        "Median period\n(yr; 5–95%)",
        "Median similarity\n(5–95%)",
        "Ensemble recovery",
    ]
    table_rows = [[row_label] for row_label in row_labels]
    cell_colours = [["#eeeeee"] for _ in row_labels]
    missing_colour = "#f4cccc"
    matched_colour = "#ffffff"

    mode_colours = []
    for mode_index, mode_key in enumerate(mode_keys):
        mode_info = syn_suite_info[mode_key]
        true_period = mode_info.get("true_period", mode_info.get("period"))
        mode_colour = mode_info.get(
            "colour",
            tol_muted[mode_index % len(tol_muted)],
        )
        mode_colours.append(mode_colour)

        non_perturbed_mode = non_perturbed_results.get(mode_key, False)
        if isinstance(non_perturbed_mode, dict):
            non_perturbed_period = f"{non_perturbed_mode['match_period']:.2f}"
            non_perturbed_similarity = (
                f"{non_perturbed_mode['match_similarity']:.3f}"
            )
            non_perturbed_colours = (matched_colour, matched_colour)
        else:
            non_perturbed_period = "No match"
            non_perturbed_similarity = "No match"
            non_perturbed_colours = (missing_colour, missing_colour)

        median_mode = median_results.get(mode_key, False)
        matches = (
            median_mode.get("runs_with_mode_match", 0)
            if isinstance(median_mode, dict)
            else 0
        )
        if matches > 0:
            period = median_mode["match_periods"]
            similarity = median_mode["match_similarities"]
            median_period = (
                f"{period['value']:.2f}\n"
                f"[{period['p05']:.2f}–\n"
                f"{period['p95']:.2f}]"
            )
            median_similarity = (
                f"{similarity['value']:.3f}\n"
                f"[{similarity['p05']:.3f}–\n"
                f"{similarity['p95']:.3f}]"
            )
            median_colours = (matched_colour, matched_colour)
        else:
            median_period = "No match"
            median_similarity = "No match"
            median_colours = (missing_colour, missing_colour)

        if ensemble_denominator > 0:
            recovery_percentage = 100 * matches / ensemble_denominator
            recovery = (
                f"{recovery_percentage:.1f}%\n"
                f"({matches}/{ensemble_denominator})"
            )
            recovery_colour = (
                matched_colour if matches > 0 else missing_colour
            )
        else:
            recovery = "Not available"
            recovery_colour = missing_colour

        mode_values = [
            f"{true_period:.2f}",
            non_perturbed_period,
            non_perturbed_similarity,
            median_period,
            median_similarity,
            recovery,
        ]
        mode_value_colours = [
            matched_colour,
            *non_perturbed_colours,
            *median_colours,
            recovery_colour,
        ]
        for row_index, (value, colour) in enumerate(
            zip(mode_values, mode_value_colours)
        ):
            table_rows[row_index].append(value)
            cell_colours[row_index].append(colour)

    column_labels = ["Result"] + [
        f"Mode {mode_key}"
        for mode_key in mode_keys
    ]
    column_width_weights = np.array(
        [1.55] + [1.0] * len(mode_keys),
        dtype=float,
    )
    column_widths = column_width_weights / column_width_weights.sum()

    table_font_size = min(
        font_size,
        max(8.5, font_size - 0.6 * max(0, len(mode_keys) - 4)),
    )
    # A modest height allowance prevents the percentile ranges and caption
    # from colliding without widening the table or shrinking its text.
    figure_height = 0.46 * figure_width
    fig, ax = plt.subplots(figsize=(figure_width, figure_height))
    fig.patch.set_facecolor("white")
    fig.patch.set_alpha(1.0)
    ax.set_facecolor("white")
    ax.axis("off")

    table = ax.table(
        cellText=table_rows,
        cellColours=cell_colours,
        colLabels=column_labels,
        colColours=["#eeeeee"] + mode_colours,
        colWidths=column_widths,
        cellLoc="center",
        colLoc="center",
        edges="closed",
        bbox=[0.0, 0.14, 1.0, 0.86],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(table_font_size)

    for (row_index, column_index), cell in table.get_celld().items():
        cell.set_edgecolor("#b7b7b7")
        cell.set_linewidth(0.6)
        cell.set_alpha(1.0)
        if row_index == 0:
            cell.set_text_props(weight="bold", color="black")
        elif column_index == 0:
            cell.set_text_props(weight="bold", color="black")

    for column_index, mode_colour in enumerate(mode_colours, start=1):
        red, green, blue = to_rgb(mode_colour)
        luminance = 0.2126 * red + 0.7152 * green + 0.0722 * blue
        text_colour = "black" if luminance > 0.55 else "white"
        table[(0, column_index)].set_text_props(
            color=text_colour,
            weight="bold",
        )


    fig.subplots_adjust(left=0.0, right=1.0, top=1.0, bottom=0.0)
    return fig, table

def Results_Summary_Table(
    results_dict,
    syn_suite_info,
    records_to_include=("ideal", "resolved", "background", "median"),
    ensemble_percentage_basis="attempted",
    figure_width=None,
    font_size=None,
):
    """Create an opaque summary table for DMD mode-recovery results."""

    if figure_width is None:
        figure_width = text_width
    if font_size is None:
        font_size = plt.rcParams["font.size"]

    records_to_include = list(records_to_include)
    valid_records = {"ideal", "resolved", "background", "median"}
    unknown_records = set(records_to_include) - valid_records
    if unknown_records:
        raise ValueError(f"Unknown record types: {unknown_records}")

    ensemble_results = results_dict.get("ensemble", {})
    n_attempted = (
        len(ensemble_results)
        if isinstance(ensemble_results, dict)
        else 0
    )

    median_results = results_dict.get("median", {})
    n_converged = median_results.get("convergence_count", 0)

    if ensemble_percentage_basis == "attempted":
        ensemble_denominator = n_attempted
    elif ensemble_percentage_basis == "converged":
        ensemble_denominator = n_converged
    else:
        raise ValueError(
            "ensemble_percentage_basis must be "
            "'attempted' or 'converged'."
        )

    record_labels = {
        "ideal": "Ideal",
        "resolved": "Res.",
        "background": "Bg.",
        "median": "Median",
    }

    mode_keys = sorted(
        syn_suite_info,
        key=lambda key: syn_suite_info[key]["true_period"],
    )

    column_labels = [
        "Mode",
        "True\nperiod\n(yr)",
    ]
    column_width_weights = [0.55, 0.80]

    for record_key in records_to_include:
        record_label = record_labels[record_key]
        column_labels.extend([
            f"{record_label}\nperiod\n(yr)",
            f"{record_label}\nsim.",
        ])
        column_width_weights.extend(
            [1.20, 1.15]
            if record_key == "median"
            else [0.72, 0.72]
        )

        if record_key == "median":
            column_labels.append("Recovery\n(%)")
            column_width_weights.append(0.85)

    matched_colour = "#ffffff"
    missing_colour = "#f4cccc"
    header_colour = "#eeeeee"

    table_rows = []
    cell_colours = []
    mode_colours = []

    for mode_index, mode_key in enumerate(mode_keys):
        mode_info = syn_suite_info[mode_key]
        true_period = float(mode_info["true_period"])
        mode_colour = mode_info.get(
            "colour",
            tol_muted[mode_index % len(tol_muted)],
        )

        row = [str(mode_key), f"{true_period:.2f}"]
        row_colours = [mode_colour, matched_colour]
        mode_colours.append(mode_colour)

        for record_key in records_to_include:
            if record_key == "median":
                mode_result = median_results.get(mode_key, False)
                matches = (
                    mode_result.get("runs_with_mode_match", 0)
                    if mode_result
                    else 0
                )

                if matches > 0:
                    period = mode_result["match_periods"]
                    similarity = mode_result["match_similarities"]
                    period_text = (
                        f"{period['value']:.2f}\n"
                        f"[{period['p05']:.2f}–\n"
                        f"{period['p95']:.2f}]"
                    )
                    similarity_text = (
                        f"{similarity['value']:.3f}\n"
                        f"[{similarity['p05']:.3f}–\n"
                        f"{similarity['p95']:.3f}]"
                    )
                    period_colour = matched_colour
                    similarity_colour = matched_colour
                else:
                    period_text = "No\nmatch"
                    similarity_text = "No\nmatch"
                    period_colour = missing_colour
                    similarity_colour = missing_colour

                if ensemble_denominator > 0:
                    recovery_percentage = 100 * matches / ensemble_denominator
                    recovery_text = (
                        f"{recovery_percentage:.1f}%\n"
                        f"({matches}/\n{ensemble_denominator})"
                    )
                else:
                    recovery_text = "Not\navailable"

                recovery_colour = (
                    missing_colour
                    if matches == 0 or ensemble_denominator == 0
                    else matched_colour
                )
                row.extend([period_text, similarity_text, recovery_text])
                row_colours.extend([
                    period_colour,
                    similarity_colour,
                    recovery_colour,
                ])
            else:
                record_result = results_dict.get(record_key, False)
                if not record_result:
                    row.extend(["Run\nfailed", "Run\nfailed"])
                    row_colours.extend([missing_colour, missing_colour])
                    continue

                mode_result = record_result.get(mode_key, False)
                if not mode_result:
                    row.extend(["No\nmatch", "No\nmatch"])
                    row_colours.extend([missing_colour, missing_colour])
                else:
                    row.extend([
                        f"{mode_result['match_period']:.2f}",
                        f"{mode_result['match_similarity']:.3f}",
                    ])
                    row_colours.extend([matched_colour, matched_colour])

        table_rows.append(row)
        cell_colours.append(row_colours)

    column_widths = np.asarray(column_width_weights, dtype=float)
    column_widths /= np.sum(column_widths)

    figure_height = max(
        0.66 * figure_width,
        0.42 * (len(mode_keys) + 2),
    )
    fig, ax = plt.subplots(figsize=(figure_width, figure_height))
    fig.patch.set_facecolor("white")
    fig.patch.set_alpha(1.0)
    ax.set_facecolor("white")
    ax.axis("off")

    caption_space = 0.12 if "median" in records_to_include else 0.02
    table = ax.table(
        cellText=table_rows,
        cellColours=cell_colours,
        colLabels=column_labels,
        colColours=[header_colour] * len(column_labels),
        colWidths=column_widths,
        cellLoc="center",
        colLoc="center",
        edges="closed",
        bbox=[0.0, caption_space, 1.0, 1.0 - caption_space],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(font_size)

    for (row_index, column_index), cell in table.get_celld().items():
        cell.set_edgecolor("#b7b7b7")
        cell.set_linewidth(0.6)
        cell.set_alpha(1.0)
        if row_index == 0:
            cell.set_text_props(weight="bold", color="black")

    for row_index, mode_colour in enumerate(mode_colours, start=1):
        mode_cell = table[(row_index, 0)]
        red, green, blue = to_rgb(mode_colour)
        luminance = 0.2126 * red + 0.7152 * green + 0.0722 * blue
        text_colour = "black" if luminance > 0.55 else "white"
        mode_cell.set_text_props(color=text_colour, weight="bold")

    if "median" in records_to_include:
        caption = (
            f"Ensemble recovery uses {ensemble_denominator} "
            f"{ensemble_percentage_basis} runs; "
            f"{n_converged}/{n_attempted} converged."
        )
        ax.text(
            0.0,
            0.015,
            caption,
            transform=ax.transAxes,
            ha="left",
            va="bottom",
            fontsize=font_size,
            color="black",
        )

    fig.subplots_adjust(left=0.0, right=1.0, top=1.0, bottom=0.0)
    return fig, table

# ---------------------------------------------------------------
# SYNTHETIC MODE SNAPSHOT COMPARISON
# ---------------------------------------------------------------

def Phase_Align_Phasor(reference_phasor, candidate_phasor):
    """Phase-align a complex candidate phasor without changing its amplitude."""

    reference_phasor = np.asarray(reference_phasor, dtype=complex).ravel()
    candidate_phasor = np.asarray(candidate_phasor, dtype=complex).ravel()

    if reference_phasor.shape != candidate_phasor.shape:
        raise ValueError(
            f"Phasor shapes differ: reference={reference_phasor.shape}, "
            f"candidate={candidate_phasor.shape}."
        )

    finite = (
        np.isfinite(reference_phasor.real)
        & np.isfinite(reference_phasor.imag)
        & np.isfinite(candidate_phasor.real)
        & np.isfinite(candidate_phasor.imag)
    )
    if not np.any(finite):
        raise ValueError("The phasors contain no shared finite values.")

    overlap = np.vdot(
        reference_phasor[finite],
        candidate_phasor[finite],
    )
    overlap_scale = (
        np.linalg.norm(reference_phasor[finite])
        * np.linalg.norm(candidate_phasor[finite])
    )
    if (
        overlap_scale == 0
        or np.abs(overlap) <= np.finfo(float).eps * overlap_scale
    ):
        return candidate_phasor.copy()

    return candidate_phasor * np.exp(-1j * np.angle(overlap))


def _Phasor_Similarity_Finite(reference_phasor, candidate_phasor):
    """Evaluate complex-phasor similarity on shared finite grid points."""

    reference_phasor = np.asarray(reference_phasor, dtype=complex).ravel()
    candidate_phasor = np.asarray(candidate_phasor, dtype=complex).ravel()
    if reference_phasor.shape != candidate_phasor.shape:
        raise ValueError(
            f"Phasor shapes differ: reference={reference_phasor.shape}, "
            f"candidate={candidate_phasor.shape}."
        )

    finite = (
        np.isfinite(reference_phasor.real)
        & np.isfinite(reference_phasor.imag)
        & np.isfinite(candidate_phasor.real)
        & np.isfinite(candidate_phasor.imag)
    )
    if not np.any(finite):
        return np.nan

    return Complex_Phasor_Compare(
        reference_phasor[finite],
        candidate_phasor[finite],
    )


def _Recovered_Mode_Entry(
    recovered_modes,
    recovered_index,
    reference_phasor,
):
    """Package one recovered mode for snapshot comparison."""

    candidate_phasor = np.asarray(
        recovered_modes["phasors"][recovered_index],
        dtype=complex,
    ).ravel()
    similarity = _Phasor_Similarity_Finite(
        reference_phasor,
        candidate_phasor,
    )
    aligned_phasor = Phase_Align_Phasor(
        reference_phasor,
        candidate_phasor,
    )

    return {
        "phasor": aligned_phasor,
        "eigenvalue": complex(
            recovered_modes["continuous_eigenvalues"][recovered_index]
        ),
        "period": float(recovered_modes["periods"][recovered_index]),
        "similarity": float(similarity),
        "recovered_index": int(recovered_index),
    }


def _Matched_Recovered_Mode_Entry(
    mode_key,
    match_results,
    recovered_modes,
    reference_phasor,
):
    """Locate the recovered-mode array entry selected by existing matching."""

    if not isinstance(match_results, dict):
        return None
    mode_match = match_results.get(mode_key, False)
    if not isinstance(mode_match, dict):
        return None

    recovered_periods = np.asarray(
        recovered_modes["periods"],
        dtype=float,
    )
    matched_period = float(mode_match["match_period"])
    period_candidates = np.flatnonzero(
        np.isclose(
            recovered_periods,
            matched_period,
            rtol=1e-10,
            atol=1e-12,
        )
    )
    if period_candidates.size == 0:
        period_candidates = np.array([
            int(np.nanargmin(np.abs(recovered_periods - matched_period)))
        ])

    if period_candidates.size == 1:
        recovered_index = int(period_candidates[0])
    else:
        similarities = [
            _Phasor_Similarity_Finite(
                reference_phasor,
                recovered_modes["phasors"][candidate_index],
            )
            for candidate_index in period_candidates
        ]
        recovered_index = int(
            period_candidates[int(np.nanargmax(similarities))]
        )

    return _Recovered_Mode_Entry(
        recovered_modes,
        recovered_index,
        reference_phasor,
    )


def _Select_Ensemble_Median_Similarity_Members(
    ensemble_results,
    mode_numbers,
):
    """Select the successful ensemble member nearest each median similarity."""

    if not isinstance(ensemble_results, dict):
        raise TypeError("ensemble_results must be a dictionary.")

    selections = {}
    for mode_key in mode_numbers:
        successful_members = []
        for ensemble_key, member_results in ensemble_results.items():
            if not isinstance(member_results, dict):
                continue
            mode_result = member_results.get(mode_key, False)
            if not isinstance(mode_result, dict):
                continue

            similarity = float(mode_result["match_similarity"])
            if not np.isfinite(similarity):
                continue
            successful_members.append((int(ensemble_key), similarity))

        if not successful_members:
            selections[mode_key] = None
            continue

        similarities = np.asarray(
            [member[1] for member in successful_members],
            dtype=float,
        )
        median_similarity = float(np.median(similarities))
        selected_index, selected_similarity = min(
            successful_members,
            key=lambda member: (
                abs(member[1] - median_similarity),
                member[0],
            ),
        )
        selections[mode_key] = {
            "ensemble_index": int(selected_index),
            "median_similarity": median_similarity,
            "selected_similarity": float(selected_similarity),
            "successful_match_count": len(successful_members),
        }

    return selections


def Build_Synthetic_Mode_Visualisation_Data(
    mode_numbers,
    nmax,
    resolved_setup,
    signal_background_setup,
    null_background_setup,
    n_ensemble=50,
    null_period_limits=(3, 100),
    null_min_quality_factor=None,
    null_seed=42,
):
    """Build four-column mode-comparison data at one common degree.

    The returned columns are the ideal input; the successful resolved-record
    ensemble member nearest the median match similarity; the equivalent
    resolved-plus-background ensemble member; and the feasible background-only
    mode with the closest period. Recovered phasors are phase-aligned to their
    reference while retaining amplitude.
    """

    mode_numbers = [str(mode_number) for mode_number in mode_numbers]
    if not mode_numbers:
        raise ValueError("mode_numbers cannot be empty.")
    if not 1 <= int(nmax) <= 15:
        raise ValueError("nmax must be an integer between 1 and 15.")
    nmax = int(nmax)
    if (
        isinstance(n_ensemble, bool)
        or int(n_ensemble) != n_ensemble
        or n_ensemble <= 0
    ):
        raise ValueError("n_ensemble must be a positive integer.")
    n_ensemble = int(n_ensemble)

    null_period_min, null_period_max = null_period_limits
    if not null_period_min < null_period_max:
        raise ValueError("null_period_limits must be increasing.")

    setups = {
        "resolved": resolved_setup,
        "signal_background": signal_background_setup,
        "null_background": null_background_setup,
    }
    for setup_name, setup in setups.items():
        if not isinstance(setup, dict):
            raise TypeError(f"{setup_name}_setup must be a dictionary.")
        missing_keys = {"svd_rank", "dmd_settings"} - set(setup)
        if missing_keys:
            raise KeyError(
                f"{setup_name}_setup is missing keys: {missing_keys}"
            )

    record_dict, syn_suite_info = Record_Dict_Construct(
        mode_numbers=mode_numbers,
        nmax=nmax,
    )
    design_matrix = Truncate_Gauss_Coeffs(A_15_r, tmax=nmax)
    perturbation_path = (
        f"{CHAOS_COV_DIR}/"
        "CHAOS_0806_nmax15_BSpl_nT_perturbations_Hleft_N1000.h5"
    )
    with h5py.File(perturbation_path, "r") as perturbation_file:
        available_ensemble_members = perturbation_file[
            "perturbations"
        ].shape[0]
    if n_ensemble > available_ensemble_members:
        raise ValueError(
            f"n_ensemble={n_ensemble} exceeds the "
            f"{available_ensemble_members} stored perturbations."
        )

    def run_record(gnm_record, setup):
        truncated_record = Truncate_Gauss_Coeffs(
            gnm_record,
            tmax=nmax,
        )
        physical_record = design_matrix @ truncated_record.T
        return Run_DMD(
            physical_record,
            times_used_relative,
            svd_rank=setup["svd_rank"],
            dmd_settings=setup["dmd_settings"].copy(),
        )

    def ensemble_results_for_record(record_key, setup):
        all_results = Pipeline_Run(
            record_dict,
            syn_suite_info,
            dmd_settings=setup["dmd_settings"].copy(),
            nmax=nmax,
            svd_rank=setup["svd_rank"],
            ensemble_settings={
                "ensemble_flag": True,
                "n_ensemble": n_ensemble,
                "record_for_ensemble": record_key,
            },
        )
        return all_results["ensemble"]

    resolved_ensemble_results = ensemble_results_for_record(
        "resolved",
        resolved_setup,
    )
    resolved_selections = _Select_Ensemble_Median_Similarity_Members(
        resolved_ensemble_results,
        mode_numbers,
    )

    signal_background_ensemble_results = ensemble_results_for_record(
        "background",
        signal_background_setup,
    )
    signal_background_selections = (
        _Select_Ensemble_Median_Similarity_Members(
            signal_background_ensemble_results,
            mode_numbers,
        )
    )

    reference_phasors = {
        mode_key: (
            design_matrix
            @ Truncate_Gauss_Coeffs(
                syn_suite_info[mode_key]["gnm_phasor"],
                tmax=nmax,
            ).T
        )
        for mode_key in mode_numbers
    }

    def reconstruct_selected_members(
        base_record,
        record_key,
        setup,
        selections,
    ):
        entries = {mode_key: None for mode_key in mode_numbers}
        modes_by_ensemble_index = {}
        for mode_key, selection in selections.items():
            if selection is None:
                continue
            modes_by_ensemble_index.setdefault(
                selection["ensemble_index"],
                [],
            ).append(mode_key)

        reconstructed_modes = {}
        with h5py.File(perturbation_path, "r") as perturbation_file:
            for ensemble_index, selected_mode_keys in (
                modes_by_ensemble_index.items()
            ):
                spline_perturbation = perturbation_file[
                    "perturbations"
                ][ensemble_index]
                gnm_perturbation = H_sv @ spline_perturbation
                gnm_perturbation = Truncate_Gauss_Coeffs(
                    gnm_perturbation,
                    tmax=nmax,
                )
                realised_record = (
                    Truncate_Gauss_Coeffs(base_record, tmax=nmax)
                    + gnm_perturbation
                )
                recovered_modes = run_record(realised_record, setup)
                match_results, _ = Match_Input_Output_Modes(
                    syn_suite_info,
                    recovered_modes,
                    nmax=nmax,
                    syn_record=True,
                    record_key=record_key,
                )
                reconstructed_modes[ensemble_index] = recovered_modes

                for mode_key in selected_mode_keys:
                    entry = _Matched_Recovered_Mode_Entry(
                        mode_key,
                        match_results,
                        recovered_modes,
                        reference_phasors[mode_key],
                    )
                    if entry is None:
                        raise RuntimeError(
                            f"Mode '{mode_key}' matched in ensemble member "
                            f"{ensemble_index} but not when that member was "
                            "reconstructed."
                        )
                    selection = selections[mode_key]
                    if not np.isclose(
                        entry["similarity"],
                        selection["selected_similarity"],
                        rtol=1e-8,
                        atol=1e-10,
                    ):
                        raise RuntimeError(
                            f"Reconstructed similarity for mode "
                            f"'{mode_key}' does not reproduce ensemble "
                            f"member {ensemble_index}."
                        )
                    entry.update(selection)
                    entries[mode_key] = entry

        return entries, reconstructed_modes

    resolved_entries, resolved_reconstructed_modes = (
        reconstruct_selected_members(
            record_dict["resolved"],
            "resolved",
            resolved_setup,
            resolved_selections,
        )
    )
    signal_background_entries, signal_background_reconstructed_modes = (
        reconstruct_selected_members(
            record_dict["background"],
            "background",
            signal_background_setup,
            signal_background_selections,
        )
    )

    gnm_chaos = CHAOS_Full_SV_Record_Obtain(nmax=nmax)
    null_background_record = Non_Wave_Spectral_Infill(
        gnm_chaos,
        np.zeros_like(gnm_chaos),
        nmax=nmax,
        dt_sample=dt_sample,
        seed=null_seed,
    )
    null_background_modes = run_record(
        null_background_record,
        null_background_setup,
    )
    null_periods = np.asarray(
        null_background_modes["periods"],
        dtype=float,
    )
    null_eigenvalues = np.asarray(
        null_background_modes["continuous_eigenvalues"],
        dtype=complex,
    )
    null_feasible_mask = (
        np.isfinite(null_periods)
        & (null_periods > null_period_min)
        & (null_periods < null_period_max)
    )
    if null_min_quality_factor is not None:
        null_quality_factors = Mode_Quality_Factor(null_eigenvalues)
        null_feasible_mask &= (
            null_quality_factors > null_min_quality_factor
        )
    null_feasible_indices = np.flatnonzero(null_feasible_mask)

    visualisation_rows = {}
    for mode_key in mode_numbers:
        mode_info = syn_suite_info[mode_key]
        reference_phasor = reference_phasors[mode_key]
        reference_eigenvalue = complex(mode_info["true_eigenvalue"])
        reference_period = float(mode_info["true_period"])

        ideal_entry = {
            "phasor": np.asarray(reference_phasor, dtype=complex).ravel(),
            "eigenvalue": reference_eigenvalue,
            "period": reference_period,
            "similarity": 1.0,
            "recovered_index": None,
        }
        resolved_entry = resolved_entries[mode_key]
        signal_background_entry = signal_background_entries[mode_key]

        if null_feasible_indices.size == 0:
            null_background_entry = None
        else:
            closest_position = int(np.argmin(
                np.abs(
                    null_periods[null_feasible_indices]
                    - reference_period
                )
            ))
            null_index = int(null_feasible_indices[closest_position])
            null_background_entry = _Recovered_Mode_Entry(
                null_background_modes,
                null_index,
                reference_phasor,
            )

        visualisation_rows[mode_key] = {
            "reference_period": reference_period,
            "ideal": ideal_entry,
            "resolved": resolved_entry,
            "signal_background": signal_background_entry,
            "null_background": null_background_entry,
        }

    recovered_mode_sets = {
        "n_ensemble": n_ensemble,
        "resolved_ensemble_results": resolved_ensemble_results,
        "signal_background_ensemble_results": (
            signal_background_ensemble_results
        ),
        "resolved_selections": resolved_selections,
        "signal_background_selections": signal_background_selections,
        "resolved_reconstructed_modes": resolved_reconstructed_modes,
        "signal_background_reconstructed_modes": (
            signal_background_reconstructed_modes
        ),
        "null_background": null_background_modes,
        "null_feasible_mask": null_feasible_mask,
    }

    return visualisation_rows, syn_suite_info, recovered_mode_sets


def Build_Recovery_Visualisation_Data(
    mode_numbers,
    nmax,
    wave_only_setup,
    background_only_setup,
    waves_background_setup,
    n_ensemble=100,
    spurious_target_period=5.2,
    feasible_period_limits=(3, 100),
):
    """Build the selected input, matched, and spurious snapshot entries.

    The wave-only, exact spectral-infill background-only, and combined
    records each run their own perturbation ensemble. Each setup must supply
    a fractional SVD energy threshold; :func:`Pipeline_Run` then applies the
    ceiling of that ensemble's median effective rank to its corresponding
    unperturbed record.
    """

    mode_numbers = [str(mode_number) for mode_number in mode_numbers]
    required_mode_numbers = {"62", "48", "1", "6", "13", "45"}
    missing_mode_numbers = required_mode_numbers - set(mode_numbers)
    if missing_mode_numbers:
        raise ValueError(
            "mode_numbers is missing required visualisation modes: "
            f"{sorted(missing_mode_numbers)}"
        )
    if not 1 <= int(nmax) <= 15:
        raise ValueError("nmax must be an integer between 1 and 15.")
    nmax = int(nmax)
    if (
        isinstance(n_ensemble, bool)
        or int(n_ensemble) != n_ensemble
        or n_ensemble <= 0
    ):
        raise ValueError("n_ensemble must be a positive integer.")
    n_ensemble = int(n_ensemble)

    period_min, period_max = feasible_period_limits
    if not period_min < period_max:
        raise ValueError("feasible_period_limits must be increasing.")
    if not np.isfinite(spurious_target_period):
        raise ValueError("spurious_target_period must be finite.")

    setups = {
        "wave_only": wave_only_setup,
        "background_only": background_only_setup,
        "waves_background": waves_background_setup,
    }
    for setup_name, setup in setups.items():
        if not isinstance(setup, dict):
            raise TypeError(f"{setup_name}_setup must be a dictionary.")
        missing_keys = {"svd_rank", "dmd_settings"} - set(setup)
        if missing_keys:
            raise KeyError(
                f"{setup_name}_setup is missing keys: {missing_keys}"
            )
        svd_rank = setup["svd_rank"]
        if not (
            isinstance(svd_rank, (float, np.floating))
            and 0 < svd_rank < 1
        ):
            raise ValueError(
                f"{setup_name}_setup['svd_rank'] must be a fractional "
                "SVD energy threshold between zero and one."
            )

    record_dict, syn_suite_info = Record_Dict_Construct(
        mode_numbers=mode_numbers,
        nmax=nmax,
    )
    background_only_record = (
        np.asarray(record_dict["background"])
        - np.asarray(record_dict["resolved"])
    )

    case_definitions = {
        "wave_only": {
            "record_key": "resolved",
            "record": record_dict["resolved"],
            "setup": wave_only_setup,
        },
        "background_only": {
            "record_key": "background_only",
            "record": background_only_record,
            "setup": background_only_setup,
        },
        "waves_background": {
            "record_key": "background",
            "record": record_dict["background"],
            "setup": waves_background_setup,
        },
    }

    case_outputs = {}
    for case_key, case_definition in case_definitions.items():
        record_key = case_definition["record_key"]
        setup = case_definition["setup"]
        case_results, unmatched_periods, recovered_modes = Pipeline_Run(
            {record_key: case_definition["record"]},
            syn_suite_info,
            dmd_settings=setup["dmd_settings"].copy(),
            nmax=nmax,
            svd_rank=setup["svd_rank"],
            ensemble_settings={
                "ensemble_flag": True,
                "n_ensemble": n_ensemble,
                "record_for_ensemble": record_key,
            },
            syn_record=True,
            total_unmatched_period_return=True,
        )
        if not isinstance(recovered_modes, dict):
            raise RuntimeError(
                f"The adaptive-rank unperturbed {case_key} DMD run failed."
            )
        case_results = Ensemble_To_Median(
            case_results,
            syn_suite_info,
        )
        case_outputs[case_key] = {
            "record_key": record_key,
            "results": case_results,
            "recovered_modes": recovered_modes,
            "unmatched_ensemble_periods": unmatched_periods,
        }

    design_matrix = Truncate_Gauss_Coeffs(A_15_r, tmax=nmax)
    reference_phasors = {
        mode_key: (
            design_matrix
            @ Truncate_Gauss_Coeffs(
                syn_suite_info[mode_key]["gnm_phasor"],
                tmax=nmax,
            ).T
        )
        for mode_key in mode_numbers
    }

    def input_entry(mode_key):
        mode_info = syn_suite_info[mode_key]
        return {
            "phasor": np.asarray(
                reference_phasors[mode_key],
                dtype=complex,
            ).ravel(),
            "period": float(mode_info["true_period"]),
            "label": f"Mode {mode_key}",
            "mode_key": mode_key,
            "recovered_index": None,
        }

    def matched_entry(case_key, mode_key):
        case_output = case_outputs[case_key]
        record_results = case_output["results"][
            case_output["record_key"]
        ]
        entry = _Matched_Recovered_Mode_Entry(
            mode_key,
            record_results,
            case_output["recovered_modes"],
            reference_phasors[mode_key],
        )
        if entry is None:
            raise RuntimeError(
                f"The adaptive-rank {case_key} run did not recover "
                f"Mode {mode_key}."
            )
        entry["label"] = f"Recovered (Mode {mode_key})"
        entry["mode_key"] = mode_key
        return entry

    wave_only_matches = {
        mode_key: matched_entry("wave_only", mode_key)
        for mode_key in ("62", "48", "6")
    }
    waves_background_mode_45 = matched_entry(
        "waves_background",
        "45",
    )

    waves_background_output = case_outputs["waves_background"]
    waves_background_results = waves_background_output["results"][
        waves_background_output["record_key"]
    ]
    waves_background_modes = waves_background_output["recovered_modes"]
    matched_output_indices = set()
    for mode_key in mode_numbers:
        matched_mode = _Matched_Recovered_Mode_Entry(
            mode_key,
            waves_background_results,
            waves_background_modes,
            reference_phasors[mode_key],
        )
        if matched_mode is not None:
            matched_output_indices.add(matched_mode["recovered_index"])

    recovered_periods = np.asarray(
        waves_background_modes["periods"],
        dtype=float,
    )
    feasible_unmatched_indices = np.asarray([
        recovered_index
        for recovered_index, recovered_period in enumerate(recovered_periods)
        if (
            recovered_index not in matched_output_indices
            and np.isfinite(recovered_period)
            and period_min < recovered_period < period_max
        )
    ], dtype=int)
    if feasible_unmatched_indices.size == 0:
        raise RuntimeError(
            "The waves-with-background run has no unmatched feasible mode."
        )

    spurious_index = int(feasible_unmatched_indices[np.argmin(
        np.abs(
            recovered_periods[feasible_unmatched_indices]
            - spurious_target_period
        )
    )])
    spurious_entry = {
        "phasor": np.asarray(
            waves_background_modes["phasors"][spurious_index],
            dtype=complex,
        ).ravel(),
        "eigenvalue": complex(
            waves_background_modes["continuous_eigenvalues"][
                spurious_index
            ]
        ),
        "period": float(recovered_periods[spurious_index]),
        "similarity": np.nan,
        "recovered_index": spurious_index,
        "label": "Spurious mode",
        "mode_key": None,
    }

    snapshot_entries = {
        "input_mode_62": {
            **input_entry("62"),
            "scale_group": "mode_62",
        },
        "wave_only_recovered_mode_62": {
            **wave_only_matches["62"],
            "scale_group": "mode_62",
        },
        "input_mode_48": {
            **input_entry("48"),
            "scale_group": "mode_48",
        },
        "wave_only_recovered_mode_48": {
            **wave_only_matches["48"],
            "scale_group": "mode_48",
        },
        "input_mode_1": {
            **input_entry("1"),
            "scale_group": "modes_1_6_13",
        },
        "input_mode_6": {
            **input_entry("6"),
            "scale_group": "modes_1_6_13",
        },
        "input_mode_13": {
            **input_entry("13"),
            "scale_group": "modes_1_6_13",
        },
        "wave_only_recovered_mode_6": {
            **wave_only_matches["6"],
            "scale_group": "modes_1_6_13",
        },
        "input_mode_45": {
            **input_entry("45"),
            "scale_group": "mode_45_and_spurious",
        },
        "waves_background_recovered_mode_45": {
            **waves_background_mode_45,
            "scale_group": "mode_45_and_spurious",
        },
        "waves_background_spurious_mode_5p2yr": {
            **spurious_entry,
            "scale_group": "mode_45_and_spurious",
        },
    }

    return snapshot_entries, syn_suite_info, case_outputs


def Save_Recovery_Visualisation_Snapshots(
    snapshot_entries,
    output_dir=None,
    state_shape=state_shape,
    longitude=longitude,
    latitude=latitude,
    figure_width=text_width / 4,
    font_size=11,
    cmap="seismic",
    png_dpi=600,
):
    """Save individual real-phasor Mollweide snapshots with shared scales."""

    if not snapshot_entries:
        raise ValueError("snapshot_entries cannot be empty.")
    if output_dir is None:
        output_dir = RESULT_DIR / "recovery_visualisation"
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    longitude = np.asarray(longitude, dtype=float)
    latitude = np.asarray(latitude, dtype=float)
    if longitude.size != state_shape[1] or latitude.size != state_shape[0]:
        raise ValueError(
            "longitude and latitude lengths must agree with state_shape."
        )

    longitude_wrapped = ((longitude + 180.0) % 360.0) - 180.0
    longitude_order = np.argsort(longitude_wrapped)
    longitude_radians = np.deg2rad(longitude_wrapped[longitude_order])
    latitude_radians = np.deg2rad(latitude)
    longitude_grid, latitude_grid = np.meshgrid(
        longitude_radians,
        latitude_radians,
    )

    real_snapshots = {}
    group_values = {}
    expected_size = int(np.prod(state_shape))
    for snapshot_key, entry in snapshot_entries.items():
        phasor = np.asarray(entry["phasor"], dtype=complex).ravel()
        if phasor.size != expected_size:
            raise ValueError(
                f"Snapshot '{snapshot_key}' has phasor length "
                f"{phasor.size}; expected {expected_size}."
            )
        snapshot = np.real(phasor).reshape(state_shape)
        snapshot = snapshot[:, longitude_order]
        real_snapshots[snapshot_key] = snapshot
        finite_values = snapshot[np.isfinite(snapshot)]
        if finite_values.size == 0:
            raise ValueError(
                f"Snapshot '{snapshot_key}' has no finite real values."
            )
        group_values.setdefault(entry["scale_group"], []).append(
            finite_values
        )

    group_limits = {}
    for group_key, finite_arrays in group_values.items():
        colour_limit = float(np.max(np.abs(np.concatenate(finite_arrays))))
        if not np.isfinite(colour_limit) or colour_limit == 0:
            colour_limit = 1.0
        group_limits[group_key] = colour_limit

    figures = {}
    saved_paths = {}
    figure_height = 0.72 * figure_width
    for snapshot_key, entry in snapshot_entries.items():
        figure = plt.figure(figsize=(figure_width, figure_height))
        axis = figure.add_subplot(111, projection="mollweide")
        colour_limit = group_limits[entry["scale_group"]]
        axis.pcolormesh(
            longitude_grid,
            latitude_grid,
            real_snapshots[snapshot_key],
            shading="auto",
            cmap=cmap,
            vmin=-colour_limit,
            vmax=colour_limit,
            rasterized=True,
        )
        axis.grid(False)
        axis.set_xticks([])
        axis.set_yticks([])
        axis.tick_params(
            axis="both",
            which="both",
            labelbottom=False,
            labelleft=False,
            labelright=False,
            labeltop=False,
            length=0,
        )
        axis.set_title(
            f"{entry['label']}\n"
            rf"$T = {float(entry['period']):.1f}\,\mathrm{{yr}}$",
            fontsize=font_size,
            pad=3,
        )
        figure.subplots_adjust(
            left=0.01,
            right=0.99,
            bottom=0.02,
            top=0.72,
        )

        png_path = output_dir / f"{snapshot_key}.png"
        pdf_path = output_dir / f"{snapshot_key}.pdf"
        figure.savefig(
            png_path,
            dpi=png_dpi,
            facecolor="white",
        )
        figure.savefig(
            pdf_path,
            dpi=png_dpi,
            facecolor="white",
        )
        figures[snapshot_key] = figure
        saved_paths[snapshot_key] = {
            "png": png_path,
            "pdf": pdf_path,
        }

    return figures, saved_paths, group_limits


def Plot_Synthetic_Mode_Visualisation(
    visualisation_rows,
    state_shape=state_shape,
    longitude=longitude,
    latitude=latitude,
    figure_width=text_width,
    font_size=11,
    cmap="seismic",
    sort_by_period=True,
    mode_colours=None,
    row_height=1.78,
):
    """Plot phase-aligned real phasors across four record contexts.

    Each mode occupies one compact row with a shared horizontal colour bar.
    Snapshot titles report period and input-mode similarity.
    """

    if not visualisation_rows:
        raise ValueError("visualisation_rows cannot be empty.")
    if font_size < 11:
        raise ValueError("font_size must be at least 11 for thesis figures.")
    if figure_width <= 0 or row_height <= 0:
        raise ValueError("figure_width and row_height must be positive.")

    column_keys = (
        "ideal",
        "resolved",
        "signal_background",
        "null_background",
    )
    column_labels = (
        "Ideal input",
        "Resolved ensemble",
        "Resolved + background\nensemble",
        "Background only\n(closest period)",
    )
    mode_keys = list(visualisation_rows.keys())
    if mode_colours is None:
        mode_colours = {
            mode_key: tol_muted[mode_index % len(tol_muted)]
            for mode_index, mode_key in enumerate(mode_keys)
        }
    else:
        missing_colours = set(mode_keys) - set(mode_colours)
        if missing_colours:
            raise KeyError(
                "Missing display colours for modes: "
                f"{sorted(missing_colours)}"
            )
    if sort_by_period:
        mode_keys = sorted(
            mode_keys,
            key=lambda key: visualisation_rows[key]["reference_period"],
        )

    longitude = np.asarray(longitude, dtype=float)
    latitude = np.asarray(latitude, dtype=float)
    if longitude.size != state_shape[1] or latitude.size != state_shape[0]:
        raise ValueError(
            "longitude and latitude lengths must agree with state_shape."
        )

    longitude_wrapped = ((longitude + 180.0) % 360.0) - 180.0
    longitude_order = np.argsort(longitude_wrapped)
    longitude_radians = np.deg2rad(longitude_wrapped[longitude_order])
    latitude_radians = np.deg2rad(latitude)
    longitude_grid, latitude_grid = np.meshgrid(
        longitude_radians,
        latitude_radians,
    )

    def compact_colourbar_value(value):
        value = float(value)
        if np.isclose(value, 0.0):
            return "0"
        exponent = int(np.floor(np.log10(abs(value))))
        if abs(exponent) >= 3:
            mantissa = value / (10.0 ** exponent)
            return rf"${mantissa:.2g}\times10^{{{exponent}}}$"
        return f"{value:.3g}"

    def compact_metric_value(value, format_spec):
        value = float(value)
        if np.isnan(value):
            return "—"
        if np.isposinf(value):
            return "∞"
        if np.isneginf(value):
            return "−∞"
        return format(value, format_spec)

    n_rows = len(mode_keys)
    figure_height = max(row_height * n_rows + 0.48, 3.5)
    fig = plt.figure(
        figsize=(figure_width, figure_height),
    )
    grid = fig.add_gridspec(
        1 + 3 * n_rows,
        4,
        height_ratios=[0.25] + [0.30, 1.0, 0.27] * n_rows,
        hspace=0.08,
        wspace=0.06,
    )
    fig.subplots_adjust(
        left=0.015,
        right=0.985,
        top=0.985,
        bottom=0.015,
    )
    axes = np.empty((n_rows, 4), dtype=object)

    for column_index, column_label in enumerate(column_labels):
        heading_axis = fig.add_subplot(grid[0, column_index])
        heading_axis.axis("off")
        heading_axis.text(
            0.5,
            0.0,
            column_label,
            transform=heading_axis.transAxes,
            ha="center",
            va="bottom",
            fontsize=font_size,
            fontweight="bold",
        )

    for row_index, mode_key in enumerate(mode_keys):
        row_results = visualisation_rows[mode_key]
        mode_heading_row = 1 + 3 * row_index
        snapshot_row = mode_heading_row + 1
        colourbar_row = mode_heading_row + 2

        mode_heading_axis = fig.add_subplot(
            grid[mode_heading_row, :]
        )
        mode_heading_axis.axis("off")
        mode_heading_axis.text(
            0.5,
            0.95,
            f"Mode {mode_key}",
            transform=mode_heading_axis.transAxes,
            ha="center",
            va="top",
            fontsize=font_size,
            fontweight="bold",
            color=mode_colours[mode_key],
        )

        first_three_snapshots = []
        for column_key in column_keys[:3]:
            entry = row_results.get(column_key)
            if entry is None:
                continue
            snapshot = np.real(
                np.asarray(entry["phasor"], dtype=complex)
            )
            first_three_snapshots.append(snapshot[np.isfinite(snapshot)])

        if not first_three_snapshots:
            raise ValueError(
                f"Mode '{mode_key}' has no finite snapshots in columns 1--3."
            )
        colour_limit = float(np.max(np.abs(np.concatenate(
            first_three_snapshots
        ))))
        if not np.isfinite(colour_limit) or colour_limit == 0:
            colour_limit = 1.0

        row_mesh = None
        null_exceeds_scale = False
        for column_index, column_key in enumerate(column_keys):
            axis = fig.add_subplot(
                grid[snapshot_row, column_index],
                projection="mollweide",
            )
            axes[row_index, column_index] = axis
            entry = row_results.get(column_key)

            axis.grid(
                True,
                linestyle=":",
                linewidth=0.6,
                alpha=0.5,
            )
            axis.tick_params(
                axis="both",
                which="both",
                labelbottom=False,
                labelleft=False,
                labelright=False,
                labeltop=False,
                length=0,
            )

            if entry is None:
                axis.text(
                    0.5,
                    0.5,
                    "No match",
                    transform=axis.transAxes,
                    ha="center",
                    va="center",
                    fontsize=font_size,
                    fontweight="bold",
                    color="0.35",
                )
                axis.set_title(
                    "T=— yr | S=—",
                    fontsize=font_size,
                    pad=4,
                )
                continue

            phasor = np.asarray(entry["phasor"], dtype=complex).ravel()
            if phasor.size != int(np.prod(state_shape)):
                raise ValueError(
                    f"Mode '{mode_key}', column '{column_key}' has "
                    f"phasor length {phasor.size}; expected "
                    f"{int(np.prod(state_shape))}."
                )
            snapshot = np.real(phasor).reshape(state_shape)
            snapshot = snapshot[:, longitude_order]
            if column_key == "null_background":
                null_exceeds_scale = (
                    np.nanmax(np.abs(snapshot)) > colour_limit
                )

            row_mesh = axis.pcolormesh(
                longitude_grid,
                latitude_grid,
                snapshot,
                shading="auto",
                cmap=cmap,
                vmin=-colour_limit,
                vmax=colour_limit,
                rasterized=True,
            )

            period = float(entry["period"])
            similarity = float(entry["similarity"])
            period_text = compact_metric_value(period, ".1f")
            similarity_text = compact_metric_value(similarity, ".2g")
            axis.set_title(
                f"T={period_text} yr | S={similarity_text}",
                fontsize=font_size,
                pad=4,
            )

        colourbar_host = fig.add_subplot(grid[colourbar_row, :])
        colourbar_host.axis("off")
        colourbar_axis = colourbar_host.inset_axes(
            [0.15, 0.64, 0.70, 0.16]
        )
        colourbar = fig.colorbar(
            row_mesh,
            cax=colourbar_axis,
            orientation="horizontal",
            extend="both" if null_exceeds_scale else "neither",
        )
        colourbar.set_ticks([])
        colourbar_axis.set_xlabel(
            r"SV amplitude (nT yr$^{-1}$)",
            fontsize=font_size,
            labelpad=2,
        )
        colourbar_host.text(
            0.14,
            0.72,
            compact_colourbar_value(-colour_limit),
            transform=colourbar_host.transAxes,
            ha="right",
            va="center",
            fontsize=font_size,
        )
        colourbar_host.text(
            0.86,
            0.72,
            compact_colourbar_value(colour_limit),
            transform=colourbar_host.transAxes,
            ha="left",
            va="center",
            fontsize=font_size,
        )

    return fig, axes


# plotting modes to visualise phasors

def Plot_Mode_Phasors(
    mode_suite,
    state_shape=state_shape,
    design_matrix=A_15_r,
    nmax=None,
    longitude=longitude,
    latitude=latitude,
    figure_width=text_width,
    font_size=11,
    cmap="seismic",
    scale="per_mode",
    sort_by_period=True,
):
    """
    Plot every mode phasor with:

        left column  = real component
        right column = imaginary component

    Supports mode dictionaries containing either:

        "gnm_phasor" : spherical-harmonic coefficient phasor

    or:

        "phasor" : already projected gridded physical-space phasor

    Period may be supplied as:

        "true_period"
        "period"

    or inferred from:

        "true_eigenvalue"
        "eigenvalue"

    Parameters
    ----------
    mode_suite : dict
        Synthetic-suite, control-suite, or similarly structured recovered-mode
        dictionary.

    state_shape : tuple
        Shape of one gridded spatial field, normally (n_latitude, n_longitude).

    design_matrix : ndarray
        Full radial-field design matrix, normally A_15_r.

    nmax : int or None
        Optional degree to which coefficient phasors should be truncated before
        projection. If None, the degree is inferred from each coefficient vector.

    scale : {"per_mode", "global"}
        "per_mode" gives each row its own symmetric colour scale.
        "global" uses one symmetric scale for the entire figure.

    Returns
    -------
    fig, axes
    """

    def get_period(mode_info):
        """Extract or calculate the mode period."""

        if "true_period" in mode_info:
            return float(mode_info["true_period"])

        if "period" in mode_info:
            return float(mode_info["period"])

        if "true_eigenvalue" in mode_info:
            eigenvalue = complex(mode_info["true_eigenvalue"])

        elif "eigenvalue" in mode_info:
            eigenvalue = complex(mode_info["eigenvalue"])

        else:
            return np.nan

        if np.isclose(eigenvalue.imag, 0.0):
            return np.inf

        return 2.0 * np.pi / np.abs(eigenvalue.imag)


    def infer_nmax_from_gauss(gauss_phasor):
        """Infer nmax from Ng = nmax * (nmax + 2)."""

        n_coefficients = np.asarray(gauss_phasor).size
        inferred_nmax = int(np.sqrt(n_coefficients + 1) - 1)

        expected_coefficients = inferred_nmax * (inferred_nmax + 2)

        if expected_coefficients != n_coefficients:
            raise ValueError(
                f"Cannot infer a complete spherical-harmonic degree from "
                f"{n_coefficients} coefficients."
            )

        return inferred_nmax


    def get_grid_phasor(mode_info):
        """Return one complex gridded physical-space phasor."""

        # Already in gridded physical space
        if "phasor" in mode_info:
            grid_phasor = np.asarray(
                mode_info["phasor"],
                dtype=complex,
            ).ravel()

        # Stored as Gauss coefficients
        elif "gnm_phasor" in mode_info:
            gauss_phasor = np.asarray(
                mode_info["gnm_phasor"],
                dtype=complex,
            ).ravel()

            available_nmax = infer_nmax_from_gauss(gauss_phasor)

            if nmax is None:
                nmax_used = available_nmax
            else:
                nmax_used = min(nmax, available_nmax)

            gauss_phasor = Truncate_Gauss_Coeffs(
                gauss_phasor,
                tmax=nmax_used,
            )

            A_used = Truncate_Gauss_Coeffs(
                design_matrix,
                tmax=nmax_used,
            )

            grid_phasor = A_used @ gauss_phasor

        else:
            raise KeyError(
                "Each mode must contain either 'phasor' or 'gnm_phasor'."
            )

        expected_size = np.prod(state_shape)

        if grid_phasor.size != expected_size:
            raise ValueError(
                f"Gridded phasor has length {grid_phasor.size}, "
                f"but state_shape={state_shape} requires {expected_size}."
            )

        return grid_phasor.reshape(state_shape)


    # Do not accidentally plot summary entries
    ignored_keys = {
        "match_count",
        "convergence_count",
    }

    mode_keys = [
        key
        for key, value in mode_suite.items()
        if (
            key not in ignored_keys
            and isinstance(value, dict)
            and (
                "phasor" in value
                or "gnm_phasor" in value
            )
        )
    ]

    if not mode_keys:
        raise ValueError(
            "No entries containing 'phasor' or 'gnm_phasor' were found."
        )

    periods = {
        key: get_period(mode_suite[key])
        for key in mode_keys
    }

    if sort_by_period:
        mode_keys = sorted(
            mode_keys,
            key=lambda key: periods[key],
        )

    grid_phasors = {
        key: get_grid_phasor(mode_suite[key])
        for key in mode_keys
    }

    n_modes = len(mode_keys)

    fig_height = max(
        2.0 * n_modes,
        0.65 * figure_width,
    )

    fig, axes = plt.subplots(
        nrows=n_modes,
        ncols=2,
        figsize=(figure_width, fig_height),
        squeeze=False,
        constrained_layout=True,
    )

    # Optional single global colour limit
    if scale == "global":
        global_limit = np.nanmax([
            np.nanmax(np.abs(grid_phasors[key].real))
            for key in mode_keys
        ] + [
            np.nanmax(np.abs(grid_phasors[key].imag))
            for key in mode_keys
        ])

    elif scale != "per_mode":
        raise ValueError(
            "scale must be either 'per_mode' or 'global'."
        )

    for row, mode_key in enumerate(mode_keys):

        phasor_grid = grid_phasors[mode_key]

        real_grid = phasor_grid.real
        imaginary_grid = phasor_grid.imag

        if scale == "per_mode":
            colour_limit = np.nanmax(
                np.abs(
                    np.concatenate([
                        real_grid.ravel(),
                        imaginary_grid.ravel(),
                    ])
                )
            )
        else:
            colour_limit = global_limit

        if not np.isfinite(colour_limit) or colour_limit == 0:
            colour_limit = 1.0

        period = periods[mode_key]

        if np.isfinite(period):
            period_text = f"Period = {period:.2f} years"
        else:
            period_text = "Static mode"

        real_image = axes[row, 0].pcolormesh(
            longitude,
            latitude,
            real_grid,
            shading="auto",
            cmap=cmap,
            vmin=-colour_limit,
            vmax=colour_limit,
        )

        axes[row, 1].pcolormesh(
            longitude,
            latitude,
            imaginary_grid,
            shading="auto",
            cmap=cmap,
            vmin=-colour_limit,
            vmax=colour_limit,
        )

        axes[row, 0].set_title(
            f"{mode_key}: real component\n{period_text}",
            fontsize=font_size,
        )

        axes[row, 1].set_title(
            f"{mode_key}: imaginary component\n{period_text}",
            fontsize=font_size,
        )

        for ax in axes[row, :]:
            ax.set_xlim(0, 360)
            ax.set_ylim(latitude.min(), latitude.max())
            ax.set_xlabel(
                "Longitude (°)",
                fontsize=font_size,
            )
            ax.tick_params(
                labelsize=font_size,
            )

        axes[row, 0].set_ylabel(
            "Latitude (°)",
            fontsize=font_size,
        )

        colourbar = fig.colorbar(
            real_image,
            ax=axes[row, :],
            location="right",
            shrink=0.85,
            pad=0.02,
        )

        colourbar.set_label(
            "SV amplitude",
            fontsize=font_size,
        )
        colourbar.ax.tick_params(
            labelsize=font_size,
        )

    return fig, axes


import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter
from pathlib import Path


def Plot_Mode_Suite_Video_Mollweide(
    mode_suite,
    times_used_relative,
    output_path,
    mode_keys=None,
    state_shape=state_shape,
    longitude=longitude,
    latitude=latitude,
    design_matrix=A_15_r,
    nmax=None,
    fps=5,
    dpi=150,
    figure_width=10,
    cmap="seismic",
    scale="per_mode",
    show_colorbars=False,
    sort_by_period=True,
    font_size=11,
):
    """
    Create one video showing the real reconstructed field for several modes,
    with one Mollweide subplot per row.

    Each row shows:
        x(t) = Re[ phasor * exp(eigenvalue * t) ]

    Compatible with dictionaries containing either:
        - 'gnm_phasor' and 'true_eigenvalue'
        - 'phasor' and 'eigenvalue'

    Period may be stored as:
        - 'true_period'
        - 'period'
    """

    times = np.asarray(times_used_relative, dtype=float).ravel()
    if times.size == 0:
        raise ValueError("times_used_relative is empty.")

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    expected_grid_size = int(np.prod(state_shape))

    # ------------------------------------------------------------------
    # Helper functions
    # ------------------------------------------------------------------

    def infer_nmax_from_gauss(gauss_phasor):
        n_coefficients = np.asarray(gauss_phasor).size
        inferred_nmax = int(np.sqrt(n_coefficients + 1) - 1)
        expected_coefficients = inferred_nmax * (inferred_nmax + 2)

        if expected_coefficients != n_coefficients:
            raise ValueError(
                f"Cannot infer nmax from {n_coefficients} coefficients."
            )

        return inferred_nmax

    def get_eigenvalue(mode_info):
        if "true_eigenvalue" in mode_info:
            return complex(mode_info["true_eigenvalue"])
        if "eigenvalue" in mode_info:
            return complex(mode_info["eigenvalue"])

        raise KeyError(
            "Mode entry must contain 'true_eigenvalue' or 'eigenvalue'."
        )

    def get_period(mode_info, eigenvalue):
        if "true_period" in mode_info:
            return float(mode_info["true_period"])
        if "period" in mode_info:
            return float(mode_info["period"])

        if np.isclose(eigenvalue.imag, 0.0):
            return np.inf

        return 2.0 * np.pi / np.abs(eigenvalue.imag)

    def get_grid_phasor(mode_info):
        if "gnm_phasor" in mode_info:
            gauss_phasor = np.asarray(
                mode_info["gnm_phasor"],
                dtype=complex,
            ).ravel()

            available_nmax = infer_nmax_from_gauss(gauss_phasor)

            if nmax is None:
                nmax_used = available_nmax
            else:
                nmax_used = min(int(nmax), available_nmax)

            gauss_phasor = Truncate_Gauss_Coeffs(
                gauss_phasor,
                tmax=nmax_used,
            )

            A_used = Truncate_Gauss_Coeffs(
                design_matrix,
                tmax=nmax_used,
            )

            grid_phasor = A_used @ gauss_phasor

        elif "phasor" in mode_info:
            grid_phasor = np.asarray(
                mode_info["phasor"],
                dtype=complex,
            ).ravel()

        else:
            raise KeyError(
                "Mode entry must contain either 'gnm_phasor' or 'phasor'."
            )

        if grid_phasor.size != expected_grid_size:
            raise ValueError(
                f"Mode phasor has length {grid_phasor.size}, "
                f"but expected {expected_grid_size}."
            )

        return grid_phasor

    # ------------------------------------------------------------------
    # Select valid modes
    # ------------------------------------------------------------------

    ignored_keys = {"match_count", "convergence_count"}

    valid_mode_keys = [
        key
        for key, value in mode_suite.items()
        if (
            key not in ignored_keys
            and isinstance(value, dict)
            and ("phasor" in value or "gnm_phasor" in value)
        )
    ]

    if mode_keys is None:
        mode_keys = valid_mode_keys
    else:
        mode_keys = list(mode_keys)

    if len(mode_keys) == 0:
        raise ValueError("No valid modes were found.")

    # ------------------------------------------------------------------
    # Prepare longitude ordering for Mollweide
    # ------------------------------------------------------------------

    # Convert 0..360 to -180..180 and sort for proper Mollweide display
    lon_wrapped_deg = ((np.asarray(longitude) + 180) % 360) - 180
    lon_sort_idx = np.argsort(lon_wrapped_deg)
    lon_plot_deg = lon_wrapped_deg[lon_sort_idx]
    lat_plot_deg = np.asarray(latitude)

    lon_plot_rad = np.deg2rad(lon_plot_deg)
    lat_plot_rad = np.deg2rad(lat_plot_deg)

    lon2d, lat2d = np.meshgrid(lon_plot_rad, lat_plot_rad)

    # ------------------------------------------------------------------
    # Build mode cubes
    # ------------------------------------------------------------------

    mode_data = {}

    for mode_key in mode_keys:
        mode_info = mode_suite[mode_key]

        eigenvalue = get_eigenvalue(mode_info)
        period = get_period(mode_info, eigenvalue)
        phasor = get_grid_phasor(mode_info)

        physical_series = np.real(
            phasor[None, :] * np.exp(eigenvalue * times[:, None])
        )

        physical_cube = physical_series.reshape(
            len(times),
            *state_shape,
        )

        # reorder longitude dimension for Mollweide
        physical_cube = physical_cube[:, :, lon_sort_idx]

        mode_data[mode_key] = {
            "eigenvalue": eigenvalue,
            "period": period,
            "cube": physical_cube,
        }

    if sort_by_period:
        mode_keys = sorted(
            mode_keys,
            key=lambda k: mode_data[k]["period"]
        )

    # ------------------------------------------------------------------
    # Colour scales
    # ------------------------------------------------------------------

    if scale == "per_mode":
        colour_limits = {
            key: np.nanmax(np.abs(mode_data[key]["cube"]))
            for key in mode_keys
        }
    elif scale == "global":
        global_limit = np.nanmax([
            np.nanmax(np.abs(mode_data[key]["cube"]))
            for key in mode_keys
        ])
        colour_limits = {key: global_limit for key in mode_keys}
    else:
        raise ValueError("scale must be 'per_mode' or 'global'.")

    for key in mode_keys:
        if not np.isfinite(colour_limits[key]) or colour_limits[key] == 0:
            colour_limits[key] = 1.0

    # ------------------------------------------------------------------
    # Figure
    # ------------------------------------------------------------------

    n_modes = len(mode_keys)
    fig_height = 3.2 * n_modes

    fig = plt.figure(figsize=(figure_width, fig_height))

    axes = []
    meshes = []

    for i, mode_key in enumerate(mode_keys):
        ax = fig.add_subplot(
            n_modes,
            1,
            i + 1,
            projection="mollweide",
        )
        axes.append(ax)

        cube = mode_data[mode_key]["cube"]
        period = mode_data[mode_key]["period"]
        sigma = mode_data[mode_key]["eigenvalue"].real
        clim = colour_limits[mode_key]

        mesh = ax.pcolormesh(
            lon2d,
            lat2d,
            cube[0],
            shading="auto",
            cmap=cmap,
            vmin=-clim,
            vmax=clim,
        )
        meshes.append(mesh)

        if np.isfinite(period):
            period_text = f"{period:.2f} yr"
        else:
            period_text = "static"

        ax.set_title(
            f"{mode_key}   |   Period = {period_text}   |   σ = {sigma:.3f} yr$^{{-1}}$",
            fontsize=font_size,
            pad=12,
        )

        ax.grid(True, alpha=0.4)

        if show_colorbars:
            cbar = fig.colorbar(
                mesh,
                ax=ax,
                orientation="horizontal",
                pad=0.08,
                fraction=0.05,
            )
            cbar.ax.tick_params(labelsize=font_size - 1)

    time_text = fig.suptitle(
        f"t = {times[0]:.2f} years",
        fontsize=font_size + 1,
        y=0.995,
    )

    fig.subplots_adjust(
        left=0.05,
        right=0.95,
        top=0.97,
        bottom=0.03,
        hspace=0.30,
    )

    # ------------------------------------------------------------------
    # Animation
    # ------------------------------------------------------------------

    def update(frame):
        for mode_key, mesh in zip(mode_keys, meshes):
            data = mode_data[mode_key]["cube"][frame]
            mesh.set_array(data.ravel())

        time_text.set_text(f"t = {times[frame]:.2f} years")
        return meshes + [time_text]

    animation = FuncAnimation(
        fig,
        update,
        frames=len(times),
        interval=1000 / fps,
        blit=False,
    )

    writer = FFMpegWriter(fps=fps)
    animation.save(output_path, writer=writer, dpi=dpi)

    plt.close(fig)

    print(f"Saved video to: {output_path}")

    return output_path
