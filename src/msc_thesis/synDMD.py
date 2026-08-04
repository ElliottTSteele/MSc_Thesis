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
from matplotlib.lines import Line2D
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

def Build_Exact_DMD(svd_rank=-1, fbdmd=False):

    return DMD(
        svd_rank=svd_rank,
        exact=True,
        forward_backward=fbdmd,
    )

def Build_BOPDMD(svd_rank=-1, bopdmd=False, bopdmd_params=None):

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
                    "maxiter": 200,
                    "tol": 1e-3,
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

    if weight_flag:
        recovered_modes["phasors"] = [Unweight_Phasor(phasor) for phasor in recovered_modes["phasors"]]

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
    periods = [2.0 * np.pi / np.abs(eig.imag) for eig in continuous_eigenvalues]

    return {
        "phasors": output_phasors,
        "discrete_eigenvalues": discrete_eigenvalues,
        "continuous_eigenvalues": continuous_eigenvalues,
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

# for single mode
def Match_Input_Output_Mode(mode_info, recovered_modes, nmax):
    # have input and output phasors + periods

    # define projection
    A_r = Truncate_Gauss_Coeffs(A_15_r, tmax=nmax)

    # defining nmax based threshold
    nmax_key = str(nmax)
    full_sim_threshold = similarity_thresholds["full_match"][nmax_key]
    par_sim_threshold = similarity_thresholds["partial_match"][nmax_key]

    output_periods = recovered_modes["periods"]
    output_phasors = recovered_modes["phasors"]

    input_gnm_phasor = mode_info["gnm_phasor"]
    input_phasor = A_r @ input_gnm_phasor.T
    input_period = mode_info["true_period"]

    match_made = False
    best_similarity = 0

    for output_idx, output_period in enumerate(output_periods):

        period_error = np.abs((output_period - input_period)/(input_period))

        if period_error < period_frac_threshold:

            output_phasor = output_phasors[output_idx]
            similarity = Complex_Phasor_Compare(input_phasor, output_phasor)

            if similarity >= full_sim_threshold and similarity > best_similarity:

                match_made = True
                match_period_error = period_error
                match_similarity = similarity
                best_similarity = similarity

    if match_made:
        return {
            "period_error":match_period_error, 
            "similarity":match_similarity
        }
    else:
        return None


import numpy as np


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
def Match_Input_Output_Modes(syn_suite_info, recovered_modes, nmax, syn_record=True):
    # have input and output phasors + periods

    # define projection
    A_r = Truncate_Gauss_Coeffs(A_15_r, tmax=nmax)

    # defining nmax based threshold
    nmax_key = str(nmax)

    sim_threshold = similarity_thresholds["full_match"][nmax_key]

    # match all inputs to an output

    output_periods = recovered_modes["periods"]
    output_phasors = recovered_modes["phasors"]

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

            period_error = np.abs((output_period - input_period)/(input_period))

            if period_error < period_frac_threshold:

                output_phasor = output_phasors[output_idx]

                similarity = Complex_Phasor_Compare(input_phasor, output_phasor)

                if similarity >= sim_threshold and similarity > best_similarity:

                    match_made = True
                    match_period = output_period
                    match_period_error = period_error
                    match_similarity = similarity
                    best_similarity = similarity

        # if there is a match, save its properties
        if match_made:
            match_results[mode_key] = {
                "match_period_error":match_period_error, 
                "match_similarity":match_similarity,
                "match_period":match_period
            }
            match_count += 1
        else:
            match_results[mode_key] = False

    # saving match count
    match_results["match_count"] = match_count

    return(match_results)

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
def Perform_DMD_on_Record(syn_suite_info, record_physical, nmax, svd_rank, dmd_settings, syn_record):

    # running DMD
    try:
        # recovering physical modes from dmd
        recovered_modes = Run_DMD(record_physical, times_used_relative, svd_rank, dmd_settings)

        # matching input and output modes
        match_results = Match_Input_Output_Modes(syn_suite_info, recovered_modes, nmax=nmax, syn_record=syn_record)

    except Exception:
        match_results = None

    return(match_results)

# -------------------------------------
# GIVEN SET OF RECORDS, APPLY DMD RETURN RESULTS
# -------------------------------------

def Pipeline_Run(record_dict_global, syn_suite_info_global, dmd_settings, nmax, svd_rank, ensemble_settings, syn_record=True):

    # first truncating all signals to relevant nmax
    A_r = Truncate_Gauss_Coeffs(A_15_r, tmax=nmax)

    # truncting input records
    record_dict = {}
    for record_key in record_dict_global:
        record_dict[record_key] = Truncate_Gauss_Coeffs(record_dict_global[record_key],
                                                        tmax=nmax)

    results_dict = {}

    for record_key in record_dict:

        record_used = record_dict[record_key]
        # projecting to physical space
        record_physical = A_r @ record_used.T

        record_match_results = Perform_DMD_on_Record(syn_suite_info_global, record_physical, 
                                                     nmax, svd_rank, dmd_settings, syn_record)
        results_dict[record_key] = record_match_results

    # Running perturbation ensemble
    ensemble_flag = ensemble_settings["ensemble_flag"]

    if ensemble_flag:

        record_for_ensemble = ensemble_settings["record_for_ensemble"]
        n_ensemble = ensemble_settings["n_ensemble"]

        results_dict["ensemble"] = {}

        perturbation_path = (f"{CHAOS_COV_DIR}/CHAOS_0806_nmax15_SV_nTyr_perturbations_N1000_Nt53.h5")

        with h5py.File(perturbation_path, "r") as f:

            for ensemble_idx in tqdm(range(n_ensemble), desc="ensembles"):

                # loading in i-th perturbation gauss time series
                gnm_perturbation_i = f["perturbations"][ensemble_idx, :, :].T
                # truncation perturbation gauss time series to nmax
                gnm_perturbation_i = Truncate_Gauss_Coeffs(gnm_perturbation_i, tmax=nmax)
                # forming realised perturbed time series
                gnm_perturbed_i = record_dict[record_for_ensemble] + gnm_perturbation_i
                # projecting to physical space
                record_physical = A_r @ gnm_perturbed_i.T

                perturbation_match_results = Perform_DMD_on_Record(syn_suite_info_global, record_physical, 
                                                                   nmax, svd_rank, dmd_settings, syn_record)
                results_dict["ensemble"][str(ensemble_idx)] = perturbation_match_results
    else:
        results_dict["ensemble"] = False

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

        ensemble_convergence_count=0

        ensemble_dict = results_dict["ensemble"]

        for ensemble_idx in ensemble_dict:

            ensemble_i_dict = ensemble_dict[ensemble_idx]

            if ensemble_i_dict:

                ensemble_convergence_count +=1

                for mode_key in ensemble_i_dict:

                    if mode_key == "match_count":

                        suite_ensemble_result["match_count"].append(ensemble_i_dict[mode_key])

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
        match_count_median, match_count_p05, match_count_p95 = Percentile_Summary(
            suite_ensemble_result["match_count"], p=5
        )
        results_dict["median"]["match_count"] = {}
        results_dict["median"]["match_count"]["value"] = match_count_median
        results_dict["median"]["match_count"]["p05"] = match_count_p05
        results_dict["median"]["match_count"]["p95"] = match_count_p95

        # converting to median with error intervals
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

    return(results_dict)


#
# PLOTTING RESULTS
#

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

from matplotlib.colors import BoundaryNorm

def Plot_Match_Count_Heatmap(
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
        max(0, similarity_threshold - similarity_margin),
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
            true_period = mode_info["true_period"]
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

from matplotlib.colors import to_rgb

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

