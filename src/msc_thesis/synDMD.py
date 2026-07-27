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
                          dt=dt_years,
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


# ------------------------------------------------------
# LONG-PERIOD REMOVAL FILTER
# ------------------------------------------------------

def Long_Period_Taper_Filter(
    time_series,
    dt,
    pass_period=10.0,
    stop_period=20.0,
    axis=0,
    passband_loss_db=1.0,
    stopband_attenuation_db=40.0,
):
    """
    Remove poorly constrained long-period variability with a zero-phase
    Butterworth high-pass filter.

    The filter is applied forwards and backwards using Gustafsson initial
    conditions. This avoids treating the finite record as periodic, does not
    invent a reflected continuation, reduces endpoint transients, and cancels
    phase delay. Butterworth filtering gives a maximally flat passband.

    The supplied attenuation specifications describe the effective
    forward-backward response. The one-way design uses half of each dB
    specification because forward-backward filtering squares the magnitude
    response.

    Default design response
    -----------------------
    Period <= 10 years:
        Passband, with no more than 1 dB attenuation at the 10-year edge.

    10 < period < 20 years:
        Monotonic Butterworth transition band.

    Period >= 20 years:
        Stopband, with at least 40 dB attenuation at the 20-year edge.

    Static/DC content is removed.

    These dB values describe the theoretical steady-state zero-phase response.
    A finite record containing only one or two cycles near the transition
    cannot attain that response everywhere; endpoint uncertainty remains, but
    Gustafsson initialisation avoids periodic-wrap and padding artefacts.

    Parameters
    ----------
    time_series : ndarray
        Real time series. Gauss-coefficient input normally has shape
        ``(Nt, Ng)`` with time on ``axis=0``.

    dt : float
        Temporal sampling interval in years.

    pass_period : float
        Short-period/passband edge in years.

    stop_period : float
        Long-period/stopband edge in years.

    axis : int
        Axis corresponding to time.

    passband_loss_db : float
        Maximum effective attenuation at the passband edge.

    stopband_attenuation_db : float
        Minimum effective attenuation at the stopband edge.

    Returns
    -------
    filtered : ndarray
        Filtered time series with the same shape as the input.

    response : ndarray
        Effective zero-phase magnitude response evaluated at the positive FFT
        frequencies of this record. This is diagnostic only; filtering itself
        is performed by ``filtfilt``.

    frequencies : ndarray
        Positive FFT frequencies in cycles/year.
    """

    time_series = np.asarray(
        time_series,
        dtype=np.float64,
    )

    if dt <= 0:
        raise ValueError(
            "dt must be positive."
        )

    if pass_period <= 0 or stop_period <= 0:
        raise ValueError(
            "pass_period and stop_period must both be positive."
        )

    if stop_period <= pass_period:
        raise ValueError(
            "stop_period must be greater than pass_period. "
            "For example: pass_period=10, stop_period=20."
        )

    if passband_loss_db <= 0:
        raise ValueError(
            "passband_loss_db must be positive."
        )

    if stopband_attenuation_db <= 0:
        raise ValueError(
            "stopband_attenuation_db must be positive."
        )

    f_stop = 1.0 / stop_period
    f_pass = 1.0 / pass_period
    sample_frequency = 1.0 / dt
    nyquist = 0.5 * sample_frequency

    if f_pass >= nyquist:
        raise ValueError(
            "The passband edge must lie below the Nyquist frequency. "
            f"Received f_pass={f_pass}, Nyquist={nyquist}."
        )

    n_time = (
        time_series.shape[
            axis
        ]
    )

    if n_time < 2:
        raise ValueError(
            "The time axis must contain at least two samples."
        )

    # Forward-backward filtering squares the magnitude response, doubling
    # attenuation in dB. Design the one-way filter to half the requested
    # effective specifications.
    filter_order, critical_frequency = buttord(
        wp=f_pass,
        ws=f_stop,
        gpass=0.5 * passband_loss_db,
        gstop=0.5 * stopband_attenuation_db,
        fs=sample_frequency,
    )

    numerator, denominator = butter(
        filter_order,
        critical_frequency,
        btype="highpass",
        fs=sample_frequency,
        output="ba",
    )

    # Remove DC exactly before solving the finite-record filter problem.
    # Keeping the singleton time dimension preserves broadcasting for any
    # supported time axis.
    temporal_mean = np.mean(
        time_series,
        axis=axis,
        keepdims=True,
    )
    time_series_zero_mean = (
        time_series
        - temporal_mean
    )

    # Gustafsson's method chooses initial states so forward-backward and
    # backward-forward filtering agree. It handles the finite endpoints
    # without periodic wrapping or synthetic reflection padding.
    filtered = filtfilt(
        numerator,
        denominator,
        time_series_zero_mean,
        axis=axis,
        method="gust",
    )

    frequencies = np.fft.rfftfreq(
        n_time,
        d=dt,
    )

    _, one_way_response = freqz(
        numerator,
        denominator,
        worN=frequencies,
        fs=sample_frequency,
    )

    response = np.abs(
        one_way_response
    )**2

    return (
        filtered,
        response,
        frequencies,
    )

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

def DMD_Mode_Pair(
    dmd,
    dt,
    atol=1e-4,
    rtol=1e-4,
    include_unpaired=False, # dont include unpaired modes
):

    eigenvalues = np.asarray(dmd.eigs, dtype=complex)
    modes = np.asarray(dmd.modes, dtype=complex)
    amplitudes = np.asarray(dmd.amplitudes, dtype=complex)

    if modes.shape[1] != eigenvalues.size:
        raise ValueError(
            "The number of columns in dmd.modes must equal the number "
            "of eigenvalues."
        )

    if amplitudes.size != eigenvalues.size:
        raise ValueError(
            "dmd.amplitudes must contain one amplitude per DMD mode."
        )

    # Complete dynamically scaled modal coefficients b_j phi_j.
    scaled_modes = modes * amplitudes[np.newaxis, :]

    used = set()
    output_phasors = []
    output_eigenvalues = []
    mode_information = []

    for i, eig_i in enumerate(eigenvalues):

        if i in used:
            continue

        # ---------------------------------------------------------
        # Real/non-oscillatory eigenvalue
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
        # Find the closest unused conjugate eigenvalue
        # ---------------------------------------------------------
        candidate_indices = [
            j
            for j in range(eigenvalues.size)
            if j != i and j not in used
        ]

        if candidate_indices:

            conjugate_target = np.conj(eig_i)

            errors = np.array([
                np.abs(eigenvalues[j] - conjugate_target)
                for j in candidate_indices
            ])

            best_position = np.argmin(errors)
            j = candidate_indices[best_position]
            pair_error = errors[best_position]

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
        if is_pair:

            # Positive frequency corresponds to positive discrete angle.
            if np.angle(eig_i) > 0:
                positive_idx = i
                negative_idx = j
            else:
                positive_idx = j
                negative_idx = i

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

            if include_unpaired:

                output_phasors.append(scaled_modes[:, i])
                output_eigenvalues.append(eig_i)

                mode_information.append({
                    "type": "unpaired_complex",
                    "indices": (i,),
                    "pair_error": pair_error,
                })

                warnings.warn(
                    f"DMD mode {i} with eigenvalue {eig_i} has no "
                    "conjugate partner.",
                    RuntimeWarning,
                )

    if len(output_phasors) == 0:
        output_phasors = np.empty(
            (modes.shape[0], 0),
            dtype=complex,
        )
        output_eigenvalues = np.asarray(
            [],
            dtype=complex,
        )
    else:
        output_phasors = np.column_stack(output_phasors)
        output_eigenvalues = np.asarray(output_eigenvalues)

    continuous_eigenvalues = np.asarray(
        D_To_C_Eigenvalue_Converter(
            output_eigenvalues,
            dt=dt,
        )
    )

    return {
        "phasors": output_phasors,
        "discrete_eigenvalues": output_eigenvalues,
        "continuous_eigenvalues": continuous_eigenvalues,
        "mode_information": mode_information,
    }

def DMD_Mode_Pair_Old(dmd_out, tol=1e-8):

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
# Generalised function that outputs general dmd mode candidates
# ---------------------------------------------------------------

def finalise_candidate_suite(
    continuous_eigenvalues,
    phasors,
    n_physical=None,
    candidate_period_min=None,
    candidate_period_max=None,
):
    """
    Convert an already-standardised set of continuous-time eigenvalues
    and physical-space phasors into the common candidate suite.

    Parameters
    ----------
    continuous_eigenvalues : array-like, shape (n_modes,)
        Continuous-time eigenvalues lambda = sigma + i*omega.

    phasors : array-like, shape (n_space, n_modes)
        Amplitude-scaled complex spatial phasors in PHYSICAL state space.

    n_physical : int or None
        Optional expected physical state dimension. If supplied, validates
        that the recovered phasors have the correct spatial dimension.

    candidate_period_min, candidate_period_max : float or None
        Optional strict bounds for oscillatory periods. With the default
        ``None`` values, oscillatory modes of every positive finite period
        are retained. Static modes are always retained and assigned
        ``period=np.inf``.

    Returns
    -------
    candidate_eigs : ndarray, shape (n_valid,)
        Valid continuous-time eigenvalues, including static modes.

    candidate_modes : ndarray, shape (n_space, n_valid)
        Corresponding physical-space complex phasors.

    candidate_periods : ndarray, shape (n_valid,)
        Corresponding positive oscillation periods. Static modes use infinity.
    """

    candidate_eigs = np.asarray(
        continuous_eigenvalues,
        dtype=complex,
    )

    candidate_modes = np.asarray(
        phasors,
        dtype=complex,
    )

    # ---------------------------------------------------------
    # HANDLE EMPTY RESULT
    # ---------------------------------------------------------

    if candidate_eigs.size == 0:

        if n_physical is None:

            if candidate_modes.ndim == 2:
                n_rows = candidate_modes.shape[0]
            else:
                n_rows = 0

        else:
            n_rows = n_physical

        return (
            np.asarray([], dtype=complex),
            np.empty(
                (n_rows, 0),
                dtype=complex,
            ),
            np.asarray([], dtype=float),
        )

    # ---------------------------------------------------------
    # VALIDATE SHAPES
    # ---------------------------------------------------------

    if candidate_modes.ndim != 2:
        raise ValueError(
            "Candidate phasors must be a 2-D array with shape "
            "(n_physical, n_modes). "
            f"Received shape {candidate_modes.shape}."
        )

    if (
        candidate_modes.shape[1]
        != candidate_eigs.size
    ):
        raise ValueError(
            "Number of candidate phasor columns must equal "
            "the number of eigenvalues. "
            f"{candidate_modes.shape[1]} vs "
            f"{candidate_eigs.size}."
        )

    if (
        n_physical is not None
        and candidate_modes.shape[0] != n_physical
    ):
        raise ValueError(
            "Candidate modes are not in the expected physical "
            "state space. "
            f"Expected {n_physical} rows, received "
            f"{candidate_modes.shape[0]}."
        )

    # ---------------------------------------------------------
    # CONTINUOUS EIGENVALUE -> PERIOD
    #
    # lambda = sigma + i*omega
    # T = 2*pi / |omega|
    # ---------------------------------------------------------

    candidate_periods = np.full(
        candidate_eigs.size,
        np.inf,
        dtype=float,
    )

    angular_frequency = np.abs(
        np.imag(candidate_eigs)
    )

    oscillatory = (
        angular_frequency > 0
    )

    candidate_periods[
        oscillatory
    ] = (
        2.0
        * np.pi
        / angular_frequency[
            oscillatory
        ]
    )

    # ---------------------------------------------------------
    # RETAIN FINITE PHYSICAL MODES
    # ---------------------------------------------------------

    valid = (
        np.isfinite(candidate_eigs.real)
        & np.isfinite(candidate_eigs.imag)
        & np.all(
            np.isfinite(candidate_modes.real)
            & np.isfinite(candidate_modes.imag),
            axis=0,
        )
    )

    if candidate_period_min is not None:
        valid &= (
            ~oscillatory
            | (candidate_periods > candidate_period_min)
        )

    if candidate_period_max is not None:
        valid &= (
            ~oscillatory
            | (candidate_periods < candidate_period_max)
        )

    return (
        candidate_eigs[valid],
        candidate_modes[:, valid],
        candidate_periods[valid],
    )

def filter_candidate_period_range(
    candidate_eigs,
    candidate_modes,
    candidate_periods,
    lower_period,
    upper_period,
):
    """
    Retain only oscillatory candidates inside an inclusive period interval.

    Static modes use ``period=np.inf`` and are therefore excluded whenever
    this optional finite-period window is applied.
    """

    candidate_eigs = np.asarray(
        candidate_eigs,
        dtype=complex,
    ).ravel()
    candidate_modes = np.asarray(
        candidate_modes,
        dtype=complex,
    )
    candidate_periods = np.asarray(
        candidate_periods,
        dtype=float,
    ).ravel()

    if not (
        np.isfinite(lower_period)
        and np.isfinite(upper_period)
        and 0.0 < lower_period <= upper_period
    ):
        raise ValueError(
            "Period limits must be finite and satisfy "
            "0 < lower_period <= upper_period."
        )

    if (
        candidate_modes.ndim != 2
        or candidate_modes.shape[1]
        != candidate_eigs.size
        or candidate_periods.size
        != candidate_eigs.size
    ):
        raise ValueError(
            "Candidate eigenvalues, mode columns, and periods must align."
        )

    keep = (
        np.isfinite(candidate_periods)
        & (candidate_periods >= lower_period)
        & (candidate_periods <= upper_period)
    )

    return (
        candidate_eigs[keep],
        candidate_modes[:, keep],
        candidate_periods[keep],
    )

# ---------------------------------------------------------------
# standard DMD output handling
# ---------------------------------------------------------------

exact_svd_rank = -1
exact_tlsq_rank = 0
exact_opt = False
exact_rescale_mode = None
exact_sorted_eigs = False
exact_tikhonov_regularization = None

def build_exact_dmd(svd_rank=-1):
    """Construct the configured PyDMD Exact DMD instance."""

    return DMD(
        svd_rank=svd_rank,
        tlsq_rank=exact_tlsq_rank,
        exact=True,
        opt=exact_opt,
        rescale_mode=exact_rescale_mode,
        forward_backward=False,
        sorted_eigs=exact_sorted_eigs,
        tikhonov_regularization=exact_tikhonov_regularization,
    )

fbdmd_svd_rank = -1
fbdmd_tlsq_rank = 0
fbdmd_exact = True
fbdmd_opt = False
fbdmd_rescale_mode = None
fbdmd_sorted_eigs = False

def build_fbdmd(svd_rank=-1):
    """Construct the configured PyDMD Forward-Backward DMD instance."""

    return FbDMD(
        svd_rank=svd_rank,
        tlsq_rank=fbdmd_tlsq_rank,
        exact=fbdmd_exact,
        opt=fbdmd_opt,
        rescale_mode=fbdmd_rescale_mode,
        sorted_eigs=fbdmd_sorted_eigs,
    )

def extract_standard_dmd_candidates(
    dmd,
    dt_snapshot,
    n_physical=None,
    embedding_d=None,
    candidate_period_min=None,
    candidate_period_max=None,
):
    """
    Extract candidates from standard discrete-time DMD variants.

    Intended for:
        - Exact DMD / DMD
        - Forward-backward DMD / FbDMD

    These methods return discrete-time DMD eigenvalues, so the existing
    DMD_Mode_Pair function is used to:

        1. amplitude-scale the modes,
        2. combine conjugate pairs,
        3. convert discrete eigenvalues to continuous eigenvalues.

    If ``embedding_d`` is supplied, the paired augmented phasors are mapped
    to their zero-delay physical block before common candidate finalisation.
    """

    recovered_dict = DMD_Mode_Pair(
        dmd,
        dt=dt_snapshot,
    )

    continuous_eigenvalues = np.asarray(
        recovered_dict[
            "continuous_eigenvalues"
        ],
        dtype=complex,
    )

    physical_phasors = np.asarray(
        recovered_dict[
            "phasors"
        ],
        dtype=complex,
    )

    if embedding_d is not None:
        physical_phasors = physicalise_embedded_phasors(
            augmented_phasors=physical_phasors,
            n_physical=n_physical,
            expected_d=embedding_d,
        )

    return finalise_candidate_suite(
        continuous_eigenvalues=
            continuous_eigenvalues,

        phasors=
            physical_phasors,

        n_physical=
            n_physical,

        candidate_period_min=
            candidate_period_min,

        candidate_period_max=
            candidate_period_max,
    )

# ---------------------------------------------------------------
# Optimised DMD handling functions
# ---------------------------------------------------------------

bopdmd_svd_rank = 8

# Operator / projection settings
bopdmd_compute_A = False
bopdmd_use_proj = True
bopdmd_init_alpha = None
bopdmd_proj_basis = None

# ---------------------------------------------------------
# BAGGING
# ---------------------------------------------------------
#
# 0 = ordinary Optimized DMD
# >0 = BOP-DMD with that many bagging trials
#
bopdmd_num_trials = 0
bopdmd_trial_size = 0.6

# ---------------------------------------------------------
# EIGENVALUE HANDLING
# ---------------------------------------------------------

bopdmd_eig_sort = "auto"

# Examples:
#
# None
# {"conjugate_pairs"}
# {"stable", "conjugate_pairs"}
#
bopdmd_eig_constraints = {
    "conjugate_pairs"
}

bopdmd_mode_prox = None

# ---------------------------------------------------------
# BAGGING FAILURE HANDLING
# ---------------------------------------------------------

bopdmd_remove_bad_bags = False
bopdmd_bag_warning = 100
bopdmd_bag_maxfail = 200

# ---------------------------------------------------------
# VARIABLE PROJECTION
# ---------------------------------------------------------

bopdmd_varpro_opts_dict = None
bopdmd_real_eig_limit = None
bopdmd_varpro_flag = True

def build_bopdmd(svd_rank=-1):
    """
    Construct the configured PyDMD BOPDMD instance.

    Notes
    -----
    bopdmd_num_trials = 0:
        Standard Optimized DMD / variable-projection DMD.

    bopdmd_num_trials > 0:
        Bagging Optimized DMD (BOP-DMD).
    """

    return BOPDMD(
        svd_rank=svd_rank,
        compute_A=bopdmd_compute_A,
        use_proj=bopdmd_use_proj,
        init_alpha=bopdmd_init_alpha,
        proj_basis=bopdmd_proj_basis,
        num_trials=bopdmd_num_trials,
        trial_size=bopdmd_trial_size,
        eig_sort=bopdmd_eig_sort,
        eig_constraints=bopdmd_eig_constraints,
        mode_prox=bopdmd_mode_prox,
        remove_bad_bags=bopdmd_remove_bad_bags,
        bag_warning=bopdmd_bag_warning,
        bag_maxfail=bopdmd_bag_maxfail,
        varpro_opts_dict=bopdmd_varpro_opts_dict,
        real_eig_limit=bopdmd_real_eig_limit,
        varpro_flag=bopdmd_varpro_flag,
    )

def Optimized_DMD_Mode_Pair(
    dmd,
    atol=1e-8,
    rtol=1e-6,
    include_unpaired=False,
):
    """
    Pair conjugate modes from Optimized DMD / BOPDMD.

    IMPORTANT
    ---------
    Unlike standard DMD, BOPDMD eigenvalues are already CONTINUOUS-TIME:

        lambda = sigma + i*omega

    Therefore NO discrete-to-continuous logarithm is applied here.

    Returns the same conceptual representation as DMD_Mode_Pair:
        - amplitude-scaled complex phasors
        - continuous-time eigenvalues
        - mode pairing metadata
    """

    eigenvalues = np.asarray(
        dmd.eigs,
        dtype=complex,
    )

    modes = np.asarray(
        dmd.modes,
        dtype=complex,
    )

    amplitudes = np.asarray(
        dmd.amplitudes,
        dtype=complex,
    )

    # ---------------------------------------------------------
    # VALIDATE
    # ---------------------------------------------------------

    if modes.ndim != 2:
        raise ValueError(
            "dmd.modes must be a 2-D array."
        )

    if (
        modes.shape[1]
        != eigenvalues.size
    ):
        raise ValueError(
            "The number of columns in dmd.modes must equal "
            "the number of eigenvalues."
        )

    if (
        amplitudes.size
        != eigenvalues.size
    ):
        raise ValueError(
            "dmd.amplitudes must contain one amplitude "
            "per DMD mode."
        )

    # ---------------------------------------------------------
    # COMPLETE DYNAMICALLY-SCALED SPATIAL COEFFICIENT
    #
    # phi_j * b_j
    # ---------------------------------------------------------

    scaled_modes = (
        modes
        * amplitudes[
            np.newaxis,
            :
        ]
    )

    used = set()

    output_phasors = []
    output_eigenvalues = []
    mode_information = []

    # ---------------------------------------------------------
    # LOOP OVER CONTINUOUS-TIME EIGENVALUES
    # ---------------------------------------------------------

    for i, eig_i in enumerate(
        eigenvalues
    ):

        if i in used:
            continue

        # -----------------------------------------------------
        # REAL / NON-OSCILLATORY EIGENVALUE
        # -----------------------------------------------------

        if np.isclose(
            eig_i.imag,
            0.0,
            atol=atol,
            rtol=rtol,
        ):

            output_phasors.append(
                scaled_modes[:, i]
            )

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

        # -----------------------------------------------------
        # FIND CLOSEST UNUSED CONJUGATE
        # -----------------------------------------------------

        candidate_indices = [
            j
            for j in range(
                eigenvalues.size
            )
            if (
                j != i
                and j not in used
            )
        ]

        if candidate_indices:

            conjugate_target = np.conj(
                eig_i
            )

            errors = np.asarray([
                np.abs(
                    eigenvalues[j]
                    - conjugate_target
                )
                for j
                in candidate_indices
            ])

            best_position = np.argmin(
                errors
            )

            j = candidate_indices[
                best_position
            ]

            pair_error = errors[
                best_position
            ]

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

        # -----------------------------------------------------
        # COMBINE CONJUGATE PAIR
        # -----------------------------------------------------

        if is_pair:

            # For CONTINUOUS-TIME eigenvalues:
            #
            # lambda = sigma + i*omega
            #
            # positive frequency is simply omega > 0.

            if eig_i.imag > 0:

                positive_idx = i
                negative_idx = j

            else:

                positive_idx = j
                negative_idx = i

            eig_positive = (
                eigenvalues[
                    positive_idx
                ]
            )

            # Same phasor convention used in DMD_Mode_Pair.
            #
            # For an exact conjugate pair:
            #
            # scaled_negative
            #     = conj(scaled_positive)
            #
            # giving 2 * scaled_positive.

            phasor = (
                scaled_modes[
                    :,
                    positive_idx
                ]
                + np.conj(
                    scaled_modes[
                        :,
                        negative_idx
                    ]
                )
            )

            output_phasors.append(
                phasor
            )

            output_eigenvalues.append(
                eig_positive
            )

            mode_information.append({
                "type":
                    "conjugate_pair",

                "indices":
                    (
                        positive_idx,
                        negative_idx,
                    ),

                "pair_error":
                    pair_error,
            })

            used.add(i)
            used.add(j)

        # -----------------------------------------------------
        # COMPLEX MODE WITHOUT CONJUGATE PARTNER
        # -----------------------------------------------------

        else:

            used.add(i)

            if include_unpaired:

                # Always orient an unpaired mode to positive
                # frequency for a consistent convention.
                if eig_i.imag > 0:

                    eig_output = eig_i
                    phasor_output = (
                        scaled_modes[:, i]
                    )

                else:

                    eig_output = np.conj(
                        eig_i
                    )

                    phasor_output = np.conj(
                        scaled_modes[:, i]
                    )

                output_phasors.append(
                    phasor_output
                )

                output_eigenvalues.append(
                    eig_output
                )

                mode_information.append({
                    "type":
                        "unpaired_complex",

                    "indices":
                        (i,),

                    "pair_error":
                        pair_error,
                })

                warnings.warn(
                    f"Optimized DMD mode {i} with "
                    f"eigenvalue {eig_i} has no "
                    "conjugate partner.",
                    RuntimeWarning,
                )

    # ---------------------------------------------------------
    # HANDLE NO RETURNED MODES
    # ---------------------------------------------------------

    if len(
        output_phasors
    ) == 0:

        output_phasors = np.empty(
            (
                modes.shape[0],
                0,
            ),
            dtype=complex,
        )

        output_eigenvalues = np.asarray(
            [],
            dtype=complex,
        )

    else:

        output_phasors = np.column_stack(
            output_phasors
        )

        output_eigenvalues = np.asarray(
            output_eigenvalues,
            dtype=complex,
        )

    # ---------------------------------------------------------
    # CRITICAL DIFFERENCE FROM DMD_Mode_Pair:
    #
    # These are ALREADY continuous-time eigenvalues.
    # DO NOT call D_To_C_Eigenvalue_Converter.
    # ---------------------------------------------------------

    continuous_eigenvalues = (
        output_eigenvalues
    )

    return {
        "phasors":
            output_phasors,

        "continuous_eigenvalues":
            continuous_eigenvalues,

        "mode_information":
            mode_information,
    }

def extract_optimized_dmd_candidates(
    dmd,
    n_physical=None,
    embedding_d=None,
    candidate_period_min=None,
    candidate_period_max=None,
    atol=1e-8,
    rtol=1e-6,
    include_unpaired=False,
):
    """
    Extract candidates from variable-projection Optimized DMD.

    Intended for PyDMD BOPDMD used as Optimized DMD, normally with:

        num_trials = 0

    BOPDMD eigenvalues are already continuous-time, so this uses
    Optimized_DMD_Mode_Pair rather than DMD_Mode_Pair.

    If ``embedding_d`` is supplied, the paired augmented phasors are mapped
    to their zero-delay physical block before common candidate finalisation.
    """

    recovered_dict = (
        Optimized_DMD_Mode_Pair(
            dmd,
            atol=atol,
            rtol=rtol,
            include_unpaired=
                include_unpaired,
        )
    )

    continuous_eigenvalues = np.asarray(
        recovered_dict[
            "continuous_eigenvalues"
        ],
        dtype=complex,
    )

    physical_phasors = np.asarray(
        recovered_dict[
            "phasors"
        ],
        dtype=complex,
    )

    if embedding_d is not None:
        physical_phasors = physicalise_embedded_phasors(
            augmented_phasors=physical_phasors,
            n_physical=n_physical,
            expected_d=embedding_d,
        )

    return finalise_candidate_suite(
        continuous_eigenvalues=
            continuous_eigenvalues,

        phasors=
            physical_phasors,

        n_physical=
            n_physical,

        candidate_period_min=
            candidate_period_min,

        candidate_period_max=
            candidate_period_max,
    )

# ---------------------------------------------------------------
# OPTIONAL HANKEL EMBEDDING + LEGACY HANKELDMD COMPATIBILITY
# ---------------------------------------------------------------

# default parameters used - JUST FOR REFERENCE REPLACE THIS LATER
# hankel_d = 5 # DEFINE AND PASS THIS IN THE RUNTIME SCRIPT
hankel_svd_rank = -1
hankel_tlsq_rank = 0
hankel_exact = True
hankel_opt = False
hankel_rescale_mode = None
hankel_forward_backward = False
hankel_sorted_eigs = False
hankel_reconstruction_method = "first"
hankel_tikhonov_regularization = None

# CHECK WHY THIS WORKS
hankel_physical_mode_method = "first"

def apply_hankel_embedding(
    dmd,
    enabled=False,
    d=1,
    reconstruction_method="first",
):
    """
    Optionally apply PyDMD time-delay/Hankel preprocessing to any DMD model.

    The returned wrapper delegates eigenvalues, modes, and amplitudes to the
    supplied base estimator. With ``enabled=False`` the original object is
    returned unchanged.
    """

    if not enabled:
        return dmd

    if not isinstance(d, (int, np.integer)) or d < 1:
        raise ValueError(
            "Hankel embedding depth d must be a positive integer."
        )

    return hankel_preprocessing(
        dmd,
        d=int(d),
        reconstruction_method=reconstruction_method,
    )

# build hankel object
def build_hankel_dmd(hankel_d, svd_rank=-1):
    """Compatibility constructor retained for older experiment scripts."""

    return HankelDMD(
        svd_rank=svd_rank,
        tlsq_rank=hankel_tlsq_rank,
        exact=hankel_exact,
        opt=hankel_opt,
        rescale_mode=hankel_rescale_mode,
        forward_backward=hankel_forward_backward,
        d=hankel_d,
        sorted_eigs=hankel_sorted_eigs,
        reconstruction_method=hankel_reconstruction_method,
        tikhonov_regularization=hankel_tikhonov_regularization,
    )

def physicalise_embedded_phasors(
    augmented_phasors,
    n_physical,
    expected_d,
):
    """
    Map delay-embedded DMD phasors back to the original physical state.

    Parameters
    ----------
    augmented_phasors : ndarray, shape (n_augmented, n_modes)
        Amplitude-scaled / paired phasors returned by DMD_Mode_Pair
        or Optimized_DMD_Mode_Pair after fitting an embedded DMD model.

    n_physical : int
        Number of spatial degrees of freedom in the original
        snapshot matrix.

    expected_d : int
        Delay-embedding depth used during preprocessing.

    Returns
    -------
    physical_phasors : ndarray, shape (n_physical, n_modes)
        Complex phasors in the same physical state space as the
        original snapshots / theoretical target phasors.
    """

    augmented_phasors = np.asarray(
        augmented_phasors,
        dtype=complex,
    )

    # ---------------------------------------------------------
    # VALIDATE INPUT SHAPE
    # ---------------------------------------------------------

    if augmented_phasors.ndim != 2:
        raise ValueError(
            "Expected augmented_phasors to be a 2-D array with shape "
            "(n_state, n_modes). "
            f"Received shape {augmented_phasors.shape}."
        )

    if (
        not isinstance(
            n_physical,
            (int, np.integer),
        )
        or n_physical <= 0
    ):
        raise ValueError(
            "n_physical must be a positive integer."
        )

    if (
        not isinstance(
            expected_d,
            (int, np.integer),
        )
        or expected_d <= 0
    ):
        raise ValueError(
            "expected_d must be a positive integer."
        )

    n_augmented = (
        augmented_phasors.shape[0]
    )

    # ---------------------------------------------------------
    # ALREADY IN PHYSICAL SPACE
    #
    # This covers:
    #   - d = 1
    #   - any PyDMD behaviour that already returns physical-space
    #     modes rather than explicitly stacked modes
    # ---------------------------------------------------------

    if n_augmented == n_physical:

        return augmented_phasors

    # ---------------------------------------------------------
    # CHECK AUGMENTED DIMENSION
    # ---------------------------------------------------------

    if (
        n_augmented
        % n_physical
        != 0
    ):

        raise ValueError(
            "Recovered Hankel mode dimension is not an integer "
            "multiple of the original physical state dimension. "
            f"Recovered rows={n_augmented}, "
            f"physical rows={n_physical}."
        )

    inferred_d = (
        n_augmented
        // n_physical
    )

    if inferred_d != expected_d:

        raise ValueError(
            "Recovered Hankel mode dimension implies a delay depth "
            "different from the configured value. "
            f"inferred_d={inferred_d}, "
            f"expected_d={expected_d}, "
            f"mode_shape={augmented_phasors.shape}, "
            f"n_physical={n_physical}."
        )

    # ---------------------------------------------------------
    # MAP BACK TO ORIGINAL PHYSICAL STATE
    #
    # The augmented Hankel state has the form conceptually:
    #
    #     [x(t)]
    #     [x(t+dt)]
    #     [x(t+2dt)]
    #       ...
    #
    # For direct comparison with a theoretical phasor defined in
    # the original state space, retain the first / zero-delay block.
    # ---------------------------------------------------------

    physical_phasors = (
        augmented_phasors[
            :n_physical,
            :
        ]
    )

    return physical_phasors

def physicalise_hankel_phasors(
    augmented_phasors,
    n_physical,
    expected_d,
):
    """Compatibility wrapper for older experiment scripts."""

    return physicalise_embedded_phasors(
        augmented_phasors=augmented_phasors,
        n_physical=n_physical,
        expected_d=expected_d,
    )

def extract_hankel_candidates(
    dmd,
    dt_snapshot,
    n_physical,
    hankel_d,
    candidate_period_min=None,
    candidate_period_max=None,
):
    """
    Extract a standardised candidate suite from a fitted HankelDMD model.

    Output is identical in structure to the Exact/FbDMD/Optimized-DMD
    candidate extractors:

        candidate_eigs
            Continuous-time eigenvalues.

        candidate_modes
            Amplitude-scaled complex phasors in ORIGINAL PHYSICAL
            state space.

        candidate_periods
            Positive oscillation periods.

    Processing flow
    ---------------
    HankelDMD
        -> DMD_Mode_Pair
        -> continuous-time eigenvalues
        -> delay-augmented paired phasors
        -> physicalise_hankel_phasors
        -> finalise_candidate_suite
    """

    # ---------------------------------------------------------
    # STANDARD DISCRETE-DMD MODE PAIRING
    #
    # HankelDMD still uses the standard discrete-time DMD
    # eigenvalue convention, so DMD_Mode_Pair is appropriate here.
    #
    # This:
    #   1. applies modal amplitudes,
    #   2. combines conjugate pairs,
    #   3. converts discrete eigenvalues to continuous time.
    # ---------------------------------------------------------

    recovered_dict = DMD_Mode_Pair(
        dmd,
        dt=dt_snapshot,
    )

    continuous_eigenvalues = np.asarray(
        recovered_dict[
            "continuous_eigenvalues"
        ],
        dtype=complex,
    )

    augmented_phasors = np.asarray(
        recovered_dict[
            "phasors"
        ],
        dtype=complex,
    )

    # ---------------------------------------------------------
    # HANDLE EMPTY RECOVERY
    # ---------------------------------------------------------

    if continuous_eigenvalues.size == 0:

        return finalise_candidate_suite(
            continuous_eigenvalues=
                continuous_eigenvalues,

            phasors=np.empty(
                (
                    n_physical,
                    0,
                ),
                dtype=complex,
            ),

            n_physical=
                n_physical,

            candidate_period_min=
                candidate_period_min,

            candidate_period_max=
                candidate_period_max,
        )

    # ---------------------------------------------------------
    # HANKEL-SPECIFIC STEP:
    # DELAY-AUGMENTED -> PHYSICAL SPACE
    # ---------------------------------------------------------

    physical_phasors = (
        physicalise_hankel_phasors(
            augmented_phasors=
                augmented_phasors,

            n_physical=
                n_physical,

            expected_d=
                hankel_d,
        )
    )

    # ---------------------------------------------------------
    # COMMON FINALISATION
    #
    # From this point onward Hankel is treated identically to every
    # other DMD variant.
    # ---------------------------------------------------------

    return finalise_candidate_suite(
        continuous_eigenvalues=
            continuous_eigenvalues,

        phasors=
            physical_phasors,

        n_physical=
            n_physical,

        candidate_period_min=
            candidate_period_min,

        candidate_period_max=
            candidate_period_max,
    )

# ---------------------------------------------------------------
# success metric related
# ---------------------------------------------------------------


def _finite_complex_array(values):
    """
    Convert an iterable of complex values to a 1-D complex ndarray,
    retaining only entries with finite real and imaginary parts.
    """

    values = np.asarray(
        values,
        dtype=complex,
    ).ravel()

    valid = (
        np.isfinite(values.real)
        & np.isfinite(values.imag)
    )

    return values[valid]


def _get_mode_colours(
    DMD_recovery,
    cmap_name="viridis",
):
    """
    Assign one discrete colour to each theoretical input mode.

    Modes are ordered by true period so colours change systematically
    from shorter to longer-period modes.
    """

    mode_numbers = list(
        DMD_recovery.keys()
    )

    mode_numbers = sorted(
        mode_numbers,
        key=lambda mode: (
            DMD_recovery[
                mode
            ]["true_period"]
        ),
    )

    cmap = plt.colormaps[
        cmap_name
    ].resampled(
        max(
            len(mode_numbers),
            1,
        )
    )

    colours = {
        mode_number: cmap(i)
        for i, mode_number
        in enumerate(mode_numbers)
    }

    return (
        mode_numbers,
        colours,
    )
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

    return errors
def plot_quality_factor_recovery(
    DMD_recovery,
    cmap_name="viridis",
    show_noise_members=True,
    show_noise_median=True,
    figsize=(8, 6),
    annotate_modes=True,
    log_q=False,
    upper_lim_yr=10,
    lower_lim_yr=1.0,
):
    """
    Plot true and recovered DMD behaviour as:

        x-axis = oscillation period (years)
        y-axis = quality factor Q

    For a continuous-time eigenvalue

        lambda = sigma + i*omega

    quality factor is defined as

        Q = |omega| / (2 * |sigma|)

    where:
        sigma = growth / decay rate
        omega = angular frequency

    Large Q:
        weak growth/decay relative to the oscillation frequency.

    Small Q:
        strong growth/decay relative to the oscillation frequency.

    Parameters
    ----------
    DMD_recovery : dict

        Expected structure:

        DMD_recovery[mode] = {
            "true_period": ...,
            "true_eigenvalue": complex,

            "DMD": {
                "similarity": ...,
                "eigenvalue": complex,
                "recovered_period": ...,
            },

            "DMD_noised": {
                "similarity": [...],
                "eigenvalue": [...],
                "recovered_period": [...],
            },
        }

        All eigenvalues must already be in continuous-time form.

    log_q : bool
        If True, use a logarithmic y-axis.

    Returns
    -------
    fig, ax
    """

    # ---------------------------------------------------------
    # QUALITY FACTOR HELPER
    # ---------------------------------------------------------

    def quality_factor(eigenvalues):

        eigenvalues = np.asarray(
            eigenvalues,
            dtype=complex,
        )

        omega = np.abs(
            eigenvalues.imag
        )

        sigma = np.abs(
            eigenvalues.real
        )

        Q = np.full(
            eigenvalues.shape,
            np.inf,
            dtype=float,
        )

        nonzero_decay = sigma > 0

        Q[nonzero_decay] = (
            omega[nonzero_decay]
            / (
                2.0
                * sigma[nonzero_decay]
            )
        )

        return Q


    mode_numbers, colours = (
        _get_mode_colours(
            DMD_recovery,
            cmap_name=cmap_name,
        )
    )

    fig, ax = plt.subplots(
        figsize=figsize
    )

    for mode_number in mode_numbers:

        results = DMD_recovery[
            mode_number
        ]

        colour = colours[
            mode_number
        ]

        # =====================================================
        # TRUE THEORETICAL MODE
        # =====================================================

        if "true_eigenvalue" not in results:

            raise KeyError(
                f"Mode {mode_number} has no "
                "'true_eigenvalue'."
            )

        true_eig = complex(
            results[
                "true_eigenvalue"
            ]
        )

        true_period = float(
            results[
                "true_period"
            ]
        )

        true_Q = quality_factor(
            [true_eig]
        )[0]

        if (
            np.isfinite(true_period)
            and np.isfinite(true_Q)
        ):

            ax.scatter(
                true_period,
                true_Q,
                marker="o",
                s=100,
                facecolors="none",
                edgecolors=colour,
                linewidths=2.0,
                zorder=5,
            )

            if annotate_modes:

                ax.annotate(
                    str(mode_number),
                    (
                        true_period,
                        true_Q,
                    ),
                    xytext=(5, 5),
                    textcoords="offset points",
                    fontsize=8,
                    color=colour,
                )

        # =====================================================
        # IDEAL DMD RECOVERY (NO P, R, H_sv, OR PERTURBATION)
        # =====================================================

        ideal_result = results.get(
            "DMD_ideal",
            {},
        )

        if (
            ideal_result
            and "eigenvalue" in ideal_result
        ):

            ideal_eig = complex(
                ideal_result[
                    "eigenvalue"
                ]
            )

            ideal_Q = quality_factor(
                [ideal_eig]
            )[0]

            ideal_period = float(
                ideal_result.get(
                    "recovered_period",
                    np.nan,
                )
            )

            if (
                np.isfinite(ideal_period)
                and np.isfinite(ideal_Q)
            ):

                ax.scatter(
                    ideal_period,
                    ideal_Q,
                    marker="+",
                    s=120,
                    linewidths=2.0,
                    color=colour,
                    zorder=9,
                )

        # =====================================================
        # CLEAN RESOLVED DMD RECOVERY
        # =====================================================

        clean_result = results.get(
            "DMD",
            {},
        )

        if (
            clean_result
            and "eigenvalue"
            in clean_result
        ):

            clean_eig = complex(
                clean_result[
                    "eigenvalue"
                ]
            )

            clean_Q = quality_factor(
                [clean_eig]
            )[0]

            # Prefer the period already calculated during
            # candidate extraction.
            if (
                "recovered_period"
                in clean_result
            ):

                clean_period = float(
                    clean_result[
                        "recovered_period"
                    ]
                )

            else:

                omega = abs(
                    clean_eig.imag
                )

                clean_period = (
                    2.0 * np.pi / omega
                    if omega > 0
                    else np.nan
                )

            if (
                np.isfinite(clean_period)
                and np.isfinite(clean_Q)
            ):

                # White under-stroke.
                ax.scatter(
                    clean_period,
                    clean_Q,
                    marker="x",
                    s=100,
                    linewidths=4.0,
                    color="white",
                    zorder=7,
                )

                ax.scatter(
                    clean_period,
                    clean_Q,
                    marker="x",
                    s=100,
                    linewidths=2.0,
                    color=colour,
                    zorder=8,
                )

        # =====================================================
        # PERTURBED ENSEMBLE
        # =====================================================

        noised_result = results.get(
            "DMD_noised",
            {},
        )

        noised_eigs = np.asarray(
            noised_result.get(
                "eigenvalue",
                [],
            ),
            dtype=complex,
        ).ravel()

        noised_periods = np.asarray(
            noised_result.get(
                "recovered_period",
                [],
            ),
            dtype=float,
        ).ravel()

        # Eigenvalue and period must correspond
        # member-by-member.
        if (
            noised_eigs.size
            != noised_periods.size
        ):

            raise ValueError(
                f"Mode {mode_number}: number of "
                "perturbed eigenvalues does not match "
                "number of recovered periods: "
                f"{noised_eigs.size} vs "
                f"{noised_periods.size}."
            )

        if noised_eigs.size == 0:
            continue

        noised_Q = quality_factor(
            noised_eigs
        )

        # Keep all three arrays aligned.
        valid = (
            np.isfinite(
                noised_eigs.real
            )
            & np.isfinite(
                noised_eigs.imag
            )
            & np.isfinite(
                noised_periods
            )
            & np.isfinite(
                noised_Q
            )
        )

        noised_periods = (
            noised_periods[
                valid
            ]
        )

        noised_Q = (
            noised_Q[
                valid
            ]
        )

        if noised_periods.size == 0:
            continue

        # -----------------------------------------------------
        # INDIVIDUAL PERTURBED RECOVERIES
        # -----------------------------------------------------

        if show_noise_members:

            ax.scatter(
                noised_periods,
                noised_Q,
                marker=".",
                s=25,
                color=colour,
                alpha=0.25,
                zorder=1,
            )

        # -----------------------------------------------------
        # MEDIAN PERTURBED RECOVERY
        #
        # Important:
        # Median Q is calculated from individual Q values,
        # rather than calculating Q from median sigma/omega.
        # -----------------------------------------------------

        if show_noise_median:

            median_period = np.median(
                noised_periods
            )

            median_Q = np.median(
                noised_Q
            )

            ax.scatter(
                median_period,
                median_Q,
                marker="s",
                s=60,
                color=colour,
                edgecolors="black",
                linewidths=0.5,
                zorder=6,
            )

    # =========================================================
    # LABELS
    # =========================================================

    ax.set_xlabel(
        "Period (years)"
    )
    ax.set_xlim(
        lower_lim_yr,
        upper_lim_yr,
    )
    ax.set_ylabel(
        r"Quality factor $Q = |\omega|/(2|\sigma|)$"
    )

    ax.set_title(
        "Period and quality-factor recovery"
    )

    if log_q:

        ax.set_yscale(
            "log"
        )

    ax.grid(
        alpha=0.25
    )

    # =========================================================
    # LEGEND
    # =========================================================

    handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="None",
            markerfacecolor="none",
            markeredgecolor="black",
            markeredgewidth=1.5,
            markersize=8,
            label="True input",
        ),

        Line2D(
            [0],
            [0],
            marker="+",
            linestyle="None",
            color="black",
            markersize=9,
            markeredgewidth=2.0,
            label="Ideal DMD",
        ),

        Line2D(
            [0],
            [0],
            marker="x",
            linestyle="None",
            color="black",
            markersize=8,
            markeredgewidth=2.0,
            label="Resolved DMD",
        ),
    ]

    if not any(
        bool(results.get("DMD_ideal", {}))
        for results in DMD_recovery.values()
    ):
        handles.pop(1)

    if show_noise_members:

        handles.append(
            Line2D(
                [0],
                [0],
                marker=".",
                linestyle="None",
                color="black",
                markersize=7,
                alpha=0.4,
                label="Perturbed recovery",
            )
        )

    if show_noise_median:

        handles.append(
            Line2D(
                [0],
                [0],
                marker="s",
                linestyle="None",
                color="black",
                markersize=6,
                label="Perturbed median",
            )
        )

    ax.legend(
        handles=handles,
        loc="best",
    )

    fig.tight_layout()

    return fig, ax

def plot_continuous_eigenvalue_recovery(
    DMD_recovery,
    cmap_name="viridis",
    show_noise_members=True,
    show_noise_median=True,
    figsize=(8, 6),
    annotate_modes=True,
    upper_lim_yr=10,
    lower_lim_yr=1.0,
):
    """
    Plot true and recovered DMD behaviour as:

        x-axis = oscillation period (years)
        y-axis = continuous growth / decay rate sigma (yr^-1)

    Parameters
    ----------
    DMD_recovery : dict
        Expected structure:

        DMD_recovery[mode] = {
            "true_period": ...,
            "true_eigenvalue": complex,

            "DMD": {
                "similarity": ...,
                "eigenvalue": complex,
                "recovered_period": ...,
            },

            "DMD_noised": {
                "similarity": [...],
                "eigenvalue": [...],
                "recovered_period": [...],
            },
        }

        All stored eigenvalues are assumed to already be in
        CONTINUOUS-TIME form:

            lambda = sigma + i*omega

        with:

            period = 2*pi / |omega|

    Returns
    -------
    fig, ax
    """

    mode_numbers, colours = (
        _get_mode_colours(
            DMD_recovery,
            cmap_name=cmap_name,
        )
    )

    fig, ax = plt.subplots(
        figsize=figsize
    )

    for mode_number in mode_numbers:

        results = DMD_recovery[
            mode_number
        ]

        colour = colours[
            mode_number
        ]

        # =====================================================
        # TRUE THEORETICAL EIGENVALUE
        # =====================================================

        if "true_eigenvalue" not in results:

            raise KeyError(
                f"Mode {mode_number} has no 'true_eigenvalue'. "
                "Store the original theoretical continuous eigenvalue "
                "in DMD_recovery before plotting."
            )

        true_eig = complex(
            results[
                "true_eigenvalue"
            ]
        )

        true_period = float(
            results[
                "true_period"
            ]
        )

        true_growth = (
            true_eig.real
        )

        if (
            np.isfinite(true_period)
            and np.isfinite(true_growth)
        ):

            ax.scatter(
                true_period,
                true_growth,
                marker="o",
                s=100,
                facecolors="none",
                edgecolors=colour,
                linewidths=2.0,
                zorder=5,
            )

            if annotate_modes:

                ax.annotate(
                    str(mode_number),
                    (
                        true_period,
                        true_growth,
                    ),
                    xytext=(5, 5),
                    textcoords="offset points",
                    fontsize=8,
                    color=colour,
                )

        # =====================================================
        # IDEAL DMD RECOVERY (NO P, R, H_sv, OR PERTURBATION)
        # =====================================================

        ideal_result = results.get(
            "DMD_ideal",
            {},
        )

        if (
            ideal_result
            and "eigenvalue" in ideal_result
        ):

            ideal_eig = complex(
                ideal_result[
                    "eigenvalue"
                ]
            )

            ideal_growth = ideal_eig.real
            ideal_period = float(
                ideal_result.get(
                    "recovered_period",
                    np.nan,
                )
            )

            if (
                np.isfinite(ideal_period)
                and np.isfinite(ideal_growth)
            ):

                ax.scatter(
                    ideal_period,
                    ideal_growth,
                    marker="+",
                    s=120,
                    linewidths=2.0,
                    color=colour,
                    zorder=9,
                )

        # =====================================================
        # CLEAN RESOLVED DMD RECOVERY
        # =====================================================

        clean_result = results.get(
            "DMD",
            {},
        )

        if (
            clean_result
            and "eigenvalue" in clean_result
        ):

            clean_eig = complex(
                clean_result[
                    "eigenvalue"
                ]
            )

            clean_growth = (
                clean_eig.real
            )

            # Prefer the period already calculated during
            # candidate extraction / matching.
            if "recovered_period" in clean_result:

                clean_period = float(
                    clean_result[
                        "recovered_period"
                    ]
                )

            else:

                omega = abs(
                    clean_eig.imag
                )

                if omega > 0:

                    clean_period = (
                        2.0
                        * np.pi
                        / omega
                    )

                else:

                    clean_period = np.nan

            if (
                np.isfinite(clean_period)
                and np.isfinite(clean_growth)
            ):

                # White under-stroke so overlapping crosses remain visible.
                ax.scatter(
                    clean_period,
                    clean_growth,
                    marker="x",
                    s=100,
                    linewidths=4.0,
                    color="white",
                    zorder=7,
                )

                ax.scatter(
                    clean_period,
                    clean_growth,
                    marker="x",
                    s=100,
                    linewidths=2.0,
                    color=colour,
                    zorder=8,
                )

        # =====================================================
        # PERTURBED ENSEMBLE
        # =====================================================

        noised_result = results.get(
            "DMD_noised",
            {},
        )

        noised_eigs = np.asarray(
            noised_result.get(
                "eigenvalue",
                [],
            ),
            dtype=complex,
        ).ravel()

        noised_periods = np.asarray(
            noised_result.get(
                "recovered_period",
                [],
            ),
            dtype=float,
        ).ravel()

        # -----------------------------------------------------
        # Check that eigenvalues and periods correspond
        # member-by-member.
        # -----------------------------------------------------

        if (
            noised_eigs.size
            != noised_periods.size
        ):

            raise ValueError(
                f"Mode {mode_number}: number of perturbed "
                "eigenvalues does not match number of recovered "
                "periods: "
                f"{noised_eigs.size} vs "
                f"{noised_periods.size}."
            )

        if noised_eigs.size == 0:
            continue

        # Keep paired period/eigenvalue entries aligned.
        valid = (
            np.isfinite(
                noised_eigs.real
            )
            & np.isfinite(
                noised_eigs.imag
            )
            & np.isfinite(
                noised_periods
            )
        )

        noised_eigs = (
            noised_eigs[
                valid
            ]
        )

        noised_periods = (
            noised_periods[
                valid
            ]
        )

        if noised_eigs.size == 0:
            continue

        noise_growth = (
            noised_eigs.real
        )

        # -----------------------------------------------------
        # INDIVIDUAL PERTURBED RECOVERIES
        # -----------------------------------------------------

        if show_noise_members:

            ax.scatter(
                noised_periods,
                noise_growth,
                marker=".",
                s=25,
                color=colour,
                alpha=0.25,
                zorder=1,
            )

        # -----------------------------------------------------
        # MEDIAN PERTURBED RECOVERY
        # -----------------------------------------------------

        if show_noise_median:

            median_period = np.median(
                noised_periods
            )

            median_growth = np.median(
                noise_growth
            )

            ax.scatter(
                median_period,
                median_growth,
                marker="s",
                s=60,
                color=colour,
                edgecolors="black",
                linewidths=0.5,
                zorder=6,
            )

    # =========================================================
    # REFERENCE LINE
    # =========================================================

    ax.axhline(
        0.0,
        color="black",
        linestyle="--",
        linewidth=1.0,
        alpha=0.7,
    )

    # =========================================================
    # LABELS
    # =========================================================

    ax.set_xlabel(
        "Period (years)"
    )
    ax.set_xlim(
        lower_lim_yr,
        upper_lim_yr,
    )
    ax.set_ylabel(
        r"Growth / decay rate "
        r"$\sigma$ (yr$^{-1}$)"
    )

    ax.set_title(
        "Continuous eigenvalue recovery"
    )

    ax.grid(
        alpha=0.25
    )

    # =========================================================
    # MARKER LEGEND
    # =========================================================

    handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="None",
            markerfacecolor="none",
            markeredgecolor="black",
            markeredgewidth=1.5,
            markersize=8,
            label="True input",
        ),

        Line2D(
            [0],
            [0],
            marker="+",
            linestyle="None",
            color="black",
            markersize=9,
            markeredgewidth=2.0,
            label="Ideal DMD",
        ),

        Line2D(
            [0],
            [0],
            marker="x",
            linestyle="None",
            color="black",
            markersize=8,
            markeredgewidth=2.0,
            label="Resolved DMD",
        ),
    ]

    if not any(
        bool(results.get("DMD_ideal", {}))
        for results in DMD_recovery.values()
    ):
        handles.pop(1)

    if show_noise_members:

        handles.append(
            Line2D(
                [0],
                [0],
                marker=".",
                linestyle="None",
                color="black",
                markersize=7,
                alpha=0.4,
                label="Perturbed recovery",
            )
        )

    if show_noise_median:

        handles.append(
            Line2D(
                [0],
                [0],
                marker="s",
                linestyle="None",
                color="black",
                markersize=6,
                label="Perturbed median",
            )
        )

    ax.legend(
        handles=handles,
        loc="best",
    )

    fig.tight_layout()

    return fig, ax

def plot_similarity_vs_recovered_period(
    DMD_recovery,
    cmap_name="viridis",
    show_noise_members=True,
    show_noise_median=True,
    connect_clean_to_truth=True,
    figsize=(8, 6),
    annotate_modes=True,
    upper_lim_yr=10,
    lower_lim_yr=1.0,
):
    """
    Plot spatial similarity against recovered period for each input mode.

    Theoretical input:
        x = true period
        y = 1

    Unperturbed DMD:
        x = recovered period
        y = spatial similarity

    Perturbed ensemble:
        x = each recovered period
        y = corresponding spatial similarity

    This shows both temporal and spatial recovery in a single diagnostic.

    Returns
    -------
    fig, ax
    """

    mode_numbers, colours = (
        _get_mode_colours(
            DMD_recovery,
            cmap_name=cmap_name,
        )
    )

    fig, ax = plt.subplots(
        figsize=figsize
    )

    for mode_number in mode_numbers:

        results = DMD_recovery[
            mode_number
        ]

        colour = colours[
            mode_number
        ]

        true_period = float(
            results[
                "true_period"
            ]
        )

        # =====================================================
        # TRUE INPUT
        # =====================================================

        ax.scatter(
            true_period,
            1.0,
            marker="o",
            s=100,
            facecolors="none",
            edgecolors=colour,
            linewidths=2.0,
            zorder=7,
        )

        if annotate_modes:

            ax.annotate(
                str(mode_number),
                (
                    true_period,
                    1.0,
                ),
                xytext=(5, -12),
                textcoords="offset points",
                fontsize=8,
                color=colour,
            )

        # =====================================================
        # IDEAL DMD RECOVERY (NO P, R, H_sv, OR PERTURBATION)
        # =====================================================

        ideal_result = results.get(
            "DMD_ideal",
            {},
        )

        if (
            ideal_result
            and "recovered_period" in ideal_result
            and "similarity" in ideal_result
        ):

            ideal_period = float(
                ideal_result[
                    "recovered_period"
                ]
            )

            ideal_similarity = float(
                ideal_result[
                    "similarity"
                ]
            )

            if (
                np.isfinite(ideal_period)
                and np.isfinite(ideal_similarity)
            ):

                if connect_clean_to_truth:

                    ax.plot(
                        [
                            true_period,
                            ideal_period,
                        ],
                        [
                            1.0,
                            ideal_similarity,
                        ],
                        color=colour,
                        linewidth=1.0,
                        linestyle=":",
                        alpha=0.45,
                        zorder=1,
                    )

                ax.scatter(
                    ideal_period,
                    ideal_similarity,
                    marker="+",
                    s=120,
                    linewidths=2.0,
                    color=colour,
                    zorder=10,
                )

        # =====================================================
        # CLEAN RESOLVED RECOVERY
        # =====================================================

        clean_result = results.get(
            "DMD",
            {},
        )

        if (
            clean_result
            and "recovered_period"
            in clean_result
            and "similarity"
            in clean_result
        ):

            clean_period = float(
                clean_result[
                    "recovered_period"
                ]
            )

            clean_similarity = float(
                clean_result[
                    "similarity"
                ]
            )

            if (
                np.isfinite(clean_period)
                and np.isfinite(
                    clean_similarity
                )
            ):

                # Optional line linking truth to recovered result.
                if connect_clean_to_truth:

                    ax.plot(
                        [
                            true_period,
                            clean_period,
                        ],
                        [
                            1.0,
                            clean_similarity,
                        ],
                        color=colour,
                        linewidth=1.0,
                        alpha=0.45,
                        zorder=1,
                    )

                # White under-stroke.
                ax.scatter(
                    clean_period,
                    clean_similarity,
                    marker="x",
                    s=100,
                    linewidths=4.0,
                    color="white",
                    zorder=8,
                )

                ax.scatter(
                    clean_period,
                    clean_similarity,
                    marker="x",
                    s=100,
                    linewidths=2.0,
                    color=colour,
                    zorder=9,
                )

        # =====================================================
        # PERTURBED ENSEMBLE
        # =====================================================

        noised_result = results.get(
            "DMD_noised",
            {},
        )

        recovered_periods = np.asarray(
            noised_result.get(
                "recovered_period",
                [],
            ),
            dtype=float,
        )

        similarities = np.asarray(
            noised_result.get(
                "similarity",
                [],
            ),
            dtype=float,
        )

        # These should always have been appended together.
        if (
            recovered_periods.size
            != similarities.size
        ):

            raise ValueError(
                f"Mode {mode_number}: number of perturbed "
                "recovered periods does not equal number of "
                "similarity values: "
                f"{recovered_periods.size} vs "
                f"{similarities.size}."
            )

        valid = (
            np.isfinite(
                recovered_periods
            )
            & np.isfinite(
                similarities
            )
        )

        recovered_periods = (
            recovered_periods[
                valid
            ]
        )

        similarities = (
            similarities[
                valid
            ]
        )

        if recovered_periods.size == 0:
            continue

        # Individual covariance-perturbed recoveries.
        if show_noise_members:

            ax.scatter(
                recovered_periods,
                similarities,
                marker=".",
                s=30,
                color=colour,
                alpha=0.25,
                zorder=2,
            )

        # Ensemble median.
        if show_noise_median:

            median_period = np.median(
                recovered_periods
            )

            median_similarity = np.median(
                similarities
            )

            ax.scatter(
                median_period,
                median_similarity,
                marker="s",
                s=60,
                color=colour,
                edgecolors="black",
                linewidths=0.5,
                zorder=6,
            )

    ax.set_xlabel(
        "Period (years)"
    )

    ax.set_ylabel(
        "Spatial phasor similarity"
    )

    ax.set_ylim(
        0.0,
        1.05,
    )

    ax.set_xlim(
        lower_lim_yr,
        upper_lim_yr,
    )

    ax.set_title(
        "DMD temporal and spatial recovery"
    )

    ax.grid(
        alpha=0.25
    )

    handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="None",
            markerfacecolor="none",
            markeredgecolor="black",
            markeredgewidth=1.5,
            markersize=8,
            label="True input",
        ),
        Line2D(
            [0],
            [0],
            marker="+",
            linestyle="None",
            color="black",
            markeredgewidth=2.0,
            markersize=9,
            label="Ideal DMD",
        ),
        Line2D(
            [0],
            [0],
            marker="x",
            linestyle="None",
            color="black",
            markeredgewidth=2.0,
            markersize=8,
            label="Resolved DMD",
        ),
    ]

    if not any(
        bool(results.get("DMD_ideal", {}))
        for results in DMD_recovery.values()
    ):
        handles.pop(1)

    if show_noise_members:

        handles.append(
            Line2D(
                [0],
                [0],
                marker=".",
                linestyle="None",
                color="black",
                markersize=7,
                alpha=0.4,
                label="Perturbed recovery",
            )
        )

    if show_noise_median:

        handles.append(
            Line2D(
                [0],
                [0],
                marker="s",
                linestyle="None",
                color="black",
                markersize=6,
                label="Perturbed median",
            )
        )

    ax.legend(
        handles=handles,
        loc="best",
    )

    fig.tight_layout()

    return fig, ax

def plot_modal_period_power(
    modal_results,
    figsize=(9, 6),
    marker_size=85.0,
    show_ideal=False,
    period_xlim=None,
):
    """
    Plot oscillatory true and recovered modes in period-power space.

    By default only clean resolved DMD candidates are shown. Set
    ``show_ideal=True`` to add the analytical ideal candidates. Static modes
    remain stored in ``modal_results`` but are not plotted. ``period_xlim``
    optionally sets a positive two-value period range on the logarithmic
    x-axis.
    """

    true_results = modal_results.get(
        "true",
        {},
    )

    fig, ax = plt.subplots(
        figsize=figsize
    )

    true_coordinates = {}

    for mode_number, result in true_results.items():
        period = float(
            result["period"]
        )
        power = float(
            result["power"]
        )

        if not (
            np.isfinite(period)
            and period > 0.0
            and np.isfinite(power)
            and power > 0.0
        ):
            continue

        true_coordinates[
            str(mode_number)
        ] = (
            period,
            power,
        )

        ax.scatter(
            period,
            power,
            s=marker_size,
            marker="o",
            facecolors="none",
            edgecolors="black",
            linewidths=1.8,
            zorder=6,
        )

    category_markers = {
        "resolved": "x",
    }

    if show_ideal:
        category_markers = {
            "ideal": "+",
            **category_markers,
        }

    for category, marker in category_markers.items():
        results = modal_results.get(
            category,
            {},
        )

        periods = np.asarray(
            results.get(
                "period",
                [],
            ),
            dtype=float,
        ).ravel()
        powers = np.asarray(
            results.get(
                "power",
                [],
            ),
            dtype=float,
        ).ravel()
        matched_inputs = results.get(
            "matched_input_modes",
            [
                []
                for _ in range(
                    periods.size
                )
            ],
        )

        if not (
            periods.size
            == powers.size
            == len(matched_inputs)
        ):
            raise ValueError(
                f"{category} modal result arrays must have equal lengths."
            )

        for idx, period in enumerate(
            periods
        ):
            power = powers[idx]

            if not (
                np.isfinite(period)
                and period > 0.0
                and np.isfinite(power)
                and power > 0.0
            ):
                continue

            is_matched = bool(
                matched_inputs[idx]
            )
            colour = (
                "green"
                if is_matched
                else "red"
            )

            ax.scatter(
                period,
                power,
                s=marker_size,
                marker=marker,
                color=colour,
                linewidths=2.0,
                zorder=7 if is_matched else 4,
            )

            if is_matched:
                for mode_number in matched_inputs[idx]:
                    target = true_coordinates.get(
                        str(mode_number)
                    )
                    if target is None:
                        continue
                    ax.plot(
                        [target[0], period],
                        [target[1], power],
                        linestyle="--",
                        color="green",
                        linewidth=0.9,
                        alpha=0.55,
                        zorder=1,
                    )

    ax.set_xscale("log")
    if period_xlim is not None:
        period_xlim = np.asarray(
            period_xlim,
            dtype=float,
        ).ravel()
        if (
            period_xlim.size != 2
            or not np.all(
                np.isfinite(period_xlim)
            )
            or period_xlim[0] <= 0.0
            or period_xlim[0] >= period_xlim[1]
        ):
            raise ValueError(
                "period_xlim must contain two finite, positive, "
                "strictly increasing values."
            )
        ax.set_xlim(
            period_xlim[0],
            period_xlim[1],
        )
    ax.set_ylim(1e1, 1e8)
    ax.set_yscale("log")
    ax.set_xlabel("Period (years)")
    ax.set_ylabel(
        r"Record-mean area-weighted SV power "
        r"[(nT/yr)$^2$]"
    )
    ax.set_title(
        "All retained DMD modes: period and record-mean SV power"
    )
    ax.grid(
        alpha=0.25,
        which="both",
    )

    handles = [
        Line2D(
            [0], [0],
            marker="o",
            linestyle="None",
            markerfacecolor="none",
            markeredgecolor="black",
            label="True input",
        ),
        Line2D(
            [0], [0],
            marker="x",
            linestyle="None",
            color="black",
            label="Resolved DMD",
        ),
        Line2D(
            [0], [0],
            marker=".",
            linestyle="None",
            color="green",
            label="Matched output",
        ),
        Line2D(
            [0], [0],
            marker=".",
            linestyle="None",
            color="red",
            label="Unmatched output",
        ),
    ]

    if show_ideal:
        handles.insert(
            1,
            Line2D(
                [0], [0],
                marker="+",
                linestyle="None",
                color="black",
                label="Ideal DMD",
            ),
        )

    ax.legend(
        handles=handles,
        loc="best",
    )
    fig.tight_layout()

    return fig, ax

def plot_resolved_mode_similarity_heatmap(
    modal_results,
    figsize=None,
    cmap="viridis",
    annotate_values=True,
):
    """
    Plot every clean resolved DMD candidate against every true input mode.

    Each cell is the absolute normalized complex inner product used by
    ``best_spatial_match``. Rows selected by at least one input are placed
    first and labelled green. Unselected rows follow and are labelled red.
    Within both groups, rows are ordered by decreasing period; static modes
    (period=infinity) therefore appear first within their group.
    """

    resolved = modal_results.get(
        "resolved",
        {},
    )

    similarities = np.asarray(
        resolved.get(
            "spatial_similarity",
            [],
        ),
        dtype=float,
    )
    periods = np.asarray(
        resolved.get(
            "period",
            [],
        ),
        dtype=float,
    ).ravel()
    matched_inputs = resolved.get(
        "matched_input_modes",
        [],
    )
    input_mode_numbers = [
        str(mode_number)
        for mode_number in resolved.get(
            "input_mode_numbers",
            [],
        )
    ]

    if similarities.ndim != 2:
        raise ValueError(
            "resolved spatial_similarity must be a 2-D array."
        )

    n_candidates, n_inputs = similarities.shape

    if periods.size != n_candidates:
        raise ValueError(
            "resolved periods must contain one value per heatmap row."
        )

    if len(matched_inputs) != n_candidates:
        raise ValueError(
            "resolved matched_input_modes must contain one entry "
            "per heatmap row."
        )

    if len(input_mode_numbers) != n_inputs:
        raise ValueError(
            "resolved input_mode_numbers must contain one label "
            "per heatmap column."
        )

    if n_candidates == 0 or n_inputs == 0:
        if figsize is None:
            figsize = (8.0, 4.5)

        fig, ax = plt.subplots(
            figsize=figsize
        )
        ax.text(
            0.5,
            0.5,
            "No retained resolved candidates or input modes.",
            transform=ax.transAxes,
            ha="center",
            va="center",
        )
        ax.set_axis_off()
        fig.tight_layout()

        return (
            fig,
            ax,
            np.asarray([], dtype=int),
        )

    def period_sort_value(period):
        if np.isposinf(period):
            return -np.inf
        if np.isfinite(period):
            return -float(period)
        return np.inf

    row_order = np.asarray(
        sorted(
            range(n_candidates),
            key=lambda idx: (
                0 if matched_inputs[idx] else 1,
                period_sort_value(periods[idx]),
                idx,
            ),
        ),
        dtype=int,
    )

    ordered_similarity = similarities[
        row_order,
        :
    ]

    if figsize is None:
        figsize = (
            max(
                8.0,
                0.55 * n_inputs + 4.0,
            ),
            max(
                4.5,
                0.32 * n_candidates + 2.5,
            ),
        )

    fig, ax = plt.subplots(
        figsize=figsize
    )

    image = ax.imshow(
        ordered_similarity,
        aspect="auto",
        interpolation="nearest",
        cmap=cmap,
        vmin=0.0,
        vmax=1.0,
    )

    ax.set_xticks(
        np.arange(n_inputs)
    )
    ax.set_xticklabels(
        [
            f"Mode {mode_number}"
            for mode_number in input_mode_numbers
        ],
        rotation=45,
        ha="right",
    )

    row_labels = []

    for original_idx in row_order:
        period = periods[
            original_idx
        ]

        if np.isposinf(period):
            period_label = "static"
        elif np.isfinite(period):
            period_label = (
                f"T={period:.3g} yr"
            )
        else:
            period_label = "T=undefined"

        selected_by = matched_inputs[
            original_idx
        ]

        if selected_by:
            match_label = (
                "matched to "
                + ",".join(
                    str(mode_number)
                    for mode_number in selected_by
                )
            )
        else:
            match_label = "unmatched"

        row_labels.append(
            f"DMD {original_idx} | "
            f"{period_label} | "
            f"{match_label}"
        )

    ax.set_yticks(
        np.arange(n_candidates)
    )
    ax.set_yticklabels(
        row_labels
    )

    for tick, original_idx in zip(
        ax.get_yticklabels(),
        row_order,
    ):
        tick.set_color(
            "green"
            if matched_inputs[original_idx]
            else "red"
        )

    if annotate_values:
        annotation_size = (
            7
            if n_candidates * n_inputs <= 400
            else 5
        )

        for row_idx in range(
            n_candidates
        ):
            for column_idx in range(
                n_inputs
            ):
                value = ordered_similarity[
                    row_idx,
                    column_idx,
                ]

                if not np.isfinite(value):
                    label = "nan"
                    text_colour = "black"
                else:
                    label = f"{value:.2f}"
                    text_colour = (
                        "white"
                        if value < 0.45
                        else "black"
                    )

                ax.text(
                    column_idx,
                    row_idx,
                    label,
                    ha="center",
                    va="center",
                    fontsize=annotation_size,
                    color=text_colour,
                )

    colourbar = fig.colorbar(
        image,
        ax=ax,
        pad=0.02,
    )
    colourbar.set_label(
        "Absolute complex spatial similarity"
    )

    ax.set_xlabel(
        "True input mode"
    )
    ax.set_ylabel(
        "Retained clean resolved DMD mode"
    )
    ax.set_title(
        "Resolved DMD-to-input spatial phasor similarity"
    )

    fig.tight_layout()

    return fig, ax, row_order
