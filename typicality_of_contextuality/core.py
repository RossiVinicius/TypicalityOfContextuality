"""Code accompanying a manuscript on typicality of contextuality certification.

This code accompanies a manuscript titled "Typicality of contextuality 
certification in prepare and measure scenarios" and is published under 
MIT License.

Please consult the README and the manuscript for further details on 
motivation and technicalities.
"""

__license__ = "MIT"
__author__ = "Vinicius P Rossi"
__email__ = "prettirossi.vinicius@gmail.com"
__version__ = "1.1"


import numpy as np
import time
import math
import itertools

#For randomly sampling states, effects and unitaries
import qutip as qt 

#For assessments of contextuality
#Vendored and modified from https://github.com/pjcavalcanti/SimplexEmbeddingGPT
from .preprocessing import fromListOfMatrixToListOfVectors
from .simplexEmbedding import SimplexEmbedding

#For multiprocessing of typicality frequencies
import multiprocessing 
from functools import partial
from tqdm import tqdm

############################## Random sampling routines #######################################

def random_density_matrix(dim: int, upperbound: float, lowerbound: float, pure: bool) -> np.ndarray:
    """
    Generate a random density matrix of dimension ``dim`` using QuTiP.

    Parameters
    ----------
    dim : int
        The dimension of the Hilbert space in which the operator lives.
    upperbound : float
        A real number between 0 and 1 which is the upper bound on the
        purity of the operator (if sampling pure operators, any value
        can be assigned).
    lowerbound : float
        A real number between 0 and 1 which is the lower bound on the
        purity of the operator (if sampling pure operators, any value
        can be assigned).
    pure : bool
        If True, sample pure operators; otherwise, sample mixed operators.

    Returns
    -------
    dm_clean : np.ndarray
        A randomly sampled density operator (pure or mixed).  
        If mixed, its purity is between ``lowerbound`` and ``upperbound``.  
        Shape = (dim, dim).

    Notes
    -----
    - If ``lowerbound != 0`` and ``upperbound != 1``, the sampling of mixed
      operators will not be uniform. For narrow intervals
      [lowerbound, upperbound] we recommend other sampling methods (such
      as Markov Chain Monte Carlo).
    """
    #Sampling pure density operators
    if pure:
        dm=qt.rand_ket(dim,distribution="haar").full() #Samples a random vector in the Hilbert space using qutip
        dm_final=np.outer(np.conjugate(dm),dm)         #Constructs the density operator from the vector
        return dm_final
    #Sampling mixed density operators
    else:
        accepted = False
        while not accepted:
            dm = qt.rand_dm(dim, distribution="ginibre").full()  #Samples a ranndom full-rank density operator using qutip
            # Clean numerical noise to ensure hermiticity
            dm_final = np.real(dm) + 1j * np.where(np.abs(np.imag(dm)) < 1e-7, 0, np.imag(dm)) #1e-7 is arbitrarily chosen
            purity = np.trace(dm_clean @ dm_clean).real 
            if purity <= upperbound and purity >= lowerbound: #Rejection method: if purity is not within the interval, repeat
                accepted = True   
                
        return dm_final
 
def random_unitary(dim: int) -> np.ndarray:
    """
    Generate a Haar-random dim x dim unitary matrix using the Mezzadri algorithm.

    Steps (dimension-independent):
        1. Draw a dim x dim complex Ginibre matrix with i.i.d. complex normals.
        2. Perform QR decomposition: Z = Q R.
        3. Fix the phases using the diagonal of R so Q becomes Haar-distributed.
        4. Convert to Qobj.
        5. Optionally remove global phase so det(U) = 1 (returns SU(d)).

    Parameters
    ----------
    dim : int
        Dimension of the unitary.

    Returns
    -------
    Uq : array
        A dim x dim Haar-random unitary in SU(d).
        Shape = (dim, dim)
    """

    # Ginibre matrix (complex Gaussian)
    Z = (np.random.randn(dim, dim) + 1j * np.random.randn(dim, dim)) / np.sqrt(2)
    # R decomposition
    Q, R = np.linalg.qr(Z)
    # Fix phases so Q is Haar distributed
    diagR = np.diag(R)
    phases = diagR / np.abs(diagR)
    U = Q @ np.diag(np.conj(phases))
    # Remove global phase and enforce det(U) = 1 (SU(d))
    det = np.linalg.det(U)
    Uq = U * (det ** (-1.0/dim))

    return Uq
 
def random_effects(dim: int, m: int, upperbound: float, lowerbound: float, pure: bool, outcome: int = 2) -> np.ndarray:
    """
    Generate a set of random quantum unsharp measurements.

    An effect here is assumed to be a positive semidefinite operator ``E``
    satisfying 0 <= E <= I. For each POVM, its elements E sum up to I.

    Parameters
    ----------
    dim : int
        The dimension of the Hilbert space over which the measurement is performed.
    m : int
        The number of measurements to be sampled.  
        Note: 2*m total effects will be output.
    upperbound : float
        A real number between 0 and 1 which is the upper bound in sharpness
        of the POVM (if sampling projectors, any value can be assigned).
    lowerbound : float
        A real number between 0 and 1 which is the lower bound in sharpness
        of the POVM (if sampling projectors, any value can be assigned).
    pure : bool
        If True, sample projectors; otherwise, sample POVM elements.
    outcome : int
        Number of outcomes in case of POVM sampling. Automatically set to
        dichotomic POVMs (outcome = 2).

    Returns
    -------
    effects : np.ndarray
        An array of randomly sampled POVM elements with sharpness between
        ``lowerbound`` and ``upperbound``, together with their complementary
        effects. Shape = (m*outcomes, dim, dim).
        
    Notes
    -----
        - The sharpness of a POVM ``(E1,...,Ek)`` is calculated as the sum
        of ``np.trace(Ei @ Ei)`` weigthed by the dimension ``d``. If the set
        constitutes a projective measurement, ``k == d`` and the sharpness is
        equal to 1.
        - Similarly to states, bounding the sharpness implies on non-uniform
        sampling. For narrow intervals [lowerbound, upperbound] we recommend
        other sampling methods.
    """
    effects=[];
    rng = np.random.default_rng()
    for i in range(m):
        if pure:
            ONB = [np.diag(np.eye(dim)[i]) for i in range(dim)]
            U = random_unitary(dim)
            eff = [U @ e @ np.conjugate(U).T for e in ONB]
            effects.append(eff)
        else:
            accepted = False
            while not accepted:
                G = rng.normal(size=(outcome,dim,dim))+1j*rng.normal(size=(outcome,dim,dim))
                POVM = np.einsum('kij,klj->kil',G,G.conj())
                S = POVM.sum(axis=0)
                eigvals, eigvecs = np.linalg.eigh(S)
                S_inv_sqrt = eigvecs @ np.diag(1.0 / np.sqrt(eigvals)) @ eigvecs.conj().T
                normPOVM = np.einsum('ab,kbc,cd->kad', S_inv_sqrt, POVM, S_inv_sqrt)
                traces = np.einsum('kij,kji->k', normPOVM, normPOVM)
                sharpness = np.real(traces.sum()) / dim
                if lowerbound <= sharpness <= upperbound:
                    effects.append(normPOVM)
                    accepted = True
    return np.array(effects).reshape(-1,dim,dim)

def fixed_effects() -> np.ndarray:
    """
    Generate a fixed grid of rank-1 quantum effects on a qubit Hilbert space.

    The construction sweeps over 21 values of the polar angle (via i) and
    11 values of the azimuthal phase (via k) to produce a set of effects
    of the form E = |s><s|, where

        |s>= [ sin(i * pi / 20), exp(1j * k * pi / 5) * cos(i * pi / 20) ].

    Parameters
    ----------
    None

    Returns
    -------
    effects : np.ndarray
        An array of equally distributed projectors over the Bloch sphere.  
        Shape = (231, 2, 2).

    Notes
    -----
    These effects lie on a discrete polar-azimuthal grid of the Bloch sphere.
    Some of them are repeated, summing up to 168 distinct effects.
    """
    effects=[];
    for i in range(21):
        for k in range(11):
            eff=np.array([np.sin(i*np.pi/20), np.exp(1j*k*np.pi/5)*np.cos(i*np.pi/20)]); # Construct a normalized state vector on the Bloch sphere
            eff=np.outer(eff, eff.conj()); #Construct the corresponding density operator.
            effects.append(eff);
    return np.array(effects)
    

############################ Typicality routines ####################################


def Typicality_oneshot(n: int, m: int, dim: int, upperbound: float, lowerbound: float,pure_prep: bool, pure_meas: bool) -> int:
    """
    Perform a single-shot typicality test by sampling random states and effects.

    The routine assesses simplex embeddability and returns a classification
    flag indicating whether the sampled scenario is embeddable or not. This
    function is used as the basic building block for estimating the frequency
    of non-embeddable scenarios.

    Parameters
    ----------
    n : int
        Number of random quantum states to sample.
    m : int
        Number of random quantum measurements to sample.
    upperbound : float
        A real number between 0 and 1 which is the upper bound in purity of 
        the states (for pure states this value is ignored) or sharpness of 
        the measurements (for projective measurements this value is ignored).
    lowerbound : float
        A real number between 0 and 1 which is the lower bound in purity of 
        the states (for pure states this value is ignored) or sharpness of 
        the measurements (for projective measurements this value is ignored).
    pure_prep : bool
        If True, sample pure states; otherwise sample mixed states.
    pure_meas : bool
        If True, sample projective measurements; otherwise sample POVMs.

    Returns
    -------
    count : int
        Classification flag based on simplex embedding:
        - 2 : Assessment failed (r is None or exceeds 1).
        - 1 : Scenario is non-embeddable (r > 1e-7).
        - 0 : Scenario is embeddable (r <= 1e-7).

    Notes
    -----
    - The threshold r > 1e-7 is based on numerical stability and can be
      adjusted for different experimental setups.
    - One can modify the routine to force projective measurements even
      when states are mixed.
    - This routine requires `SimplexEmbedding`, available at:
      https://github.com/pjcavalcanti/SimplexEmbeddingGPT
    """
    states=[];
    for i in range(n):
        stat=random_density_matrix(dim, upperbound, lowerbound, pure_prep)
        states.append(stat) #Construct the set of states
    effects=random_effects(dim,m, upperbound, lowerbound, pure_meas) #Construct set of effects
    s,e,u,mms=fromListOfMatrixToListOfVectors(states,effects); #Construct GPT fragment from sampled states/effects
    r=SimplexEmbedding(s, e, u, mms) #Assess simplex embedding of the GPT fragment
    if (r==None or r>1): #If LP returns incorrect value of robustness of contextuality for any reason
        count=2
    else:
        count=int(r>1e-6) #count = 0 if r is below threshold and 1 otherwise
        #if r<=1e-7:
        #    print(states,effects) #Debug
    return count

def Typicality_worker(args: tuple, iterations_per_worker: int) -> tuple:
    """
    Worker routine for parallelised typicality estimation.

    Repeatedly calls `Typicality_oneshot(...)` and accumulates statistics
    over a fixed number of iterations. Results with return value 2 
    (i.e. invalid or unclassifiable cases) are excluded from the denominator.

    Parameters
    ----------
    args : tuple
        Tuple of parameters (n, m, dim, upperbound, lowerbound, pure)
        passed directly to `Typicality_oneshot(...)`.
    iterations_per_worker : int
        Number of typicality trials to perform in this worker.

    Returns
    -------
    count : int
        Number of contextual cases where `Typicality_oneshot` returned 1.
    total : int
        Number of valid evaluations (excluding those that returned 2).

    Notes
    -----
    This function is designed to run under multiprocessing.
    """
    n, m, dim, upperbound, lowerbound, pure_preps, pure_meas = args
    count = 0
    total = 0
    for _ in range(iterations_per_worker):
        r = Typicality_oneshot(n, m, dim, upperbound, lowerbound, pure_preps, pure_meas)
        if r != 2: #Exclude cases in which r is miscalculated for any reason
            total += 1
            count += r # r is 0 or 1 in valid cases
    return count, total

def Parallel_Typicality(n: int, m: int, dim: int, upperbound: float, lowerbound: float, pure_preps: bool, pure_meas: bool, total_iterations: int, num_workers: int = 200) -> float:
    """
    Parallelised estimation of typicality by distributing trials across multiple workers.

    Distributes `total_iterations` calls to `Typicality_oneshot(...)` across multiple
    workers and aggregates the results.

    Parameters
    ----------
    n : int
        Number of random states per trial.
    m : int
        Number of random effects per trial.
    dim : int
        Hilbert space dimension.
    upperbound : float
        Upper bound on purity for sampled states/effects. Ignored if `pure_preps` and
        `pure_meas` are True.
    lowerbound : float
        Lower bound on purity for sampled states/effects. Ignored if `pure_preps` and
        `pure_meas` are True.
    pure_preps : bool
        If True, sample pure states; otherwise, sample mixed states.
    pure_meas : bool
        If True, sample projective measurements; otherwise, sample POVMs.
    total_iterations : int
        Total number of one-shot trials to perform across all workers.
    num_workers : int, optional
        Number of worker processes to spawn. Defaults to `multiprocessing.cpu_count()`,
        the total number of available CPU cores.

    Returns
    -------
    freq : float
        Estimated frequency of typicality.

    Notes
    -----
    - Adjust `num_workers` manually if running on a shared or resource-limited system.
    - If `total_iterations` is not divisible by `num_workers`, the remainder iterations
      are discarded.
    - Uses `tqdm` for progress reporting.
    """
    if m < 2:
       raise ValueError(f"Number of measurements m={m} is less than 2. Contextuality requires at least 2 measurements.")
   
    if num_workers is None:
        num_workers = multiprocessing.cpu_count()
        
    iterations_per_worker = total_iterations//num_workers # Divide the total number of iterations evenly among workers
    # Wrap the worker function with fixed iterations per worker
    worker_func = partial(Typicality_worker, 
                         iterations_per_worker=iterations_per_worker)
    # Create a multiprocessing pool with num_workers workers
    # Each worker is passed the same arguments (n, m, dim, etc.)
    with multiprocessing.Pool(num_workers) as pool:
        results = list(tqdm(pool.imap(worker_func, [(n, m, dim, upperbound, lowerbound, pure_preps, pure_meas)] * num_workers),
                           total=num_workers,
                           desc=f"Processing (n={n}, m={m}, dim={dim})"))
    
    total_count = sum(r[0] for r in results) # contextual cases
    total_valid = sum(r[1] for r in results) # valid cases
    
    return total_count / total_valid if total_valid > 0 else 0
    
def Fixed_typicality_oneshot(n: int, upperbound: float, lowerbound: float, pure: bool) -> int:
    """
    Perform a single-shot typicality test with random qubit states and fixed projectors.
    
    This routine uses random qubit states and a fixed set of 84 projectors to assess 
    simplex embeddability, returning a classification flag that can be used to estimate 
    the frequency of non-embeddable scenarios over multiple trials.

    Parameters
    ----------
    n : int
        Number of random quantum states to sample.
    upperbound : float
        Upper bound on the purity of the sampled states (between 0 and 1).
        Ignored if `pure` is True.
    lowerbound : float
        Lower bound on the purity of the sampled states (between 0 and 1).
        Ignored if `pure` is True.
    pure : bool
        If True, sample pure states; otherwise, sample mixed states.

    Returns
    -------
    count : int
        Classification flag based on simplex embedding assessment:
        - Returns 2 if the assessment failed (None or r > 1).
        - Returns 1 if r > 1e-7.
        - Returns 0 if r <= 1e-7.

    Notes
    -----
    - The threshold r > 1e-7 is discussed in the accompanying paper 
      and can be adjusted for specific experimental setups.
    - This routine requires `SimplexEmbedding`, available at:
      https://github.com/pjcavalcanti/SimplexEmbeddingGPT
    """
    states=[];
    for i in range(n):
        stat=random_density_matrix(2, upperbound, lowerbound, pure);
        states.append(stat); #Construct the set of states
    effects=fixed_effects(); 
    s,e,u,mms=fromListOfMatrixToListOfVectors(states,effects); #Construct the GPT fragment from the sets of states/effects
    r=SimplexEmbedding(s, e, u, mms); #Assess simplex embedding of the GPT fragment
    if (r==None or r>1): #If LP returns incorrect value of robustness of contextuality for any reason
        count=2;
    else:
        count=int(r>1e-7); #count = 0 if r below threshold, count = 1 otherwise
    return count

def Typicality_worker_fixed(args: tuple, iterations_per_worker: int) -> tuple:
    """
    Perform a single-shot typicality test with random qubit states and fixed projectors.

    Samples `n` random qubit states and 84 fixed projectors, assesses
    simplex embeddability, and returns a classification flag used to
    estimate the frequency of non-embeddable scenarios.

    Parameters
    ----------
    n : int
        Number of random quantum states to sample.
    upperbound : float
        Upper bound in purity of the states (ignored if `pure` is True).
    lowerbound : float
        Lower bound in purity of the states (ignored if `pure` is True).
    pure : bool
        If True, sample pure states; otherwise, sample mixed states.

    Returns
    -------
    count : int
        Classification flag based on simplex embedding:
        - 2 : Assessment failed (r is None or exceeds 1)
        - 1 : Scenario is non-embeddable (r > 1e-7)
        - 0 : Scenario is embeddable (r <= 1e-7)

    Notes
    -----
    - The threshold r > 1e-7 is discussed in the paper accompanying this code
      and can be adjusted for different experimental setups.
    - This routine requires `SimplexEmbedding`, available at:
      https://github.com/pjcavalcanti/SimplexEmbeddingGPT
    """
    n, upperbound, lowerbound, pure = args
    count = 0
    total = 0
    for _ in range(iterations_per_worker):
        r = Fixed_typicality_oneshot(n, upperbound, lowerbound, pure)
        if r != 2: #Exclude cases in which r is miscalculated for any reason
            total += 1
            count += r #r is 0 or 1 in valid cases
    return count, total

def Parallel_Typicality_fixed(n: int, upperbound: float, lowerbound: float, pure: bool, total_iterations: int, num_workers: int = None) -> float:
    """
    Parallelised estimation of typicality using fixed projectors.

    Distributes `total_iterations` calls to `Fixed_typicality_oneshot(...)`
    across multiple workers and aggregates the results.

    Parameters
    ----------
    n : int
       Number of random states per trial.
    upperbound : float
       Upper bound on purity for sampled states. Ignored if `pure` is True.
    lowerbound : float
       Lower bound on purity for sampled states. Ignored if `pure` is True.
    pure : bool
       If True, sample pure states; otherwise, sample mixed states.
    total_iterations : int
       Total number of one-shot trials to perform across all workers.
    num_workers : int, optional
       Number of worker processes to spawn. Defaults to `multiprocessing.cpu_count()`,
       the total number of available CPU cores.

    Returns
    -------
    freq : float
       Estimated frequency of nontrivial typicality.

    Notes
    -----
    - Adjust `num_workers` manually if running on a shared or resource-limited system.
    - If `total_iterations` is not divisible by `num_workers`, the remainder iterations
      are discarded.
    - Uses `tqdm` for progress reporting.
    """
    if num_workers is None:
        num_workers = multiprocessing.cpu_count()
    
    iterations_per_worker = total_iterations // num_workers # Divide the total number of iterations evenly among workers
    # Wrap the worker function with fixed iterations per worker
    worker_func = partial(Typicality_worker_fixed, 
                         iterations_per_worker=iterations_per_worker)
    # Create a multiprocessing pool with num_workers workers
    # Each worker is passed the same arguments (n, upperbound, etc)
    with multiprocessing.Pool(num_workers) as pool:
        results = list(tqdm(pool.imap(worker_func, [(n, upperbound, lowerbound, pure)] * num_workers),
                           total=num_workers,
                           desc=f"Processing n={n}"))
    total_count = sum(r[0] for r in results) #contextual cases
    total_valid = sum(r[1] for r in results) #valid cases
    
    return total_count / total_valid if total_valid > 0 else 0

#################### Tool for assessing minimal preparations ##################

def Minimalpreps(m: int, dim: int, upperbound: float, lowerbound: float, pure_meas: bool, iterations: int, num_workers: int = 200) -> int:
    """
    Compute the minimal number of random mixed states required for typicality > 0.99.

    Parameters
    ----------
    m : int
        Number of randomly sampled measurements.
    dim : int
        Hilbert space dimension.
    upperbound : float
        Upper bound in purity of the states (and POVMs if `pure_meas=False`).
    lowerbound : float
        Lower bound in purity of the states (and POVMs if `pure_meas=False`).
    iterations : int
        Number of iterations of sampling random states and effects.
       num_workers : int, optional
          Number of worker processes to spawn. Defaults to `multiprocessing.cpu_count()`,
          the total number of available CPU cores.

    Returns
    -------
    minimal_preps : int
        Minimal number of sampled states for typicality > 0.99.

    Notes
    -----
    - To sample only projective measurements, comment line 246 and uncomment line 247
      in the repository.
    - `SimplexEmbedding(...)` scales poorly with dimension in its current version.
    - A larger `iterations` value improves statistical reliability. This can be
      compensated by increasing the number of measurements `m`.
    """
    minimal_preps=0
    n=5
    while minimal_preps == 0:
            t=Parallel_Typicality(n, m, dim, upperbound, lowerbound, False, pure_meas, iterations)
            if t > 0.99:
                minimal_preps = n
            n+=1
    return minimal_preps

###################### Typicality analysis in the paper ########################

def Typicality_PnM_parallel(pure_preps: bool, pure_meas: bool, docname: str):
    """
    Compute the typicality of random states and effects over varying numbers of preparations and measurements.

    This routine generates the numerical data used to produce Figure 1
    of the paper.

    Parameters
    ----------
    pure_preps : bool
        If True, sample pure states; if False, sample mixed states.
    pure_meas: bool
        If True, sample projective measurments; if False, sample POVMs.
    docname : str
        Filename for saving numerical results (in .txt format), e.g., 'typicality.txt'.

    Returns
    -------
    None
        The function produces a text file containing the numerical data points.

    Notes
    -----
    - The number of states varies from 4 to 19 (inclusive).
    - The number of measurements varies from 2 to 19 (inclusive), i.e.,
      the number of effects ranges from 4 to 38 (inclusive).
    """
    points = [(x, y, 2, 1, 0, pure_preps, pure_meas, 10**6) for x in range(4,20) for y in range(2, 20)]
    results = [Parallel_Typicality(*point) for point in points]
    x_values = [point[0] for point in points]
    y_values = [point[1] for point in points]
    np.savetxt(docname, np.column_stack((x_values, y_values, results)), header="Preparations  Measurements  Typicality", fmt="%.6f")

def Typicality_n_states_fixed_parallel(pure: bool, docname: str):
    """
    Compute the typicality of random states with a fixed set of effects.

    This routine generates the numerical data used to produce Figure 2
    of the paper. The effects are fixed and normalised projectors.

    Parameters
    ----------
    pure : bool
        If True, sample pure states; if False, sample mixed states.
    docname : str
        Filename for saving numerical results (in .txt format), e.g., 'typicality.txt'.

    Returns
    -------
    None
        The function produces a text file containing the numerical data points.

    Notes
    -----
    - The number of states varies from 4 to 14 (inclusive).
    - The effects used are fixed and normalised projectors.
    """
    points = [4,5,6,7,8,9,10,11,12,13,14]
    results=np.zeros(11) 
    for k in range(11):
        results[k]=Parallel_Typicality_fixed(k+4, 1, 0, pure, 10**6)
    x_values = [i+4 for i in range(len(points))]
    np.savetxt(docname, np.column_stack((x_values, results)), header="Dimension  Typicality", fmt="%.6f")

def Typicality_minimalpreps(upperbound: float, pure_meas: bool, docname: str):
    """
    Compute the minimal number of random mixed states for typicality > 0.99 with 20 random effects and varying lower bounds on purity.

    This routine generates the numerical data used to produce Figure 4
    of the paper. The lower bound on purity is considered in [0, 0.5, 0.7, 0.9].

    Parameters
    ----------
    upperbound : float
        Upper bound in purity of the states (and POVMs if `pure_meas=False`).
    pure_meas : bool
        If True, samples projective measurements; if False, samples POVMs.
    docname : str
        Filename for saving numerical results (in .txt format), e.g., 'typicality.txt'.

    Returns
    -------
    None
        The function produces a text file containing the numerical data points.

    Notes
    -----
    - The `upperbound` must be greater than 0.9.
    """
    minimal_preps=np.zeros(3);
    lowerbound_values=np.array([0.5, 0.7, 0.9])
    for k in range(3):
        minimal_preps[k] = Minimalpreps(20, 2, upperbound, lowerbound_values[k], pure_meas, 10**6)
    np.savetxt(docname, np.column_stack((lowerbound_values, minimal_preps)), header="LowerBound  MinimalPreps", fmt="%.6f")
    
def Typicality_iterations_parallel(pure_preps: bool, pure_meas: bool, docname: str):
    """
    Run parallelised typicality estimation for increasing numbers of iterations.

    The number of iterations increases as 10^N with N ∈ {2, …, 7}, i.e., from 100
    up to 10^7 iterations. Used to generate the plots in Figures 5, 6, and 7 of
    the paper.

    Parameters
    ----------
    pure_preps : bool
        If True, sample pure states; otherwise, sample mixed states.
    pure_meas : bool
        If True, sample projective measurements; otherwise, sample POVMs.
    docname : str
        Filename for saving numerical results (in .txt format), e.g., 'typicality.txt'.

    Returns
    -------
    None
        The function produces a text file containing the numerical data points.

    Notes
    -----
    - Typicality values are stored alongside the runtime required for each run.
    """
    points = [(4, 2, 2, 1, 0, pure_preps, pure_meas, 10**N) for N in range(2, 8)]
    results = [] #Store typicality results
    runtimes = [] #Store elapsed time for each run
    #Run typicality estimation for each iteration value
    for point in points:
        start = time.time()
        res = Parallel_Typicality(*point)
        end = time.time()
        results.append(res)
        runtimes.append(end - start) #Store runtime in seconds

    x_values = [point[7] for point in points] #Extract iteration counts
    y_values = np.array(results)
    np.savetxt(docname,np.column_stack((x_values, y_values, runtimes)), header="Iterations  Typicality  RuntimeSeconds", fmt="%.6f")


def wilson_score_interval(successes: int,
                          iterations: int,
                          confidence: float = 0.99,
                          z_value: float = None) -> tuple:
    """
    Estimates Wilson score interval.
    
    Implements the Wilson score interval with continuity correction
    for an observed number of successes in a number of Bernoulli trials.

    Parameters
    ----------
    successes : int
        Number of runs with robustness > threshold.
    iterations : int
        Total number of independent samples.
    confidence : float, optional
        Desired confidence level (0 < confidence < 1). Default is 0.95.
        Common choices: 0.90, 0.95, 0.99.
    z_value : float, optional
        If provided, this z-score will be used instead of looking up from
        `confidence`. This allows arbitrary confidence levels if you already
        know the corresponding z.

    Returns
    -------
    p_hat : float
        Observed proportion = successes / trials (0 if trials == 0).
    lower : float
        Lower bound of the Wilson confidence interval (clipped to [0,1]).

    Notes
    -----
    - If `trials == 0`, the function returns (0.0, 0.0) as a convention:
      no data -> no information (p_hat=0, full interval).
    - Supported `confidence` values by lookup: 0.80, 0.90, 0.95, 0.975, 0.99, 0.995.
      For other confidence levels, the user may pass `z_value` directly.
    """
    # Quick validation
    if iterations < 0 or successes < 0:
        raise ValueError("`successes` and `iterations` must be non-negative integers.")
    if successes > iterations:
        raise ValueError("`successes` cannot exceed `iterations`.")
    if iterations == 0:
        # No iterations -> undefined p_hat; return convention (0, full interval)
        return 0.0, 0.0

    # Observed frequence
    p_hat = successes / iterations

    # Determine z: either provided or lookup common values
    if z_value is None:
        # lookup table for common confidence levels
        z_lookup = {
            0.80: 1.281551565545,   # ~ z_{0.80}
            0.90: 1.644853626951,   # ~ z_{0.90}
            0.95: 1.959963984540,   # ~ z_{0.95}
            0.975: 1.959963984540,  # ~ (two-sided 0.975)
            0.99: 2.575829303548,   # ~ z_{0.99}
            0.995: 2.807033768345   # ~ z_{0.995}
        }
        try:
            z = z_lookup[confidence]
        except KeyError:
            # If user asked for an uncommon confidence and did not provide z,
            # raise a clear error asking them to provide z_value.
            raise ValueError(
                "Unsupported `confidence`. Use one of "
                f"{sorted(z_lookup.keys())} or provide `z_value` explicitly."
            )
    else:
        z = float(z_value)

    #Wilson formula with continuity correction
    center = (p_hat + z**2 / (2.0 * iterations)) / (1.0 + z**2 / iterations)
    half = (z * math.sqrt((p_hat * (1.0 - p_hat) / iterations) + (z**2 / (4.0 * iterations**2)))) / (1.0 + z**2 / iterations)

    lower = max(0.0, center - half)

    return p_hat, lower

def Typicality_POM(iterations: int) -> tuple:
    """
    Estimate average success rate, variance, average robustness of contextuality and typicality for a fixed set of 8 optimal qubit states and 3 optimal measurements rotated by random unitaries.

    For each iteration, the function applies a random Haar-distributed
    unitary to the measurements, computes the robustness of contextuality,
    and accumulates statistics.

    Parameters
    ----------
    iterations : int
        Number of independent trials to run.

    Returns
    -------
    av : float
        Average success rate.
    sigma : float
        Standard deviation of the average success rate.
    avr: float
        Average robustness of contextuality.
    sigmar : float
        Standard deviation of the average robustness of contextuality
    av_op : float
        Average operational success rate (maximised over permutation of measurements)
    sigma_op : float
        Standard deviation of the average operational success rate
    t : float
        Typicality of contextuality.

    Notes
    -----
    - The set of states is a fixed collection of 8 qubit pure states defined
      explicitly at the start of the function.
    - By default, the optimal measurements are rotated by a random Haar unitary
      in each iteration. To use random projective measurements or POVMs, uncomment
      the alternative effects lines in the code.
    - The success rate is calculated from the robustness of contextuality based
      on Phys. Rev. A 111, 022217 (2025).
    """
    #Define optimal qubit states for 3-to-1 POM
    s000=0.5*(np.array([[1,0],[0,1]])+np.sqrt(3/2)/2*np.array([[0,1],[1,0]])+np.sqrt(3/2)/2*np.array([[0,-1j],[1j,0]])+0.5*np.array([[1,0],[0,-1]]))
    s001=0.5*(np.array([[1,0],[0,1]])+np.sqrt(3/2)/2*np.array([[0,1],[1,0]])+np.sqrt(3/2)/2*np.array([[0,-1j],[1j,0]])-0.5*np.array([[1,0],[0,-1]]))
    s010=0.5*(np.array([[1,0],[0,1]])+np.sqrt(3/2)/2*np.array([[0,1],[1,0]])-np.sqrt(3/2)/2*np.array([[0,-1j],[1j,0]])+0.5*np.array([[1,0],[0,-1]]))
    s011=0.5*(np.array([[1,0],[0,1]])+np.sqrt(3/2)/2*np.array([[0,1],[1,0]])-np.sqrt(3/2)/2*np.array([[0,-1j],[1j,0]])-0.5*np.array([[1,0],[0,-1]]))
    s100=0.5*(np.array([[1,0],[0,1]])-np.sqrt(3/2)/2*np.array([[0,1],[1,0]])+np.sqrt(3/2)/2*np.array([[0,-1j],[1j,0]])+0.5*np.array([[1,0],[0,-1]]))
    s101=0.5*(np.array([[1,0],[0,1]])-np.sqrt(3/2)/2*np.array([[0,1],[1,0]])+np.sqrt(3/2)/2*np.array([[0,-1j],[1j,0]])-0.5*np.array([[1,0],[0,-1]]))
    s110=0.5*(np.array([[1,0],[0,1]])-np.sqrt(3/2)/2*np.array([[0,1],[1,0]])-np.sqrt(3/2)/2*np.array([[0,-1j],[1j,0]])+0.5*np.array([[1,0],[0,-1]]))
    s111=0.5*(np.array([[1,0],[0,1]])-np.sqrt(3/2)/2*np.array([[0,1],[1,0]])-np.sqrt(3/2)/2*np.array([[0,-1j],[1j,0]])-0.5*np.array([[1,0],[0,-1]]))
    states = np.array([s000,s001,s010,s011,s100,s101,s110,s111])
    #Define fixed optimal measurements for the task
    POM_eff=e=0.5*np.array([[[2,0],[0,0]],[[0,0],[0,2]],[[1,1],[1,1]],[[1,-1],[-1,1]],[[1,-1j],[1j,1]],[[1,1j],[-1j,1]]])
    
    ops1=0;
    ops2=0;
    s1=0;
    s2=0;
    r1=0
    r2=0
    count=0;
    for i in range(iterations):
        #Apply random Haar-distributed unitary (for rotated POVM)
        U=random_unitary(2) #Comment if needed.
        effects = [U @ e @ np.conj(U).T for e in POM_eff] #Analysis for rotated measurements. Comment if needed.
        #Alternatively:
        #effects = random_effects(2, 3, 1, 0, True) #Uncomment if analysis for projective measurements
        #effects = random_effects(2, 3, 1, 0, False); #Uncomment if analysis for random POVMs
        meas = [(effects[0], effects[1]), (effects[2], effects[3]), (effects[4], effects[5])]
    
        # There are 3! permutations and 2^3 sign flips
        perms = list(itertools.permutations(range(3)))
        flips = list(itertools.product([1, -1], repeat=3))
        
        best_s = 0.0
        for perm in perms:
            for flip in flips:
                s_sum = 0.0
                for x in range(8):
                    bits = np.array(list(map(int, f"{x:03b}")))  # [x1,x2,x3]
                    for y in range(3):
                        # Which physical measurement corresponds to logical bit y
                        phys_y = perm[y]
                        Eplus, Eminus = meas[phys_y]
                        # Flip outcome label if needed
                        if flip[y] == -1:
                            Eplus, Eminus = Eminus, Eplus
                        # Success prob for this (x,y)
                        E_corr = Eplus if bits[y] == 1 else Eminus
                        s_sum += np.real(np.trace(states[x] @ E_corr))
                s_avg = s_sum / (8 * 3)
                best_s = max(best_s, s_avg)
        s,e,u,mms=fromListOfMatrixToListOfVectors(states, effects);
        ops1+=best_s
        ops2+=best_s**2
        r=SimplexEmbedding(s,e,u,mms);
        if r>1e-7:
            sample= 0.5*(4/3-r)/(1-r);
            rsample = r
            s1+=sample
            s2+=sample**2
            r1+=rsample
            r2+=rsample**2
            count+=1
        else:
            s1+=2/3;
            s2+=(2/3)**2;
    av_op=ops1/10**6
    av2_op=ops2/10**6
    sigma_op=np.sqrt(av2_op-av_op**2)
    av=s1/10**6
    av2=s2/10**6
    sigma=np.sqrt(av2-av**2);
    avr1=r1/10**6
    avr2=r2/10**6
    sigmar=np.sqrt(avr2-avr1**2)
    t=count/10**6
    return av, sigma, avr1, sigmar, av_op, sigma_op, t
