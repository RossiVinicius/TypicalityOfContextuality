"""
This file is vendored and modified from:
https://github.com/RossiVinicius/SimplexEmbeddingGPT-v2

Original authors:
P. J. Cavalcanti, V. P. Rossi

License: MIT
"""

from scipy.linalg import lu
from scipy.sparse import csr_matrix
import numpy as np
import cvxpy as cp

from .math_tools import rref, FindStateConeFacets, FindEffectConeFacets


def DefineAccessibleGPTFragment(statesOrEffects):
    """
    Constructs the accessible Generalized Probabilistic Theory (GPT) fragment from a given set of states or effects.
    Uses the Reduced Row Echelon Form (RREF) of the matrix of states/effects to construct the inclusion map (RREF without the kernel)
    and projection (pseudo inverse of the inclusion), and the set of states/effects represented in the accessible fragment.

    Args:
    statesOrEffects (np.ndarray): A numpy array representing states or effects. Should be a 2D array.
        shape = (gpt fragment dimension, number of states or effects)

    Returns:
    tuple: A tuple containing:
        - inclusionMap (np.ndarray): The inclusion map matrix derived from the Reduced Row Echelon Form (RREF).
            shape = (GPT fragment dimension, accessible fragment dimension)

        - projectionMap (np.ndarray): The projection map matrix, which is the pseudo-inverse of the inclusion map.
            shape = (accessible fragment dimension, GPT fragment dimension)

        - accessibleFragment (np.ndarray): The accessible fragment, computed as the projection of statesOrEffects.
            shape = (accessible fragment dimension, number of states or effects)
    """
    REF = rref(statesOrEffects.T)
    
    P, L, U = lu(statesOrEffects)
    r = len(np.unique(csr_matrix(U).indptr)) - 1
    
    inclusionMap = REF[:r, :].T
    projectionMap = np.linalg.pinv(inclusionMap.T@inclusionMap)@inclusionMap.T

    return inclusionMap, projectionMap, projectionMap @ statesOrEffects


def SimplicialConeEmbedding(H_S, H_E, accessibleFragmentBornRule, depolarizingMap):
    """
    Solves Linear Program 2 from the paper by testing whether a simplicial cone embedding exists
    given the GPT's state and effect cone facets, the bilinear form giving the Born rule in
    the accessible fragment, and its depolarizing map.

    Args:
    H_S (np.ndarray): A numpy array representing the state cone facets.
        shape = (number of state cone facets, dimension of states in the GPT accessible fragment)
        
    H_E (np.ndarray): A numpy array representing the effect cone facets.
        shape = (dimension of effects in the GPT accessible fragment, number of effect cone facets)
        
    accessibleFragmentBornRule (np.ndarray): The bilinear form giving the Born rule in the accessible fragment.
        shape = (dimension of states in the GPT accessible fragment, dimension of effects in the GPT accessible fragment)
        
    depolarizingMap (np.ndarray): The depolarizing map, typically the outer product of accessible fragment unit and MMS.
        shape = (dimension of effects in the GPT accessible fragment, dimension of states in the GPT accessible fragment)

    Returns:
    tuple: A tuple containing:
        - robustness (float): The minimum amount of noise such that the depolarizing map causes a simplicial cone embedding to exist. 
        - sigma (np.ndarray): The sigma matrix obtained from the optimization problem.
    """
    H_S = np.array(H_S, dtype=float)
    H_E = np.array(H_E, dtype=float)
    robustness, sigma = cp.Variable(nonneg=True), cp.Variable(
         shape=(H_E.shape[1], H_S.shape[0]), nonneg=True
     )

    problem = cp.Problem(
         cp.Minimize(robustness),
         [
             robustness * depolarizingMap
             + (1 - robustness) * accessibleFramentBornRule
             - H_E @ sigma @ H_S
             == 0
         ],
     )

     try:
         # Try solving with ECOS first
         problem.solve(solver=cp.ECOS, verbose=False)
         # If ECOS fails, try CLARABEL as a fallback
         if problem.status in ["infeasible", "unbounded"]:
             problem.solve(solver=cp.CLARABEL, verbose=False)
     except SolverError:
         # If both solvers fail, return robustness=2 and sigma=None (or any placeholder)
         return 2, None

     #Check if the solution is valid (robustness should be between 0 and 1)
     if problem.status not in ["optimal", "optimal_inaccurate"] or robustness.value is None:
         return 2, None  # Invalid solution, return default
     else:
         return robustness.value, sigma.value


def SimplexEmbedding(states, effects, unit, mms, debug=False):
    """
    Constructs a noncontextual ontological model for the (possibly depolarized) GPT fragment.
    Tests whether a simplex embedding exists given the GPT's states, effects, unit, and maximally mixed state.
    

    Args:
    states (np.ndarray): A numpy array of states.
    effects (np.ndarray): A numpy array of effects.
    unit (np.ndarray): The unit effect.
    mms (np.ndarray): The maximally mixed state.
    debug (bool, optional): Flag to print debug information. Default is False.

    Returns:
    robustness (float): The robustness value.
    """

    inclusion_S, projection_S, accessibleFragmentStates = DefineAccessibleGPTFragment(
        states
    )
    inclusion_E, projection_E, accessibleFragmentEffects = DefineAccessibleGPTFragment(
        effects
    )
    accessibleFragmentUnit = projection_E @ unit
    # MMS: Maximally mixed state
    accessibleFragmentMMS = projection_S @ mms

    H_S = FindStateConeFacets(accessibleFragmentStates)
    H_E = FindEffectConeFacets(accessibleFragmentEffects)

    # Bilinear form giving the Born rule in the accessible fragment:
    accessibleFragmentBornRule = inclusion_E.T @ inclusion_S
    depolarizingMap = np.outer(accessibleFragmentUnit, accessibleFragmentMMS)
    robustness, sigma = SimplicialConeEmbedding(
        H_S, H_E, accessibleFragmentBornRule, depolarizingMap
    )
    return robustness
