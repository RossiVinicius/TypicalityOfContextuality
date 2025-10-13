# TypicalityOfContextuality
This repository contains the Python code accompanying the manuscript "Typicality of contextuality certification in prepare and measure scenarios" [].

The code provides tools for studying how often quantum contextuality is present in prepare-and-measure scenarios by performing large-scale numerical sampling of random quantum states and measurements, and assessing their simplex embeddability.

The main features include:
- Random sampling of quantum states (pure/mixed) and measurements (projective/POVMs);
- Simplex embedding assessment to certify contextuality;
- Parallelized computation for efficient large-scale sampling;
- Typicality analysis across various parameter regimes;
- Statistical analysis with confidence intervals.

# Requirements
This code requires Python 3.7+, as well as the packages `numpy`, `qutip`, `tqdm` and `multiprocessing`.
Additionally, it requires the implementation of the Linear Program introduced in Physical Review Letters 132 (5), 050202 (2024), available at <https://github.com/pjcavalcanti/SimplexEmbeddingGPT>. Please follow the installation guidelines provided in that repository and make sure that the functions `fromListOfMatrixToListOfVectors` and `SimplexEmbedding` are working properly.

# Installation
  1. Clone the repository containing the linear program implementation for testing simplex-embedability:
     ```
     git clone https://github.com/pjcavalcanti/SimplexEmbeddingGPT.git
     ```
  2. Install the dependencies for the linear program:
     ```
      pip install numpy scipy cvxpy itertools pycddlib
     ```
     Please follow troubleshoot instructions provided in the README of SimplexEmbeddingGPT.git for a proper installation of `pycddlib`. Make sure that the functions `fromListOfMatrixToListOfVectors` and `SimplexEmbedding` are properly working. Follow the README for example usage.
  3. Clone the present repository:
    ```
    git clone https://github.com/RossiVinicius/TypicalityOfContextuality.git
    ```
  4. Install the remaining dependencies:
     ```
     pip install qutip tqdm multiprocessing
     ```

  In order to reproduce the computations in the paper, please also consider modifying the code provided in SimplexEmbeddingGPT.git accordingly:
  - In the file `mathtools.py`, replace the line `import cdd` to
     ```
     import cdd.gmp as cdd
     ```
  - In the file `mathtools.py`, modify the function `FindStateConeFacets` to the following:
     ```
    def FindStateConeFacets(S):
        S = np.array([[Rational(x).limit_denominator() for x in row] for row in S])
        C = cdd.matrix_from_array(S.T)
        C.rep_type = cdd.RepType.GENERATOR
        H_S=cdd.copy_inequalities(cdd.polyhedron_from_matrix(C))
        H_S=np.array(H_S.array)
        H_S[np.abs(H_S) < 1e-8] = 0
        return H_S
     ```
  - In the file `mathtools.py`, modify the function `FindEffectConeFacets` to the following:
     ```
    def FindEffectConeFacets(E):
        E = np.array([[Rational(x).limit_denominator() for x in row] for row in E])
        C = cdd.matrix_from_array(E.T)
        C.rep_type = cdd.RepType.INEQUALITY
        H_E=np.array(cdd.copy_generators(cdd.polyhedron_from_matrix(C)).array).T
        H_E[np.abs(H_E) < 1e-8] = 0    
        return H_E
     ```
     These steps ensure that the polytope description is done symbolically, increasing stability at the expense of performance.
   - In the file `simplexEmbedding.py`, modify the function `SimplicialConeEmbedding` to the following:
     ```
      def SimplicialConeEmbedding(H_S, H_E, accessibleFramentBornRule, depolarizingMap):
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
     ```
     This ensures that the main solver for the linear program will be ECOS, with an alternative try with CLARABEL. If none of the solvers manages to resolve the program, it will return a robustness of contextuality `r = 2`, which will later be ignore when computing typicality frequencies.

# Usage
     
