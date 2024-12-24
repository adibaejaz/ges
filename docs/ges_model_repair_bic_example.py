import sys
print("Using Python interpreter:", sys.executable)

import ges
import sempler
import numpy as np


def generate_starting_graphs(A):
    """
    Generates a sequence of adjacency matrices (numpy arrays) starting 
    from the all-zero matrix up to A, changing one entry at a time 
    (only where A has a non-zero value).
    
    The order of added edges is row-major: we iterate over rows (top to bottom)
    and columns (left to right), adding edges in that order.

    Parameters
    ----------
    A : numpy.ndarray
        Target adjacency matrix of shape (n, n).

    Returns
    -------
    list of numpy.ndarray
        A list of adjacency matrices, starting with the all-zero matrix
        and ending with A, each time adding exactly one non-zero entry.
    """
    # Ensure A is a numpy array
    A = np.array(A)
    n, m = A.shape
    assert n == m, "A must be a square matrix"
    
    # Create the all-zero matrix
    current = np.zeros_like(A)
    
    # Prepare a list to store the matrices in the sequence
    sequence = []
    
    # 1. Include the initial all-zero matrix
    sequence.append(current.copy())
    
    # 2. Find the non-zero positions in row-major order
    #    (row by row, then column by column).
    for i in range(n):
        for j in range(m):
            if A[i, j] != 0:
                # Add this edge in the current matrix
                current[i, j] = A[i, j]
                # Store a copy of the updated matrix in the sequence
                sequence.append(current.copy())
    
    return sequence


# Generate observational data from a Gaussian SCM using sempler
A = np.array([[0, 0, 1, 0, 0],
              [0, 0, 1, 0, 0],
              [0, 0, 0, 1, 1],
              [0, 0, 0, 0, 1],
              [0, 0, 0, 0, 0]])
W = A * np.random.uniform(1, 2, A.shape) # sample weights
data = sempler.LGANM(W,(1,2), (1,2)).sample(n=5000)

# Generate all hypothetical starting graphs
starting_graphs = generate_starting_graphs(A)
print(starting_graphs)
# Convert each DAG in starting_graphs to its CPDAG
cpdags = [ges.dag_to_cpdag(graph) for graph in starting_graphs]
print(cpdags)

# Run GES with the gaussian BIC score
estimate, score = ges.fit_bic(data, A)

print(estimate, score)

# Output
# [[0 0 1 0 0]
#  [0 0 1 0 0]
#  [0 0 0 1 1]
#  [0 0 0 0 1]
#  [0 0 0 1 0]] 21511.315220683457
