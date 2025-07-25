# Sum_module/pagerank.py
# This module defines the PageRankCalculator class to compute PageRank scores based on a connection matrix
import numpy as np

class PageRankCalculator:
    def __init__(self, connection_matrix, damping=0.85, max_iterations=100, tolerance=1e-6):
        """
        Initialize the PageRank calculator.
        
        Args:
            connection_matrix (np.ndarray or list of lists): Adjacency or connection matrix (square).
            damping (float): Damping factor, usually 0.85.
            max_iterations (int): Maximum number of iterations to run PageRank.
            tolerance (float): Threshold for convergence (L1 norm difference).
        """
        self.connection_matrix = np.array(connection_matrix, dtype=float)
        self.damping = damping
        self.max_iterations = max_iterations
        self.tolerance = tolerance
        self.num_nodes = self.connection_matrix.shape[0]
        self.transition_matrix = self._build_transition_matrix()
        self.pagerank_scores = np.ones(self.num_nodes)  # Initial scores

    def _build_transition_matrix(self):
        """Build the stochastic transition matrix from the connection matrix."""
        row_sums = np.sum(self.connection_matrix, axis=1)
        transition = np.zeros_like(self.connection_matrix, dtype=float)

        for i in range(self.num_nodes):
            if row_sums[i] > 0:
                transition[i, :] = self.connection_matrix[i, :] / row_sums[i]
            # If a row sums to zero, that row remains zero (dangling node)

        return transition

    def calculator(self):
        """
        Run the PageRank algorithm until convergence or max iterations.

        Returns:
            np.ndarray: Final PageRank scores.
        """
        for iteration in range(self.max_iterations):
            new_scores = (1 - self.damping) / self.num_nodes \
                         + self.damping * self.transition_matrix.T @ self.pagerank_scores

            if np.linalg.norm(new_scores - self.pagerank_scores, ord=1) < self.tolerance:
                print(f"PageRank converged after {iteration + 1} iterations.")
                break
            self.pagerank_scores = new_scores

        return self.pagerank_scores
