import numpy as np

class CosineSimilarityConnector:
    def __init__(self, threshold=0.2):
        """
        Initialize with a cosine similarity threshold for connection.
        """
        self.threshold = threshold
        self.similarity_matrix = None
        self.connection_matrix = None

    def cosine_similarity_matrix(self, tfidf_matrix):
        """
        Calculate the cosine similarity matrix from TF-IDF vectors.
        Args:
            tfidf_matrix (np.ndarray): TF-IDF matrix (n_sentences, n_features)
        Returns:
            np.ndarray: Cosine similarity matrix (n_sentences, n_sentences)
        """
        # Normalize each row (sentence vector) to unit length 
        # computes the L2 norm (Euclidean length) of each row vector.
        norm = np.linalg.norm(tfidf_matrix, axis=1, keepdims=True)
        # Avoid division by zero
        # Some sentences might have all-zero TF-IDF values (e.g., empty or stopword-only sentences), resulting in a norm of 0
        norm[norm == 0] = 1
        normalized_matrix = tfidf_matrix / norm
        # Cosine similarity is the dot product of normalized vectors
        similarity = np.dot(normalized_matrix, normalized_matrix.T)
        self.similarity_matrix = similarity
        return similarity

    def create_connection_matrix(self, tfidf_matrix):
        """
        Create a boolean connection matrix based on cosine similarity threshold.
        Args:
            tfidf_matrix (np.ndarray): TF-IDF matrix (n_sentences, n_features)
        Returns:
            np.ndarray: Connection matrix (n_sentences, n_sentences)
        """
        similarity = self.cosine_similarity_matrix(tfidf_matrix)
        connection = (similarity > self.threshold).astype(int)
        np.fill_diagonal(connection, 0)  # Remove self-connections
        self.connection_matrix = connection
        return connection
