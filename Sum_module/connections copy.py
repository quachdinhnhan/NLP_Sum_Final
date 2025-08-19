# Sum_module/connections.py
# This module defines the ConnectionMatrix class to create a connection matrix based on sentence similarities.
import re
import numpy as np
import math

class ConnectionMatrix:
    def __init__(self, sentences, min_common_words=4, max_common_words=1000, weighted=False):
        """
        Initialize the ConnectionMatrix class.
        
        Args:
            sentences (list of str): List of preprocessed sentence texts.
            min_common_words (int): Minimum number of common words required for connection.
            max_common_words (int): Maximum number of common words allowed for connection.
        """
        self.sentences = sentences
        self.min_common_words = min_common_words
        self.max_common_words = max_common_words
        self.weighted = weighted
        self.matrix = None

    def similarity_score(self, sent1, sent2):
        """
        Computes the weighted similarity between two sentences
        using the formula:
            |common words| / (log(|sent1|) + log(|sent2|))
        """
        words1 = set(re.findall(r'\b\w+\b', sent1.lower()))
        words2 = set(re.findall(r'\b\w+\b', sent2.lower()))
        num_common = words1.intersection(words2)
        if num_common == 0 or len(words1) == 0 or len(words2) == 0:
            return 0.0
        denominator = math.log(len(words1)) + math.log(len(words2))
        if denominator == 0:
            return 0.0
        return num_common / denominator

    def has_connection(self, sentence1, sentence2):
        """
        Check if two sentences have a connection based on common words.
        
        Args:
            sentence1 (str): The first sentence.
            sentence2 (str): The second sentence.
            
        Returns:
            bool: True if sentences have connection, else False.
        """
        words1 = set(re.findall(r'\b\w+\b', sentence1.lower()))
        words2 = set(re.findall(r'\b\w+\b', sentence2.lower()))
        
        common_words = words1.intersection(words2)
        # check if the lenghth of common words is equal to lenght of words1 or words2
        if len(common_words) == len(words1):
            return False
        return self.min_common_words <= len(common_words) <= self.max_common_words

    
    
    def create_matrix(self):
        """
        Create a symmetric connection matrix where matrix[i][j] is True if sentences i and j connect.
        
        Returns:
            np.ndarray: matrix of shape (n, n).
        """
        n = len(self.sentences)
        dtype = float if self.weighted else int
        matrix = np.zeros((n, n), dtype=dtype)

        for i in range(n):
            for j in range(i + 1, n):
                if self.weighted:
                    score = self.similarity_score(self.sentences[i], self.sentences[j])
                    if score > 0:
                        matrix[i][j] = score
                        matrix[j][i] = score
                else:
                    connection = self.has_connection(self.sentences[i], self.sentences[j])
                    matrix[i][j] = int(connection)  # 0 or 1
                    matrix[j][i] = int(connection)
        
        self.matrix = matrix
        return matrix
