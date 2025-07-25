# Sum_module/summarizer.py
# Summarizer module for extracting summary sentences based on PageRank scores.

class Summarizer:
    def __init__(self, sentences_dict, pagerank_scores, top_percent=0.1):
        """
        Initialize the Summarizer.

        Args:
            sentences_dict (dict): Dictionary of sentence_id -> sentence metadata (includes 'sentence_text').
                                   Should be ordered or index-accessible by sentence_id.
            pagerank_scores (list or np.ndarray): List/array of PageRank scores corresponding to sentences.
            top_percent (float): Fraction of top sentences to include in summary (e.g., 0.1 for top 10%).
        """
        self.sentences_dict = sentences_dict
        self.pagerank_scores = pagerank_scores
        self.top_percent = top_percent

    def get_top_sentence_ids(self):
        """
        Get the sentence IDs with top PageRank scores.

        Returns:
            List[int]: List of sentence IDs for the top scoring sentences.
        """
        num_sentences = len(self.pagerank_scores)
        top_n = max(1, int(self.top_percent * num_sentences))  # At least one sentence
        # Sort indices by score descending and take top_n
        sorted_indices = sorted(range(num_sentences), key=lambda i: self.pagerank_scores[i], reverse=True)
        return sorted_indices[:top_n]

    def get_summary_sentences(self):
        """
        Get the summary sentences as a list of strings.

        Returns:
            List[str]: List of top sentences (by sentence_text).
        """
        top_sentence_ids = self.get_top_sentence_ids()
        # Return sentences' text in selection order
        return [self.sentences_dict[i]['sentence_text'] for i in top_sentence_ids]

    def print_summary(self):
        """
        Print the summary sentences with their IDs and scores.
        """
        # top_sentence_ids = self.get_top_sentence_ids()
        # print("\nTotal Summary Sentences:", len(top_sentence_ids))
        # print("Top Percent:", self.top_percent * 100, "%")
        # print("Summary Sentences:")
        # for i in top_sentence_ids:
        #     print(f"Sentence ID {i}: {self.pagerank_scores[i]:.8f} - {self.sentences_dict[i]['sentence_text']}")
        print("\nSummary Complete.")
        return self.get_summary_sentences()