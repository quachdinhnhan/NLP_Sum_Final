import numpy as np

class TFIDFVectorizer:
    """
    A simple TF-IDF Vectorizer for sentence-level features
    """

    def __init__(self):
        self.word_index = {}
        self.idf = {}
        self.all_words = []

    def transform(self, processed_sentence_text_dict):
        """
        Fit the TF-IDF model and return the TF-IDF matrix and supporting dicts.
        Args:
            processed_sentence_text_dict: dict of {sentence_id: preprocessed_text}
           
        Returns:
            tf_idf_matrix: np.ndarray, shape (num_sentences, num_words)
            word_index: dict mapping word to col index in tfidf matrix
            idf (dict): inverse document frequency for each word
        """
        
        # Step 1: Compute tf for each sentence
        tf_dict = {}
        for sentence_id, text in processed_sentence_text_dict.items():
            words = text.split()
            tf = {}
            for word in words:
                tf[word] = tf.get(word, 0) + 1
            total_words = len(words)
            if total_words > 0:
                for word in tf:
                    tf[word] /= total_words
            tf_dict[sentence_id] = tf

        # Step 2: Compute document frequency (df) for each word_ in this project df is sentence freq
        df = {}
        for tf in tf_dict.values():
            for word in tf:
                df[word] = df.get(word, 0) + 1

        # Step 3: Compute idf
        N = len(processed_sentence_text_dict)
        idf = {}
        for word, freq in df.items():
            idf[word] = np.log(N / freq)
        self.idf = idf

        # Step 4: Compute TF-IDF per sentence
        tf_idf_sentence_dict = {}
        for sentence_id, tf in tf_dict.items():
            tf_idf = {}
            for word, tf_value in tf.items():
                tf_idf[word] = tf_value * idf[word]
            tf_idf_sentence_dict[sentence_id] = tf_idf

        # Step 5: Build vocabulary and index
        all_words = set()
        for tfidf in tf_idf_sentence_dict.values():
            all_words.update(tfidf.keys())
        all_words = sorted(all_words)
        self.all_words = all_words

        word_index = {word: idx for idx, word in enumerate(all_words)}
        self.word_index = word_index

        # Step 6: Build TF-IDF matrix
        num_sentences = len(tf_idf_sentence_dict)
        num_words = len(all_words)
        tf_idf_matrix = np.zeros((num_sentences, num_words))

        
        # Fill the TF-IDF matrix
        for sent_id, tfidf in tf_idf_sentence_dict.items():
            
            for word, value in tfidf.items():
                idx = word_index[word]
                tf_idf_matrix[sent_id, idx] = float(value)

        return tf_idf_matrix, word_index, idf

    