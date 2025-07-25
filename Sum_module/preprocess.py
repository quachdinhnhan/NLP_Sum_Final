import nltk
from nltk.corpus import stopwords

from nltk.stem import WordNetLemmatizer
import string

class Preprocessor:
    def __init__(self, use_lemmatizer=True, language='english'):
        self.stop_words = set(stopwords.words(language))
        self.lemmatizer = WordNetLemmatizer() if use_lemmatizer else None
        # You can extend this later to support Stemmer or other methods

    def preprocess_text(self, text):
        # Remove punctuation
        text = text.translate(str.maketrans('', '', string.punctuation))
        # Convert to lowercase
        text = text.lower()
        # Tokenize the text into words
        words = text.split()
        # Remove stopwords and lemmatize words
        if self.lemmatizer:
            processed_words = [self.lemmatizer.lemmatize(word) for word in words if word not in self.stop_words]
        else:
            processed_words = [word for word in words if word not in self.stop_words]
        # Join back to a cleaned string
        return " ".join(processed_words)
    def preprocess_dict(self, sentences_dict):
        """
        sentences_dict: dict of sentence_id: sentence_text or dict data that contains 'sentence_text' field
        Returns a dictionary mapping sentence_id to processed text
        """
        # If input dict structure is {id: { 'sentence_text': str, ...}}
        # Use this:
        # text_dict = {sid: data['sentence_text'] for sid, data in sentences_dict.items()}
        # If input dict is directly {id: text}, comment line above and uncomment below:
        # text_dict = sentences_dict

        # Check if values are dicts or strings
        first_value = next(iter(sentences_dict.values()))
        if isinstance(first_value, dict) and 'sentence_text' in first_value:
            text_dict = {sid: data['sentence_text'] for sid, data in sentences_dict.items()}
        else:
            text_dict = sentences_dict  # assume it's already {id: text}

        processed_dict = {sid: self.preprocess_text(text) for sid, text in text_dict.items()}
        return processed_dict
    