import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import string


class Preprocessor:
    def __init__(self, use_lemmatizer=True, language='english'):
        self.stop_words = set(stopwords.words(language))
        self.lemmatizer = WordNetLemmatizer() if use_lemmatizer else None
    
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
        sentences_dict: dict of sentence_id: dict containing 'sentence_text' field
        Returns a dictionary mapping sentence_id to processed text
        """
        #Extract sentence_text from each sentence dict
        text_dict = {sid: data['sentence_text'] for sid, data in sentences_dict.items()}

        #Preprocess each sentence text
        processed_dict = {sid: self.preprocess_text(text) for sid, text in text_dict.items()}
        return processed_dict