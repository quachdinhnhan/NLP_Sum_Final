from Sum_module.file_reader import FileReader
from Sum_module.parse_doc import ParseDoc
from Sum_module.preprocess import Preprocessor
from Sum_module.tfidf_vectorizer import TFIDFVectorizer
from Sum_module.cosine_connector import CosineSimilarityConnector
from Sum_module.pagerank import PageRankCalculator
from Sum_module.summarizer import Summarizer
from Sum_module.output_writer import OutputWriter
from Sum_module.evaluation import Evaluator

import numpy as np
import json
import os

file_name = 'd112h'

# Define paths for input files
input_file_path = f'Data/DUC_TEXT/test/{file_name}'
preference_file_path = f'Data/DUC_SUM/{file_name}'

doc_file = FileReader(input_file_path).read_file()
sentences_dict = ParseDoc.parse_doc(doc_file)    

preprocessor = Preprocessor(use_lemmatizer=True, language='english')
processed_sentence_text_dict = preprocessor.preprocess_dict(sentences_dict)

tfidf_vectorizer = TFIDFVectorizer()

tfidf_matrix, idf, all_words = tfidf_vectorizer.transform(processed_sentence_text_dict)

cosine_connector = CosineSimilarityConnector(threshold=0.2)
connection_matrix = cosine_connector.cosine_similarity_matrix(tfidf_matrix)
np.fill_diagonal(connection_matrix, 0)  # Remove self-connections
# print("Connection Matrix:\n", connection_matrix)
print(type(connection_matrix))
pagerank_calculator = PageRankCalculator(connection_matrix)
pagerank_scores = pagerank_calculator.calculator()