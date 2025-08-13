from Sum_module.file_reader import FileReader
from Sum_module.parse_doc import ParseDoc
from Sum_module.preprocess import Preprocessor
from Sum_module.connections import ConnectionMatrix
from Sum_module.pagerank import PageRankCalculator
from Sum_module.summarizer import Summarizer
from Sum_module.output_writer import OutputWriter
from Sum_module.evaluation import Evaluator

import json
import os

file_name = 'd112h'
# Define paths for input files
input_file_path = f'Data/DUC_TEXT/test/{file_name}'
preference_file_path = f'Data/DUC_SUM/{file_name}'
# input_file_path = 'Data/DUC_TEXT/test/d112h'
# preference_file_path = 'Data/DUC_SUM/d112h'
print(f"Input File Path: {input_file_path}")
# Read the input document file
doc_file = FileReader(input_file_path).read_file()
# Parse the document to extract sentences and their metadata
# ParseDoc.parse_doc returns a dictionary with sentence_id as keys and metadata as values
sentences_dict = ParseDoc.parse_doc(doc_file)
print("Sentences Dict:", sentences_dict)

# preprocess the sentences for further analysis
preprocessor = Preprocessor(use_lemmatizer=True, language='english')
processed_sentence_text_dict = preprocessor.preprocess_dict(sentences_dict)
# print("Processed Sentences:", processed_sentence_text_dict)


# Create a connection matrix based on common words in sentences
connection_matrix = ConnectionMatrix(
    sentences=list(processed_sentence_text_dict.values()),
    min_common_words=4,
    max_common_words=10
).create_matrix()
# print ("Connection Matrix:", connection_matrix)

# Calculate PageRank scores based on the connection matrix
pagerank_calculator = PageRankCalculator(connection_matrix)
pagerank_scores = pagerank_calculator.calculator()
# print("PageRank Scores:", pagerank_scores)

# Create a summarizer instance to extract top sentences based on PageRank scores
summarizer = Summarizer(
    sentences_dict=sentences_dict,
    pagerank_scores=pagerank_scores,
    top_percent=0.1
)
summary_sentences = summarizer.get_summary_sentences()
summarizer.print_summary()

# Write the summary sentences to an output file
output_writer = OutputWriter(
    sentences_dict=sentences_dict,
    output_dir='output'
)
output_file_path = output_writer.write_summary(
    summary_sentence_ids=summarizer.get_top_sentence_ids(),
    input_file_path=input_file_path,
    suffix='_commonword'
)    

# Parse the preference summary file
preference_doc_file = FileReader(preference_file_path).read_file()
preference_sum_dict = ParseDoc.parse_doc(preference_doc_file)

# Evaluate the summary against the reference summary
evaluator = Evaluator(
    sentences_dict=sentences_dict,
    summary_sentence_ids=summarizer.get_top_sentence_ids(),
    preference_sum_dict=preference_sum_dict
)
evaluation_results = evaluator.evaluate()
print("Evaluation Results:", evaluation_results)
# # write evaluation results to a JSON file inlcuding filename and scores
# evaluation_output_path = 'output/evaluation_results.json'
# with open(evaluation_output_path, 'w', encoding='utf-8') as eval_file:
#     json.dump({
#         'file_name': file_name,
#         'evaluation_results': evaluation_results
#     }, eval_file, ensure_ascii=False, indent=4)

