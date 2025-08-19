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

def process_file(file_name, base_text_dir='Data/DUC_TEXT/test', base_preference_dir='Data/DUC_SUM'):
    # Build full file paths
    input_file_path = os.path.join(base_text_dir, file_name)
    preference_file_path = os.path.join(base_preference_dir, file_name)

    # Read the input document file
    doc_file = FileReader(input_file_path).read_file()

    # Parse the document to extract sentences and their metadata
    sentences_dict = ParseDoc.parse_doc(doc_file)

    # Preprocess the sentences for further analysis
    preprocessor = Preprocessor(use_lemmatizer=True, language='english')
    processed_sentence_text_dict = preprocessor.preprocess_dict(sentences_dict)

    # Create a connection matrix based on common words in sentences
    connection_matrix = ConnectionMatrix(
        sentences=list(processed_sentence_text_dict.values()),
        min_common_words=4,
        # max_common_words=5000
    ).create_matrix()
    #----------------------------------------------------------------
    # Calculate PageRank scores based on the connection matrix
    pagerank_calculator = PageRankCalculator(connection_matrix)
    pagerank_scores = pagerank_calculator.calculator()
    #----------------------------------------------------------------
    # Create a summarizer instance to extract top sentences based on PageRank scores
    summarizer = Summarizer(
        sentences_dict=sentences_dict,
        pagerank_scores=pagerank_scores,
        top_percent=0.1
    )
    
    # summary_sentences = summarizer.get_summary_dict()
    summarizer.print_summary()
    #----------------------------------------------------------------
    # Write the summary sentences to an output file
    output_writer = OutputWriter(
        sentences_dict=sentences_dict,
        output_dir='output'
    )
    
    output_writer.write_summary(
        summary_sentence_ids=summarizer.get_top_sentence_ids(),
        input_file_path=input_file_path,
        suffix='_commonwords_test' 
    )
    # Parse the preference summary file
    preference_doc_file = FileReader(preference_file_path).read_file()
    preference_sum_dict = ParseDoc.parse_doc(preference_doc_file)
    # Create an Evaluator instance to evaluate the summary
    evaluator = Evaluator(
        sentences_dict=sentences_dict,
        summary_sentence_ids=summarizer.get_top_sentence_ids(),
        preference_sum_dict=preference_sum_dict
    )
    evaluation_results = evaluator.evaluate()
    # write evaluation results to a JSON file inlcuding filename and scores of each file in the same JSON file
    evaluation_output_path = 'output/evaluation_commonwords_test.json'
    # write or append evaluation results to the JSON file
    if os.path.exists(evaluation_output_path):
        with open(evaluation_output_path, 'r+', encoding='utf-8') as eval_file:
            data = json.load(eval_file)
            data[file_name] = evaluation_results
            eval_file.seek(0)
            json.dump(data, eval_file, ensure_ascii=False, indent=4)
    else:
        # If file does not exist, create it with the first entry
        with open(evaluation_output_path, 'w', encoding='utf-8') as eval_file:
            json.dump({file_name: evaluation_results}, eval_file, ensure_ascii=False, indent=4) 
    

def main():
    
    # file_names = [
    #     'd112h',
    #     'd113h',
    #     'd114h',
    #     'd115i',   
    #     'd116i',
    #     'd117i',
    #     'd118i',
    #     'd119i',
    #     'd120i',
    # ]
    test_dir = 'Data/DUC_TEXT/test'
    file_names = [f for f in os.listdir(test_dir) if os.path.isfile(os.path.join(test_dir, f))]
    file_names = sorted(file_names)
    print(file_names)

    for file_name in file_names:
        print(f"Processing file: {file_name}")
        process_file(file_name)
    
    

if __name__ == "__main__":
    main()