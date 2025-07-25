import re

class ParseDoc:
    def __init__(self,doc_file):
        self.doc_file = doc_file

    # Create a dictionary save all the metadata of sentences
    def parse_doc (doc_file):
        """
        Parse the document file and extract sentences metadata.
        
        Args:
            doc_file (str): The content of the document file.
            
        Returns:
            dict: A dictionary with sentence_id as keys and metadata as values.
        """
        sentences_dict = {}
        # Create unique sentence_id
        # Initialize sentence_id to 0
        sentence_id = 0

        # Find all sentences tags and their content in the document
        sentence_matches = re.findall(r'<s\s+docid="([^"]+)"\s+num="([^"]+)"\s+wdcount="([^"]+)">\s*(.*?)\s*</s>', doc_file, re.DOTALL)

        for doc_id, num, wdcount, sentence_text in sentence_matches:
            sentences_dict[sentence_id] = {
                "doc_id": doc_id,
                "num": num,
                "wdcount": int(wdcount),
                "sentence_text": sentence_text.strip()
            }
            sentence_id += 1    
        return sentences_dict