# Sum_module/output_writer.py
# OutputWriter module for writing summary sentences to a file.
import os

class OutputWriter:
    def __init__(self, sentences_dict, output_dir='output'):
        """
        Initialize the OutputWriter.
        
        Args:
            sentences_dict (dict): Dictionary mapping sentence_id to sentence metadata including
                                   'doc_id', 'wdcount', 'num', and 'sentence_text'.
            output_dir (str): Directory to save output files. Created if not exists.
        """
        self.sentences_dict = sentences_dict
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
    
    def write_summary(self, summary_sentence_ids, input_file_path, suffix='_commonword'):
        """
        Write the selected summary sentences to an output file.
        
        Args:
            summary_sentence_ids (list[int]): List of sentence IDs to write.
            input_file_path (str): Original input file path (used to derive output file name).
            suffix (str): Suffix to add to output file name (default '_commonword').
        
        Returns:
            str: The output file path that the summary is written to.
        """
        input_filename = os.path.splitext(os.path.basename(input_file_path))[0]
        output_file_path = os.path.join(self.output_dir, f'{input_filename}{suffix}')
        
        with open(output_file_path, 'w', encoding='utf-8') as outfile:
            for sentence_id in summary_sentence_ids:
                data = self.sentences_dict[sentence_id]
                doc_id = data.get('doc_id', 'unknown')
                wdcount = data.get('wdcount', '0')
                num = data.get('num', '0')
                sentence_text = data.get('sentence_text', '')
                # Write sentence in original tag format
                outfile.write(f'<s doc_id="{doc_id}" num="{num}" wdcount="{wdcount}"> {sentence_text}</s>\n')
        
        print(f"\nTop {len(summary_sentence_ids)} sentences written to {output_file_path}")
        return output_file_path
