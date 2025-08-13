# nlp_module/evaluation.py
from typing import Dict, List, Tuple, Set

class Evaluator:
    def __init__(self, sentences_dict: Dict[int, Dict], summary_sentence_ids: List[int], preference_sum_dict: Dict[int, Dict]):
        """
        Initialize the Evaluator
        
        Args:
            sentences_dict (dict): Mapping sentence_id -> metadata including 'doc_id' and 'num'.
            summary_sentence_ids (list[int]): List of sentence IDs selected in the summary.
            preference_sum_dict (dict): Parsed reference summary dictionary with sentence metadata.
        """
        self.sentences_dict = sentences_dict
        self.summary_sentence_ids = summary_sentence_ids
        self.preference_sum_dict = preference_sum_dict
        
        # Precompute set of (doc_id, num) tuples in reference summary for quick lookup
        self.preference_keys: Set[Tuple[str, str]] = set(
            (v['doc_id'], v['num']) for v in self.preference_sum_dict.values()
        )

    def evaluate(self) -> Dict[str, float]:
        """
        Evaluate the summary against the reference summary.
        
        Returns:
            dict: {
                "matched": int, number of matched sentences,
                "recall": float (%),
                "precision": float (%),
                "f1": float (%)
            }
        """
        matched = 0
        for sid in self.summary_sentence_ids:
            sent = self.sentences_dict[sid]
            if (sent['doc_id'], sent['num']) in self.preference_keys:
                # print(f"Matched: {sent['doc_id']}, {sent['num']}, {sent['sentence_text']}")
                matched += 1

        pref_len = len(self.preference_sum_dict)
        sum_len = len(self.summary_sentence_ids)
        recall = (matched / pref_len) * 100 if pref_len > 0 else 0.0
        precision = (matched / sum_len) * 100 if sum_len > 0 else 0.0
        f1 = (
            (2 * recall * precision) / (recall + precision)
            if (recall + precision) > 0 else 0.0
        )
        
        # print(f"Number of matched sentences: {matched}")
        # print(f"Recall: {recall:.2f}%")
        # print(f"Precision: {precision:.2f}%")
        # print(f"F1 Score: {f1:.2f}%")
        
        # only return 2digit for readability
        recall = round(recall, 2)
        precision = round(precision, 2)
        f1 = round(f1, 2)
        return {
            "labeled": pref_len,
            "extracted": len(self.summary_sentence_ids),
            "matched": matched,
            "recall": recall,
            "precision": precision,
            "f1": f1
        }
