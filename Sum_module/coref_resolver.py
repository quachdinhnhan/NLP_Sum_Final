from stanza.server import CoreNLPClient

class CorefResolver:
    def __init__(self, corenlp_path: str = None, memory: str = '4G', timeout: int = 60000):
        """
        Initialize the CorefResolver.

        Args:
            corenlp_path (str, optional): Path to Stanford CoreNLP directory. If None,
                relies on CORENLP_HOME env variable or default installation.
            memory (str): Java heap size for CoreNLP server (e.g., '4G').
            timeout (int): Timeout for annotation in milliseconds.
        """
        self.corenlp_path = corenlp_path
        self.memory = memory
        self.timeout = timeout
        self.client = None

    def __enter__(self):
        """Start CoreNLPClient upon entering context."""
        self.client = CoreNLPClient(
            annotators=['tokenize', 'ssplit', 'pos', 'lemma', 'ner', 'parse', 'coref'],
            memory=self.memory,
            timeout=self.timeout,
            corenlp_path=self.corenlp_path
        )
        self.client.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Shutdown CoreNLPClient upon exiting context."""
        if self.client:
            self.client.stop()

    def build_coref_resolved_text(self, ann):
        """
        Construct coreference-resolved text by replacing mentions with their representative mention.

        Args:
            ann: Stanford CoreNLP annotation object.

        Returns:
            str: Coreference-resolved text.
        """
        replace_map = {}

        for chain in ann.corefChain:
            rep_mention = chain.mention[chain.representative]
            rep_sent_idx = rep_mention.sentenceIndex
            rep_start = rep_mention.beginIndex
            rep_end = rep_mention.endIndex
            rep_tokens = ann.sentence[rep_sent_idx].token[rep_start:rep_end]
            rep_text = " ".join(token.word for token in rep_tokens)

            for mention in chain.mention:
                if mention == rep_mention:
                    continue
                key = (mention.sentenceIndex, mention.beginIndex, mention.endIndex)
                replace_map[key] = rep_text

        resolved_sentences = []

        for sent_idx, sentence in enumerate(ann.sentence):
            tokens = sentence.token
            resolved_tokens = []
            i = 0
            while i < len(tokens):
                replaced = False
                for (s_idx, start, end), rep_text in replace_map.items():
                    if s_idx == sent_idx and start == i:
                        resolved_tokens.append(rep_text)
                        i = end
                        replaced = True
                        break
                if not replaced:
                    resolved_tokens.append(tokens[i].word)
                    i += 1
            resolved_sentences.append(" ".join(resolved_tokens))

        return " ".join(resolved_sentences)

    def resolve(self, text: str) -> str:
        """
        Annotate the input text, perform coreference resolution, and return resolved text.

        Args:
            text (str): Input raw text.

        Returns:
            str: Coreference-resolved text.
        """
        if not self.client:
            raise RuntimeError("CoreNLPClient has not been started. Use the class as a context manager.")

        ann = self.client.annotate(text)
        return self.build_coref_resolved_text(ann)


# Usage example:
if __name__ == "__main__":
    sample_text = ("Barack was born in Hawaii. His wife Michelle was born in Milan. "
                   "He says that she is very smart.")

    # Optionally specify corenlp_path if necessary
    corenlp_dir = None  # e.g., '/Users/dindan/stanford-corenlp-4.5.10'

    with CorefResolver(corenlp_path=corenlp_dir) as resolver:
        resolved_text = resolver.resolve(sample_text)
        print("Original text:", sample_text)
        print("Coref resolved text:", resolved_text)
