from stanza.server import CoreNLPClient

def build_coref_resolved_text(ann):
    # Create a mapping of (sentenceIdx, startIdx, endIdx) to representative mention text
    replace_map = {}

    for chain in ann.corefChain:
        # Find the representative mention (index 0 usually)
        rep_mention = chain.mention[chain.representative]
        rep_sent_idx = rep_mention.sentenceIndex
        rep_start = rep_mention.beginIndex
        rep_end = rep_mention.endIndex
        # Get representative mention text
        rep_tokens = ann.sentence[rep_sent_idx].token[rep_start:rep_end]
        rep_text = " ".join([token.word for token in rep_tokens])

        # For all mentions except the representative one, map to rep_text
        for mention in chain.mention:
            # Skip the representative mention itself
            if mention == rep_mention:
                continue
            key = (mention.sentenceIndex, mention.beginIndex, mention.endIndex)
            replace_map[key] = rep_text

    # Now, rebuild all sentences, replacing mentions where applicable
    resolved_sentences = []

    for sent_idx, sentence in enumerate(ann.sentence):
        tokens = sentence.token
        resolved_tokens = []
        i = 0
        while i < len(tokens):
            replaced = False
            # Check if any mention starts at this token index
            for (s_idx, start, end), rep_text in replace_map.items():
                if s_idx == sent_idx and start == i:
                    # Replace the tokens from start to end with rep_text
                    resolved_tokens.append(rep_text)
                    i = end  # Move index after the mention span
                    replaced = True
                    break
            if not replaced:
                resolved_tokens.append(tokens[i].word)
                i += 1
        # Join tokens in this sentence
        resolved_sentence = " ".join(resolved_tokens)
        resolved_sentences.append(resolved_sentence)

    # Join all sentences with space (or use original delimiters)
    return " ".join(resolved_sentences)

import os
#read the string from file
# with open("output/full_text_d112h", "r", encoding="utf-8") as f:
#     test_text = f.read()
test_text = "Sir Robert Clark, chairman of MGN, lists a series of transactions - some of which he stresses may have been perfectly legitimate - that took place in the past few months. He says legal action may be taken against a number of organisations, including Goldman Sachs, the US investment bank, over more than Pounds 40m in transfers from MGN if the bank was aware that they were 'effected for improper purposes."
# Usage example:
with CoreNLPClient(
    annotators=['tokenize', 'ssplit', 'pos', 'lemma', 'ner', 'parse', 'coref'],
    memory='8G',
    timeout=180000,
    endpoint='http://localhost:9000'  # or omit if letting client start server
) as client:
    # test_text = full_text_d112h
    ann = client.annotate(test_text)

    resolved_text = build_coref_resolved_text(ann)
    # print("Coreference resolved text:")
    # print(resolved_text)
    with open("output/test_coref", "w", encoding="utf-8") as f:
        f.write(test_text)
    print("Text has been written to d112h_coref")
