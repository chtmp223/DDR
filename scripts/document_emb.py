# Author: Chau Pham (chtmp223.github.io)
# Last modified: 2023-04-28
# Script to generate vector representation of documents
# Usage: python document_emb.py <model_path> <model_type> <model_format> <doc_path> <doc_output_path> <oov_output_path>

from nltk import word_tokenize
import numpy as np
import sys
import pandas as pd
from load_model import load_pretrained_model
        

def doc_emb(doc_path, pretrained_model, vocab, dim): 
    # Document representation ----
    with open(doc_path, 'r') as f: 
        document_lines = f.readlines()

    # Tokenize document and compute document embedding
    doc_emb, oov = [], []
    for line in document_lines: 
        line = word_tokenize(line)                        
        line_emb = []
        for word in line:
            word = word.lower().strip().replace("…", '')        # Clean up words
            if word in vocab:
                line_emb.append(pretrained_model[word])
            else: 
                oov.append(word)
        doc_emb.append(np.mean(line_emb, axis=0))

    print("Finished loading document")
    return doc_emb, oov


def main(): 
    model_path = sys.argv[1]
    model_type = sys.argv[2]
    model_format = sys.argv[3]
    doc_path = sys.argv[4]
    doc_output_path = sys.argv[5]
    oov_output_path = sys.argv[6]

    model, vocab, dim = load_pretrained_model(model_path, model_type, model_format)
    doc, oov = doc_emb(doc_path, model, vocab, dim)

    doc_df = pd.DataFrame(doc).to_csv(doc_output_path, index=False)
    oov_df = pd.DataFrame(oov).to_csv(oov_output_path, index=False)


if __name__ == "__main__":
    main()