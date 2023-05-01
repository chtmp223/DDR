# Author: Chau Pham (chtmp223.github.io)
# Last modified: 2023-04-28
# Script to generate vector representation of seed words
# Usage: python concept_emb.py <model_path> <model_type> <model_format> <seed_path> <term_output_path> <oov_output_path>

import os
import sys
import numpy as np
import pandas as pd
from load_model import load_pretrained_model


def load_terms_dic(seed_path): 
    '''
    Given a path to a directory containing seed words,
    return a dictionary {file_name: [seed words]}
    '''
    terms_dic = {}
    for file in os.listdir(seed_path):
        with open(seed_path + file, 'r') as f:
            cat = file.split(".")[0]
            terms_dic[cat] = f.read().split()
    return terms_dic


def concept_emb(seed_path, pretrained_model, vocab, dim): 
    '''
    Return a dictionary {file_name: [seed words' embeddings]}
    a dictionary {file_name: [oov words]}
    '''
    concept = {}
    oov = {}
    terms_dic = load_terms_dic(seed_path)
    for topic in terms_dic: 
        topic_emb, oov_words = [], []
        for words in terms_dic[topic]:
            if words in vocab:
                topic_emb.append(pretrained_model[words])
            else: 
                oov_words.append(words)
        concept[topic] = np.mean(topic_emb, axis=0)
        oov[topic] = oov_words

    print("Finished loading seed words")
    return concept, oov


def main():
    model_path = sys.argv[1]
    model_type = sys.argv[2]
    model_format = sys.argv[3]
    seed_path = sys.argv[4]
    term_output_path = sys.argv[5]
    oov_output_path = sys.argv[6]

    model, vocab, dim = load_pretrained_model(model_path, model_type, model_format)
    concept, oov = concept_emb(seed_path, model, vocab, dim)

    concept_df = pd.DataFrame(concept).to_csv(term_output_path, index=False)
    oov_df = pd.DataFrame(oov).to_csv(oov_output_path, index=False)


if __name__ == "__main__":
    main()
