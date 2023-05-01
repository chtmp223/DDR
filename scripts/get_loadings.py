# Author: Chau Pham (chtmp223.github.io)
# Last modified: 2023-04-28
# Script to calculate the loadings of each document for each concept
# Usage: python3 get_loadings.py <model_path> <model_type> <model_format> <doc_emb_path> <concept_emb_path> <output_path>

import numpy as np
import sys
import pandas as pd
from load_model import load_pretrained_model
from concept_emb import concept_emb
from document_emb import doc_emb


def get_loadings(doc_emb, concept_emb): 
    # Compute similarity between each document and concepts
    similarity = {}
    for topic in concept_emb: 
        topic_sim = []
        for doc in doc_emb: 
            doc_term_sim = np.dot(doc, concept_emb[topic]) / (np.linalg.norm(doc) * np.linalg.norm(concept_emb[topic]))
            topic_sim.append(doc_term_sim)
        similarity[topic] = topic_sim
    return similarity


def main(): 
    model_path = sys.argv[1]
    model_type = sys.argv[2]
    model_format = sys.argv[3]
    seed_path = sys.argv[4]
    doc_path = sys.argv[5]
    output_path = sys.argv[6]

    # Getting results ----
    model, vocab, dim = load_pretrained_model(model_path, model_type, model_format)
    doc, oov_doc = doc_emb(doc_path, model, vocab, dim)
    concept, oov_concept = concept_emb(seed_path, model, vocab, dim)
    similarity = get_loadings(doc, concept)

    # Writing results ----
    pd.DataFrame(doc).to_csv(output_path + "doc_emb.csv", index=False)
    pd.DataFrame(concept).to_csv(output_path + "concept_emb.csv", index=False)
    pd.DataFrame(oov_doc).to_csv(output_path + "oov_doc.csv", index=False)
    pd.DataFrame(oov_concept).to_csv(output_path + "oov_concept.csv", index=False)
    pd.DataFrame(similarity).to_csv(output_path + "loadings.csv", index=False)

    print("Finished calculating loadings!")

if __name__ == "__main__":
    main()
    

