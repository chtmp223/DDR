# Author: Chau Pham (chtmp223.github.io)
# Last modified: 2023-04-28
# Script to load pretrained embeddings (word2vec, GloVe, fasttext)
# Usage: python load_model.py <model_path> <model_type> <model_format> <model_output_path>

import sys
import numpy as np
import gensim
from gensim.models import KeyedVectors as kv
import io


def load_glove(model_path): 
    '''
    Load pretrained GloVe embeddings from model_path 
    Return a dictionary {word: embedding}, a list of words, 
    and the embedding dimension
    '''
    embeddings = {}
    with open(model_path, 'r') as f: 
        for line in f: 
            values = line.split() 
            word = values[0] 
            coefs = np.asarray(values[1:], dtype='float32') 
            embeddings[word] = coefs
    vocab = list(embeddings.keys())
    dim = len(embeddings[vocab[0]])
    return embeddings, list(embeddings.keys()), len(embeddings[vocab[0]])


def load_word2vec(model_path, binary):
    '''
    Load pretrained word2vec embeddings from model_path
    Return a dictionary {word: embedding}, a list of words,
    and the embedding dimension
    '''
    model = kv.load_word2vec_format(model_path, binary=binary)
    if gensim.__version__ < '4.0.0':
        return model, set(model.index2word), model.vector_size
    else: 
        return model, set(model.index_to_key), model.vector_size
    

def load_fasttext(model_path, binary):
    '''
    Load pretrained fasttext embeddings from model_path
    Return a dictionary {word: embedding}, a list of words,
    and the embedding dimension
    ''' 
    fin = io.open(model_path, 'r', encoding='utf-8', newline='\n', errors='ignore')
    n, d = map(int, fin.readline().split())
    data = {}
    for line in fin:
        tokens = line.rstrip().split(' ')
        data[tokens[0]] = map(float, tokens[1:])
    return data, list(data.keys()), d


def load_pretrained_model(model_path, model_type, binary): 
    '''
    Load pretrained embeddings from model_path
    '''
    if model_type in ["glove", "word2vec", "fasttext"]:
        if model_type == 'glove':
            emb, vocab, dim = load_glove(model_path)
        elif model_type == 'word2vec':
            emb, vocab, dim = load_word2vec(model_path, binary)
        elif model_type == 'fasttext':
            emb, vocab, dim = load_fasttext(model_path, binary)
        print('Loaded {} embeddings of {} dimensions'.format(model_type, dim))
    else: 
        raise ValueError('Invalid model type')
    
    return emb, vocab, dim


def main(): 
    model_path = sys.argv[1]
    model_type = sys.argv[2]        # 'glove', 'word2vec', 'fasttext'
    model_format = sys.argv[3]      # 'bin', 'txt'
    output_path = sys.argv[4]
    binary = model_format == 'bin'

    emb, vocab, dim = load_pretrained_model(model_path, model_type, binary)
    np.save(output_path, emb)

if __name__ == '__main__': 
    main()
