# DDR

This repository contains an updated implementation of the distributed dictionary representation (DDR) algorithm, which returns an estimation of the semantic similarity between a set of documents and latent constructs (seed words/dictionary). 


## Background
The DDR algorithm was first introduced by [Garten 2018, Dictionaries and distributions: Combining expert knowledge and large scale textual data content analysis : Distributed dictionary representation](https://pubmed.ncbi.nlm.nih.gov/28364281/). The original implementation can be found [here](https://github.com/USC-CSSL/DDR). 

The implementation in this repository is based on the original implementation, but has been modified to work with other types of pretrained embeddings (GloVe, FastText, Word2vec) and Python 3. Instead of the original `split()` method, `nltk` word tokenization method (`word_tokenize()`) is used to tokenize input documents. 


## Installation Guidelines
(Adopted from the original repository)

Currently, this package requires an installation of python 3, as well as the following dependencies:
```
numpy
nltk
pandas
gensim
```
To install these dependencies, you can use the following command: `pip install -r requirements.txt`


## Usage
To start, you should create a `data` folder to store all the necessary files for the DDR algorithm. In the `data` folder, there should be 3 subdirectories: `dictionary`, `document` and `output`. 
- `dictionary` should store .txt files that contain the seed words. Each dictionary dimension is represented by a separate text file and each text file contains the seed terms for that dimension (separated by whitespace). 
- `document` should store a .txt file that contain all documents (one document for each line).  
- `output` should be the destination directory storing the result of the DDR algorithm. 

Once you've set up the necessary folders and datasets, run the code stored in `run.sh`. You will need to modify the data paths and activate your virtual environment (optional) to match your setup. Then, assuming you are inside the `DDR` folder, run the script with `sh run.sh`. This script will store files containing document and concept representation, missing words, and final loadings in the designated `output` directory. 


## Reference
Garten J, Hoover J, Johnson KM, Boghrati R, Iskiwitch C, Dehghani M. Dictionaries and distributions: Combining expert knowledge and large scale textual data content analysis : Distributed dictionary representation. Behav Res Methods. 2018 Feb;50(1):344-361. doi: 10.3758/s13428-017-0875-9. PMID: 28364281.
