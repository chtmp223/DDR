# DDR

This repository contains my implementation of the distributed dictionary representation (DDR) algorithm, which returns an estimation of the semantic similarity between a set of documents and latent constructs. 


## Background
The DDR algorithm was first introduced by [Garten 2018, Dictionaries and distributions: Combining expert knowledge and large scale textual data content analysis : Distributed dictionary representation](https://pubmed.ncbi.nlm.nih.gov/28364281/). The original implementation can be found [here](https://github.com/USC-CSSL/DDR). The implementation in this repository is based on the original implementation, but has been modified to work with other types of pretrained embeddings (e.g. GloVe, FastText, etc.) and Python 3.


## Installation Guidelines
(Adopted from the original repository)

Currently, this package requires an installation of python 3, as well as the following dependencies:
```
numpy
nltk
pandas
gensim (>= 3.0.0)
```
To install these dependencies, you can use the following command: `pip install -r requirements.txt`


## Usage
To get the loadings for each document, please refer to `run.sh`. You will need to modify the data paths and activate your virtual environment (optional) to match your setup. Finally, assuming you are inside the `DDR` folder, run the script with `sh run.sh`.

If you want to get a csv file for document/concept embeddings, please run `document_emb.py` and `concept_emb.py` individually. Refer to the usage comments in the files for more details. 


## Reference
Garten J, Hoover J, Johnson KM, Boghrati R, Iskiwitch C, Dehghani M. Dictionaries and distributions: Combining expert knowledge and large scale textual data content analysis : Distributed dictionary representation. Behav Res Methods. 2018 Feb;50(1):344-361. doi: 10.3758/s13428-017-0875-9. PMID: 28364281.
