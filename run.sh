#!/bin/bash
# Author: Chau Pham (chtmp223.github.io)
# Last modified: 2023-05
# Script to get loadings for each document
# Usage: sh run.sh

# Update the following variables to match your setup ---- 
MODEl_PATH="~/glove/glove.6B.50d.txt"                   # path to pretrained representation
TYPE="glove"                                            # 'glove', 'fastext' or 'word2vec'
FORMAT="txt"                                            # 'bin' or 'txt'
SEED_PATH="../data/dictionary/hero_villain/"            # path to directory containing .txt file of terms 
                                                        # (separated by whitespace)
DOC_PATH="../data/document/character_description.txt"   # path to directory containing .txt file of documents   
                                                        # (separated by newline)
OUTPUT_PATH="../data/output/"                           # path to directory to store result

# Placeholder: activate your virtual environment 
pip3 install --upgrade pip
pip3 install -r requirements.txt

# Run script ---- 
# This script will store document, concept representation, 
# missing words and loadings file in the designated output directory
python3 scripts/get_loadings.py $GLOVE $TYPE $FORMAT $SEED_PATH $DOC_PATH $OUTPUT_PATH