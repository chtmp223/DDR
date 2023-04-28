#!/bin/bash
# Author: Chau Pham (chtmp223.github.io)
# Last modified: 2023-04-28
# Script to get loadings for each document
# Usage: sh run.sh

# Update the following variables to match your setup
GLOVE="~/glove/glove.6B.50d.txt"
TYPE="glove"
FORMAT="txt"
SEED_PATH="../data/dictionary/hero_villain/"
DOC_PATH="../data/document/character_description.txt"
OUTPUT_PATH="../data/output/glove_loadings.csv"

# Placeholder: activate your virtual environment
pip3 install --upgrade pip
pip3 install -r requirements.txt
python3 scripts/get_loadings.py $GLOVE $TYPE $FORMAT $SEED_PATH $DOC_PATH $OUTPUT_PATH