# VEOB and ST

## Introduction
This repository provides implementation of a fast machine learning method with vector embedding on orthonormal basis and spectral transform. The algorithm and math mechanism is explained in the file **VEOB-and-ST.doc**.

## Source code files
Source code files are located under **examples** and **src** folders.

### examples folder contains Pluto notebook files
- preprocess_imdb.jl - prepare word and sentence data from the original IMDB files
- word_spelling_cos.jl - word orthography/spelling embedding using SVD and DCT
- word_ctx2_imdb.jl - word semantic/meaning with SVD
- word_simple_en_dict.jl - word semantic/meaning embedding combining orthography and dictionary entries with synonyms and antonyms
- sent_ctx2_cos.jl - sentence embedding with word vectors and DCT
- query_imdb_chromadb.jl - load and query word and sentence embeddings with ChromaDB
- svd_cos_z_mnist.jl - MNIST data dimension reduction with 2D DCT and SVD
- svd_mnist_chromadb.jl - load and query MNIST embedding with ChromaDB

### src folder
- partition.jl - data partition into clustered cells
- my_utils.jl - utility functions used in the notebook

## Setup steps
The prerequisite is Julia and Python languages are installed on the machine, then rin the following the steps:
1. clone the project repository
git clone https://github.com/louisyulu/veob-and-st.git
2. go to the project folder and start Julia
cd veob-and-st
julia --project=.
3. switch to package mode and instantiate the required packages
]
instantiate
4. switch back to Julia mode and include conda package, then back to package mode
[backspace key]
using CondaPkg
]
5. install the python packages under the local conda environment, download spacy language model, 
switch back to Julia mode, then exit
conda run pip install spacy chromadb deeplake
conda run python -m spacy download en_core_web_sm
[backspace key]
exit()
## Prepare dataset
1. create data and text folder
mkdir data text
2. download imdb samples data from https://ai.stanford.edu/~amaas/data/sentiment/ 
3. unzip the data and move it into text folder resulting path text/aclImdb
4. download the json file https://github.com/nightblade9/simple-english-dictionary/blob/main/processed/merged.json
rename and and copy it into the file text/simple-english-dictionary.json
## Start a program
1. start Julia in a command line
julia --project=.
2. import Pluto and run it
import Pluto
Pluto.run()
3. When the browser screen shows up, open a Pluto notebook file in examples folder to run

*In the notebooks, the one time run cell can be trigger by the checkbox in prevous cell, uncheck it after run*