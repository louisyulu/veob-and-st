# Vector Embedding on Orthonormal Basis and Spectral Transform (VEOB & ST)

## Overview
This repository contains an implementation of a fast machine learning method that utilizes vector embedding on an orthonormal basis and spectral transform. The algorithm and mathematical principles are detailed in the **VEOB-and-ST.doc** file.

## Code Structure
The source code files are housed in the **examples** and **src** directories.

### Examples Directory
This directory contains Pluto notebook files:
- preprocess_imdb.jl - Prepares word and sentence data from original IMDB files.
- word_spelling_cos.jl - Embeds word orthography/spelling using SVD and DCT.
- word_ctx2_imdb.jl - Embeds word semantics/meanings with SVD.
- word_simple_en_dict.jl - Combines orthography and dictionary entries with synonyms and antonyms for semantic/meaning embedding.
- sent_ctx2_cos.jl - Embeds sentences with word vectors and DCT.
- query_imdb_chromadb.jl - Loads and queries word and sentence embeddings with ChromaDB.
- svd_cos_z_mnist.jl - Reduces MNIST data dimension with 2D DCT and SVD.
- svd_mnist_chromadb.jl - Loads and queries MNIST embedding with ChromaDB.
- knowledge_graph_chromadb.jl - Calculate the knowledge graph vectors embeddings and query with ChromaDB.

### Src Directory
This directory contains:
- partition.jl - Partitions data into clustered cells.
- my_utils.jl - Contains utility functions used in the notebooks.

## Installation Steps
Ensure Julia and Python are installed, then follow these steps:
1. Clone the repository:    
git clone https://github.com/louisyulu/veob-and-st.git
2. Navigate to the project folder and start Julia:     
cd veob-and-st    
julia --project=.
3. Switch to package mode and instantiate the required packages:     
]    
instantiate
4. Switch back to Julia mode, include the Conda package, then return to package mode again:    
[backspace key]    
using CondaPkg    
]
5. Install Python packages under the local conda environment, download the spacy language model, switch back to Julia mode, then exit:    
conda run pip install spacy chromadb deeplake    
conda run python -m spacy download en_core_web_sm    
[backspace key]    
exit()

## Dataset Preparation
1. Create data and text folders: 
mkdir data text
2. Download IMDB samples data from [here](https://ai.stanford.edu/~amaas/data/sentiment/) 
3. Unzip the data and move it into the text folder: `text/aclImdb`
4. Download the [json file](https://github.com/nightblade9/simple-english-dictionary/blob/main/processed/merged.json), rename it, and copy it into the file: `text/simple-english-dictionary.json`
5. Download the [csv file](https://github.com/resource-watch/graph/blob/master/import_db_csv_files/conceptEdges.csv) and move it to `text/conceptEdges.csv`

## Running a Program
1. Start Julia in the command line:    
julia --project=.
2. Import Pluto and run it:    
import Pluto    
Pluto.run()
3. When the browser screen appears, open a Pluto notebook file in the examples folder to run.

*Note: In the notebooks, one-time run cells can be triggered by checking the box in the previous cell. Uncheck it after running.*