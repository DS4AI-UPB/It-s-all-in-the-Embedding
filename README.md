# It's all in the Embedding! Fake News Detection using Document Embeddings

## Article:

Ciprian-Octavian Truică, Elena-Simona Apostol. *It's all in the Embedding! Fake News Detection using Document Embeddings*.  Mathematics, 11(3):1-29(508), ISSN 2227-7390, January 2023. DOI: [10.3390/math11030508](http://doi.org/10.3390/math11030508 )

## Code 

Packages needed:
- pandas
- numpy
- ntlk
- sklearn
- spacy
- glove_python
- mittens
- gensim
- tensorflow
- keras
- bert-for-tf2
- tensorflow-hub
- transformers 
- datasets
- simpletransformers
- sentence_transformers
- xgboost

The dataset is the corpus.csv. The header is: 
- id - a unique identifier for the news article
- content - the textual content of the news article
- label - the label of the news article: reliable or fake

Note: All the datasets whould have the same header if you plan to run the code on other datasets

### Models

Run all the models. Found in src directory.

Parameters:
- DIR_EMBS - directory with the document embedings (all the embeddings are needed for the code to run)
- NUM_ITER: number of iterations

Run: `` python3.7 models.py DIREMBS NUM_ITER``

### Create embeddings

Creates all the document embeddings. Found in src directory.

Parameters:
- FIN - corpus file in csv format with the header id, content, lable
- DIR_EMBS - directory where to save the document embedings
- USE_CUDA - 1 use CUDA for embeddings, 0 do not use CUDA for embedding

Run: ``python3.7 create_embeddings.py FIN DIROUT USE_CUDA``


### Statistics

Found in dataset_stats

dataset_stats.py - get the corpus statistics and unigrams and save into an output file (stats.csv)

Parameters:
- FIN - corpus file in csv format with the header id, content, lable
- FOUT - statistics file (also needed for topic modeling)

Run: ``python3.7 dataset_stats.py FIN FOUT``

### Topic Modeling

Found in dataset_stats


extract_topics.py - get the topics for the corpus, require the number of topics, number of terms, and the number of iterations. Use the stats.csv as input (the csv resulted from the statistics step)

Parameters:
- FOUT - statistics file (the result o the dataset_stats.py)
- NUM_TOPICS - number of topics
- NUM_KW - number of keywords
- NUM_ITER - number of iterations

Run: ``python3.7 extract_topics.py FOUT 1 10 10``
