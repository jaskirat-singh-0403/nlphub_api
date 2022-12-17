# NLP Hub - One stop for day-to-day NLP operations

This is the API for the project NLPHub which processes the requests from the main NLPHub website and returns respective results. It manages the following features:

## 1) Text Summarization:

For Text Summarization following is the basic workflow:

- ### Getting the Required Files
This includes English Core processor of SpaCy as well as Stopwords etc.

- ### Getting query data from text
Here the request is processed to extract the text to be summarized

- ### Fnding Maximum Frequency Words
The words who have the maximum frequency are found

- ### Generation of Sentence Scores
Sentences are generated and scored based on word frequency

- ### Generation of Summary
Sentences with highest scores are displayed as summary

## 2) Topic Modelling:

For Topic Modelling pretrained models known as Latent Dirichlet Allocation and NMF have been used.
Data is preprocessed by:
- Removing stopwords
- Removing Punctuations
- Parsing
- Lemmatization
- POS tagging
- Filtering based on POS tags

The results are returned back to the user.

## 3) Sentiment Analysis:

For Sentiment Analysis, VADER Sentiment Analyser has been used that automatically preprocesses the data and gives the sentiment result to the user.
