import urllib.request
from flask import Flask
from flask import request
from flask import render_template
import numpy as np
import os
import spacy
from spacy.lang.en.stop_words import STOP_WORDS
from string import punctuation
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import string
from sklearn.decomposition import NMF, LatentDirichletAllocation, TruncatedSVD
from sklearn.feature_extraction.text import CountVectorizer
import warnings
warnings.filterwarnings('ignore')
import spacy
from spacy.lang.en.stop_words import STOP_WORDS
from spacy.lang.en import English

app = Flask(__name__, static_url_path='/static', static_folder='static/')


@app.route('/summarise', methods=['GET','POST'])
def my_form_post():
    data = request.args
    data =data.get("text")
    
    
    stopwords = list(STOP_WORDS)
    
    doc1=data
    nlp =spacy.load('en_core_web_sm')
    
    docx = nlp(doc1)
    
    mytokens = [token.text for token in docx]
    
    word_freq={}
    for word in docx:
        if word.text not in stopwords:
            if word.text not in word_freq.keys():
                word_freq[word.text]=1;
                
            else:
                word_freq[word.text]+=1;
                
    max_freq=max(word_freq.values())

    for word in word_freq:
        word_freq[word]=(word_freq[word]/max_freq)     
        
    sentence_list = [ sentence for sentence in docx.sents]
    
    [w.text.lower() for t in sentence_list for w in t ]
    
    
    sentence_scores={}
    for sentence in sentence_list:
        for word in sentence:
            if word.text in word_freq.keys():
                if len(sentence.text.split(' ')) < 30:
                    if sentence not in sentence_scores.keys():
                        sentence_scores[sentence] = word_freq[word.text]
                    else:
                        sentence_scores[sentence] += word_freq[word.text]
                        
                        
    from heapq import nlargest
    
    sum_sentences = nlargest(8, sentence_scores, key=sentence_scores.get)
    
    final_sentences = [ w.text for w in sum_sentences ] 
    
    
    summary = ' '.join(final_sentences)
    
    print(len(doc1))
    print(len(summary))
    return render_template('Summarisation.html', message = summary)

@app.route('/sentiment', methods=['GET','POST'])
def sentiment_vader():
    data = request.args
    sentence =data.get("text")
    # Create a SentimentIntensityAnalyzer object.
    sid_obj = SentimentIntensityAnalyzer()

    sentiment_dict = sid_obj.polarity_scores(sentence)
    negative = sentiment_dict['neg']
    neutral = sentiment_dict['neu']
    positive = sentiment_dict['pos']
    compound = sentiment_dict['compound']

    if sentiment_dict['compound'] >= 0.05 :
        overall_sentiment = "Positive"

    elif sentiment_dict['compound'] <= - 0.05 :
        overall_sentiment = "Negative"

    else :
        overall_sentiment = "Neutral"
  
    return render_template('Sentiment.html', message = overall_sentiment)

@app.route('/topic', methods=['GET','POST'])
def fun1():
    
    data = request.args
    sentence =data.get("text")
    punctuations = string.punctuation
    stopwords = list(STOP_WORDS)
    nlp=spacy.load("en_core_web_lg")
    parser=English()
    sentence=nlp(sentence)
    mytokens = parser(sentence)
    print([word.lemma_ for word in mytokens])
    mytokens = [ word.lemma_.lower().strip() if word.lemma_ != "-PRON-" else word.lower_ for word in mytokens ]
    print(mytokens)
    mytokens = [ word for word in mytokens if word not in stopwords and word not in punctuations ]
    
    sent = " ".join([i for i in mytokens])
    vectorizer = CountVectorizer(min_df=1, max_df=1, stop_words='english', lowercase=True, token_pattern='[a-zA-Z\-][a-zA-Z\-]{2,}')
    data_vectorized = vectorizer.fit_transform([sent])
    NUM_TOPICS = 10
    lda = LatentDirichletAllocation(n_components=NUM_TOPICS, max_iter=10, learning_method='online',verbose=True)
    data_lda = lda.fit_transform(data_vectorized)
    nmf = NMF(n_components=NUM_TOPICS)
    data_nmf = nmf.fit_transform(data_vectorized) 
    return render_template('Topics.html',message=[(vectorizer.get_feature_names()[i]).title() for i in lda.components_[0].argsort()[:-10 - 1:-1]]) 

    

if __name__ == '__main__':
    app.run()
