import re
import string
from prettytable import PrettyTable
from sh import find
import pandas as pd 
import numpy as np
import math 
import nltk
from nltk.tokenize import word_tokenize
import pickle

def remove_punctuation(s):
    "see http://stackoverflow.com/questions/265960/best-way-to-strip-punctuation-from-a-string-in-python"
    table = string.maketrans("","")
    return s.translate(table, string.punctuation)

def tokenize(text):
    dictionary = {}
    text = remove_punctuation(text)
    text = text.lower()
    listy = re.split("\W+", text)
    for i in listy:
        dictionary[i] = True 
    return dictionary

def tokenize_2(text):
    text = remove_punctuation(text)
    text = text.lower()
    return re.split("\W+", text)

def count_words(words):
    wc = {}
    for word in words:
        wc[word] = wc.get(word, 0.0) + 1.0
    return wc

def is_string(text):
    return isinstance(text, basestring)

MTurk_results = pd.read_csv('MTurk_Results.csv')
MTurk_results_with_indices = pd.read_csv('sample_articles_with_indices.csv')
MTurk_results_with_indices_merged = MTurk_results_with_indices.merge(MTurk_results, on=['text'])
MTurk_results_with_indices_merged['is_string'] = MTurk_results_with_indices_merged.text.apply(is_string)
MTurk_clean = MTurk_results_with_indices_merged[MTurk_results_with_indices_merged['is_string']]


# setup some structures to store our data
vocab = {}
word_counts = {
    "negative": {},
    "positive": {},
    "neutral": {}
}
priors = {  
    "negative": 0.,
    "positive": 0.,
    "neutral": 0.
}

docs = []
all_words = set(word.lower() for passage in MTurk_clean['text'] for word in tokenize_2(passage))

training = []
for i in MTurk_results_with_indices_merged['index']:

    if not np.isnan(i):

        f = open('../../text_scraping/article_text/' + str(int(i)) + '.txt', 'r')
        text = f.read()
        if MTurk_results_with_indices_merged[MTurk_results_with_indices_merged['index'] == i]['Avg'][j] > 0.5:
            category = 'positive'
        elif MTurk_results_with_indices_merged[MTurk_results_with_indices_merged['index'] == i]['Avg'][j] < -0.5:
            category = 'negative'
        else:
            category = 'neutral'
    
        if category == 'positive':
            training.append((({word: (word in tokenize_2(text)) for word in all_words}), 'positive'))
        elif category == 'negative':
            training.append((({word: (word in tokenize_2(text)) for word in all_words}), 'negative'))
        else:
            training.append((({word: (word in tokenize_2(text)) for word in all_words}), 'neutral'))

classifier = nltk.NaiveBayesClassifier.train(training)
pickle.dump( classifier, open( "naive_bayes_nltk.pkl", "wb" ) )


