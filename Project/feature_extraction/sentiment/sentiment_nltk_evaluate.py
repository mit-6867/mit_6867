import re
import string
from prettytable import PrettyTable
import pandas as pd 
import pickle 
import nltk
import csv

classifier = pickle.load(open('naive_bayes_nltk.pkl', 'rb'))

def is_string(text):
    return isinstance(text, basestring)

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

MTurk_results = pd.read_csv('MTurk_Results.csv')
MTurk_results_with_indices = pd.read_csv('sample_articles_with_indices.csv')
MTurk_results_with_indices_merged = MTurk_results_with_indices.merge(MTurk_results, on=['text'])
MTurk_results_with_indices_merged['is_string'] = MTurk_results_with_indices_merged.text.apply(is_string)
MTurk_clean = MTurk_results_with_indices_merged[MTurk_results_with_indices_merged['is_string']]

all_words = set(word.lower() for passage in MTurk_clean['text'] for word in tokenize_2(passage))
articles = pd.read_csv('articles.csv')

articles_for_validation = articles.head(n=1000)

k = 0

def naive_bayes_nltk(text):
	global k 
	k += 1 
	print k 
	if isinstance(text, basestring):
		text_for_classification = {word: (word in tokenize_2(text)) for word in all_words}
		return classifier.classify(text_for_classification)
	else:
		return 'neutral'

articles_for_validation['naive_bayes_nltk'] = articles_for_validation.text.apply(naive_bayes_nltk)

articles_for_validation[['index', 'naive_bayes_nltk']].to_csv('naive_bayes_nltk.csv', index=False, quotechar='', quoting=csv.QUOTE_NONE, escapechar=' ', sep=',', encoding='utf8')