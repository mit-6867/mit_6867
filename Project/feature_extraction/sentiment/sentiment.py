import re
import string
from prettytable import PrettyTable
from sh import find
import pandas as pd 
import numpy as np
import math 
import csv

def remove_punctuation(s):
    "see http://stackoverflow.com/questions/265960/best-way-to-strip-punctuation-from-a-string-in-python"
    table = string.maketrans("","")
    return s.translate(table, string.punctuation)

def tokenize(text):
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

j = 0
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
            docs.append(('positive', i))
        elif category == 'negative':
            docs.append(('negative', i))
        else:
            docs.append(('neutral', i))
    
        priors[category] += 1
        words = tokenize(text)
        counts = count_words(words)
        for word, count in counts.items():
            # if we haven't seen a word yet, let's add it to our dictionaries with a count of 0
            if word not in vocab:
                vocab[word] = 0.0 # use 0.0 here so Python does "correct" math
            if word not in word_counts[category]:
                word_counts[category][word] = 0.0
            vocab[word] += count
            word_counts[category][word] += count

    j += 1

print docs 
print priors
print word_counts

articles = pd.read_csv('articles.csv')

k = 0

def naive_bayes(text):
    if isinstance(text, basestring):
        words = tokenize(text)
        counts = count_words(words)
    
        prior_positive = (priors['positive'] / sum(priors.values()))
        prior_negative = (priors['negative'] / sum(priors.values()))
        prior_neutral = (priors['neutral'] / sum(priors.values()))
    
        log_prob_positive = 0.0
        log_prob_negative = 0.0
        log_prob_neutral = 0.0
    
        for w, cnt in counts.items():
        # skip words that we haven't seen before, or words less than 3 letters long
            if not w in vocab or len(w) <= 3:
                continue
            # calculate the probability that the word occurs at all
            p_word = vocab[w] / sum(vocab.values())
            # for both categories, calculate P(word|category), or the probability a 
            # word will appear, given that we know that the document is <category>
            p_w_given_positive = word_counts["positive"].get(w, 0.0) / sum(word_counts["positive"].values())
            p_w_given_negative = word_counts["negative"].get(w, 0.0) / sum(word_counts["negative"].values())
            p_w_given_neutral = word_counts["neutral"].get(w, 0.0) / sum(word_counts["neutral"].values())
        # add new probability to our running total: log_prob_<category>. if the probability 
        # is 0 (i.e. the word never appears for the category), then skip it
        if 'p_w_given_positive' not in locals() and 'p_w_given_negative' not in locals() and 'p_w_given_neutral' not in locals():
            global k 
            k += 1 
            print k
            return 'neutral'
        else:
            if p_w_given_positive > 0:
                log_prob_positive += math.log(cnt * p_w_given_positive / p_word)
            if p_w_given_negative > 0:
                log_prob_negative += math.log(cnt * p_w_given_negative / p_word)
            if p_w_given_neutral > 0:
                log_prob_neutral += math.log(cnt * p_w_given_neutral / p_word)
        
            global k
            k += 1 
            print k 
    
            if max([math.exp(log_prob_positive + math.log(prior_positive)), math.exp(log_prob_negative + math.log(prior_negative)), math.exp(log_prob_neutral + math.log(prior_neutral))]) == math.exp(log_prob_positive + math.log(prior_positive)):
                return 'positive'
            elif max([math.exp(log_prob_positive + math.log(prior_positive)), math.exp(log_prob_negative + math.log(prior_negative)), math.exp(log_prob_neutral + math.log(prior_neutral))]) == math.exp(log_prob_negative + math.log(prior_negative)):
                return 'negative'
            else:
                return 'neutral'
    else:
        return 'neutral'

articles['naive_bayes'] = articles.text.apply(naive_bayes)
articles[['index', 'naive_bayes']].to_csv('naive_bayes.csv', index=False, quotechar='', quoting=csv.QUOTE_NONE, escapechar=' ', sep=',', encoding='utf8')