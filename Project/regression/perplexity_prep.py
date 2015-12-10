import pickle 
import pandas as pd 

trigramPerplexity = pickle.load(open('trigram-nn-perplexity.pkl', 'rb'))
trigramPerplexity.to_csv('trigram_perplexity.csv')

bigramPerplexity = pickle.load(open('bigram-perplexity.pkl', 'rb'))
bigramPerplexity.to_csv('bigram_perplexity.csv')
