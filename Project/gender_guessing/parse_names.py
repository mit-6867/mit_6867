import pickle
import pandas
import itertools
import requests
from bs4 import BeautifulSoup
import time
import csv
import pickle

def get_popularity(author):
	print author
	r = requests.get('http://www.bing.com/search',
	                 params={'q':'"'+author+'"',
	                         "tbs":"li:1"}
	                )
	
	soup = BeautifulSoup(r.text, "html.parser")
	raw_results_text = soup.find('div',{'id':'b_tween'}).text
	time.sleep(20)
	return [int(raw_results_text.split()[0].replace(',', ''))]
	
def remove_quotes(string):
	return string.replace('"', '')

df = pandas.read_pickle('/Users/dholtz/Downloads/articleData-withWords.pkl')

authors = list(itertools.chain(*df['authors']))
unique_authors = set(authors)

authors_df = pandas.DataFrame(list(unique_authors), columns = ['author_name'])
authors_df['author_name'] = authors_df.author_name.apply(remove_quotes)
authors_df['author_name'] = authors_df.author_name.apply(remove_quotes)

popularities = []
popularities = pickle.load(open('popularities.pkl', 'rb'))
for i in range(len(popularities), len(authors_df['author_name'])):
	popularities += get_popularity(authors_df['author_name'][i])
	pickle.dump(popularities, open('popularities.pkl', 'wb'))
	print popularities

authors_df['popularity'] = pandas.Series(popularities)
authors_df.to_csv('author_list.tsv', index=False, quotechar='', quoting=csv.QUOTE_NONE, escapechar=' ', sep='\t', encoding='utf8')