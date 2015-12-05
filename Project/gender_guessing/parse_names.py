import pickle
import pandas
import itertools
import requests
from bs4 import BeautifulSoup
import time

def get_popularity(author):
	print author
	r = requests.get('http://www.bing.com/search',
	                 params={'q':'"'+author+'"',
	                         "tbs":"li:1"}
	                )
	
	soup = BeautifulSoup(r.text, "html.parser")
	raw_results_text = soup.find('div',{'id':'b_tween'}).text
	time.sleep(30)
	return raw_results_text.replace(' resultsAny time', '').replace(',', '').replace(' ', '')
	
def remove_quotes(string):
	return string.replace('"', '')

df = pandas.read_pickle('/Users/dholtz/Downloads/articleData.pkl')

authors = list(itertools.chain(*df['authors']))
unique_authors = set(authors)

authors_df = pandas.DataFrame(list(unique_authors), columns = ['author_name'])
authors_df['author_name'] = authors_df.author_name.apply(remove_quotes)
authors_df['author_name'] = authors_df.author_name.apply(remove_quotes)
authors_df.to_csv('author_list.csv', index=False, encoding='utf-8')

popularities = []
for i in authors_df['author_name']:
	popularities += get_popularity(i)
