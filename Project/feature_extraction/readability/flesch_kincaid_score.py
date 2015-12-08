from textstat.textstat import textstat
import os
import csv
import numpy as np 
import pandas as pd

flesch_kincaid_scores = pd.DataFrame(columns = ('index_value', 'flesch-kincaid_score'))

for i in os.listdir('../../text_scraping/article_text'):
	if i.endswith(".txt"):
		with open ('../../text_scraping/article_text/' + i, "r") as myfile:
			data=myfile.read().replace('\n', '')
			if data != '' and '.' in data:
				data_append = [{'index_value': i.replace('.txt', ''), 'flesch-kincaid_score': textstat.flesch_reading_ease(data)}]
				df_append = pd.DataFrame(data_append)
				flesch_kincaid_scores = pd.concat([flesch_kincaid_scores, df_append])
			else:
				data_append = [{'index_value': i.replace('.txt', ''), 'flesch-kincaid_score': None}]
				df_append = pd.DataFrame(data_append)
				flesch_kincaid_scores = pd.concat([flesch_kincaid_scores, df_append])

flesch_kincaid_scores.to_csv('flesch_kincaid.txt', index=False)