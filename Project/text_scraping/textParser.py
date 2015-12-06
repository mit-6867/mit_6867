from newspaper import Article
import pandas as pd 

article_urls = pd.read_csv('article_list.txt', header=None)
url = 'http://www.nytimes.com/2013/08/31/sports/ncaafootball/lewan-staying-at-michigan-with-renewed-focus.html'

for i in range(len(article_urls)):
	print i 
	article = Article(url=article_urls[0][i])
	article.download()
	article.parse()
	text = article.text
	text = text.replace('Photo\n', '')
	text = text.replace('Photo \n', '')
	text = text.replace('Advertisement Continue reading the main story\n', '')
	text = text.replace('\n\n', ' ')
	text = text[1:]
	f = open('article_text/' + str(i + 1) + '.txt', 'w')
	f.write(text.encode('utf8'))
	f.close()