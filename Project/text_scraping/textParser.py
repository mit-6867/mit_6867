from newspaper import Article
import sys
import re
import codecs

blk = int(sys.argv[1])

article_urls = codecs.open('list_fixed.txt', encoding = 'utf-8').read().splitlines()

def wrapper(block):
    for i in block:
        print i 
        article = Article(url=article_urls[i])
        article.download()
        article.parse()
        text = article.text
        text = text.replace('Photo\n', '')
        text = text.replace('Photo \n', '')
        text = text.replace('Advertisement Continue reading the main story\n', '')
        text = text.replace('Photo Advertisement Continue reading the main story\n', '')
        text = re.sub("\n+" , " ", text)
        text = re.sub('^ ', '', text)
        f = open('article_text/' + str(i + 1) + '.txt', 'w')
        f.write(text.encode('utf-8'))
        f.close()

if blk == 1:
    wrapper(xrange(1670))
    
elif blk == 2:
    wrapper(xrange(1670, 3341))
    
elif blk == 3:
    wrapper(xrange(3341,5011))
    
elif blk == 4:
    wrapper(xrange(5011,6682))
