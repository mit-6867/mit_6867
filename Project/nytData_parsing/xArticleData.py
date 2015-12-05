import numpy as np
import re
import os
import time
import simplejson as json
import sys
import pandas as pd
reload(sys)
sys.setdefaultencoding('utf-8')

month = sys.argv[1]
half = int(sys.argv[2])

if half == 1:
    path = '/mnt/nfs6/sinan_data.proj/NYT/' + month + '/days1-15/'
if half == 2:
    path = '/mnt/nfs6/sinan_data.proj/NYT/' + month + '/days16-30/'

listpath = '/srv/ml_project/article_list/08-list'
url_list = open(listpath,'r').read().splitlines()
n = len(url_list)
col_list = ['id', 'count', 'publishedDate', 'headline', 'authors', 'typeOfMaterial', 'desk', 'type', 'section', 'desFacet', 'perFacet', 'orgFacet', 'geoFacet']
df = pd.DataFrame({'id': range(n), 'count': np.zeros(n), 'publishedDate': np.zeros(n), 'headline': ['None'] * n, 'authors': [[] for _ in range(n)], 'typeOfMaterial': ['None'] * n, 'desk': ['None'] * n, 'type': ['None'] * n, 'section' : ['None'] * n, 'desFacet': [[] for _ in range(n)], 'perFacet': [[] for _ in range(n)], 'orgFacet': [[] for _ in range(n)], 'geoFacet': [[] for _ in range(n)]}, index = url_list, columns = col_list)
pattern = '%m-%d-%Y %H:%M'

def addToCell(cell, alist):
    for i in alist:
        cell.append(i)

for root, dirs, files in os.walk(path):
	name = os.path.join(root)
	name = re.sub('/mnt.*NYT/', '', name)
	name = re.sub('(\d\d)/days1-15', '\\1', name)
	name = re.sub('(\d\d)/days16-30', '\\1', name)
	name = re.sub('/(\d\d)/', '-\\1', name)
	name = re.sub('/', '-', name)
	date = name + '-2013 00:00'

	try:
		epoch = max((int(time.mktime(time.strptime(date, pattern)))- 7 * 86400) * 1000, 1375329600000)
	except:
		epoch = 3000000000000

	for file in files:
		f = open(os.path.join(root, file), 'r')
		print os.path.join(root, file).encode('utf-8')
		for line in f:
			try:
				obj = json.loads(line)
				pDate = obj.get('publishedDate', 0)
				if pDate < epoch:
				    continue
				ad = obj['assetData']
				url = ad['url']
				if df.loc[url,'count'] > 0:
				    df.loc[url,'count'] += 1
				elif df.loc[url, 'count'] == 0:
				    df.loc[url,'count'] += 1
				    df.loc[url,'headline'] = ad.get('headline', 'None')
				    df.loc[url,'publishedDate'] = ad.get('publishedDate')
				    df.loc[url,'desk'] = ad.get('desk', 'None').lower()
				    df.loc[url,'type'] = ad.get('type', 'None').lower()
				    df.loc[url,'section'] = ad.get('section', 'None').lower()
				    df.loc[url,'typeOfMaterial'] = ad.get('typeOfMaterial', 'None').lower()
				    addToCell(df.loc[url,'authors'], ad.get('authors', []))
				    addToCell(df.loc[url,'desFacet'], ad.get('desFacet', []))
				    addToCell(df.loc[url,'perFacet'], ad.get('perFacet', []))
				    addToCell(df.loc[url,'orgFacet'], ad.get('orgFacet', []))
				    addToCell(df.loc[url,'geoFacet'], ad.get('geoFacet', []))
			except:
			    continue
		f.close()

out = '/srv/ml_project/articleData-'+month+'-'+str(half)+'.pkl'

df.to_pickle(out)