import pandas as pd
import numpy as np

def addToCell(cell, alist):
    for i in alist:
        cell.append(i)

path = '/Users/mfzhao/NYT NLP/Data/'
df1 = pd.read_pickle(path + '1.pkl')
df2 = pd.read_pickle(path + '2.pkl')
df3 = pd.read_pickle(path + '3.pkl')

url_list = open(path+'article_list.txt','r').read().splitlines()
n = len(url_list)
col_list = ['id', 'count', 'publishedDate', 'headline', 'authors', 'typeOfMaterial', 'desk', 'type', 'section', 'desFacet', 'perFacet', 'orgFacet', 'geoFacet']
df = pd.DataFrame({'id': range(n), 'count': np.zeros(n), 'publishedDate': np.zeros(n), 'headline': ['None'] * n, 'authors': [[] for _ in range(n)], 'typeOfMaterial': ['None'] * n, 'desk': ['None'] * n, 'type': ['None'] * n, 'section' : ['None'] * n, 'desFacet': [[] for _ in range(n)], 'perFacet': [[] for _ in range(n)], 'orgFacet': [[] for _ in range(n)], 'geoFacet': [[] for _ in range(n)]}, index = url_list, columns = col_list)

for url in url_list:
    df.loc[url,'count'] = df1.loc[url,'count'] + df2.loc[url,'count'] + df3.loc[url,'count']
    if df1.loc[url,'count'] != 0:
        df.loc[url,'headline']       = df1.loc[url,'headline']
        df.loc[url,'publishedDate']  = df1.loc[url,'publishedDate']
        df.loc[url,'desk']           = df1.loc[url,'desk']
        df.loc[url,'type']           = df1.loc[url,'type']
        df.loc[url,'section']        = df1.loc[url,'section']
        df.loc[url,'typeOfMaterial'] = df1.loc[url,'typeOfMaterial']
        addToCell(df.loc[url,'authors'], df1.loc[url,'authors'])
        addToCell(df.loc[url,'desFacet'], df1.loc[url,'desFacet'])
        addToCell(df.loc[url,'perFacet'], df1.loc[url,'perFacet'])
        addToCell(df.loc[url,'orgFacet'], df1.loc[url,'orgFacet'])
        addToCell(df.loc[url,'geoFacet'], df1.loc[url,'geoFacet'])
    elif df2.loc[url,'count'] != 0:
        df.loc[url,'headline']       = df2.loc[url,'headline']
        df.loc[url,'publishedDate']  = df2.loc[url,'publishedDate']
        df.loc[url,'desk']           = df2.loc[url,'desk']
        df.loc[url,'type']           = df2.loc[url,'type']
        df.loc[url,'section']        = df2.loc[url,'section']
        df.loc[url,'typeOfMaterial'] = df2.loc[url,'typeOfMaterial']
        addToCell(df.loc[url,'authors'], df2.loc[url,'authors'])
        addToCell(df.loc[url,'desFacet'], df2.loc[url,'desFacet'])
        addToCell(df.loc[url,'perFacet'], df2.loc[url,'perFacet'])
        addToCell(df.loc[url,'orgFacet'], df2.loc[url,'orgFacet'])
        addToCell(df.loc[url,'geoFacet'], df2.loc[url,'geoFacet'])
    elif df3.loc[url,'count'] != 0:
        df.loc[url,'headline']       = df3.loc[url,'headline']
        df.loc[url,'publishedDate']  = df3.loc[url,'publishedDate']
        df.loc[url,'desk']           = df3.loc[url,'desk']
        df.loc[url,'type']           = df3.loc[url,'type']
        df.loc[url,'section']        = df3.loc[url,'section']
        df.loc[url,'typeOfMaterial'] = df3.loc[url,'typeOfMaterial']
        addToCell(df.loc[url,'authors'], df3.loc[url,'authors'])
        addToCell(df.loc[url,'desFacet'], df3.loc[url,'desFacet'])
        addToCell(df.loc[url,'perFacet'], df3.loc[url,'perFacet'])
        addToCell(df.loc[url,'orgFacet'], df3.loc[url,'orgFacet'])
        addToCell(df.loc[url,'geoFacet'], df3.loc[url,'geoFacet'])
    print df.loc[url]

df.to_pickle(path + 'articleData.pkl')
