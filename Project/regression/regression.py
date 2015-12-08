import pandas as pd 
import pickle 
import numpy as np 

def create_dummy_variables(df, columns):
    dummies = pd.get_dummies(articleData[columns], prefix=columns)
    df_reduced = df.ix[:, df.columns - columns]
    new_df = pd.concat([df_reduced, dummies], axis=1)
    return new_df

def clean_up_whitespace(string):
    while '  ' in string:
        string = string.replace('  ', ' ')
    return string    .encode('utf-8', 'ignore')



articleData = pickle.load(open('articleData-withWords.pkl', 'rb'))

authorData = pd.DataFrame.from_csv('../gender_guessing/authors_with_genders.tsv', sep='\t', index_col=None, encoding='utf-8')
authorData['author_name'] = authorData.author_name.apply(clean_up_whitespace)

fleschKincaid = pd.DataFrame.from_csv('../feature_extraction/readability/flesch_kincaid.txt', sep=',', index_col=None)
fleschKincaid['id'] = fleschKincaid['index_value']
fleschKincaid = fleschKincaid.ix[:, fleschKincaid.columns - ['index_value']]

bigramPerplexity = pickle.load(open('bigram-perplexity.pkl', 'rb'))
articleData = pd.concat([articleData, bigramPerplexity], axis=1)

def get_author_gender(author_array):
    genders = []

    if len(author_array) > 0:
        for i in author_array:
            j = clean_up_whitespace(i)
            prop_male = authorData[authorData['author_name'] == j]['prop_male']
            if min(prop_male) >= .9:
                genders += [1.] 
            elif max(prop_male) <= .1:
                genders += [0.] 
            else: 
                genders += [.5]

    else: 
        genders += [.5]

    if max(genders) == 1. and min(genders) == 1.:
        return 'male'
    elif min(genders) == 0. and max(genders) == 0.:
        return 'female'
    else:
        return 'ambiguous / unknown'

def get_author_popularity(author_array):
    popularities = []

    if len(author_array) > 0:
        for i in author_array:
            j = clean_up_whitespace(i)
            popularity = authorData[authorData['author_name'] == j]['popularity'].tolist()
            popularities += popularity

    else: 
        popularities += [int(authorData['popularity'].median())]

    return sum(popularities) / float(len(popularities))


articleData['author_gender'] = articleData.authors.apply(get_author_gender)
articleData['popularity_pre_log'] = articleData.authors.apply(get_author_popularity)

dummyColumns = ['typeOfMaterial', 'desk', 'type', 'section', 'author_gender']
articleDataDummies = create_dummy_variables(articleData, dummyColumns)
articleDataDummies = articleDataDummies.merge(fleschKincaid, on=['id'])
articleDataDummies['flesch-kincaid_score'].fillna(articleDataDummies['flesch-kincaid_score'].median(), inplace=True)

articleDataDummies['log_count'] = articleDataDummies['count'].apply(np.log)
articleDataDummies['log_popularity'] = articleDataDummies.popularity_pre_log.apply(np.log)

columnstoRemove = ['authors', 'desFacet', 'geoFacet', 'headline', 'id', 'orgFacet', 'perFacet', 'publishedDate', 'count', 'popularity_pre_log', 'log_count']
articleDataDummiesRegression = articleDataDummies.ix[:, articleDataDummies.columns - columnstoRemove]
yValues = articleDataDummies['log_count']

X = articleDataDummiesRegression.values
y = yValues.values

XtX = np.dot(np.transpose(X), X)
XtXinv = np.linalg.pinv(XtX)

beta = np.dot(np.dot(XtXinv, np.transpose(X)), y)

def predict(beta, X):
    return np.dot(np.reshape(beta, (1, -1)), np.transpose(X))

def MSE(predictions, actuals):
	print np.reshape(actuals, (1, -1)).shape
	print predictions.shape
	mse = np.dot((np.reshape(actuals, (1, -1))-predictions), np.transpose(np.reshape(actuals, (1, -1))-predictions))/float(predictions.size)

	return mse 

yPredict = predict(beta, X)

print MSE(yPredict, y)