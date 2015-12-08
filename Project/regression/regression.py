import pandas as pd 
import pickle 
import numpy as np 
import random
import math
import csv

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


def scaleFeatures(X, mean = None, sigma= None):
	if mean == None and sigma == None:
		mean = np.mean(X, axis=0)
		sigma = np.std(X, axis=0)

	meanArray = np.repeat(mean.reshape(1, -1), X.shape[0], axis=0)
	sigmaArray = np.repeat(sigma.reshape(1, -1), X.shape[0], axis=0)

	scaledFeatures = (X - meanArray)/sigmaArray 
	
	return scaledFeatures, mean, sigma

def scale_features(array):
	scaled_array = (array - np.mean(array))/np.std(array)
	return scaled_array

articleData['author_gender'] = articleData.authors.apply(get_author_gender)
articleData['popularity_pre_log'] = articleData.authors.apply(get_author_popularity)

dummyColumns = ['typeOfMaterial', 'desk', 'type', 'section', 'author_gender']
articleDataDummies = create_dummy_variables(articleData, dummyColumns)
articleDataDummies = articleDataDummies.merge(fleschKincaid, on=['id'])
articleDataDummies['flesch-kincaid_score'].fillna(articleDataDummies['flesch-kincaid_score'].median(), inplace=True)

articleDataDummies['log_count'] = articleDataDummies['count'].apply(np.log)
articleDataDummies['log_popularity'] = articleDataDummies.popularity_pre_log.apply(np.log)
articleDataDummies['log_wcount'] = articleDataDummies.wcount.apply(np.log)

columnstoRemove = ['authors', 'desFacet', 'geoFacet', 'headline', 'id', 'orgFacet', 'perFacet', 'publishedDate', 
'count', 'popularity_pre_log', 'log_count', 'author_gender_male', 'desk_Business', 'section_Arts', 'type_Article', 
'typeOfMaterial_Interview', 'wcount']
articleDataDummiesRegression = articleDataDummies.ix[:, articleDataDummies.columns - columnstoRemove]
intercept = pd.Series(np.ones(len(articleDataDummiesRegression)))
articleDataDummiesRegression = articleDataDummiesRegression.apply(scale_features, axis=0)
articleDataDummiesRegression['intercept'] = intercept

yValues = articleDataDummies['log_count']


training_rows = random.sample(articleDataDummiesRegression.index, int(9*math.floor(len(articleDataDummiesRegression)/10.)))
training_data = articleDataDummiesRegression.ix[training_rows]
holdout_data = articleDataDummiesRegression.drop(training_rows)
training_labels = yValues.ix[training_rows]
holdout_labels = yValues.drop(training_rows)

X = training_data.values
y = training_labels.values

def predict(beta, X):
    return np.dot(np.reshape(beta, (1, -1)), np.transpose(X))

def MSE(predictions, actuals):
    mse = np.dot((np.reshape(actuals, (1, -1))-predictions), np.transpose(np.reshape(actuals, (1, -1))-predictions))/float(predictions.size)
    return mse 

training_MSES = []
holdout_MSES = []
best_weights = []
best_holdout_MSE = 100000
best_l = 0
ls = []

for l in [0, .1, 1, 2, 3, 4, 5, 10, 25, 50, 100]:
    XtX = np.dot(np.transpose(X), X) + l*np.identity(X.shape[1])
    XtXinv = np.linalg.pinv(XtX)

    beta = np.dot(np.dot(XtXinv, np.transpose(X)), y)

    yPredict = predict(beta, X)
    yPredictHoldout = predict(beta, holdout_data.values)

    training_MSES += list(MSE(yPredict, y)[0])
    holdout_MSES += list(MSE(yPredictHoldout, holdout_labels.values)[0])
    ls += [l] 

    if MSE(yPredictHoldout, holdout_labels.values) < best_holdout_MSE:
    	best_weights = beta
    	best_l = l
    	best_holdout_MSE = MSE(yPredictHoldout, holdout_labels.values)

print ', '.join(str(v) for v in best_weights)
print '", "'.join(str(v) for v in articleDataDummiesRegression.columns.values)
print training_MSES
print holdout_MSES 
print best_holdout_MSE
print best_l 

MSE_performance = pd.Series(training_MSES).to_frame(name='training_mse')
MSE_performance['holdout_mse'] = pd.Series(holdout_MSES)
MSE_performance['lambdas'] = pd.Series(ls)
MSE_performance.to_csv('mse_performance.csv', index=False, quotechar='', quoting=csv.QUOTE_NONE, escapechar=' ', sep=',', encoding='utf8')