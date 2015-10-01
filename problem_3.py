import numpy as np
import scipy.linalg as linalg 
import pylab
import homework1
import sys
import pandas as pd
import matplotlib.pyplot as plt
from ggplot import *

def createFeatures(X=np.zeros(1), M=1):
	features = X**0
	for i in range(1, M+1):
		features = np.hstack([features, X**i])
	return features

def ridgeRegression(X, Y, l=0, M=1, features=False):
	# Closed form solution from http://www.hongliangjie.com/notes/lr.pdf
	if not features:
		features = createFeatures(X, M)
	else:
		features = X
		M = features.shape[1]-1

	firstTerm = linalg.pinv(np.dot(np.transpose(features), features) + l*np.identity(M+1))
	secondTerm = np.transpose(features)
	thirdTerm = Y

	theta = np.dot(firstTerm, np.dot(secondTerm, thirdTerm))

	return theta

def predictRidge(X, theta, features=False):
	if not features:
		features = createFeatures(X, M=theta.size-1)
	else:
		features = X
	
	predictions = np.dot(features, theta)

	return predictions

def MSE(predictions, actuals):
	mse = np.dot(np.transpose(predictions-actuals), (predictions-actuals))/float(predictions.size)

	return mse 

def gridSearch(lambdas, Ms, X_train, X_test, X_valid, Y_train, Y_test, Y_valid):
	results = np.empty([4,1])
	bestLambda = lambdas[0]
	bestM = Ms[0]
	bestTheta = ridgeRegression(X_train, Y_train, l=lambdas[0], M=Ms[0])
	bestPredictions = predictRidge(X_test, bestTheta)
	minMSE = MSE(bestPredictions, Y_test)
	print Y_test 
	print Y_train

	for i in lambdas:
		for j in Ms:
			theta = ridgeRegression(X_train, Y_train, l=i, M=j)
			predictions = predictRidge(X_test, theta)
			mse = MSE(predictions, Y_test)
			print 'Lambda = %f, M = %d, Test MSE = %f, Train MSE = %f, Validation MSE = %f' % (i, j, mse, MSE(predictRidge(X_train, theta), Y_train), MSE(predictRidge(X_valid, theta), Y_valid))
			results = np.hstack([results, np.array([i, j, mse, MSE(predictRidge(X_train, theta), Y_train)]).reshape(4,1)])
			if mse < minMSE:
				minMSE = mse
				bestLambda = i
				bestM = j 
				bestTheta = theta 
				bestPredictions = predictions

	result = 'Lambda = %f, M = %f, MSE = %f ' % (bestLambda, bestM, minMSE)
	print result

	return bestTheta, np.transpose(results)

def gridSearchBlog(lambdas, X_train, X_test, X_valid, Y_train, Y_test, Y_valid):
	lambda_values = []
	mse_values = []
	bestLambda = lambdas[0]
	scaled_X_train_blog = X_train
	X_train_blog_final = addInterceptTerm(scaled_X_train_blog)
	scaled_X_valid = X_valid
	scaled_X_test = X_test
	X_valid_blog_final = addInterceptTerm(scaled_X_valid)
	X_test_blog_final = addInterceptTerm(scaled_X_test)

	print lambdas[0]
	print 'getting theta...'
	bestTheta = ridgeRegression(X_train_blog_final, Y_train, l=lambdas[0], features=True)
	print 'predicting...'
	bestPredictions = predictRidge(X_test_blog_final, bestTheta, features=True)
	print 'calculating MSE...'
	minMSE = MSE(bestPredictions, Y_test)

	for i in lambdas[1:]:
		print i 
		print 'getting theta...'
		theta = ridgeRegression(X_train_blog_final, Y_train, l=i, features=True)
		print 'predicting...'
		predictions = predictRidge(X_test_blog_final, theta, features=True)
		print 'calculating MSE...'
		mse = MSE(predictions, Y_test)
		print 'Test MSE', mse 
		print 'Train MSE', MSE(predictRidge(X_train_blog_final, theta, features=True), Y_train)
		mse_values += [mse]
		lambda_values += [i]
		if mse < minMSE:
			minMSE = mse
			bestLambda = i
			bestTheta = theta 
			bestPredictions = predictions

	result = 'Lambda = %f MSE = %f ' % (bestLambda, minMSE)
	print result
	print lambda_values 
	print mse_values

	return bestTheta, lambda_values, mse_values

def addInterceptTerm(X_array):
	data = np.concatenate((np.ones(X_array.shape[0]).reshape(-1, 1), X_array), axis=1)

	return data

def scaleFeatures(X, mean = None, sigma= None):
	if mean == None and sigma == None:
		mean = np.mean(X, axis=0)
		sigma = np.std(X, axis=0)

	meanArray = np.repeat(mean.reshape(1, -1), X.shape[0], axis=0)
	sigmaArray = np.repeat(sigma.reshape(1, -1), X.shape[0], axis=0)

	scaledFeatures = (X - meanArray)/sigmaArray 
	
	return scaledFeatures, mean, sigma

##########

X, Y = homework1.getData('/Users/dholtz/Downloads/6867_hw1_data/curvefitting.txt')
theta = ridgeRegression(X, Y, l=0, M=10)	

lambdas = np.array([0, .01, .1, 1, 10, 100])
Ms = np.array([1, 2, 3, 4, 5])

X_train, Y_train = homework1.getData('/Users/dholtz/Downloads/6867_hw1_data/regress_train.txt')
X_test, Y_test = homework1.getData('/Users/dholtz/Downloads/6867_hw1_data/regress_test.txt')
X_validate, Y_validate = homework1.getData('/Users/dholtz/Downloads/6867_hw1_data/regress_validate.txt')

theta = ridgeRegression(X_train, Y_train, M=100)
predictions = predictRidge(X_train, theta)
print MSE(predictions, Y_train)

thetaOLS = ridgeRegression(X_train, Y_train, l=0, M=10)
predictions = predictRidge(X_train, thetaOLS)

pylab.plot(X_train, Y_train, 'ro',
	X_train, predictRidge(X_train, thetaOLS), 'bo')
pylab.show()

predictions = predictRidge(X, theta)

theta = ridgeRegression(X, Y)

pylab.plot(X, Y, 'ro',
	X, predictRidge(X, theta), 'bo')
pylab.show()

bestTheta, resultsArray = gridSearch(lambdas, Ms, X_train, X_test, X_validate, Y_train, Y_test, Y_validate)

resultsArray = pd.DataFrame(resultsArray, columns=np.array(['lambda', 'M', 'MSE_test', 'MSE_train']))
resultsArray = resultsArray[1:len(resultsArray.index)]

resultsArray['lambda'] = resultsArray['lambda'].astype('category')

bestTheta, resultsArray = gridSearch(lambdas, Ms, X, X, X, Y, Y, Y)

resultsArray = pd.DataFrame(resultsArray, columns=np.array(['lambda', 'M', 'MSE_test', 'MSE_train']))
resultsArray = resultsArray[1:len(resultsArray.index)]

resultsArray['lambda'] = resultsArray['lambda'].astype('category')

print ggplot(aes(x='M', y='MSE_test', colour=str('lambda')), data=resultsArray) + geom_line() +  \
	ylab('MSE') + xlab('M (polynomial order)') + ggtitle('MSE dependence on M, lambda (Bishop 1.4 Data)')

order = np.argsort(X_test, axis=0)

theta = ridgeRegression(X_train, Y_train, M=5)

bishop = {'x_data': pd.Series(X.flatten()), 'y_data': pd.Series(Y.flatten())}
bishopDF = pd.DataFrame(bishop)
print bishopDF.dtypes
print ggplot(aes(x='x_data', y='y_data'), data=bishopDF) + geom_point() + \
 xlab('x') + ylab('y') + ggtitle('Bishop Figure 1.4 Data')

hyperparams = {'lambdaM': pd.Series(['Lambda = 0, M = 5', 'Lambda = 0, M = 5', 'Lambda = 0, M = 5', 'Lambda = 1, M = 1', 'Lambda = 1, M = 1', \
	'Lambda = 1, M = 1', 'Lambda = 10, M = 3', 'Lambda = 10, M = 3', 'Lambda = 10, M = 3']), 'dataset': pd.Series(['Train', 'Test', 'Validation', \
		'Train', 'Test', 'Validation', 'Train', 'Test', 'Validation']), 'MSE': pd.Series([0.078139, 25.007056, 13.883744, 1.460208, 1.776578, 1.414984, \
		1.518378, 2.785566, 1.131175])}
hyperparamsDF = pd.DataFrame(hyperparams)


#pylab.plot(list(X_train[order].flatten()), list(Y_train[order].flatten()), 'ro',
#	X_train[order].flatten(), predictRidge(X_train, bestTheta)[order].flatten(), 'bo')
#pylab.show()
#
#pylab.plot(list(X_test[order].flatten()), list(Y_test[order].flatten()), 'ro',
#	X_test[order].flatten(), predictRidge(X_test, bestTheta)[order].flatten(), 'bo')
#pylab.show()
#
#pylab.plot(list(X_validate[order].flatten()), list(Y_validate[order].flatten()), 'ro',
#	X_validate[order].flatten(), predictRidge(X_validate, bestTheta)[order].flatten(), 'bo')
#pylab.show()

###
### Blog Feedback Data
###


X_train_blog = np.genfromtxt('/Users/dholtz/Downloads/6867_hw1_data/BlogFeedback_data/x_train.csv', delimiter=",")
Y_train_blog = np.genfromtxt('/Users/dholtz/Downloads/6867_hw1_data/BlogFeedback_data/y_train.csv', delimiter=",")
X_test_blog = np.genfromtxt('/Users/dholtz/Downloads/6867_hw1_data/BlogFeedback_data/x_test.csv', delimiter=",")
Y_test_blog = np.genfromtxt('/Users/dholtz/Downloads/6867_hw1_data/BlogFeedback_data/y_test.csv', delimiter=",")
X_valid_blog = np.genfromtxt('/Users/dholtz/Downloads/6867_hw1_data/BlogFeedback_data/x_val.csv', delimiter=",")
Y_valid_blog = np.genfromtxt('/Users/dholtz/Downloads/6867_hw1_data/BlogFeedback_data/y_val.csv', delimiter=",")

#scaled_X_train_blog, mean_scaling, sigma_scaling = scaleFeatures(X_train_blog)
#X_train_blog_final = addInterceptTerm(scaled_X_train_blog)

#thetaBlog = ridgeRegression(X_train_blog, Y_train_blog, l=.01, features=True)
#predictBlog = predictRidge(X_train_blog, thetaBlog, features=True)
#seBlog = MSE(predictBlog, Y_train_blog)
10000000.000000

lambdas = 10**np.array(range(15))

mses, lambda_values, mse_values = gridSearchBlog(lambdas, X_train_blog, X_test_blog, X_valid_blog, Y_train_blog, Y_test_blog, Y_valid_blog)
print MSE(np.repeat(np.mean(Y_test_blog), Y_test_blog.size), Y_test_blog)
print Y_test_blog.size

MSE_trend = {'lambda' : pd.Series(np.array(lambda_values)), 'mse' : pd.Series(np.array(mse_values))}
MSE_DF = pd.DataFrame(MSE_trend)

print ggplot(MSE_DF, aes(x='lambda', y='mse')) + geom_line() + \
geom_hline(yintercept=[MSE(np.repeat(np.mean(Y_test_blog), Y_test_blog.size), Y_test_blog)], size=10) + \
xlab('Lambda') + ylab('MSE') + ggtitle('Blog Feedback Regression MSE as a function of Lambda') + scale_x_log()
#print createFeatures(X, M=5)
#print thetaBlog
#print mseBlog