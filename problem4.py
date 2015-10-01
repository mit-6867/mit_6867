import numpy as np
import scipy.optimize as spo
import pandas as pd 
import scipy
import scipy.io
import pylab
import statsmodels.api as sm
from ggplot import *

np.set_printoptions(precision=4)

# Generic gradient descent function
def gradientDescent(f, df, init, lr=0.3, crit=0.0001, maxIter=100000, h=0.001, verbose=False):
	count = 0
	nIter = 0
	fcall = 0
	while count < 2 and nIter <= maxIter:
		f_i = f(init)
		# Calculates the gradient
		grad = df(init, f, h)
		# Update step
		init = init - lr * grad.reshape(-1, 1)
		f_u = f(init)
		# Calculate the difference in f using initial and updated values
		diff = abs((f_i - f_u))
		nIter += 1
		fcall += 2
		if df == centdiff:
			fcall += 2*len(init)
		# Tracks successive number of times that difference is below the convergence criterion
		if diff < crit:
			count += 1
		else:
			count = 0
		print 'Loss function: ', f_u
		print 'Test MSE: ', MSE(predict(init, X_test), Y_test)
		print 'Train MSE: ', MSE(predict(init, X_train), Y_train_val)
		if verbose == True:
			print init 
	print "nIter: %d" % (nIter)
	print "Fcall: %d" % (fcall)
	return init

# Central difference approximation of a gradient, returns a vector of length of the input vector x
def centdiff(x, f, h=0.00001):
	n = len(x)
	out = np.zeros(n)
	for i in range(0, n):
		hplus = np.copy(x)
		hminus = np.copy(x)
		hplus[i] += h
		hminus[i] -= h
		# Calculates a better denominator to address potential problems with floating point arithmetic especially for small values of h
		hfix = hplus[i] - hminus[i]
		out[i] = (f(hplus) - f(hminus))/(hfix)
	return out

def ridgeRegressionLoss(theta):
	xTtheta = np.dot(np.transpose(X_train), theta)
	diff = Y_train - xTtheta
	loss = np.sum(np.dot(np.transpose(diff), diff))*(1/Y_train.size) + Lambda * np.sum(np.dot(np.transpose(theta), theta))

	return loss


def LASSOLoss(theta):
	XTtheta = np.dot(np.transpose(X_train), theta)
	diff = Y_train - XTtheta
	loss = np.sum(np.dot(np.transpose((diff)), diff))*(1/Y_train.size) + Lambda * np.sum(np.absolute(theta))

	return loss

def predict(theta, X):
	prediction = np.dot(np.transpose(X), theta)

	return prediction 

def MSE(predictions, actuals):
	mse = np.dot((np.transpose(predictions)-actuals), np.transpose(np.transpose(predictions)-actuals))/float(predictions.size)

	return mse 

def addInterceptTerm(X_array):
	data = np.vstack([X_array, np.ones(X_array.shape[1]).reshape(1, -1)])

	return data

variables = scipy.io.loadmat('/Users/dholtz/Downloads/6867_hw1_data/regress-highdim.mat')
X_train = variables['X_train']
X_train = addInterceptTerm(X_train)
X_test = variables['X_test']
X_test = addInterceptTerm(X_test)
Y_train = variables['Y_train']
Y_train_val = Y_train
Y_train = Y_train.reshape(-1, 1)	
Y_test = variables['Y_test']
W_true = variables['W_true']
print W_true.shape
print np.array([0]).reshape(1,-1).shape
W_true = np.hstack([np.array([0]).reshape(1,-1), W_true])
Lambda = 7

thetaInit = np.random.rand(13).reshape(-1, 1)

## These don't always converge - might be because the actual function is sinusoidal?
thetaLASSO = gradientDescent(f=LASSOLoss, df=centdiff, init=thetaInit, lr=0.00003, h=.00001, crit=.006, maxIter=1000000)
thetaRidge = gradientDescent(f=ridgeRegressionLoss, df=centdiff, init=thetaInit, lr=0.00003, h=.00001, crit=.006, maxIter=1000000)

LASSOPredictions = predict(thetaLASSO, X_test)
ridgePredictions = predict(thetaRidge, X_test)
LASSOPredictionsTrain = predict(thetaLASSO, X_train)
ridgePredictionsTrain = predict(thetaRidge, X_train)

X_lin = np.linspace(-1, 1, 1000)
#Sorry I'm not sorry.
X_lin = np.array([X_lin, np.sin(0.4*np.pi*X_lin*1), np.sin(0.4*np.pi*X_lin*2), np.sin(0.4*np.pi*X_lin*3), np.sin(0.4*np.pi*X_lin*4), np.sin(0.4*np.pi*X_lin*5), np.sin(0.4*np.pi*X_lin*6), np.sin(0.4*np.pi*X_lin*7), np.sin(0.4*np.pi*X_lin*8), np.sin(0.4*np.pi*X_lin*9), np.sin(0.4*np.pi*X_lin*10), np.sin(0.4*np.pi*X_lin*11)])
X_lin = addInterceptTerm(X_lin)

print W_true.shape 
print X_lin.shape
ActualPredictions = predict(np.transpose(W_true), X_lin)

YLasso_lin = predict(thetaLASSO, X_lin)
YRidge_lin = predict(thetaRidge, X_lin)

#print sm.OLS(Y_test, X_test).fit().summary()

Lambda = 0

thetaOLS = gradientDescent(f=LASSOLoss, df=centdiff, init=thetaInit, lr=0.00003, h=.00001, crit=.006, maxIter=1000000)

OLSPredictions = predict(thetaOLS, X_test)
OLSPredictionsTrain = predict(thetaOLS, X_train)
YOLS_lin = predict(thetaOLS, X_lin)

pylab.plot(X_train[0:1,].flatten(), Y_train.flatten(), 'ro',
	X_train[0:1,].flatten(), LASSOPredictionsTrain.flatten(), 'bo',
	X_train[0:1,].flatten(), ridgePredictionsTrain.flatten(), 'go',
	X_train[0:1,].flatten(), OLSPredictionsTrain.flatten(), 'co')
pylab.show()

pylab.plot(X_test[0:1,].flatten(), Y_test.flatten(), 'ro',
	X_test[0:1,].flatten(), LASSOPredictions.flatten(), 'bo',
	X_test[0:1,].flatten(), ridgePredictions.flatten(), 'go',
	X_test[0:1,].flatten(), OLSPredictions.flatten(), 'co')
pylab.show()


pylab.plot(X_lin[0:1,].flatten(), ActualPredictions.flatten(), 'r',
	X_lin[0:1,].flatten(), YLasso_lin.flatten(), 'b',
	X_lin[0:1,].flatten(), YRidge_lin.flatten(), 'g',
	X_lin[0:1,].flatten(), YOLS_lin.flatten(), 'c',
	X_train[0:1,].flatten(), Y_train.flatten(), 'bo',
	X_test[0:1,].flatten(), Y_test.flatten(), 'mo')
pylab.show()

print thetaOLS 
print thetaRidge 
print thetaLASSO
print W_true

print np.concatenate([thetaOLS, thetaRidge, thetaLASSO, np.transpose(W_true)])

d = {'weights' : pd.Series(np.concatenate([thetaOLS, thetaRidge, thetaLASSO, np.transpose(W_true)]).flatten()), \
'Models': pd.Series(np.concatenate([np.repeat(np.array(['OLS']), 13),np.repeat(np.array(['Ridge Regression']), 13),\
	np.repeat(np.array(['LASSO']), 13),np.repeat(np.array(['True']), 13)]).flatten()),\
	'Coefficient': pd.Series(np.repeat(np.array(['w1', 'w2', 'w3', 'w4', 'w5', 'w6', 'w7', 'w8', 'w9', 'w10', 'w11', 'w12', 'w13']), 4).flatten())}

data = pd.DataFrame(d)
data['Coefficient'] = data['Coefficient'].astype('category')
data['Models'] = data['Models'].astype('category')

print data.dtypes


print ggplot(data, aes(x='Coefficient', y='weights')) + geom_bar(stat="identity") + facet_wrap('Models', scales='fixed')

