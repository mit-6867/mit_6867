import numpy as np
import scipy.optimize as spo
import pandas as pd 
import scipy
import scipy.io
import pylab

np.set_printoptions(precision=4)

# Generic gradient descent function
def gradientDescent(f, df, init, lr=0.3, crit=0.0001, maxIter=100000, h=0.001):
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
	loss = np.sum(np.dot(np.transpose(diff), diff)) + Lambda * np.sum(np.dot(np.transpose(theta), theta))

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


variables = scipy.io.loadmat('/Users/dholtz/Downloads/6867_hw1_data/regress-highdim.mat')
X_train = variables['X_train']
X_test = variables['X_test']
Y_train = variables['Y_train']
Y_train_val = Y_train
Y_train = Y_train.reshape(-1, 1)	
Y_test = variables['Y_test']
Lambda = 5

thetaInit = np.random.rand(12).reshape(-1, 1)

thetaLASSO = gradientDescent(f=LASSOLoss, df=centdiff, init=thetaInit, lr=0.00003, h=.00001, crit=.006, maxIter=1000000)
thetaRidge = gradientDescent(f=ridgeRegressionLoss, df=centdiff, init=thetaInit, lr=0.00003, h=.00001, crit=.006, maxIter=1000000)
LASSOPredictions = predict(thetaLASSO, X_test)
ridgePredictions = predict(thetaRidge, X_test)
LASSOPredictionsTrain = predict(thetaLASSO, X_train)
ridgePredictionsTrain = predict(thetaRidge, X_train)

pylab.plot(X_train[:1,].flatten(), Y_train.flatten(), 'ro',
	X_train[:1,].flatten(), LASSOPredictionsTrain.flatten(), 'bo',
	X_train[:1,].flatten(), ridgePredictionsTrain.flatten(), 'go')
pylab.show()

pylab.plot(X_test[:1,].flatten(), Y_test.flatten(), 'ro',
	X_test[:1,].flatten(), LASSOPredictions.flatten(), 'bo',
	X_test[:1,].flatten(), ridgePredictions.flatten(), 'go')
pylab.show()