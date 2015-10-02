import numpy as np
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import scipy.optimize as spo
import pandas as pd
import scipy
import scipy.io
import matplotlib.pylab as pylab
import statsmodels.api as sm
from ggplot import *

np.set_printoptions(precision=4)

# Generic gradient descent function
def gradientDescent(f, df, init, lr=0.3, crit=0.0001, maxIter=100000, h=0.0001, verbose=False):
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
		#print 'Initial Loss Function: ', f_i
		XTtheta = np.dot(np.transpose(X_train), init)
		diff = Y_train - XTtheta
		#print 'Diff manual ', diff
		#print 'Loss function: ', f_u
		#print 'Squared Error: ', np.sum(np.dot(np.transpose(diff), diff))*(1./Y_train.size)
		#print 'Penalty: ', Lambda * np.sum(np.absolute(init))
		#print 'Loss: ', np.sum(np.dot(np.transpose(diff), diff))*(1./Y_train.size) + Lambda * np.sum(np.absolute(init))
		#print 'Lambda: ', Lambda
		#print 'Theta term manual', np.sum(np.absolute(init))
		#print 'Test MSE: ', MSE(predict(init, X_test), Y_test)
		#print 'Train MSE: ', MSE(predict(init, X_train), Y_train_val)
		#print 'theta', init
		#print f(init)
		if verbose == True:
			print init
	#print "nIter: %d" % (nIter)
	#print "Fcall: %d" % (fcall)
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
	loss = np.sum(np.dot(np.transpose(diff), diff))*(1./Y_train.size) + Lambda2 * np.sum(np.dot(np.transpose(theta), theta))

	return loss


def LASSOLoss(theta):
	XTtheta = np.dot(np.transpose(X_train), theta)
	diff = Y_train - XTtheta
	#print 'Diff - loss', diff
	#print 'squared error', np.sum(np.dot(np.transpose((diff)), diff))*(1./Y_train.size)
	#print 'Lambda', Lambda
	#print 'Theta term', np.sum(np.absolute(theta))
	#print 'penalty', Lambda * np.sum(np.absolute(theta))
	#print 'theta', theta
	loss = np.sum(np.dot(np.transpose((diff)), diff))*(1./Y_train.size) + Lambda1 * np.sum(np.absolute(theta))

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

variables = scipy.io.loadmat('/Users/mfzhao/Downloads/6867_hw1_data/regress-highdim.mat')
X_train = variables['X_train']
#X_train = addInterceptTerm(X_train)
X_test = variables['X_test']
#X_test = addInterceptTerm(X_test)
Y_train = variables['Y_train']
Y_train_val = Y_train
Y_train = Y_train.reshape(-1, 1)
Y_test = variables['Y_test']
W_true = variables['W_true']
#W_true = np.hstack([np.array([0]).reshape(1,-1), W_true])

#thetaInit = (.5*np.random.rand(12)-1).reshape(-1, 1)
#thetaInit = W_true.reshape(-1, 1)
thetaInit = np.repeat(.5, 12).reshape(-1, 1)
#thetaInit = np.repeat(.25, 12).reshape(-1, 1)

best_lambda1 = 0
best_lambda2 = 0
best_mse1 = 100
best_mse2 = 100
for try_lambda in [0, .1, .2, .3, .4, .5, .6, .7 ,.8, .9, 1]:
	Lambda1 = try_lambda
	Lambda2 = try_lambda
 	current_lambda = try_lambda
	print 'trying', try_lambda

	## These don't always converge - might be because the actual function is sinusoidal?
	print 'lasso'
	thetaLASSO = np.array(spo.fmin_bfgs(LASSOLoss, thetaInit, gtol=.0000001)).reshape(-1,1)
	print 'ridge'
	thetaRidge = np.array(spo.fmin_bfgs(ridgeRegressionLoss, thetaInit, gtol=.0000001)).reshape(-1,1)

	MSERidge = MSE(predict(thetaRidge, X_test), Y_test)
	print 'MSERidge', MSERidge
	MSELASSO = MSE(predict(thetaLASSO, X_test), Y_test)
	print 'MSE', MSELASSO

	if MSERidge < best_mse2:
		best_lambda2 = current_lambda
		best_mse2 = MSERidge
		print 'best lambda is', best_lambda2

	if MSELASSO < best_mse1:
		best_lambda1 = current_lambda
		best_mse1 = MSELASSO
		print 'best lambda is', best_lambda1

Lambda1 = best_lambda1
Lambda2 = best_lambda2
print Lambda1, Lambda2

## These don't always converge - might be because the actual function is sinusoidal?
print X_train
print Y_train
print thetaInit
#thetaLASSO = gradientDescent(f=LASSOLoss, df=centdiff, init=thetaInit, lr=0.00003, h=.000001, crit=.006, maxIter=1000000)
#print MSE(predict(thetaLASSO, X_test), Y_test)
#print 'theta'
#thetaRidge = gradientDescent(f=ridgeRegressionLoss, df=centdiff, init=thetaInit, lr=0.00003, h=.000001, crit=.006, maxIter=1000000)
#print 'ridge'
thetaLASSO = np.array(spo.fmin_bfgs(LASSOLoss, thetaInit, gtol=.0000001)).reshape(-1,1)
thetaRidge = np.array(spo.fmin_bfgs(ridgeRegressionLoss, thetaInit, gtol=.0000001)).reshape(-1,1)


LASSOPredictions = predict(thetaLASSO, X_test)
print 'predicted lasso test'
ridgePredictions = predict(thetaRidge, X_test)
print 'predicted ridge test'
LASSOPredictionsTrain = predict(thetaLASSO, X_train)
ridgePredictionsTrain = predict(thetaRidge, X_train)

X_lin = np.linspace(-1, 1, 1000)
#Sorry I'm not sorry.
X_lin = np.array([X_lin, np.sin(0.4*np.pi*X_lin*1), np.sin(0.4*np.pi*X_lin*2), np.sin(0.4*np.pi*X_lin*3), np.sin(0.4*np.pi*X_lin*4), np.sin(0.4*np.pi*X_lin*5), np.sin(0.4*np.pi*X_lin*6), np.sin(0.4*np.pi*X_lin*7), np.sin(0.4*np.pi*X_lin*8), np.sin(0.4*np.pi*X_lin*9), np.sin(0.4*np.pi*X_lin*10), np.sin(0.4*np.pi*X_lin*11)])
#X_lin = addInterceptTerm(X_lin)

ActualPredictions = predict(np.transpose(W_true), X_lin)

YLasso_lin = predict(thetaLASSO, X_lin)
YRidge_lin = predict(thetaRidge, X_lin)

#thetaBFGS = spo.fmin_bfgs(LASSOLoss, thetaInit, gtol=.0000001)
#print thetaBFGS

Lambda1 = 0

thetaOLS = gradientDescent(f=LASSOLoss, df=centdiff, init=thetaInit, lr=0.00003, h=.00001, crit=.006, maxIter=1000000)
#thetaOLS = np.array(thetaBFGS).reshape(-1,1)
thetaOLS = np.array(spo.fmin_bfgs(LASSOLoss, thetaInit, gtol=.0000001)).reshape(-1,1)


OLSPredictions = predict(thetaOLS, X_test)
OLSPredictionsTrain = predict(thetaOLS, X_train)
YOLS_lin = predict(thetaOLS, X_lin)

pylab.plot(X_train[0:1,].flatten(), Y_train.flatten(), 'ro', label = 'Actual')
pylab.plot(X_train[0:1,].flatten(), LASSOPredictionsTrain.flatten(), 'bo', label = 'LASSO')
pylab.plot(X_train[0:1,].flatten(), ridgePredictionsTrain.flatten(), 'go', label = 'Ridge')
pylab.plot(X_train[0:1,].flatten(), OLSPredictionsTrain.flatten(), 'co', label = 'OLS')
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Predicted training data values for various models')
plt.legend(loc=1)
pylab.show()

pylab.plot(X_test[0:1,].flatten(), Y_test.flatten(), 'ro', label = 'Actual')
pylab.plot(X_test[0:1,].flatten(), LASSOPredictions.flatten(), 'bo', label = 'LASSO')
pylab.plot(X_test[0:1,].flatten(), ridgePredictions.flatten(), 'go', label = 'Ridge')
pylab.plot(X_test[0:1,].flatten(), OLSPredictions.flatten(), 'co', label = 'OLS')
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Predicted test data values for various models')
plt.legend(loc=1)
pylab.show()


pylab.plot(X_lin[0:1,].flatten(), ActualPredictions.flatten(), 'r', label = 'Actual')
pylab.plot(X_lin[0:1,].flatten(), YLasso_lin.flatten(), 'b', label = 'LASSO')
pylab.plot(X_lin[0:1,].flatten(), YRidge_lin.flatten(), 'g', label = 'Ridge')
pylab.plot(X_lin[0:1,].flatten(), YOLS_lin.flatten(), 'c', label = 'OLS')
pylab.plot(X_train[0:1,].flatten(), Y_train.flatten(), 'bo', label = 'Training Data')
pylab.plot(X_test[0:1,].flatten(), Y_test.flatten(), 'mo', label = 'Test Data')
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Predicted f(x) for various models')
plt.legend(loc=1)
pylab.show()

#print thetaOLS
#print thetaRidge
#print thetaLASSO
#print W_true

d = {'weights' : pd.Series(np.concatenate([thetaOLS, thetaRidge, thetaLASSO, np.transpose(W_true)]).flatten()), \
'Models': pd.Series(np.concatenate([np.repeat(np.array(['OLS']), 12),np.repeat(np.array(['Ridge Regression']), 12),\
	np.repeat(np.array(['LASSO']), 12),np.repeat(np.array(['True']), 12)]).flatten()),\
	'Coefficient': pd.Series(np.repeat(np.array(['w1', 'w2', 'w3', 'w4', 'w5', 'w6', 'w7', 'w8', 'w9', 'w10', 'w11', 'w12']), 4).flatten())}
data = pd.DataFrame(d)
data['Coefficient'] = data['Coefficient'].astype('category')
data['Models'] = data['Models'].astype('category')
print ggplot(data, aes(x='Coefficient', y='weights')) + geom_bar(stat="identity") + facet_wrap('Models', scales='fixed') + \
xlab('Feature') + ylab('Weights') + ggtitle('Feature Weight Distribution by Model')


-0.5964 +-0.3849 +-0.261  +-0.3697 +-0.2602 +-0.5029 +-0.8768 +-0.8314 +-0.824  +-1.1407 +-1.0579 +-0.6955
