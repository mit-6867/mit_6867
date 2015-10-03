import numpy as np
import pandas as pd
import random
import scipy as sp
import homework1
import sys
import matplotlib as mpl
from matplotlib import pylab as pl
from ggplot import *

np.set_printoptions(precision=4)
# Maximum Likelihood Weight Vector Function assuming Gaussian Errors (aka OLS)
def OLS(X, Y, M=1):
	phi = createTFeatures(X, M)
	XtX = np.dot(np.transpose(phi), phi)
	XtY = np.dot(np.transpose(phi), Y)
	theta = sp.linalg.solve(XtX, XtY)
	return theta

def yPred(X, Y, M=1):
    pl.plot(X.T.tolist()[0],Y.T.tolist()[0], 'gs')
    phi = createTFeatures(X, M)
    w = OLS(X, Y, M)
    pts = np.array([[p] for p in pl.linspace(min(X), max(X), 100)])
    yhat = np.dot(createTFeatures(pts, M), w)
    return yhat

def plot(yhat):
	pts = np.array([[p] for p in pl.linspace(min(X), max(X), 100)])
	true = np.sin(2*np.pi*pts)
	DF = pd.DataFrame({'X': pd.Series(X.flatten()), 'Y': pd.Series(Y.flatten()), 'axis': pd.Series(pts.flatten()), 'yhat': pd.Series(yhat.flatten()), 'true': pd.Series(true.flatten())})
	mpl.rcParams["figure.figsize"] = "2, 4"
	print ggplot(DF, aes(x='axis', y='yhat')) + geom_line(color="red", size=4) + geom_line(aes(y='true'), color='green', size=2) + geom_point(aes(x='X', y='Y'), size = 200, alpha=0.5) + xlim(-0.1,1.1) + xlab('X') + ylab('Y')

def createTFeatures(X=np.zeros(1), M=1):
    features = X**0
    for i in range(1, M+1):
        features = np.hstack([features, np.sin(2*np.pi*X*i)])
    return features

def predict(X, theta):
	features = createTFeatures(X, M=theta.size-1)
	yhat = np.dot(features, theta)
	return yhat

def SSE(theta):
	yhat = predict(X, theta)
	res = yhat - Y.flatten()
	return np.dot(res, res)

def gSSE(theta, *args):
	phi = createFeatures(X, M=theta.size-1)
	return 2*np.dot(phi.T, (predict(X, theta) - Y.flatten()))

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

def gradientDescent(f, df, init, lr=0.3, crit=0.0001, maxIter=100, h=0.001):
	count = 0
	nIter = 0
	fcall = 0
	while count < 2 and nIter < maxIter:
		f_i = f(init)
		# Calculates the gradient
		grad = df(init, f, h)
		# Update step
		init = init - lr * grad
		f_u = f(init)
		# Calculate the difference in f using initial and updated values
		diff = abs((f_i - f_u))
		nIter += 1
		fcall += 2
		if df == centdiff:
			fcall += 2*init.size
		# Tracks successive number of times that difference is below the convergence criterion
		if diff < crit:
			count += 1
		else:
			count = 0

	print "nIter: %d" % (nIter)
	print "Fcall: %d" % (fcall)
	return init

X, Y = homework1.getData('/Users/mfzhao/Downloads/6867_hw1_data/curvefitting.txt')

t0 = OLS(X,Y,0)
t1 = OLS(X,Y,1)
t3 = OLS(X,Y,3)
t9 = OLS(X,Y,9)

t0 = pd.Series(t0.flatten())
t1 = pd.Series(t1.flatten())
t3 = pd.Series(t3.flatten())
t9 = pd.Series(t9.flatten())

DF = pd.DataFrame({'0': t0, '1': t1, '3': t3, '9': t9})
print DF

# 2.1
pts = np.array([[p] for p in pl.linspace(min(X), max(X), 100)])

Y0 = yPred(X,Y,0)
Y1 = yPred(X,Y,1)
Y3 = yPred(X,Y,3)
Y9 = yPred(X,Y,9)

plot(Y0)
plot(Y1)
plot(Y3)
plot(Y9)

# 2.2
random.seed(1)

def rInit(n, M, lr, crit, maxIter):
	err = np.array([])
	for i in range(0,n):
 		s = np.random.uniform(-100, 100, M+1)
		print i
 		theta = gradientDescent(SSE, gSSE, s, lr, crit, maxIter)
		err = np.append(err, SSE(theta))
	return err

def rbfgs(n, M):
	err = np.array([])
	for i in range(0,n):
 		s = np.random.uniform(-100, 100, M+1)
		print i
 		theta = sp.optimize.fmin_bfgs(SSE, s)
		err = np.append(err, SSE(theta))
	return err

SSE(t.flatten())
rInit(20, 9, 0.025, 10**-7, 100000)
rbfgs(20, 9)
