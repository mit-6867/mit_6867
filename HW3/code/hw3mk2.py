import numpy as np 
import scipy.io 
import sys
from scipy.special import expit
import random

def sigm(x):
    x = np.clip(x, -500, 500)
    return expit(x)

def dsigm(x):
    return sigm(x)*(1-sigm(x))

def fprop(w1, w2, b1, b2, x):
    z2 = np.dot(x,w1) + b1
    a2 = sigm(z2)
    z3 = np.dot(a2, w2) + b2
    a3 = sigm(z3)
    return z2, a2, z3, a3

def oneHot(Y):
	Y_1H = np.zeros((Y.shape[0], len(np.unique(Y))))
	Y_train_locations = np.hstack([np.arange(Y.shape[0]).reshape(-1, 1), Y-1])
	Y_1H[Y_train_locations[:,0:1].astype(int), Y_train_locations[:,1:].astype(int)] = 1
	return Y_1H

def LossFn(py, Y):
    return -Y*np.log(py)-(1-Y)*np.log(1-py)
    
def NNet(X, Y, m, l, a, crit=1e-5, maxiter = 100000):
    n = Y.shape[0]
    d = X.shape[1] 
    k = Y.shape[1]
    w1 = (2*np.random.random((d,m))-1)*0.5
    w2 = (2*np.random.random((m,k))-1)*0.5
    b1 = np.zeros((1,m))
    b2 = np.zeros((1,k))
    it = 0
    count = 0
    lossPrev = 10000000
    lossNow = 100000
    while count < 2:
        lossPrev = lossNow
        z2, a2, z3, a3 = fprop(w1, w2, b1, b2, X)
        lossNow = np.sum(LossFn(a3, Y))
        if np.abs(lossPrev - lossNow) < crit:
            count += 1
        else:
            count = 0
        delta3 = a3 - Y
        w2update = np.dot(np.transpose(a2), delta3)/n + 2*l*w2
        delta2 = np.dot(delta3, np.transpose(w2))*dsigm(z2)
        w1update = np.dot(np.transpose(X), delta2)/n + 2*l*w1
        
        w2 -= a*w2update
        w1 -= a*w1update
        b2 -= delta3.sum(axis=0)/n
        b1 -= delta2.sum(axis=0)/n
        it += 1
        if it == maxiter:
            print 'Maximum number of iterations reached. Convergence Criterion not met.'
            break
    return w1, w2, b1, b2

def sgdNNet(X, Y, m, l, a, n, crit=1e-5, maxiter = 100000):
    d = X.shape[1] 
    k = Y.shape[1]
    w1 = (2*np.random.random((d,m))-1)*0.5
    w2 = (2*np.random.random((m,k))-1)*0.5
    b1 = np.zeros((1,m))
    b2 = np.zeros((1,k))
    it = 0
    count = 0
    lossPrev = 10000000
    lossNow = 100000
    random.seed(84721)
    while count < 2:
        ind = random.sample(range(300), n)
        lossPrev = lossNow
        z2, a2, z3, a3 = fprop(w1, w2, b1, b2, X[ind])
        lossNow = np.sum(LossFn(a3, Y[ind]))
        if np.abs(lossPrev - lossNow) < crit:
            count += 1
        else:
            count = 0
        delta3 = a3 - Y[ind]
        w2update = np.dot(np.transpose(a2), delta3)/n + 2*l*w2
        delta2 = np.dot(delta3, np.transpose(w2))*dsigm(z2)
        w1update = np.dot(np.transpose(X[ind]), delta2)/n + 2*l*w1
        
        w2 -= a*w2update
        w1 -= a*w1update
        b2 -= delta3.sum(axis=0)/n
        b1 -= delta2.sum(axis=0)/n
        it += 1
        if it == maxiter:
            print 'Maximum number of iterations reached. Convergence Criterion not met.'
            break
    return w1, w2, b1, b2

def nnPredict(w1, w2, b1, b2, Xv):
    z2, a2, z3, a3 = fprop(w1, w2, b1, b2, Xv)
    maxClass = np.argmax(a3, axis=1)
    return maxClass + 1

def classificationError(yPredict, Y):
    classificationError = 1-np.sum(np.array(yPredict.reshape(-1, 1) == Y.reshape(-1, 1), dtype=bool))/float(Y.shape[0])
    return classificationError

