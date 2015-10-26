from numpy import *
import scipy.optimize
import matplotlib as mpl
from ggplot import *
import pandas as pd
import scipy.io

def scaling(X):
    for i in range(11):
        xmax = max(X[:,i])
        xmin = min(X[:,i])
        X[:,i] =  (X[:,i] - xmin)/(xmax - xmin)

# Logistic Regression Loss Function
def LRLoss(w, X, Y, l=0):
    return sum(log(1+exp(-hstack(Y)*(dot(X,w[1:len(w)])+w[0]))))+l*dot(w[1:len(w)], w[1:len(w)])

# Prediction Function
def LRPredict(w, X):
    return 1/(1+exp(-(dot(X,w[1:len(w)])+w[0])))
    
# Training Function
def Train(X, Y, l=0):
    w0 = random.randn(12)
    w_opt = scipy.optimize.fmin_bfgs(LRLoss, w0, args=(X,Y,l))
    return w_opt

# Classification Error Function
def classifyErr(w, X, Y, db):
    py = LRPredict(w,X)
    yhat = ones(len(py))
    for i in range(len(py)):
        if py[i] < db:
            yhat[i] = -1
    return 1-mean(Y.flatten() == yhat)

def loadTitan(name, typ):
    file = 'data_'+name+'_'+typ+'.csv'
    try:
        data = scipy.io.loadmat('/Users/dholtz/Downloads/hw2_resources/data/'+file)['data']
    except:
        data = scipy.io.loadmat('/Users/mfzhao/Downloads/hw2_resources/data/'+file)['data']
    X = data[:,0:11]
    scaling(X)
    Y = data[:,11:12]
    return X, Y

def GridLR(Xt, Yt, Xv, Yv, l, db=0.5):
    l_opt = 0
    bestLoss = 10000
    w0 = random.randn(12)
    w_opt = zeros(12)
    vErr = []
    for i in range(len(l)):
        w = scipy.optimize.fmin_bfgs(LRLoss, w0, args=(Xt,Yt,l[i]))
        Loss = LRLoss(w, Xv, Yv, 0)/len(Yv)
        vErr.append(Loss)
        if Loss < bestLoss:
            bestLoss = Loss
            l_opt = l[i]
            w_opt = w
    return w_opt, l_opt

def SVMErr(yhat, y):
    return 1 - mean((yhat == y))

Xtr, Ytr= loadTitan('titanic', 'train')
Xv, Yv= loadTitan('titanic', 'validate')
Xtest, Ytest = loadTitan('titanic', 'test')

lamb = linspace(.3,.4,11)
w,l = GridLR(Xtr, Ytr, Xv, Yv, lamb)
CE_LRtr = classifyErr(w, Xtr, Ytr, 0.5)
CE_LRv = classifyErr(w, Xv, Yv, 0.5)
CE_LRtest = classifyErr(w, Xtest, Ytest, 0.5)

print '=========================LR========================='
print 'Optimal w: ', w
print 'Optimal lambda', l

print 'LR Training Classification Error: ',CE_LRtr
print 'LR Validation Classification Error: ',CE_LRv
print 'LR Test Classification Error: ',CE_LRtest
print '===================================================='
