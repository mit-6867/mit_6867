from numpy import *
from plotBoundary import *
import scipy.optimize
import matplotlib as mpl
from ggplot import *
import pandas as pd
import scipy.io
import cvxopt
from cvxopt import matrix
from problem2 import *

def scaling(X):
    list = (4,5,6,7)
    for i in list:
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
    w_opt = scipy.optimize.fmin(LRLoss, w0, args=(X,Y,l))
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
    Y = data[:,11:12]
    return array(X), array(Y)

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

def SVMGrid(Xtr, Ytr, Xv, Yv):
    C = linspace(0,100,101)
    bestErr = 1
    kernel = gaussian_kernel
    bestC = 0
    for i in range(len(C)):
        n_features, K, alpha, sv, sv_y, sv_bool, ind = trainSVM(Xtr, Ytr, kernel, C=C[i])
        yhat = predictSVM(Xv)
        cErr = SVMErr(yhat, Yv)
        if cErr < bestErr:
            bestErr = cErr
            bestC = C[i]
            n_featuresb, Kb, alphab, svb, sv_yb, sv_boolb, indb = n_features, K, alpha, sv, sv_y, sv_bool, ind
    return n_featuresb, Kb, alphab, svb, sv_yb, sv_boolb, indb, bestC

Xtr, Ytr= loadTitan('titanic', 'train')
Xv, Yv= loadTitan('titanic', 'validate')
Xtest, Ytest = loadTitan('titanic', 'test')
scaling(Xtr)
scaling(Xv)
scaling(Xtest)

XtrC = Xtr.copy()
XvC = Xv.copy()
XtestC = Xtest.copy()
YtrC = Ytr.copy()
YvC = Yv.copy()
YtestC = Ytest.copy()

lamb = linspace(0,1,101)
w,l = GridLR(XtrC, YtrC, Xv, Yv, lamb)
CE_LRtr = classifyErr(w, Xtr, Ytr, 0.5)
CE_LRv = classifyErr(w, Xv, Yv, 0.5)
CE_LRtest =classifyErr(w, Xtest, Ytest, 0.5)

print '=========================LR========================='
print 'Optimal w: ', w
print 'Optimal lambda', l

print 'LR Training Classification Error: ',CE_LRtr
print 'LR Validation Classification Error: ',CE_LRtr
print 'LR Test Classification Error: ',CE_LRtr
print '===================================================='
kernel = gaussian_kernel
n_features, K, alpha, sv, sv_y, sv_bool, ind = trainSVM(XtrC, YtrC, kernel, 1)
#n_features, K, alpha, sv, sv_y, sv_bool, ind = SVMGrid(XtrC, YtrC, XvC, YvC)
yhattr = predictSVM(XtrC)
yhatv = predictSVM(XvC)
yhattest = predictSVM(XtestC)
CE_SVMtr = SVMErr(yhattr, YtrC)
CE_SVMv = SVMErr(yhatc, YtrC)
CE_SVMtest = SVMErr(yhattest, YtrC)

w_SVM = np.zeros(n_features)
	for n in range(len(alpha)):
		w_SVM += alpha[n] * sv_y[n] * sv[n]

print '=========================SVM========================='
print 'optimal w:', w_SVM

print 'SVM Training Classification Error: ',CE_LRtr
print 'SVM Validation Classification Error: ',CE_LRtr
print 'SVM Test Classification Error: ',CE_LRtr

print 'geometric margin':, geometricMarginSVM()
print '====================================================='
