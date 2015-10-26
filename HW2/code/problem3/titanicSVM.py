import numpy as np
from plotBoundary import *
import cvxopt
import sys
from cvxopt import matrix
import scipy.io

def scaling(X):
    for i in range(11):
        xmax = max(X[:,i])
        xmin = min(X[:,i])
        X[:,i] =  (X[:,i] - xmin)/(xmax - xmin)

def loadTitan(name, typ):
    file = 'data_'+name+'_'+typ+'.csv'
    try:
        data = scipy.io.loadmat('/Users/dholtz/Downloads/hw2_resources/data/'+file)['data']
    except:
        data = scipy.io.loadmat('/Users/mfzhao/Downloads/hw2_resources/data/'+file)['data']
    print data.shape
    X = np.array(data[:,0:11]).copy()
    scaling(X)
    Y = np.array(data[:,11:12]).copy()
    print X.shape 
    print Y.shape
    return X, Y

def linear_kernel(x1, x2):
    return np.dot(x1, x2)

def polynomial_kernel(x, y, p=3):
    return (1 + np.dot(x, y)) ** p

def gaussian_kernel(x, y):
    global sigma
    return np.exp(-linalg.norm(x-y)**2 / (2 * (sigma ** 2)))

# Carry out training, primal and/or dual
def trainSVM(X, y, kernel, C):
	n_samples, n_features = X.shape
	K = np.zeros((n_samples, n_samples))
	for i in range(n_samples):
		for j in range(n_samples):
			K[i, j] = kernel(X[i], X[j])

	P = cvxopt.matrix(np.outer(y,y) * K)
	q = cvxopt.matrix(np.ones(n_samples) * -1.)
 	A = cvxopt.matrix(y.astype(float), (1, n_samples))
	b = cvxopt.matrix(0.0)

	G = cvxopt.matrix(np.vstack([np.diag(np.ones(n_samples) * -1.), np.diag(np.ones(n_samples))]))
	h = cvxopt.matrix(np.hstack([np.zeros(n_samples), C*np.ones(n_samples)])) 
 
	# Solve QP problem
	solution = cvxopt.solvers.qp(P, q, G, h, A, b)
	alpha = np.ravel(solution['x'])

	sv_bool = alpha > .00001
	ind = np.arange(len(alpha))[sv_bool]
	alpha = alpha[sv_bool]
	sv_y = y[sv_bool]
	sv = X[sv_bool]
	print "%d support vectors out of %d points" % (len(alpha), n_samples)

	return n_features, K, alpha, sv, sv_y, sv_bool, ind

def predictSVM(X):
	global alpha 
	global sv 
	global sv_y
	global sv_bool 
	global ind 
	global K
	global n_features

	theta_0 = 0
	for n in range(len(alpha)):
		theta_0 += sv_y[n]
		theta_0 -= np.sum(alpha * sv_y * K[ind[n],sv_bool])
		theta_0 /= len(alpha)

		# Weight vector
	if kernel == linear_kernel:
		weight = np.zeros(n_features)
		for n in range(len(alpha)):
			weight += alpha[n] * sv_y[n] * sv[n]
	else:
		weight = None

	if weight != None:
		return np.sign(np.dot(X, weight) + theta_0)

	else:
		if len(np.array(X).shape) == 1:
			n_points = 1 
		else:
			n_points = np.array(X).shape[0]
		y_predict = np.zeros(n_points)
		for i in range(n_points):
			s = 0 
			for a_0, sv_y_0, sv_0 in zip(alpha, sv_y, sv):
				s += a_0*sv_y_0*kernel(X[i], sv_0)
			y_predict[i] = s
		return np.sign(y_predict + theta_0)

def geometricMarginSVM():
	global alpha 
	global sv 
	global sv_y
	global sv_bool 
	global ind 
	global K
	global n_features

	theta_0 = 0
	for n in range(len(alpha)):
		theta_0 += sv_y[n]
		theta_0 -= np.sum(alpha * sv_y * K[ind[n],sv_bool])
		theta_0 /= len(alpha)

	weight = np.zeros(n_features)
	for n in range(len(alpha)):
		weight += alpha[n] + sv_y[n] + sv[n]

	weights = np.append(weight, theta_0)
	#print np.linalg.norm(weights)
	return 1./np.linalg.norm(weights)

def SVMErr(yhat, y):
    return 1 - mean((yhat == y))

def SVMGrid(Xtr, Ytr, Xv, Yv):
    global alpha 
    global sv 
    global sv_y
    global sv_bool 
    global ind 
    global K
    global n_features
    print 'SVMGrid'
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

sigma = 1
kernel = gaussian_kernel
#n_features, K, alpha, sv, sv_y, sv_bool, ind = trainSVM(Xtr, Ytr, kernel, 1)
n_features, K, alpha, sv, sv_y, sv_bool, ind = SVMGrid(Xtr, Ytr, Xv, Yv)
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

print 'geometric margin: ', geometricMarginSVM()
print '====================================================='
