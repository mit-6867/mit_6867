import numpy as np
from plotBoundary import *
import cvxopt
import sys
# import your SVM training code

# parameters
name = 'nonsep'
print '======Training======'
# load data from csv files
try:
    train = loadtxt('/Users/dholtz/Downloads/hw2_resources/data/data_'+name+'_train.csv')
except:
    train = loadtxt('/Users/mfzhao/Downloads/hw2_resources/data/data_'+name+'_train.csv')
# use deep copy here to make cvxopt happy
X = train[:, 0:2].copy()
Y = train[:, 2:3].copy()

def linear_kernel(x1, x2):
    return np.dot(x1, x2)

def polynomial_kernel(x, y, p=3):
    return (1 + np.dot(x, y)) ** p

def gaussian_kernel(x, y):
    global sigma
    return np.exp(-linalg.norm(x-y)**2 / (2 * (sigma ** 2)))

#X = np.array([[1., 2.], [2., 2.], [0., 0.], [-2., 3.]])
#Y = np.array([1., 1., -1., -1.])
C = 1
kernel = gaussian_kernel


# Carry out training, primal and/or dual
def trainSVM(X, y, kernel, C):
	n_samples, n_features = X.shape
	K = np.zeros((n_samples, n_samples))
	for i in range(n_samples):
		for j in range(n_samples):
			K[i, j] = kernel(X[i], X[j])

	P = cvxopt.matrix(np.outer(y,y) * K)
	q = cvxopt.matrix(np.ones(n_samples) * -1.)
	A = cvxopt.matrix(y, (1, n_samples))
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

#n_features, K, alpha, sv, sv_y, sv_bool, ind = trainSVM(X, Y, kernel, C=C)

# Define the predictSVM(x) function, which uses trained parameters
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


#print 'geometric margin', geometricMarginSVM()

# plot training results
#print '======Plot Training======'
#plotDecisionBoundary(X, Y, predictSVM, [-1, 0, 1], title = 'Linear SVM, stdev4 Training')
#y_predict = predictSVM(X)
#y_predict = np.reshape(y_predict, (len(y_predict), -1))
#correct = float(np.sum(Y == y_predict))/len(Y)
#print correct
#
#
#print '======Validation=======	'
## load data from csv files
#validate = loadtxt('/Users/dholtz/Downloads/hw2_resources/data/data_'+name+'_validate.csv')
#X = validate[:, 0:2]
#Y = validate[:, 2:3]
## plot validation results
#plotDecisionBoundary(X, Y, predictSVM, [-1, 0, 1], title = 'Linear SVM, stdev4 Validation')
#y_predict = predictSVM(X)
#y_predict = np.reshape(y_predict, (len(y_predict), -1))
#correct = float(np.sum(Y == y_predict))/len(Y)
#print correct

for i in (.01, .1, 1, 10, 100):
	#for sig in (.01, .1, 1, 10, 100):
	for sig in ([1.]):
		print 'C = ', i 
		print 'sigma = ', sig 
		sigma = sig
		kernel = linear_kernel
		C = i 
		n_features, K, alpha, sv, sv_y, sv_bool, ind = trainSVM(X, Y, kernel, C=i)
		print 'geometric margin', geometricMarginSVM()