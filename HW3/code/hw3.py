import numpy as np 
import scipy.io 
import sys
from scipy.special import expit

filename = "/Users/dholtz/Downloads/hw3_resources/toy_multiclass_2_train.csv"
T = scipy.io.loadmat(filename, appendmat=False)['toy_data']

X_train = T[:,0:2]
Y_train = T[:,2:3]
print len(np.unique(Y_train))
print Y_train.shape[0]

def sigmoidFunction( signal, derivative=False ):
    signal = np.clip( signal, -500, 500 )
    signal = expit( signal )
    
    if derivative:
        return np.multiply(signal, 1-signal)
    else:
        return signal

def forwardProp(X_train, w1, w2, b1, b2, activation_functions):
    output = np.dot(X_train, w1) + b1
    activations = activation_functions[0](output)
    output2 = np.dot(activations, w2) + b2
    activations2 = activation_functions[1](output2)
    return output, activations, output2, activations2

def lossFunction(output, Y_train, l, w1, w2):
	Y_train_matrix = np.zeros((Y_train.shape[0], len(np.unique(Y_train))))
	Y_train_locations = np.hstack([np.arange(Y_train.shape[0]).reshape(-1, 1), Y_train-1])
	Y_train_matrix[Y_train_locations[:,0:1].astype(int), Y_train_locations[:,1:].astype(int)] = 1
	loss = np.sum(-Y_train_matrix*np.log(output) - (1-Y_train_matrix)*np.log(1-output)) + l*(np.sum(w1**2) + np.sum(w2**2))
	print loss
	return loss, Y_train_matrix

def neuralNetwork(X_train, Y_train, n1, n2, l, activation_functions, learning_rate):
    # m = number of observations
    # d = number of features
    # n2 = number of classes
    # n1 = number of hidden units
    m = Y_train.shape[0]
    d = X_train.shape[1]
    w1 = (2*np.random.random((d,n1)) - 1)*.001
    w2 = (2*np.random.random((n1, n2)) - 1)*.001
    b1 =  np.zeros((1, n1))
    b2 =  np.zeros((1, n2))
    
    MSE = () 
    c = 1e-5
    error = 10000
    old_error = 1000000
    while np.abs(old_error-error) > c:
    	old_error = error
    	output, activations, output2, activations2 = forwardProp(X_train, w1, w2, b1, b2, activation_functions)
    	error, Y_train_matrix = lossFunction(activations2, Y_train, l, w1, w2)

    	corrections2 = Y_train_matrix*(1-activations2) + (1-Y_train_matrix)*activations2
    	errorTerm2 = np.dot(np.transpose(activations), corrections2)
    	errorTermReg2 = errorTerm2 + 2*l*w2

        activationDeriv = sigmoidFunction(output, derivative=True)

        XtimesCorrections = np.transpose(X_train)
        corrections1 = np.dot(corrections2, np.transpose(w2))
        productTimesW2 = np.dot(np.dot(corrections1, np.transpose(activations)), activationDeriv)
        errorTermReg1 = np.dot(XtimesCorrections, productTimesW2) + 2*l*w1

        b2 = b2 - learning_rate*np.dot(np.transpose(np.ones((m, 1))), corrections2)
        b1 = b1 - learning_rate*np.dot(np.transpose(np.ones((m, 1))), productTimesW2)

        w1 = w1 - learning_rate*errorTermReg1
        w2 = w2 - learning_rate*errorTermReg2

    return w1, w2, b1, b2

def nnPredict(X_train, w1, w2, b1, b2, activation_functions):
	output, activations, output2, Y_matrix = forwardProp(X_train, w1, w2, b1, b2, activation_functions)
	print Y_matrix
	maxClass = np.argmax(Y_matrix, axis=1)
	return maxClass

def classificationError(yPredict, Y):
	classificationError = 1-np.sum(np.array(yPredict.reshape(-1, 1) == Y.reshape(-1, 1), dtype=bool))/float(Y.shape[0])
	print classificationError
	return classificationError

w1, w2, b1, b2 = neuralNetwork(X_train, Y_train, 100, 3, .001, [sigmoidFunction, sigmoidFunction], 1e-5/300)

yPredict = nnPredict(X_train, w1, w2, b1, b2, [sigmoidFunction, sigmoidFunction])

classificationError(yPredict+1, Y_train)