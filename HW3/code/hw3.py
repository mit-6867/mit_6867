import numpy as np 
import scipy.io 
import sys
from scipy.special import expit

filename = "/Users/dholtz/Downloads/hw3_resources/toy_multiclass_1_train.csv"
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

def forwardProp(X_train, w1, w2, activation_functions):
    output = np.dot(X_train, w1[1:,:]) + w1[0:1,:]
    activations = activation_functions[0](output)
    output2 = np.dot(activations, w2[1:,:]) + w2[0:1,:]
    activations2 = activation_functions[1](output2)
    return output, activations, output2, activations2

def lossFunction(output, Y_train, l, w1, w2):
	Y_train_matrix = np.zeros((Y_train.shape[0], len(np.unique(Y_train))))
	Y_train_locations = np.hstack([np.arange(Y_train.shape[0]).reshape(-1, 1), Y_train-1])
	Y_train_matrix[Y_train_locations[:,0:1].astype(int), Y_train_locations[:,1:].astype(int)] = 1
	loss = np.sum(-Y_train_matrix*np.log(output) - (1-Y_train_matrix)*np.log(1-output)) + l*(np.sum(w1[1:,:]**2) + np.sum(w2[1:,:]**2))
	print loss
	return loss, Y_train_matrix

def neuralNetwork(X_train, Y_train, n1, n2, l, activation_functions, learning_rate):
    w1 = 2*np.random.random((X_train.shape[1]+1,n1)) - 1
    w2 = 2*np.random.random((w1.shape[1]+1, n2)) - 1 
    
    MSE = () 
    c = 1e-5
    error = 10000
    old_error = 1000000
    while np.abs(old_error-error) > c:
    	old_error = error
    	output, activations, output2, activations2 = forwardProp(X_train, w1, w2, activation_functions)
    	error, Y_train_matrix = lossFunction(activations2, Y_train, l, w1, w2)

    	corrections2 = Y_train_matrix*(1-activations2) + (1-Y_train_matrix)*activations2
    	activationsBias2 = np.hstack([activations, np.ones((activations.shape[0], 1)).reshape(-1, 1)])
    	errorTerm2 = np.dot(np.transpose(activationsBias2), corrections2)
    	errorTermReg2 = errorTerm2/float(X_train.shape[0]) + 2*l*w2


        X_trainBias = np.hstack([X_train, np.ones((X_train.shape[0], 1)).reshape(-1, 1)])
        activationsBias = np.hstack([activations, np.ones((activations.shape[0], 1)).reshape(-1,1)])
        activationDeriv = sigmoidFunction(output, derivative=True)
        activationDerivBias = np.hstack([activationDeriv, np.ones((activationDeriv.shape[0], 1)).reshape(-1,1)])

        XtimesCorrections = np.dot(np.transpose(X_trainBias), corrections2)
        productTimesW2 = np.dot(XtimesCorrections, np.transpose(w2[1:,:]))
        errorTermReg1 = np.dot(np.dot(productTimesW2, np.transpose(activations)), activationDeriv)/float(X_train.shape[0]) + 2*l*w1

        w1 = w1 - learning_rate*errorTermReg1
        w2 = w2 - learning_rate*errorTermReg2

    return w1, w2

def nnPredict(X_train, w1, w2, activation_functions):
	output, activations, output2, Y_matrix = forwardProp(X_train, w1, w2, activation_functions)
	print Y_matrix
	maxClass = np.argmax(Y_matrix, axis=1)
	return maxClass

def classificationError(yPredict, Y):
	classificationError = np.sum(np.array(yPredict.reshape(-1, 1) == Y.reshape(-1, 1), dtype=bool))/Y.shape[0]
	print classificationError
	return classificationError

w1, w2 = neuralNetwork(X_train, Y_train, 10, 3, .01, [sigmoidFunction, sigmoidFunction], .00005)

yPredict = nnPredict(X_train, w1, w2, [sigmoidFunction, sigmoidFunction])

classificationError(yPredict+1, Y_train)