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
    return activations2

def lossFunction(output, Y_train, l, w1, w2):
	Y_train_matrix = np.zeros((Y_train.shape[0], len(np.unique(Y_train))))
	Y_train_locations = np.hstack([np.arange(Y_train.shape[0]).reshape(-1, 1), Y_train-1])
	Y_train_matrix[Y_train_locations[:,0:1].astype(int), Y_train_locations[:,1:].astype(int)] = 1
	loss = np.sum(-Y_train_matrix*np.log(output) - (1-Y_train_matrix)*np.log(1-output)) + l*(np.sum(w1**2) + np.sum(w2**2))
	print loss
	return loss

def neuralNetwork(X_train, Y_train, n1, n2, l, activation_functions, learning_rate):
    w1 = 2*np.random.random((X_train.shape[1]+1,n1)) - 1
    w2 = 2*np.random.random((w1.shape[1]+1, n2)) - 1 
    
    MSE = () 
    error_limit = 1e-3
    while MSE > error_limit:
    	output = forwardProp(X_train, w1, w2, activation_functions)
    	error = lossFunction(output, Y_train, l, w1, w2)
	

neuralNetwork(X_train, Y_train, 4, 3, .1, [sigmoidFunction, sigmoidFunction])