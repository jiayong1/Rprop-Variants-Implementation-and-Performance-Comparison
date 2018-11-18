import numpy as np

from pdb import set_trace

def ReluD(x):
	x[x > 0] = 1
	x[x <= 0] = 0
	return x

def checkzero(x):
	x[x == 0] = 1e-16
	x[x == 1] = 1 - 1e-16
	return x

def sigmoid(x):
	return 1 / (1 + np.exp(-x))

def sigmoidD(x):
	'''Derivative of the sigmoid function.'''
	return sigmoid(x) - np.multiply(sigmoid(x), sigmoid(x))

def softmax(x):
	x = x - np.max(x)
	softmax_x = np.exp(x) / np.sum(np.exp(x))
	return softmax_x

def softmaxD(x):
#	s = softmax(x).reshape(-1,1)
#	return np.diagflat(s) - np.dot(s, s.T)
	return softmax(x) - np.multiply(softmax(x), softmax(x))