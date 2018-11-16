import numpy as np
import random
import matplotlib.pyplot as plt
import math
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
#%matplotlib inline

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
	"""Derivative of the sigmoid function."""
	return sigmoid(x) - np.multiply(sigmoid(x), sigmoid(x))