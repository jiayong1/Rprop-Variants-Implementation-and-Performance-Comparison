import numpy as np
import random
import matplotlib.pyplot as plt
import math
#%matplotlib inline

from utils import ReluD, checkzero, sigmoid, sigmoidD

from pdb import set_trace

class net:
	def __init__(self, inputdata, outputdata, size, ss, numofiter, dim, hiddenlayerlist):
		self.input = inputdata
		self.output = outputdata
		self.size = size
		self.ss = ss
		self.iter = numofiter
		self.dim = dim
		self.nd = len(hiddenlayerlist[0])
		self.loss = []
		self.hiddenunits = hiddenlayerlist
		
		#randomly generate the weights and biases based on the layers and units
		wb = []
		wb.append(np.random.rand(dim + 1, self.hiddenunits[0][0]) * 2 - 1)
		if (self.nd > 1):
			for i in range(1,self.nd):
				wb.append(np.random.rand(self.hiddenunits[0][i - 1] + 1, self.hiddenunits[0][i]) * 2 - 1)
		
		wb.append(np.random.rand(self.hiddenunits[0][-1] + 1, 1) * 2 - 1)
		self.wb = wb
	
	#only forward to get the result
	def forwardewithcomputedW(self, testx):
		ones = np.ones((np.shape(testx)[0], 1))
		
		newinput = np.append(testx, ones, axis=1)
		
		z = np.dot(newinput, self.wb[0])
		a = np.maximum(z, 0)
		
		for i in range(1, self.nd):
			a = np.append(a, ones, axis=1)
			z = np.dot(a, self.wb[i])
			a = np.maximum(z, 0)
		
		a = np.append(a, ones, axis=1)
		z = np.dot(a, self.wb[-1])
		a = sigmoid(z)
		
		a[a > 0.5] = 1
		a[a <= 0.5] = 0
		
		return a
		
	#forward and backward to train the network
	def backpropagation(self):
		ones = np.ones((self.size, 1))
		
		for e in range(self.iter):
			#forward
			#two lists to store a and z
			alist = [self.input]
			newinput = np.append(self.input, ones, axis=1)
			zlist = []
			
			z = np.dot(newinput, self.wb[0])
			a = np.maximum(z, 0)
			alist.append(a)
			zlist.append(z)
			
			for i in range(1, self.nd):
				a = np.append(a, ones, axis=1)
				z = np.dot(a, self.wb[i])
				zlist.append(z)
				a = np.maximum(z, 0)
				alist.append(a)
		
			a = np.append(a, ones, axis=1)
			z = np.dot(a, self.wb[-1])
			
			zlist.append(z)
			a = sigmoid(z)
			a = checkzero(a)
			alist.append(a)
			
			#modified loss
			self.loss.append((-1) * np.mean(((1 - self.output) * np.log(1 - alist[-1])) + self.output * np.log(alist[-1])))
			
			#backward
			
			#modified error
			outputerror = ((1 - self.output)/(1 - alist[-1]) - self.output / alist[-1]) * sigmoidD(zlist[-1])
			
			errorlist = [outputerror]
			for j in range(1, self.nd + 1):
				
				tempW = np.delete(np.transpose(self.wb[-j]), -1, axis=1)
				error = np.multiply(np.dot(errorlist[-j], tempW), ReluD(zlist[-j - 1]))
				errorlist = [error] + errorlist
			
			newW = []
			
			#updated W and b
			for i in range(0, len(self.wb)):

				theW = self.wb[i][0 : -1, :] - (self.ss) * np.dot(np.transpose(alist[i]), errorlist[i]) / self.size
				theB = np.reshape(self.wb[i][-1, :], (1,np.shape(self.wb[i][-1, :])[0])) - (self.ss) * np.reshape(np.mean(errorlist[i], axis=0), (1, np.shape(self.wb[i][-1, :])[0])) / self.size
				newW.append(np.vstack((theW, theB)))
			
			self.wb = newW
		
		#plot the Loss
		plt.figure(3)
		plt.xlabel('Iterations')
		plt.ylabel('Loss')
		plt.title('Loss Plot')
		plt.plot(range(1, self.iter + 1), self.loss)
		plt.show()