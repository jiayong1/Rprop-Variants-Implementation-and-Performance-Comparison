import numpy as np
import random
import matplotlib.pyplot as plt
import math
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
#%matplotlib inline

from net import net
from data import generatedata
from utils import ReluD, checkzero, sigmoid, sigmoidD

from pdb import set_trace

random.seed(0)


def main():
	#set hyperparameter at here 
	hiddenlayerlist = [[16,32,16]]	#change the number of hidden layer, and nodes in the layer
	
	ss = 1e-4		   #step Size
	numofiter = 1000   #iterations
	size = 2500		  #input size
	dim = 2			 #input dimension
	margin = 0		  #change Margin at here, change this value to 0 to make the data not linear separable
	
	algorithm = input('Select algorithm: (input r+, r-, ir+ or ir-)')
	
	modeltype = input('Classification or Regression? (input c or r)')
	
	
	if modeltype == 'c':
		
		#generate the input and output for classification
		inputdata, outputdata = generatedata(size, dim, margin)

		#plot to viaualize if it is 1D
		print('Training Data Plot: ')
		plt.figure(1)
		if dim == 1:
			
			plt.scatter(inputdata[: size // 2, 0],np.ones((size // 2, 1)), color='r')
			plt.scatter(inputdata[size // 2 :, 0],np.ones((size // 2, 1)), color='b')
			plt.legend(['Label 1', 'Label 0'], loc='upper right')
		elif dim == 2:
		
			plt.scatter(inputdata[: size // 2, 0],inputdata[: size // 2, 1], color='r')
			plt.scatter(inputdata[size // 2 :, 0],inputdata[size // 2 :, 1], color='b')
			plt.legend(['Label 1', 'Label 0'], loc='upper right')
	
		network = net(inputdata, outputdata, size, ss, numofiter, dim, hiddenlayerlist, modeltype, algorithm)
		network.backpropagation()
		output = network.forwardewithcomputedW(inputdata)
	
		#plot network computed result
		output = np.append(inputdata,output, axis=1)
		print('Network computed output: ')
	
		plt.figure(4)
		if dim ==1:
		
			output1 = output[output[:, -1] == 1]
			output2 = output[output[:, -1] == 0]
			plt.scatter(output1[:, 0],np.ones((np.shape(output1)[0], 1)), color='r')
			plt.scatter(output2[:, 0],np.ones((np.shape(output2)[0], 1)), color='b')
			plt.legend(['Label 1', 'Label 0'], loc='upper right')
		
		if dim ==2:
			output1 = output[output[:, -1] == 1]
			output2 = output[output[:, -1] == 0]
			plt.scatter(output1[:, 0], output1[:, 1], color='r')
			plt.scatter(output2[:, 0], output2[:, 1], color='b')
			plt.legend(['Label 1', 'Label 0'], loc='upper right')
	
		plt.show()
	
	elif modeltype == 'r':
		#generate the input and output for regression
		inputdata, outputdata = generatedataForRegression(size,dim)
		network = net(inputdata, outputdata, size, ss, numofiter,dim, hiddenlayerlist, modeltype)
		network.backpropagation()
		if dim == 2:
			fig = plt.figure(figsize=(10,10))
			ax = plt.axes(projection='3d')
			X = np.arange(-4, 4, 0.1)
			Y = np.arange(-4, 4, 0.1)
			X, Y = np.meshgrid(X, Y)
			a = X.flatten()
			b = Y.flatten()
			testx = np.append(np.reshape(a,(len(a),1)), np.reshape(b,(len(b),1)), axis=1)
			outputy = np.reshape(network.forwardewithcomputedW(testx), np.shape(X))	 
			ax.plot_surface(X, Y, outputy,rstride=1, cstride=1,cmap=cm.coolwarm, linewidth=0, antialiased=False)
		
	


if __name__ == '__main__':
	main()