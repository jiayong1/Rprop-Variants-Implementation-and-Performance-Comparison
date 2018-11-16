import numpy as np
import random
import matplotlib.pyplot as plt
import math
#%matplotlib inline

from net import net
from data import generatedata
from utils import ReluD, checkzero, sigmoid, sigmoidD

from pdb import set_trace

random.seed(0)

def main():
	#set hyperparameter at here 
	hiddenlayerlist = [[18, 20, 25, 16, 15]]	#change the number of hidden layer, and nodes in the layer
	
	ss = 1e-4			#step Size
	numofiter = 20000	#iterations
	size = 200			#input size
	dim = 2				#input dimension
	margin = 0			#change Margin at here, change this value to 0 to make the data not linear separable
	
	#generate the input and output
	inputdata, outputdata = generatedata(size, dim, margin)
	
	#plot to viaualize if it is 1D
	print("Training Data Plot: ")
	plt.figure(1)
	if dim == 1:
		
		plt.scatter(inputdata[: size // 2, 0],np.ones((size // 2, 1)), color="r")
		plt.scatter(inputdata[size // 2 :, 0],np.ones((size // 2, 1)), color="b")
		plt.legend(['Label 1', 'Label 0'], loc='upper right')
	elif dim == 2:
		
		plt.scatter(inputdata[: size // 2, 0],inputdata[: size // 2, 1], color="r")
		plt.scatter(inputdata[size // 2 :, 0],inputdata[size // 2 :, 1], color="b")
		plt.legend(['Label 1', 'Label 0'], loc='upper right')
	
	network = net(inputdata, outputdata, size, ss, numofiter, dim, hiddenlayerlist)
	network.backpropagation()
	output = network.forwardewithcomputedW(inputdata)
	
	#plot network computed result
	output = np.append(inputdata,output, axis=1)
	print("Network computed output: ")
	
	plt.figure(4)
	if dim ==1:
		
		output1 = output[output[:, -1] == 1]
		output2 = output[output[:, -1] == 0]
		plt.scatter(output1[:, 0],np.ones((np.shape(output1)[0], 1)), color="r")
		plt.scatter(output2[:, 0],np.ones((np.shape(output2)[0], 1)), color="b")
		plt.legend(['Label 1', 'Label 0'], loc='upper right')
		
	if dim ==2:
		output1 = output[output[:, -1] == 1]
		output2 = output[output[:, -1] == 0]
		plt.scatter(output1[:, 0], output1[:, 1], color="r")
		plt.scatter(output2[:, 0], output2[:, 1], color="b")
		plt.legend(['Label 1', 'Label 0'], loc='upper right')
	
	plt.show()


if __name__ == '__main__':
	main()