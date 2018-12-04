import numpy as np
import random
import matplotlib.pyplot as plt
import math
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
#%matplotlib inline

from net import net
from data import generatedata, generatedataForRegression, get_mnist
from utils import get_one_hot, softmax

from pdb import set_trace

import sklearn
from sklearn.model_selection import train_test_split

random.seed(0)


def main():
	#set hyperparameter at here 
	hiddenlayerlist = [[16,32,16]]	#change the number of hidden layer, and nodes in the layer
	
	ss = 1e-2		   #step Size
	numofiter = 300   #iterations
	size = 2500		  #input size
	dim = 2			 #input dimension
	margin = 0		  #change Margin at here, change this value to 0 to make the data not linear separable
	
	output_unit = 1
	
	algorithm = input('Select algorithm: (input ebp, r+, r-, ir+ or ir-)')
	algorithm = 'r+'
	modeltype = input('Classification or Regression? (input c, r, mnist or bc)')
	modeltype = 'mnist'
	
	
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
	
		network = net(inputdata, outputdata, size, ss, numofiter, dim, hiddenlayerlist, modeltype, algorithm, output_unit, [])
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
		network = net(inputdata, outputdata, size, ss, numofiter,dim, hiddenlayerlist, modeltype, output_unit, [])
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
		
	elif modeltype == 'mnist':
		
		train_images, train_labels, test_images, test_labels = get_mnist()
		
#		size = train_images.shape[0]
		size = 60000
		numofiter = 20
		dim = 28**2
		hiddenlayerlist = [[1000]] # 2500, 2000, 1500, 1000, 500
		output_unit = 10
		
		print('Algorithm: ' + algorithm + '\nModel type: ' + modeltype + '\nIterations: ' + str(numofiter))
		
		# get_one_hot(train_labels[: size, :], 10)
		# train_labels[: size, :].flatten()
		network = net(train_images[: size, :], get_one_hot(train_labels[: size, :], 10), size, ss, numofiter, dim, hiddenlayerlist, modeltype, algorithm, output_unit, [])
		network.backpropagation()
		
		# load the saved model
		filename = 'wb_' + modeltype + '_' + algorithm + '_' + str(numofiter) + '.npz'
		wb_ini = np.load(filename)['arr_0'].tolist()
		network = net(train_images[: size, :], get_one_hot(train_labels[: size, :], 10), size, ss, numofiter, dim, hiddenlayerlist, modeltype, algorithm, output_unit, wb_ini)
		
		# test the accuracy
		
		tst_size = 10000
		
		tst_imgs = train_images[: tst_size]
		tst_lbls = train_labels[: tst_size].flatten()
		
		tst_out_raw = network.forwardewithcomputedW(tst_imgs)
		tst_out_cls = np.argmax(tst_out_raw, axis=1)
		
		accuracy = sum(tst_out_cls == tst_lbls) / tst_size
		print('test accuracy: ' + str(accuracy))
#		set_trace()
	
	elif modeltype == 'bc':
		data = np.genfromtxt("breastCancerData.csv", delimiter = ",")
		label = np.genfromtxt("breastCancerLabels.csv", delimiter = ",")
		MinMaxscaler = sklearn.preprocessing.MinMaxScaler()
		data = np.float32(MinMaxscaler.fit_transform(data))
		#Split Train and Test Data
		trainD, testD , trainT, testT  = train_test_split(data, label, random_state=6)

		size = np.shape(trainD)[0]
		numofiter = 1000
		dim = 9
		hiddenlayerlist = [[80,100,50]] # 2500, 2000, 1500, 1000, 500
		output_unit = 1
		
		network = net(trainD, np.reshape(trainT, (len(trainT),1)), size, ss, numofiter, dim, hiddenlayerlist, modeltype, algorithm, output_unit, [])
		network.backpropagation()
		output = network.forwardewithcomputedW(testD)
		accuracy = sum(output == np.reshape(testT, (len(testT),1))) / len(testT)
		print('test accuracy: ' + str(accuracy[0]))








if __name__ == '__main__':
	main()