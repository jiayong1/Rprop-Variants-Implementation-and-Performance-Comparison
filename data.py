import numpy as np
import matplotlib.pyplot as plt
import math
from matplotlib import cm
#%matplotlib inline

import mnist

from pdb import set_trace

def generatedata(size, dim, margin):
	size = int(size)
	ones = np.ones((size // 2, 1))
	zeros = np.zeros((size // 2, 1))
	
	#check margin here, if not zero, use margin to make data separable, if it is zero, make it not separable
	if margin != 0:
		if dim == 1:
			x1 = np.random.rand(size // 2, 1) * 3 + margin / 2
			x2 = np.random.rand(size // 2, 1) *(-3) - margin / 2
			x = np.vstack((x1, x2))
			y = np.vstack((ones, zeros))
		
		elif dim == 2:
			s1 = np.random.rand(size // 2, 1) * 2 + margin / 2 
			s2 = np.random.rand(size // 2, 1) * (-2) - margin / 2
			x1 = np.random.rand(size, 1) * 4 - 2
			coff = np.random.rand(1, 1) * 4 -2
			b = np.reshape(np.random.random(1) * 4 - 2, (1, 1))
			x2 = np.dot(x1, coff)+ np.asscalar(b) + np.vstack((s1, s2))
			x = np.append(x1, x2, axis=1)
			s1.fill(1)
			s2.fill(0)
			y = np.vstack((s1, s2))
	else:
		if dim == 1:
			x1 = np.random.rand(size // 4, 1) + 1
			x2 = np.random.rand(size // 4, 1) * (-1)
			x3 = np.random.rand(size // 4, 1)
			x4 = np.random.rand(size // 4, 1) * (-1) - 1
			
			x = np.vstack((np.vstack((np.vstack((x1, x2)), x3)), x4))
			y = np.vstack((ones, zeros))
		
		elif dim == 2:
			x1 = np.random.rand(size, 1) * 8 - 4
			s1 = np.random.rand(size // 2, 1) * 2
			s2 = np.random.rand(size // 2, 1) * (-2)
			x2 = np.reshape(3 * np.sum(np.sin(x1), axis=1), (size, 1)) + np.vstack((s1, s2))
			x = np.append(x1, x2, axis=1)
			s1.fill(1)
			s2.fill(0)
			y = np.vstack((s1, s2))
	
	return x, y

def generatedataForRegression(size,dim):
	if dim == 1:
		x =np.reshape(np.linspace(-math.pi, math.pi, num=size), (size, dim))
	else:
		#x = np.random.rand(size,dim)*10 -5
		X = np.arange(-5, 5, 0.2)
		Y = np.arange(-5, 5, 0.2)
		X, Y = np.meshgrid(X, Y)
		a = X.flatten()
		b = Y.flatten()
		x = np.append(np.reshape(a,(len(a),1)), np.reshape(b,(len(b),1)), axis=1)
		size = 2500
	
	y = np.reshape(np.sum(np.sin(x), axis=1), (size,1))
	fig = plt.figure(figsize=(10,10))
	ax = plt.axes(projection='3d')
	out = np.reshape(y, np.shape(X))
	ax.plot_surface(X, Y, out,rstride=1, cstride=1,cmap=cm.coolwarm, linewidth=0, antialiased=False)
	return x, y

def get_mnist():
	
	train_images = mnist.train_images().reshape((-1, 28**2))
	train_labels = mnist.train_labels().reshape((-1, 1))
	
	test_images = mnist.test_images().reshape((-1, 28**2))
	test_labels = mnist.test_labels().reshape((-1, 1))
	
	train_images = train_images / (train_images.max() - train_images.min())
	train_images -= train_images.mean()
	
	test_images = test_images / (test_images.max() - test_images.min())
	test_images -= test_images.mean()
	
	return train_images, train_labels, test_images, test_labels