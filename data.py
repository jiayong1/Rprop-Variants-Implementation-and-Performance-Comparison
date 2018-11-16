import numpy as np
import random
import matplotlib.pyplot as plt
import math
#%matplotlib inline

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
#			s1 = np.random.rand(size//2,1)* 2 -1
#			s2 = np.random.rand(size//2,1)* 2 -1
#			x1 = np.random.rand(size,1)*4 -2
#			coff = np.random.rand(1,1)*4 -2
#			b = np.reshape(np.random.random(1)*4 - 2, (1,1))
#			x2 = np.dot(x1,coff)+ np.asscalar(b) + np.vstack((s1,s2))
#			x = np.append(x1,x2,axis=1)
#			s1.fill(1)
#			s2.fill(0)
#			y = np.vstack((s1,s2))
			x1 = np.random.rand(size, 1) * 8 - 4
			s1 = np.random.rand(size // 2, 1) * 2
			s2 = np.random.rand(size // 2, 1) * (-2)
			x2 = np.reshape(3 * np.sum(np.sin(x1), axis=1), (size, 1)) + np.vstack((s1, s2))
			x = np.append(x1, x2, axis=1)
			s1.fill(1)
			s2.fill(0)
			y = np.vstack((s1, s2))
	
	return x, y