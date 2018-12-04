import numpy as np
import matplotlib.pyplot as plt
#%matplotlib inline

from utils import ReluD, checkzero, sigmoid, sigmoidD, softmax, softmaxD

from pdb import set_trace

class net:
	def __init__(self, inputdata, outputdata, size, ss, numofiter, dim, hiddenlayerlist, modeltype, algorithm, output_unit, wb_ini):
		self.input = inputdata
		self.output = outputdata
		self.size = size
		self.ss = ss
		self.iter = numofiter
		self.dim = dim
		self.nd = len(hiddenlayerlist[0])
		self.modeltype = modeltype
		self.algorithm = algorithm

		self.loss = []
		self.hiddenunits = hiddenlayerlist
		self.output_unit = output_unit
		
		self.val_accu = []
		
		if wb_ini == []:
			#randomly generate the weights and biases based on the layers and units
			wb = []
			wb.append(np.random.rand(dim + 1, self.hiddenunits[0][0]) * 2 - 1)
			if (self.nd > 1):
				for i in range(1,self.nd):
					wb.append(np.random.rand(self.hiddenunits[0][i - 1] + 1, self.hiddenunits[0][i]) * 2 - 1)
			
			wb.append(np.random.rand(self.hiddenunits[0][-1] + 1, self.output_unit) * 2 - 1)
			self.wb = wb
		else:
			self.wb = wb_ini
	
	#only forward to get the result
	def forwardewithcomputedW(self, testx):
		ones = np.ones((np.shape(testx)[0], 1))
		
		newinput = np.append(testx, ones, axis=1)
		
		z = np.dot(newinput, self.wb[0])
#		a = np.maximum(z, 0)
		a = sigmoid(z)
		
		for i in range(1, self.nd):
			a = np.append(a, ones, axis=1)
			z = np.dot(a, self.wb[i])
#			a = np.maximum(z, 0)
			a = sigmoid(z)
		
		a = np.append(a, ones, axis=1)
		z = np.dot(a, self.wb[-1])
		if self.modeltype == 'c' or self.modeltype == "bc":
			a = sigmoid(z)
			a[a > 0.5] = 1
			a[a <= 0.5] = 0
		elif self.modeltype == 'r':
			a = z
		elif self.modeltype == 'mnist':
			a = softmax(z)
		
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
#			a = np.maximum(z, 0)
			a = sigmoid(z)
			alist.append(a)
			zlist.append(z)
			
			for i in range(1, self.nd):
				a = np.append(a, ones, axis=1)
				z = np.dot(a, self.wb[i])
				zlist.append(z)
#				a = np.maximum(z, 0)
				a = sigmoid(z)
				alist.append(a)
			
			a = np.append(a, ones, axis=1)
			z = np.dot(a, self.wb[-1])
			
			zlist.append(z)
			
			if self.modeltype == 'c' or self.modeltype == "bc":
				a = sigmoid(z)
				a = checkzero(a)
				alist.append(a)
				#modified loss(classification)
				self.loss.append((-1) * np.mean(((1 - self.output) * np.log(1 - alist[-1])) + self.output * np.log(alist[-1])))
				outputerror = ((1 - self.output)/(1 - alist[-1]) - self.output / alist[-1]) * sigmoidD(zlist[-1])
			
			elif self.modeltype == 'r':
				#loss(Regression)
				alist.append(a)
				self.loss.append( np.mean(0.5 * np.square(self.output - zlist[-1]), axis=0))
				outputerror = (zlist[-1] - self.output)
			
			elif self.modeltype == 'mnist':
				a = softmax(z)
#				print(a[0])
				a = checkzero(a)
				alist.append(a)
				#modified loss(classification)
#				self.loss.append((-1) * np.mean(((1 - self.output) * np.log(1 - alist[-1])) + self.output * np.log(alist[-1])))
				self.loss.append(- np.mean(np.sum(self.output * np.log(alist[-1]), axis=1)))
#				outputerror = ((1 - self.output)/(1 - alist[-1]) - self.output / alist[-1]) * softmaxD(zlist[-1])
				outputerror = - (self.output/alist[-1]) * softmaxD(zlist[-1])
#				set_trace()
#				correct_logprobs = -np.log(alist[-1][range(self.size) , self.output])
#				data_loss = np.sum(correct_logprobs) / self.size
#				self.loss.append(data_loss)
#				outputerror = data_loss
			
			
			#backward
			errorlist = [outputerror]
			for j in range(1, self.nd + 1):
				
				tempW = np.delete(np.transpose(self.wb[-j]), -1, axis=1)
#				error = np.multiply(np.dot(errorlist[-j], tempW), ReluD(zlist[-j - 1]))
				error = np.multiply(np.dot(errorlist[-j], tempW), sigmoidD(zlist[-j - 1]))
				errorlist = [error] + errorlist
			
			if self.algorithm == 'ebp':
				newW = []
				
				#updated W and b
				for i in range(0, len(self.wb)):
#					set_trace()
					theW = self.wb[i][0 : -1, :] - (self.ss) * np.dot(np.transpose(alist[i]), errorlist[i]) / self.size
					theB = np.reshape(self.wb[i][-1, :], (1,np.shape(self.wb[i][-1, :])[0])) - (self.ss) * np.reshape(np.mean(errorlist[i], axis=0), (1, np.shape(self.wb[i][-1, :])[0])) / self.size
					newW.append(np.vstack((theW, theB)))
				
				self.wb = newW
			
			elif self.algorithm == 'r+':
				########################## Rprop+ algorithm begin ##########################
				#update W and b in Rprop algorithm
				npos, nneg = 1.2, 0.5
				dmax, dmin = 50.0, 0.000001
				initial_d = 0.0001

				# grad[k][i][j] means the kth layer, the gradient of w_ij
				# prevgrad means the previous gradient
				# d means the delta in the learning rule, it is always > 0
				# dw is d * sign(gradient)
				grad, prevgrad, d, dw = [], [], [], []
				for k in range(0, len(self.wb)):
					# np.shape(self.wb[k])[0] - 1, because the last row of self.wb[k] is bias, we only update weights
					grad.append( np.zeros((np.shape(self.wb[k])[0] - 1, np.shape(self.wb[k])[1])) )
					prevgrad.append( np.zeros(np.shape(grad[k])) )
					dw.append( np.zeros(np.shape(grad[k])) )
					d.append( np.ones(np.shape(grad[k])) * initial_d )
				
				for k in range(0, len(self.wb)):
					grad[k] = np.dot(np.transpose(alist[k]), errorlist[k])
					prev_grad_multiply_grad = prevgrad[k] * grad[k]
					
					gt_index = prev_grad_multiply_grad > 0
					lt_index = prev_grad_multiply_grad < 0
					eq_index = prev_grad_multiply_grad == 0
					
					## prev_grad * grad > 0 ##
					d[k][gt_index] = np.minimum(d[k][gt_index] * npos, dmax)
					dw[k][gt_index] = d[k][gt_index] * np.sign(grad[k][gt_index])
					
					## prev_grad * grad < 0 ##
					d[k][lt_index] = np.maximum(d[k][lt_index] * nneg, dmin)
					grad[k][lt_index] = 0

					## prev_grad * grad == 0 ##
					dw[k][eq_index] = d[k][eq_index] * np.sign(grad[k][eq_index])
					
					self.wb[k][0:-1, :] = self.wb[k][0:-1, :] - dw[k]
					self.wb[k][-1, :] = self.wb[k][-1, :] - self.ss * np.mean(errorlist[k], axis=0) / self.size
					
					prevgrad[k] = grad[k]
				########################## Rprop+ algorithm end ##########################
			elif self.algorithm == 'r-':
				########################## Rprop- algorithm begin ##########################
				#update W and b in Rprop algorithm
				npos, nneg = 1.2, 0.5
				dmax, dmin = 50.0, 0.000001
				initial_d = 0.0001

				# grad[k][i][j] means the kth layer, the gradient of w_ij
				# prevgrad means the previous gradient
				# d means the delta in the learning rule, it is always > 0
				# dw is d * sign(gradient)
				grad, prevgrad, d, dw = [], [], [], []
				for k in range(0, len(self.wb)):
					# np.shape(self.wb[k])[0] - 1, because the last row of self.wb[k] is bias, we only update weights
					grad.append( np.zeros((np.shape(self.wb[k])[0] - 1, np.shape(self.wb[k])[1])) )
					prevgrad.append( np.zeros(np.shape(grad[k])) )
					dw.append( np.zeros(np.shape(grad[k])) )
					d.append( np.ones(np.shape(grad[k])) * initial_d )
				
				for k in range(0, len(self.wb)):
					grad[k] = np.dot(np.transpose(alist[k]), errorlist[k])
					prev_grad_multiply_grad = prevgrad[k] * grad[k]
					
					gt_index = prev_grad_multiply_grad > 0
					lt_index = prev_grad_multiply_grad < 0
					eq_index = prev_grad_multiply_grad == 0
					
					## prev_grad * grad > 0 ##
					d[k][gt_index] = np.minimum(d[k][gt_index] * npos, dmax)
					## prev_grad * grad < 0 ##
					d[k][lt_index] = np.maximum(d[k][lt_index] * nneg, dmin)

					dw[k] = d[k] * np.sign(grad[k])
					
					self.wb[k][0:-1, :] = self.wb[k][0:-1, :] - dw[k]
					self.wb[k][-1, :] = self.wb[k][-1, :] - self.ss * np.mean(errorlist[k], axis=0) / self.size
					
					prevgrad[k] = grad[k]
				########################## Rprop- algorithm end ##########################
			elif self.algorithm == 'ir+':
				########################## iRprop+ algorithm begin ##########################
				#update W and b in Rprop algorithm
				npos, nneg = 1.2, 0.5
				dmax, dmin = 50.0, 0.000001
				initial_d = 0.0001

				# grad[k][i][j] means the kth layer, the gradient of w_ij
				# prevgrad means the previous gradient
				# d means the delta in the learning rule, it is always > 0
				# dw is d * sign(gradient)
				grad, prevgrad, d, dw = [], [], [], []
				for k in range(0, len(self.wb)):
					# np.shape(self.wb[k])[0] - 1, because the last row of self.wb[k] is bias, we only update weights
					grad.append( np.zeros((np.shape(self.wb[k])[0] - 1, np.shape(self.wb[k])[1])) )
					prevgrad.append( np.zeros(np.shape(grad[k])) )
					dw.append( np.zeros(np.shape(grad[k])) )
					d.append( np.ones(np.shape(grad[k])) * initial_d )
				
				for k in range(0, len(self.wb)):
					grad[k] = np.dot(np.transpose(alist[k]), errorlist[k])
					prev_grad_multiply_grad = prevgrad[k] * grad[k]
					
					gt_index = prev_grad_multiply_grad > 0
					lt_index = prev_grad_multiply_grad < 0
					eq_index = prev_grad_multiply_grad == 0
					
					## prev_grad * grad > 0 ##
					d[k][gt_index] = np.minimum(d[k][gt_index] * npos, dmax)
					dw[k][gt_index] = d[k][gt_index] * np.sign(grad[k][gt_index])
					
					## prev_grad * grad < 0 ##
					d[k][lt_index] = np.maximum(d[k][lt_index] * nneg, dmin)
					grad[k][lt_index] = 0
					# print(self.loss)
					try:
						if self.loss[-1] > self.loss[-2]:
							dw[k][lt_index] = -dw[k][lt_index]
						else:
							dw[k][lt_index] = 0
					except:
						dw[k][lt_index] = 0

					## prev_grad * grad == 0 ##
					dw[k][eq_index] = d[k][eq_index] * np.sign(grad[k][eq_index])
					
					self.wb[k][0:-1, :] = self.wb[k][0:-1, :] - dw[k]
					self.wb[k][-1, :] = self.wb[k][-1, :] - self.ss * np.mean(errorlist[k], axis=0) / self.size
					
					prevgrad[k] = grad[k]
				########################## iRprop+ algorithm end ##########################
			elif self.algorithm == 'ir-':
				########################## iRprop- algorithm begin ##########################
				#update W and b in Rprop algorithm
				npos, nneg = 1.2, 0.5
				dmax, dmin = 50.0, 0.000001
				initial_d = 0.0001

				# grad[k][i][j] means the kth layer, the gradient of w_ij
				# prevgrad means the previous gradient
				# d means the delta in the learning rule, it is always > 0
				# dw is d * sign(gradient)
				grad, prevgrad, d, dw = [], [], [], []
				for k in range(0, len(self.wb)):
					# np.shape(self.wb[k])[0] - 1, because the last row of self.wb[k] is bias, we only update weights
					grad.append( np.zeros((np.shape(self.wb[k])[0] - 1, np.shape(self.wb[k])[1])) )
					prevgrad.append( np.zeros(np.shape(grad[k])) )
					dw.append( np.zeros(np.shape(grad[k])) )
					d.append( np.ones(np.shape(grad[k])) * initial_d )
				
				for k in range(0, len(self.wb)):
					grad[k] = np.dot(np.transpose(alist[k]), errorlist[k])
					prev_grad_multiply_grad = prevgrad[k] * grad[k]
					
					gt_index = prev_grad_multiply_grad > 0
					lt_index = prev_grad_multiply_grad < 0
					eq_index = prev_grad_multiply_grad == 0
					
					## prev_grad * grad > 0 ##
					d[k][gt_index] = np.minimum(d[k][gt_index] * npos, dmax)
					## prev_grad * grad < 0 ##
					d[k][lt_index] = np.maximum(d[k][lt_index] * nneg, dmin)
					grad[k][lt_index] = 0

					dw[k] = d[k] * np.sign(grad[k])
					
					self.wb[k][0:-1, :] = self.wb[k][0:-1, :] - dw[k]
					self.wb[k][-1, :] = self.wb[k][-1, :] - self.ss * np.mean(errorlist[k], axis=0) / self.size
					
					prevgrad[k] = grad[k]
				########################## iRprop- algorithm end ##########################
			
			
			if e % 10 == 9:
				
				val_size = 5000
				val_idx = np.random.randint(0, high=self.input.shape[0], size=val_size)
#				set_trace()
				val_imgs = self.input[val_idx]
				val_lbls = self.output[val_idx]
				val_lbls_cls = np.argmax(val_lbls, axis=1)
				
				val_out = self.forwardewithcomputedW(val_imgs)
				val_out_cls = np.argmax(val_out, axis=1)
				
				accuracy = sum(val_out_cls == val_lbls_cls) / val_size
				self.val_accu.append(accuracy)
				
				print('Epoch ' + str(e + 1) + ' finished. Current loss: ' + str(self.loss[-1]) + '. Current validation accuracy: ' + str(accuracy) +'.')
		
		
		filename = self.modeltype + '_' + self.algorithm + '_' + str(self.iter)
		np.savez_compressed('wb_' + filename, self.wb)
		np.savez_compressed('loss_' + filename, self.loss)
		np.savez_compressed('valaccu_' + filename, self.val_accu)
		
		#plot the Loss
		plt.figure(3)
		plt.xlabel('Iterations')
		plt.ylabel('Loss')
		plt.title('Loss Plot')
		plt.plot(range(1, self.iter + 1), self.loss)
#		plt.show()
		
		plt.savefig(filename + '.png')