""" ReLU Layer """

import numpy as np

class ReLULayer():

	@staticmethod
	def relu(z):
		return np.maximum(np.zeros_like(z), z)
	@staticmethod
	def reluP(z):
		if(z > 0):
			return 1
		else:
			return 0
			

	def __init__(self):
		"""
		Applies the rectified linear unit function element-wise: relu(x) = max(x, 0)
		"""
		self.out = 0
		self.trainable = False # no parameters

	def forward(self, Input):

		############################################################################
	    # TODO: Put your code here
		# Apply ReLU activation function to Input, and return results.

		self.out = self.relu(Input)

		return self.out

	    ############################################################################


	def backward(self, delta):

		############################################################################
	    # TODO: Put your code here
		# Calculate the gradient using the later layer's gradient: delta

		#return np.dot(self.out, np.dot(np.transpose(self.relu(self.out)), delta))
		return delta * self.reluP(self.out)

	    ############################################################################
