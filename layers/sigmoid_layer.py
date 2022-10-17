""" Sigmoid Layer """

import numpy as np

EPS = 1e-11

class SigmoidLayer():

	def __init__(self):
		"""
		Applies the element-wise function: f(x) = 1/(1+exp(-x))
		"""
		self.out = 0
		self.err = 0
		self.trainable = False

	def forward(self, Input):

		############################################################################
	    # TODO: Put your code here
		# Apply Sigmoid activation function to Input, and return results.

		inp = (Input - np.average(Input))

		self.out = (1 / (1 + np.exp(-inp)))
		return self.out

	    ############################################################################

	def backward(self, delta):

		############################################################################
	    # TODO: Put your code here
		# Calculate the gradient using the later layer's gradient: delta

		#return np.dot(self.out, np.dot(np.transpose(self.sigmoidP(self.out)), delta))
		self.err = delta * self.out * (1 - self.out)
		return self.err

	    ############################################################################
