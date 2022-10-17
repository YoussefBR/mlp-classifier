""" Softmax Cross-Entropy Loss Layer """

import numpy as np

# a small number to prevent dividing by zero, maybe useful for you
EPS = 1e-11

class SoftmaxCrossEntropyLossLayer():
	def __init__(self):
		self.acc = 0.
		self.loss = np.zeros(1, dtype='f')
		self.logit = 0
		self.gt = 0

	def forward(self, logit, gt):
		"""
	      Inputs: (minibatch)
	      - logit: forward results from the last FCLayer, shape(batch_size, 10)
	      - gt: the ground truth label, shape(batch_size, 10)
	    """

		############################################################################
	    # TODO: Put your code here
		# Calculate the average accuracy and loss over the minibatch, and
		# store in self.accu and self.loss respectively.
		# Only return the self.loss, self.accu will be used in solver.py.


		loss = -1 * gt * np.log(np.maximum(EPS, logit))
		self.loss = np.sum(loss)
	
		log = np.argmax(logit, axis = 1)
		gt2 = np.argmax(gt, axis = 1)
		
		hits = 0

		for i in range(log.size):
			if(log[i] == gt2[i]):
				hits += 1

		self.acc = hits / log.size

		
	    ############################################################################

		return self.loss


	def backward(self):

		############################################################################
	    # TODO: Put your code here
		# Calculate and return the gradient (have the same shape as logit)

		gradient = - 1 * self.gt * np.log(np.maximum(EPS, self.logit))
		return gradient
	    ############################################################################
