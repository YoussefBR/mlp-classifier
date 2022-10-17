""" Optimizer Class """

import numpy as np

class SGD():
	def __init__(self, learningRate, weightDecay):
		self.learningRate = learningRate
		self.weightDecay = weightDecay

	# One backpropagation step, update weights layer by layer
	def step(self, model):
		layers = model.layerList
		for layer in layers:
			if layer.trainable:
				############################################################################
			    # TODO: Put your code here
				# Calculate diff_W and diff_b using layer.grad_W and layer.grad_b.
				# Do not forget the weightDecay term.

				# Probably garbage
				# w = np.dot(np.transpose(layer.out), np.dot(layer.out, layer.grad_W))
				# diff_W = (-1 * self.learningRate * w)
				# diff_b = (-1 * self.learningRate * np.dot(np.dot(layer.grad_b, np.transpose(w)), w))

				# Cleaner 3 brown 1 blue eqs
				# diff_W = layer.out.T.dot(layer.sigmoid(layer.outW)  * (1 - layer.sigmoid(layer.outW)))  * np.sum(layer.delta)
				# diff_b = layer.sigmoid(layer.outW) * (1 - layer.sigmoid(layer.outW)) * np.sum(layer.delta)

				# Towards data science eqs
				m = np.shape(layer.delta)[0]

				diff_W = - 1 * self.learningRate * layer.grad_W
				diff_b = - 1 * self.learningRate * (np.sum(layer.grad_b) / m)
    
			    ############################################################################

				# Weight update grad version that doesn't change the weights & biases
				layer.grad_W = (1 - ((self.learningRate * self.weightDecay) / m)) * layer.grad_W + (diff_W)
				layer.grad_b += diff_b
				# Weight update .W and .b version that gives me the previous error
				# layer.W = (1 - ((self.learningRate * self.weightDecay) / m)) * layer.grad_W + (diff_W)
				# layer.b += diff_b
