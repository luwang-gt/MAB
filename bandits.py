import numpy as np


class Bandits(object):
	"""
	Multi-armed bandits
	"""
	def __init__(self, k, batch_size=1000, p=None):
		self.k = k
		self.batch_size=batch_size
		self.idx = 0
		self.p = p

	def reset():
		self.idx = 0
		
		self.optimal = 