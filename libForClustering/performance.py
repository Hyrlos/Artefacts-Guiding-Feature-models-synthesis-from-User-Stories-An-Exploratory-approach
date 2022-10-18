"""
## This module provides performance analytics on various aspects of the classification.
"""

from libForClustering.dataset import Dataset
from libForClustering.preprocessing import Preprocessing
from libForClustering.model import Model
from gensim.models import CoherenceModel
import numpy as np



class Performance:

	"""
	## This class contains all of the performance methods.
	"""

	def __init__(self):
		self.training_data = []
		self.test_data = []


	def variance(self, documents):
		"""
		## This function calculates the variance of the input documents' predicted values.
		### Args :
		* `src.data_structures.Documents` documents : the documents for which you want to calculate the variance of the vectorisation.
		### Returns :
		* int : the variance of the input vectors. The larger the better.
		"""
		return np.var(documents.predicted_values)


	def mean(self, documents, vector_size):
		"""
		## This function calcultates the mean vector of the input vectors.
		### Args :
		* `src.data_structures.Documents` documents : the documents for which the mean of the vectors will be calculated.
		* int vector_size : the size of the vectors used in the model.
		### Returns :
		* int : the mean vector of the input vectors.
		"""
		coords = []
		for i in range(vector_size):
			coords.append([])
		for vector in documents.predicted_values:
			for i in range(vector_size):
				coords[i].append(vector[i])
		mean_vector = []
		for coord in coords:
			mean_vector.append(sum(coord)/vector_size)
		return mean_vector
































