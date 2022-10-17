# author : Liam RICE

"""
This module runs tests for the `src.cluster` module.
"""

import unittest
from cluster import Cluster
from data_structures import Documents



class TestCluster(unittest.TestCase):

	cluster = Cluster()

	def test_sort(self):
		documents = Documents(texts=["Document 1", "Document 2", "Document 3", "Document 4"])
		clusters = [0, 1, 0, 2]
		num_topics = 3
		result = [["Document 1", "Document 3"], ["Document 2"], ["Document 4"]]
		self.assertEqual(str(self.cluster.sort(documents, clusters, num_topics)), str(result), "Did not sort correctly")

if __name__ == '__main__':
	unittest.main()











































