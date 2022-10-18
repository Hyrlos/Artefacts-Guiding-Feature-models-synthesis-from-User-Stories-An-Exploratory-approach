
"""
## This module handles all the clustering methods for document vectors.
"""

from sklearn import cluster
from sklearn import metrics
from gensim.models import Word2Vec
import math
import matplotlib.pyplot as plt
from gensim.similarities.docsim import MatrixSimilarity
from nltk.cluster import KMeansClusterer
import nltk
from libForClustering.data_structures import Documents, Document
from libForClustering.model import Model
from libForClustering.performance import Performance



class Cluster:

	def __init__(self):
		self.clusterer = None


	def skl_kmeans(self, num_clusters, repeats=25, maximum_iterations=1000):
		"""
		## This function creates an sklearn K-means algorithm.
		### Args :
		* int num_clusters : the number of clusters.
		* @optional int repeats : the number of times the algorithm repeats with random cluster centre starts.
		* @optional int maximum_iterations : the maximum number of iterations the algorithm makes to converge on it's centres.
		### Returns :
		* KMeans : the created k-means object.
		"""
		kmeans = cluster.KMeans(n_clusters=num_clusters, n_init=repeats, max_iter=maximum_iterations)
		self.clusterer = kmeans
		return kmeans


	def skl_kmeans_fit(self, documents):
		"""
		## This function fits the algorithm to the data contained in documents.
		### Args :
		* `src.data_structures.Documents` documents : the Documents object containing the document vectors.
		### Returns :
		* bool : True if the data is fit to the clustering algorithm, and False if no clustering algorithm was generated.
		"""
		if(self.clusterer == None):
			return False
		else:
			self.clusterer.fit(documents.predicted_values)
			return True


	def skl_kmeans_get_clusters(self, documents):
		"""
		## This function gets the cluster data for the input documents.
		### Args :
		* `src.data_structures.Documents` documents : the Documents object containing the document vectors.
		### Returns :
		* int[] : the array of indexes, each item in the list points to the topic that document was classified as.
		"""
		if(self.clusterer == None):
			return []
		else:
			labels = self.clusterer.predict(documents.predicted_values)
			return labels


	def skl_kmeans_fit_and_cluster(self, documents):
		"""
		## This function is a combined version of the skl_kmeans_fit and skl_kmeans_get_clusters functions.
		### Args :
		* `src.data_structures.Documents` documents : the Documents object containing the document vectors.
		### Returns :
		* int[] : the array of indexes, each item in the list points to the topic that document was classified as.
		"""
		if(self.clusterer == None):
			return []
		else:
			labels = self.clusterer.fit_predict(documents.predicted_values)
			return labels


	def skl_kmeans_fit_and_cluster_vectors(self, vectors):
		"""
		## This function is a combined version of the skl_kmeans_fit and skl_kmeans_get_clusters functions.
		### Args :
		* float[][] vectors : the document vectors.
		### Returns :
		* int[] : the array of indexes, each item in the list points to the topic that document was classified as.
		"""
		if(self.clusterer == None):
			return []
		else:
			labels = self.clusterer.fit_predict(vectors)
			return labels


	def skl_score(self, documents):
		"""
		## This function scores the clusterer.
		### Args :
		* `src.data_structures.Documents` documents : the Documents object containing the document vectors on which the score will be based.
		### Returns :
		* float : the score of the cluster based on the input documents.
		"""
		if(self.clusterer == None):
			return None
		else:
			return self.clusterer.score(documents.predicted_values)


	def write_clustered_data(self, fname, data):
		"""
		## This function writes clustered data to a file so a user can easily view the classification.
		### Args :
		* String fname : the name of the file to write the data to.
		* String[][] data : the sorted data, a list of topics, each containing a list of user stories.
		"""
		write = ""
		for topic in data:
			t = ""
			for doc in topic:
				t = t + doc + "\n"
			t = t + "\n</>\n\n"
			write = write + t
		f = open(fname, "w")
		f.write(write)
		f.close()


	def sort(self, documents, clusters, num_topics):
		"""
		## This function associates the document titles to their clusters.
		### Args :
		* `src.data_structures.Documents` documents : the Documents containing the string reprisentation of the documents.
		* int[] clusters : the clusters to which the documents are classified.
		* int num_topics : the number of topics used in the classification.
		### Returns :
		* string[][] : each array contains the documents that were classified to that topic number.
		"""
		sorted_data = []
		for i in range(num_topics):
			sorted_data.append([])
		for i in range(len(clusters)):
			sorted_data[clusters[i]].append(documents.short_training_set()[i])
		return sorted_data

	def _sort(self, documents, clusters, num_topics, alt_tagged_documents=None):
		"""
		## This function associates the document titles and tokens to their clusters.
		### Args :
		* `src.data_structures.Documents` documents : the documents that were classified.
		* int[] clusters : the clusteres to which the documents are classified.
		* int num_topics : the number of clusters used in the kmeans classification.
		### Returns :
		* (string, string[])[][] : each array contains the documents and tokens that were classified to that topic number.
		"""
		tagged_documents = documents.tagged_documents
		if(alt_tagged_documents != None):
			tagged_documents = alt_tagged_documents
		sorted_data = []
		for i in range(num_topics):
			sorted_data.append([])
		for topic, doc, toks, tags in zip(clusters, documents.short_training_set(), documents.tokenized_documents, self.tagged_doc_to_tuple_array(tagged_documents)):
			sorted_data[topic].append((doc, toks, tags))
		return sorted_data
	

	def _sort_features(self, clusters, feature_list, num_topics):
		"""
		## This function associates the objects in feature_list to the appropriate clusters.
		### Args :
		* int[] clusters : the associated cluster of each document.
		* Tuple[] feature_list : the information wanted to be sorted for each document.
		### Returns :
		* Tuple[][][] : same tuples as present per line in feature_list.
		"""
		sorted_data = []
		for i in range(num_topics):
			sorted_data.append([])
		for topic, feature_details in zip(clusters, feature_list):
			sorted_data[topic].append(feature_details)
		return sorted_data


	def tagged_doc_to_tuple_array(self, tagged_docs):
		tuple_array = []
		for doc in tagged_docs:
			doc_array = []
			for tok, tag in zip(doc[0], doc[1]):
				doc_array.append((tok, tag))
			tuple_array.append(doc_array)
		return tuple_array


	def graph_elbow_vec(self, docs, min_clusters, max_clusters, jump=1):
		"""
		## This function generates the elbow graph for all k-means algorithms going from the minimum number of clusters to the maximum number of clusters.
		### Args :
		* `src.data_structures.Documents` docs : the Documents that will be fit to the k-means algorithms.
		* int min_clusters : the minimum number of clusters.
		* int max_clusters : the maximum number of clusters.
		* @optional int jump : the model skips this value number of k-means, so for jump=3 you would get num_clusters=1 -> 4 -> 7 -> etc...
		"""
		distortions = []
		K = range(min_clusters, max_clusters+1, jump)
		for k in K:
			model = self.skl_kmeans(k)
			model.fit(docs.predicted_values)
			distortions.append(model.inertia_)
		plt.figure(figsize=(16,8))
		plt.plot(K, distortions, 'bx-')
		plt.xlabel('k')
		plt.ylabel('Distortion')
		plt.title('The elbow method showing the optimal number of clusters')
		plt.show()


	def graph_elbow_vec_vector(self, vectors, min_clusters, max_clusters, jump=1):
		"""
		## This function generates the elbow graph for all k-means algorithms going from the minimum number of clusters to the maximum number of clusters.
		### Args :
		* float[][] docs : the vectors that will be fit to the k-means algorithms.
		* int min_clusters : the minimum number of clusters.
		* int max_clusters : the maximum number of clusters.
		* @optional int jump : the model skips this value number of k-means, so for jump=3 you would get num_clusters=1 -> 4 -> 7 -> etc...
		"""
		distortions = []
		K = range(min_clusters, max_clusters+1, jump)
		for k in K:
			model = self.skl_kmeans(k)
			model.fit(vectors)
			distortions.append(model.inertia_)
		plt.figure(figsize=(16,8))
		plt.plot(K, distortions, 'bx-')
		plt.xlabel('k')
		plt.ylabel('Distortion')
		plt.title('The elbow method showing the optimal number of clusters')
		plt.show()
	

	def get_optimal_elbow_num(self, vectors, min_clusters, max_clusters, jump=1):
		"""
		## This function finds the ideal number of clusters to generate from a set of vectors.
		### Args :
		* float[][] docs : the vectors that will be fit to the k-means algorithms.
		* int min_clusters : the minimum number of clusters.
		* int max_clusters : the maximum number of clusters.
		* @optional int jump : the model skips this value number of k-means, so for jump=3 you would get num_clusters=1 -> 4 -> 7 -> etc...
		"""
		distortions = []
		K = range(min_clusters, max_clusters+1, jump)
		for k in K:
			model = self.skl_kmeans(k)
			model.fit(vectors)
			distortions.append(model.inertia_)
		acc = []
		if len(distortions) > 0:
			prev = distortions[0]
			min = -1
			max = distortions[0]
			for i in range(1, len(distortions)):
				j = distortions[i] - prev
				prev = distortions[i]
				acc.append(j)
				if distortions[i] < min or min == -1:
					min = distortions[i]
				if distortions[i] > max:
					max = distortions[i]
			range_reach = max-min
			prev = acc[0]
			sol = 0
			found = False
			for i in range(1, len(acc)):
				if prev != 0:
					if abs(acc[i]) < range_reach * 0.01 and not found:
						sol = i+min_clusters
						found = True
					prev = acc[i]
			# if not found or the solution is the number of vectors to cluster, then graph is not in a 1/x shape and so we return the halfway point
			if not found or sol >= len(vectors)-1:
				sol = math.floor(len(vectors)/2)
			return sol
		elif len(distortions) == 1:
			return distortions[0]
		else:
			return 2



	def main(self):
		from gensim.test.utils import common_corpus, common_dictionary
		from gensim.similarities import MatrixSimilarity

		query = [(1, 2), (5, 4)]
		index = MatrixSimilarity(common_corpus, num_features=len(common_dictionary))
		sims = index[query]

		from gensim.corpora.textcorpus import TextCorpus
		from gensim.test.utils import datapath, get_tmpfile
		from gensim.similarities import Similarity

		corpus = TextCorpus(datapath('testcorpus.mm'))
		index_temp = get_tmpfile("index")
		index = Similarity(index_temp, corpus, num_features=400)  # create index
		query = next(iter(corpus))
		result = index[query]  # search similar to `query` in index
		print(len(index))

		for sims in index[corpus]:  # if you have more query documents, you can submit them all at once, in a batch
			len(sims)

		# There is also a special syntax for when you need similarity of documents in the index
		# to the index itself (i.e. queries=indexed documents themselves). This special syntax
		# uses the faster, batch queries internally and **is ideal for all-vs-all pairwise similarities**:

		for similarities in index:  # yield similarities of the 1st indexed document, then 2nd...
			len(sims)



if __name__ == '__main__':
	c = Cluster()
	c.main()



































