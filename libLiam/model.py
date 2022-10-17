# author : Liam RICE

"""
## This module handles models used for classifying or vectorising data.
### Models on offer :
* Latent Semantic Indexing
* Latent Dirichlet Allocation
* Word2Vec
* Doc2Vec
"""

import multiprocessing
import random
import numpy as np
import gensim
import gensim.downloader
from gensim.models.ldamulticore import LdaMulticore
from gensim.models.lsimodel import LsiModel
from gensim.models.word2vec import Word2Vec
from gensim.models.doc2vec import Doc2Vec
from gensim.interfaces import TransformedCorpus
from libLiam.data_structures import Documents



class DocTermMatrixIterable(object):
	"""
	## This class contains a Dictionary and LineSentence object for use in LSI and LDA stream training.
	"""
	def __init__(self, line_sentence, dictionary):
		self.dictionary = dictionary #: Dictionary : the dictionary object linking to the data contained in the LineSentence object.
		self.line_sentence = line_sentence #: LineSentence : the LineSentence object linking to a file containing the tokenized documents.

	def __iter__(self):
		for tokens in self.line_sentence:
			yield self.dictionary.doc2bow(tokens)



class Model:

	def __init__(self):
		self.model = None
		self.vec_model = Word2Vec()
		self.vec_size = 0


	### STANDARD MODELS ###


	def lda(self, documents, num_topics=50, iterations=10, seed=None):
		"""
		## This function creates and trains a Latent Dirichlet Allocation model from the input document-term matrix and dictionary.
		### Args :
		* `src.data_structures.Documents` documents : The Documents object containing the tokenized documents, document-term matrix and dictionary.
		* @Optional int num_topics : the size of the document vectors, each vector dimension reprisents a topic.
		* @Optional int iterations : the number of training iterations.
		* @Optional int seed : the random seed, used to create reproductible models.
		### Returns :
		* LdaMulticore : returns a gensim LdaMulticore object.
		* int : returns an int corresponding to the seed used to pregenerate this model.
		"""
		if(seed==None):
			seed=random.randint(0, 50000)

		lda = LdaMulticore(corpus=documents.document_term_matrix, id2word=documents.document_dictionary, num_topics=num_topics, workers=multiprocessing.cpu_count(), iterations=iterations, random_state=seed)
		self.model = lda
		self.vec_size = num_topics
		return lda, seed


	def lda_stream(self, line_sentence, documents, num_topics=50, iterations=10, chunksize=500, num_cores=multiprocessing.cpu_count(), seed=None):
		"""
		## This function creates and trains a Latent Dirichlet Allocation model from the input document-term matrix and dictionary.
		### Args :
		* LineSentence line_sentence : the LineSentence object that we will iterate over.
		* `src.data_structures.Documents` documents : The Documents object containing the tokenized documents, document-term matrix and dictionary.
		* @Optional int num_topics : the size of the document vectors, each vector dimension reprisents a topic.
		* @Optional int iterations : the number of training iterations.
		* @Optional int chunksize : the size of each chunk loaded into memory for training.
		* @Optional int seed : the random seed, used to create reproductible models.
		### Returns :
		* LdaMulticore : returns a gensim LdaMulticore object.
		* int : returns an int corresponding to the seed used to pregenerate this model.
		"""
		if(seed==None):
			seed=random.randint(0, 50000)
		documents.tokenized_documents = []
		dtmi = DocTermMatrixIterable(line_sentence, documents.document_dictionary)
		lda = LdaMulticore(corpus=dtmi, id2word=documents.document_dictionary, num_topics=num_topics, workers=num_cores, iterations=iterations, random_state=seed, chunksize=chunksize)
		self.model = lda
		self.vec_size = num_topics
		return lda, seed


	def lda_load_local(self, fname):
		"""
		## This function loads a Latent Dirichlet Allocation model off of a local model file.
		### Args :
		* String fname : the path to the file from which to load the LDA model.
		### Returns :
		* LdaMulticore : returns the pretrained gensim LdaMulticore object or None.
		"""
		try:
			lda = LdaMulticore.load(fname)
			self.model = lda
			return lda
		except:
			print("Load",fname,"failed - file not found.")
			return None


	def _lsi_lda_results_to_vec(self, documents, results, num_topics=50):
		"""
		## This function translates the results output of LSI and LDA models to standard vectors that can be used in k-means.
		### Args :
		* `src.data_structures.Documents` documents : the Documents object containing the documents to classify.
		* (int, float)[][] results : an array of the output of get_value function that gets the reprisentations of LSI and LDA models.
		* @optional int num_topics : the number of topics used in the LSI/LDA model.
		### Returns :
		* float[][] : the array of vectors reprisenting the input results
		"""
		vectors = []
		for result in results:
			vec = []
			num = 0
			for i in range(num_topics):
				if(num < len(result)):
					if(i == result[num][0]):
						vec.append(result[num][1])
						num += 1
					else:
						vec.append(0)
				else:
					vec.append(0)
			vectors.append(vec)
		documents.predicted_values = vectors
		return vectors


	def lsi(self, documents, num_topics=50, iterations=10, seed=None):
		"""
		## This function creates and trains a Latent Semantic Indexing model from the input document-term matrix and dictionary.
		### Args :
		* `src.data_structures.Documents` documents : the Documents object containing the tokenized documents, document-term matrix and dictionary.
		* @Optional int num_topics : the size of the document vectors, each vector dimension reprisents a topic.
		* @Optional int iterations : the number of training iterations.
		* @Optional int seed : the random seed, used to create reproductible models.
		### Returns :
		* LsiModel : returns a gensim LsiModel object.
		* int : returns an int corresponding to the seed used to pregenerate this model.
		"""
		if(seed==None):
			seed=random.randint(0, 50000)
		lsi = LsiModel(corpus=documents.document_term_matrix, num_topics=num_topics, id2word=documents.document_dictionary, power_iters=iterations, random_seed=seed)
		self.model = lsi
		self.vec_size = num_topics
		return lsi, seed


	def lsi_stream(self, line_sentence, documents, num_topics=50, iterations=10, chunksize=5000, seed=None):
		"""
		## This function creates and trains a Latent Semantic Indexing model from the input document-term matrix and dictionary.
		### Args :
		* LineSentence line_sentence : the LineSentence object that we will iterate over.
		* `src.data_structures.Documents` documents : the Documents object containing the tokenized documents, document-term matrix and dictionary.
		* @Optional int num_topics : the size of the document vectors, each vector dimension reprisents a topic.
		* @Optional int iterations : the number of training iterations.
		* @Optional int chunksize : the size of each chunk used during training.
		* @Optional int seed : the random seed, used to create reproductible models.
		### Returns :
		* LsiModel : returns a gensim LsiModel object.
		* int : returns an int corresponding to the seed used to pregenerate this model.
		"""
		if(seed==None):
			seed=random.randint(0, 50000)
		documents.tokenized_documents = []
		dtmi = DocTermMatrixIterable(line_sentence, documents.document_dictionary)
		lsi = LsiModel(corpus=dtmi, num_topics=num_topics, id2word=documents.document_dictionary, power_iters=iterations, random_seed=seed, chunksize=chunksize)
		self.model = lsi
		self.vec_size = num_topics
		return lsi, seed


	def lsi_load_local(self, fname):
		"""
		## This function loads a Latent Semantic Indexing model off of a local model file.
		### Args :
		* String fname : the path to the file from which to load the LSI model.
		### Returns :
		* LsiModel : returns the pretrained gensim LsiModel object or None.
		"""
		try:
			lsi = LsiModel.load(fname)
			self.model = lsi
			return lsi
		except:
			print("Load",fname,"failed - file not found.")
			return None


	def get_value(self, document):
		"""
		## This function gets the vector value of an input document from a previously generated LSI or LDA model.
		### Args :
		* (int, int)[] document : the input document, which has been processed into a document-term matrix.
		### Returns :
		* (int, float)[] : returns the vector reprisentation of the document by LSI or LDA model (whichever was generated previously).
		"""
		try:
			if(self.model != None):
				return self.model[document]
			else:
				print("Error - model not loaded.")
				return None
		except KeyError:
			return 0


	def get_values(self, documents, num_topics=50):
		"""
		## This function gets the vector value of all input documents from LSI or LDA models.
		### Args :
		* `src.data_structures.Documents` documents : the documents to get the document vectors.
		* @optional int num_topics : the number of topics used in the LSI or LDA model.
		### Returns :
		* float[][] : returns the vector reprisentation of the documents.
		"""
		data = []
		for doc in documents.document_term_matrix:
			value = self.get_value(doc)
			data.append(value)
		results = self._lsi_lda_results_to_vec(documents, data, num_topics)
		return results


	### VECTOR MODELS ###

	
	def word2vec(self, documents, vector_size=100, iterations=5, use_skip_gram=False, use_hierarchical_softmax=False, seed=None):
		"""
		## This function creates a new Word2Vec model from the input arrays of words.
		### Args :
		* `src.data_structures.Documents` documents : the Documents object containing the tokenized documents.
		* @Optional int vector_size : the size of the word vectors used.
		* @Optional int iterations : the number of training iterations over the training corpus.
		* @Optional bool use_skip_gram : if True, the model will use skip-gram rather than bag-of-words.
		* @Optional bool use_hierarchical_softmax : if True, the model will use hierarchical softmax.
		* @Optional int seed : the random seed, used to create reproductible models.
		### Returns :
		* Word2Vec : returns a gensim Word2Vec object.
		* int : returns an int corresponding to the seed used to pregenerate this model.
		"""
		if(seed==None):
			seed=random.randint(0, 50000)

		sg = 0
		if(use_skip_gram):
			sg = 1

		hs = 0
		if(use_hierarchical_softmax):
			hs = 1

		t_data = documents
		if(isinstance(documents, Documents)):
			t_data = documents.tokenized_documents
		if(isinstance(documents, TransformedCorpus)):
			t_data = documents

		w2v = Word2Vec(sentences=t_data, vector_size=vector_size, window=5, min_count=1, workers=multiprocessing.cpu_count(), sg=sg, hs=hs, epochs=iterations, compute_loss=True, seed=seed)
		self.vec_model = w2v
		self.vec_size = vector_size
		return w2v, seed


	def word2vec_stream(self, line_sentence, vector_size=100, iterations=5, use_skip_gram=False, use_hierarchical_softmax=False, chunksize=5000, seed=None):
		"""
		## This function creates a new Word2Vec model from the input arrays of words.
		### Args :
		* LineSentence line_sentence : the LineSentence object that we will iterate over.
		* @Optional int vector_size : the size of the word vectors used.
		* @Optional int iterations : the number of training iterations over the training corpus.
		* @Optional bool use_skip_gram : if True, the model will use skip-gram rather than bag-of-words.
		* @Optional bool use_hierarchical_softmax : if True, the model will use hierarchical softmax.
		* @Optional int chunksize : the size of each chunk used for training.
		* @Optional int seed : the random seed, used to create reproductible models.
		### Returns :
		* Word2Vec : returns a gensim Word2Vec object.
		* int : returns an int corresponding to the seed used to pregenerate this model.
		"""
		if(seed==None):
			seed=random.randint(0, 50000)

		sg = 0
		if(use_skip_gram):
			sg = 1

		hs = 0
		if(use_hierarchical_softmax):
			hs = 1

		w2v = Word2Vec(corpus_file=line_sentence, vector_size=vector_size, window=5, min_count=1, workers=multiprocessing.cpu_count(), sg=sg, hs=hs, epochs=iterations, compute_loss=True, seed=seed, batch_words=chunksize)
		self.vec_model = w2v
		self.vec_size = vector_size
		return w2v, seed


	def word2vec_load(self, model_to_fetch):
		"""
		## This function loads a pretrained Word2Vec model from the internet.
		### Args :
		* String model_to_fetch : the exact name of the model to fetch from the internet.
		### Returns :
		* Word2Vec : returns the pretrained gensim Word2Vec object or None.
		"""
		try:
			w2v = gensim.downloader.load(model_to_fetch)
			self.vec_model.wv = w2v
			return w2v
		except:
			print("Load",model_to_fetch,"failed - target not found.")
			return None


	def word2vec_load_local(self, fname):
		"""
		## This function loads a pretrained Word2Vec model from a local file.
		### Args :
		* String fname : the path to the file from which to load the Word2Vec model.
		### Returns :
		* Word2Vec : returns the pretrained gensim Word2Vec object or None.
		"""
		try:
			w2v = Word2Vec.load(fname)
			self.vec_model = w2v
			return w2v
		except:
			print("Load",fname,"failed - file not found.")
			return None


	def word2vec_get_vector(self, word):
		"""
		## This function gets a word's corresponding vector from the generated Word2Vec model.
		### Args :
		* String word : the word for which you want the vector.
		### Returns :
		* float[] : returns a float array that corresponds to the input word's vector or None.
		"""
		if(self.vec_model.wv != None):
			return self.vec_model.wv[word]
		else:
			print("Error - model not loaded.")
			return None


	def word2vec_get_vectors(self, documents, vector_size=100):
		"""
		## This function gets the document vectors from the word_to_vec model.
		### Args :
		* `src.data_structures.Documents` documents : the Documents object containing the documents to translate to document vectors.
		* @optional int vector_size : the size of the vectors used in the word2vec model.
		* @optional String length : the length of each document to use, "s" for short, "l" for long and "f" for full. "s" by default.
		### Returns :
		* float[][] : the document vectors for each document.
		"""
		original_documents = documents.tokenized_documents
		vectors = []
		for doc in original_documents:
			vals = []
			for i in range(vector_size):
				vals.append([])
			for word in doc:
				try:
					wv = self.word2vec_get_vector(word)
					for i in range(vector_size):
						vals[i].append(wv[i])
				except:
					for i in range(vector_size):
						vals[i].append(0)
			for i in range(vector_size):
				vec = []
				for list_coord in vals:
					vec.append(sum(list_coord))
			vectors.append(vec)
		documents.predicted_values = vectors
		return vectors


	def doc2vec(self, documents, vector_size=100, iterations=10, use_distributed_memory=False, use_hierarchical_softmax=False, use_dbow_and_skip_gram=False, seed=None):
		"""
		## This function creates a Doc2Vec model from the input tagged documents.
		### Args :
		* `src.data_structures.Documents` documents : the Documents object containing the array of TaggedDocuments instead of normal tokenized documents.
		* @Optional int vector_size : the size of the document vectors used.
		* @Optional int iterations : the number of training iterations over the training corpus.
		* @Optional bool use_distributed_memory : if True, the model will used distributed memory, else will use distributed bag-of-words.
		* @Optional bool use_dbow_and_skip_gram : if True, the model will use both bag-of-words doc-vectors and skip-gram word-vectors instead of just doc-vectors
		* @Optional int seed : the random seed, used to create (mostly) reproductible models.
		### Returns :
		* Doc2Vec : returns a gensim Doc2Vec object.
		* int : returns an int corresponding to the seed used to pregenerate this model.
		"""
		if(seed==None):
			seed=random.randint(0, 50000)

		dm = 0
		if(use_distributed_memory):
			dm = 1

		hs = 0
		if(use_hierarchical_softmax):
			hs = 1

		dbw = 0
		if(use_dbow_and_skip_gram):
			dbw = 1

		t_data = documents
		if(isinstance(documents, Documents)):
			t_data = documents.tagged_documents
		if(isinstance(documents, TransformedCorpus)):
			t_data = documents

		try:
			d2v = Doc2Vec(documents=t_data, dm=dm, vector_size=vector_size, window=5, min_count=1, workers=multiprocessing.cpu_count(), epochs=iterations, hs=hs, dm_mean=1, dbow_words=dbw, max_vocab_size=50000000, seed=seed)
			self.vec_model = d2v
			self.vec_size = vector_size
			return d2v, seed
		except:
			print("ERROR GENERATING DOC2VEC")
			print(t_data)


	def doc2vec_stream(self, line_sentence, vector_size=100, iterations=10, use_distributed_memory=False, use_hierarchical_softmax=False, use_dbow_and_skip_gram=False, seed=None):
		"""
		## This function creates a Doc2Vec model from the input tagged documents.
		### Args :
		* LineSentence line_sentence : the LineSentence object that we will iterate over.
		* @Optional int vector_size : the size of the document vectors used.
		* @Optional int iterations : the number of training iterations over the training corpus.
		* @Optional bool use_distributed_memory : if True, the model will used distributed memory, else will use distributed bag-of-words.
		* @Optional bool use_dbow_and_skip_gram : if True, the model will use both bag-of-words doc-vectors and skip-gram word-vectors instead of just doc-vectors
		* @Optional int chunksize : the size of the chunks used for training.
		* @Optional int seed : the random seed, used to create (mostly) reproductible models.
		### Returns :
		* Doc2Vec : returns a gensim Doc2Vec object.
		* int : returns an int corresponding to the seed used to pregenerate this model.
		"""
		if(seed==None):
			seed=random.randint(0, 50000)

		dm = 0
		if(use_distributed_memory):
			dm = 1

		hs = 0
		if(use_hierarchical_softmax):
			hs = 1

		dbw = 0
		if(use_dbow_and_skip_gram):
			dbw = 1

		d2v = Doc2Vec(corpus_file=line_sentence, dm=dm, vector_size=vector_size, window=5, min_count=1, workers=multiprocessing.cpu_count(), epochs=iterations, hs=hs, dm_mean=1, dbow_words=dbw, max_vocab_size=50000000, seed=seed)
		self.vec_model = d2v
		self.vec_size = vector_size
		return d2v, seed


	def doc2vec_load_local(self, fname):
		"""
		## This function loads a Doc2Vec model from a local model file.
		### Args :
		* String fname : the pathto the file from which to load the Doc2Vec model.
		### Returns :
		* Doc2Vec : returns the pretrained gensim Doc2Vec object or None.
		"""
		try:
			d2v = Doc2Vec.load(fname)
			self.vec_model = d2v
			return d2v
		except:
			print("Load",fname,"failed - file not found.")
			return None


	def doc2vec_get_vector(self, document):
		"""
		## This function gets the doc-vector for the input untagged document.
		### Args :
		* String[] document : the list of words corresponding to a document.
		### Returns :
		* float[] : returns the float array corresponding to the doc-vector of the input document or None.
		"""
		if(self.vec_model != None):
			return self.vec_model.infer_vector(document)#, epochs=50)
		else:
			print("Error - model not loaded.")
			return None


	def doc2vec_get_vectors(self, documents):
		"""
		## This function gets the document vectors trained by the doc2vec model.
		### Args :
		* `src.data_structures.Documents` documents : the documents to turn into document vectors.
		### Returns :
		* float[][] : returns the array of document vectors corresponding to the input documents.
		"""
		vectors = []
		for doc in documents.tokenized_documents:
			vectors.append(self.doc2vec_get_vector(doc))
		documents.predicted_values = vectors
		return vectors


	def vec_model_loss(self):
		"""
		## This function gets the loss function from a Word2Vec or Doc2Vec model, whichever was generated most recently.
		### Returns :
		* float : returns a float corresponding to the latest training loss value.
		"""
		return self.vec_model.get_latest_training_loss()



	### GENERIC FUNCTIONS ###

	
	def save_model(self, fname, model):
		"""
		## This function saves a model to a file, works for Word2Vec, Doc2Vec, LsiModel and LdaMulticore.
		### Args :
		* String fname : the path to the file to which to save the model.
		* LsiModel || LdaMulticore || Doc2Vec || Word2Vec model : the model to save to the local file.
		### Returns :
		* bool : returns True is the document was saved correctly, or false if not.
		"""
		try:
			model.save(fname)
			return True
		except:
			return False


	# first do topics_to_vector for LSI and LDA
	def to_ndarray(self, data, vector_size=100):
		final = []
		for obj in data:
			if(len(obj) == vector_size):
				a = np.array(obj)
				final.append(a)
		return final


	def main(self):
		import math
		from preprocessing import Preprocessing
		from dataset import Dataset

		stats = []
		for num_topics in range(10, 11):
			d = Dataset()
			p = Preprocessing()
			
			training_documents = d.read_from_file("data/user_story_training_dataset.txt")
			documents_to_sort = d.read_from_file("data/user_story_datasets/g04-recycling.txt")
			s_training_documents = p.remove_subject(training_documents)
			s_documents_to_sort = p.remove_subject(documents_to_sort)
			dictionary, training_dtm, dtm_to_sort = p.preprocess_lsi_lda(s_training_documents, s_documents_to_sort)
			self.lda(dtm_to_sort, dictionary, num_topics, 20)
			values = self.get_values(dtm_to_sort)

			final_sorted = self.lda_sort_results(values, num_topics)

			results = self.lda_associate_results_to_documents(final_sorted, documents_to_sort)

			d.write_strings_to_file(results, "output/lda_test_sorted.txt")
			#print("\tNum topics :",num_topics)
			#print("Maximum Confidence :", max(probs),"\nMinimum Confidence :", min(probs),"\nMean Confidence :", sum(probs)/len(probs), "\nMedian Confidence :", probs[math.floor(len(probs)/2)],"\n")
			#stats.append((num_topics, (max(probs), min(probs), sum(probs)/len(probs), probs[math.floor(len(probs)/ 2)])))

		"""
		max_max = 0
		max_topic = 0
		max_min = 0
		min_topic = 0
		max_mean = 0
		mean_topic = 0
		max_median = 0
		median_topic = 0
		for stat in stats:
			if(stat[1][0] > max_max):
				max_max = stat[1][0]
				max_topic = stat[0]
			if(stat[1][1] > max_min):
				max_min = stat[1][1]
				min_topic = stat[0]
			if(stat[1][2] > max_mean):
				max_mean = stat[1][2]
				mean_topic = stat[0]
			if(stat[1][3] > max_median):
				max_median = stat[1][3]
				median_topic = stat[0]
		"""

		#print("\tSummary :\nMaximum :", max_topic,"-",max_max,"\nMinimum :",min_topic,"-",max_min,"\nMean :",mean_topic,"-",max_mean,"\nMedian :",median_topic,"-",max_median)

		"""
		from preprocessing import Preprocessing
		prep = Preprocessing()
		seed = 4242069
		
		data = ["as user i want to do a thing", "as an admin i would like to be able to think", "as a human i would prefer to sleep"]
		ct = prep.simple_tokenize(data)
		td = prep.simple_tokenize_tag(data)
		dic = prep.to_dictionary(ct)
		dtm = prep.to_doc_term_matrix(dic, ct)
		lda = self.lda(dtm, dic, seed=seed)
		lsi = self.lsi(dtm, dic, seed=seed)
		w2v = self.word2vec(ct, seed=seed)
		d2v = self.doc2vec(td, seed=seed)

		print(lda[dtm[0]])

		print(self.vec_model_loss())

		self.save_model("test/lda.model", lda)
		self.save_model("test/lsi.model", lsi)
		self.save_model("test/w2v.model", w2v)
		self.save_model("test/d2v.model", d2v)

		lda = self.lda_load_local("test/lda.model")
		lsi = self.lsi_load_local("test/lsi.model")
		w2v = self.word2vec_load_local("test/w2v.model")
		d2v = self.doc2vec_load_local("test/d2v.model")

		w2v = self.word2vec_load("glove-twitter-25")
		print(sum(self.word2vec_get_vector("user")))
		"""


if __name__ == '__main__':
	model = Model()
	model.main()




















