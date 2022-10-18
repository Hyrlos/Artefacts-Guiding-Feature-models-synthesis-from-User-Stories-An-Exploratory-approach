"""
This module runs tests for the `src.model` module.
"""

import unittest
import numpy as np
from model import Model
from preprocessing import Preprocessing
from data_structures import Documents, Document

#print(get_issues_from_project(12440105))
#print(get_issues_from_project(12440105, True))

import gensim

class TestModel(unittest.TestCase):

	# Setting up constants and check values
	model = Model()
	prep = Preprocessing()
	doc = ["as user i want to do a thing", "as an admin i would like to be able to think", "as a human i would prefer to sleep"]
	wordvector = 'user'
	docvector = ["system", "response"]
	seed = 4242069

	docs = Documents(texts=doc)
	ct = prep.simple_tokenize(docs)
	dic = prep.to_dictionary(docs)
	dtm = prep.to_doc_term_matrix(docs)
	tdocs = Documents(deep_copy=docs)
	tdoc = prep.simple_tokenize_tag(tdocs)
	docs.tokenized_documents = ct
	docs.document_dictionary = dic
	docs.document_term_matrix = dtm
	loss = 0.0
	w2v_sum = -0.057463605868179
	w2v_gt_sum = -6.104655081406236
	lsi_res = [(0, 0.5946260800607388), (1, -1.0294880755223355), (2, 0.33903233119447157)]

	lda,_ = model.lda(docs, seed=seed)
	d2v,_ = model.doc2vec(tdocs, seed=seed)

	# Tests
	def test_lda(self):
		test_lda,_ = self.model.lda(self.docs, seed=self.seed)
		self.assertEqual(test_lda[[(0,1), (1,1), (2,1)]], self.lda[[(0,1), (1,1), (2,1)]], "Did not generate proper LDA model")
	def test_lda_result(self):
		test_lda,_ = self.model.lda(self.docs, seed=self.seed)
		self.assertEqual(self.model.get_value([(0,1), (1,1), (2,1)]), self.lda[[(0,1), (1,1), (2,1)]], "LDA does not provide the correct answer through Model")
	def test_lda_save(self):
		test_lda,_ = self.model.lda(self.docs, seed=self.seed)
		self.assertEqual(self.model.save_model("test/lda.model", test_lda), True, "Did not save LDA model correctly")
	def test_lda_load(self):
		test_lda = self.model.lda_load_local("test/lda.model")
		self.assertEqual(test_lda[[(0,1), (1,1), (2,1)]], self.lda[[(0,1), (1,1), (2,1)]], "Did not load pregenerated LDA model correctly")

	def test_lsi(self):
		test_lsi,_ = self.model.lsi(self.docs, seed=self.seed)
		self.assertEqual(test_lsi[[(0,1), (1,1), (2,1)]], self.lsi_res, "Did not generate proper LSI model")
	def test_lsi_result(self):
		test_lsi,_ = self.model.lsi(self.docs, seed=self.seed)
		self.assertEqual(self.model.get_value([(0,1), (1,1), (2,1)]), self.lsi_res, "LSI does not provide the correct answer through Model")
	def test_lsi_save(self):
		test_lsi,_ = self.model.lsi(self.docs, seed=self.seed)
		self.assertEqual(self.model.save_model("test/lsi.model", test_lsi), True, "Did not save LSI model correctly")
	def test_lsi_load(self):
		test_lsi = self.model.lsi_load_local("test/lsi.model")
		self.assertEqual(test_lsi[[(0,1), (1,1), (2,1)]], self.lsi_res, "Did not load pregenerated LSI model correctly")
	
	def test_w2v(self):
		test_w2v,_ = self.model.word2vec(self.docs, seed=self.seed)
		self.assertEqual(sum(test_w2v.wv[self.wordvector]), self.w2v_sum, "Did not generate proper Word2Vec model")
	def test_w2v_result(self):
		test_w2v,_ = self.model.word2vec(self.docs, seed=self.seed)
		self.assertEqual(sum(self.model.word2vec_get_vector(self.wordvector)), self.w2v_sum, "Word2Vec does not provide the correct vector through Model")
	def test_w2v_save(self):
		test_w2v,_ = self.model.word2vec(self.docs, seed=self.seed)
		self.assertEqual(self.model.save_model("test/w2v.model", test_w2v), True, "Did not save Word2Vec model correctly")
	def test_w2v_load(self):
		test_w2v = self.model.word2vec_load_local("test/w2v.model")
		self.assertEqual(sum(test_w2v.wv[self.wordvector]), self.w2v_sum, "Did not load pregenerated Word2Vec model correctly")
	def test_w2v_load_dist(self):
		test_w2v = self.model.word2vec_load("glove-twitter-25")
		self.assertEqual(sum(self.model.word2vec_get_vector(self.wordvector)), self.w2v_gt_sum, "Did not load Word2Vec model from internet correctly")

	def test_d2v(self):
		test_d2v,_ = self.model.doc2vec(self.tdocs, seed=self.seed)
		self.assertEqual(sum(test_d2v.infer_vector(self.docvector)), sum(self.d2v.infer_vector(self.docvector)), "Did not generate proper Doc2Vec model")
	def test_d2v_result(self):
		test_d2v,_ = self.model.doc2vec(self.tdocs, seed=self.seed)
		self.assertEqual(sum(self.model.doc2vec_get_vector(self.docvector)), sum(self.d2v.infer_vector(self.docvector)), "Doc2Vec does not provide the correct vector through Model")
	def test_d2v_save(self):
		test_d2v,_ = self.model.doc2vec(self.tdocs, seed=self.seed)
		self.assertEqual(self.model.save_model("test/d2v.model", test_d2v), True, "Did not save Doc2Vec model correctly")
	def test_d2v_load(self):
		test_d2v = self.model.doc2vec_load_local("test/d2v.model")
		self.assertEqual(sum(test_d2v.infer_vector(self.docvector)), sum(self.d2v.infer_vector(self.docvector)), "Did not load pregenerated Doc2Vec model correctly")
	def test_d2v_loss(self):
		test_d2v,_ = self.model.doc2vec(self.tdocs, seed=self.seed)
		self.assertEqual(self.model.vec_model_loss(), self.loss, "Did not calculate Doc2Vec model loss correctly")

	def test_to_ndarray(self):
		og_data = [[2, 3, 6], [1, 5], [2, 2, 0], [-1, -3, 6], [1, 5, 6, 8]]
		final_data = [np.array([2, 3, 6]), np.array([2, 2, 0]), np.array([-1, -3, 6])]
		self.assertEqual(str(self.model.to_ndarray(og_data, 3)), str(final_data), "Did not turn python lists into numpy ndarrays correctly")
	def test_lsi_lda_results_to_vec(self):
		docs = Documents([])
		data = [[(0, 1), (1, 2), (2, 3)], [(2, 3)], [(0, 1), (1, 2), (2, 3), (3, 4)]]
		results = "[[1, 2, 3], [0, 0, 3], [1, 2, 3]]"
		self.assertEqual(str(self.model._lsi_lda_results_to_vec(docs, data, 3)), results, "Did not convert LSI/LDA ")



if __name__ == '__main__':
	unittest.main()