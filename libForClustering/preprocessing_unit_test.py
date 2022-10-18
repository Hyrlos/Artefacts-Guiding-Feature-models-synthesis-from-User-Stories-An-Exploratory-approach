
"""
This module runs tests for the `src.preprocessing` module.
"""

import unittest
import os
from gensim import corpora
from gensim.models.doc2vec import TaggedDocument
from preprocessing import Preprocessing
from data_structures import Documents, Document



class TestPreprocessing(unittest.TestCase):

	# Setting up constants and check values
	preprocessing = Preprocessing()
	documents = Documents([Document("as a user i want to be able to change my peudonym"), Document("as an admin i would like to eat posh food"), Document("as a senior software engineer i want to be able to redesign my database")])
	dictionary = corpora.Dictionary([['as', 'user', 'want', 'to', 'be', 'able', 'to', 'change', 'my', 'peudonym'], ['as', 'an', 'admin', 'would', 'like', 'to', 'eat', 'posh', 'food'], ['as', 'senior', 'software', 'engineer', 'want', 'to', 'be', 'able', 'to', 'redesign', 'my', 'database']])
	doc_term_matrix = [[(0, 1), (1, 1), (2, 1), (3, 1), (4, 1), (5, 1), (6, 2), (7, 1), (8, 1)], [(1, 1), (6, 1), (9, 1), (10, 1), (11, 1), (12, 1), (13, 1), (14, 1), (15, 1)], [(0, 1), (1, 1), (2, 1), (4, 1), (6, 2), (8, 1), (16, 1), (17, 1), (18, 1), (19, 1), (20, 1)]]
	sol_s = [['as', 'user', 'want', 'to', 'be', 'able', 'to', 'change', 'my', 'peudonym'], ['as', 'an', 'admin', 'would', 'like', 'to', 'eat', 'posh', 'food'], ['as', 'senior', 'software', 'engineer', 'want', 'to', 'be', 'able', 'to', 'redesign', 'my', 'database']]
	sol_c_tf = [['user', 'want', 'able', 'change', 'peudonym'], ['admin', 'would', 'like', 'eat', 'posh', 'food'], ['senior', 'software', 'engineer', 'want', 'able', 'redesign', 'database']]
	sol_c_tt = [['user', 'want', 'able', 'change', 'peudonym'], ['admin', 'would', 'like', 'eat', 'posh', 'food'], ['senior', 'software', 'engineer', 'want', 'able', 'redesign', 'database']]
	sol_c_ff = [['as', 'a', 'user', 'i', 'want', 'to', 'be', 'able', 'to', 'change', 'my', 'peudonym'], ['as', 'an', 'admin', 'i', 'would', 'like', 'to', 'eat', 'posh', 'food'], ['as', 'a', 'senior', 'software', 'engineer', 'i', 'want', 'to', 'be', 'able', 'to', 'redesign', 'my', 'database']]
	sol_c_ft = [['as', 'a', 'user', 'i', 'want', 'to', 'be', 'able', 'to', 'change', 'my', 'peudonym'], ['as', 'an', 'admin', 'i', 'would', 'like', 'to', 'eat', 'posh', 'food'], ['as', 'a', 'senior', 'software', 'engineer', 'i', 'want', 'to', 'be', 'able', 'to', 'redesign', 'my', 'database']]
	sol_sv = [TaggedDocument(words=['as', 'user', 'want', 'to', 'be', 'able', 'to', 'change', 'my', 'peudonym'], tags=[0]), TaggedDocument(words=['as', 'an', 'admin', 'would', 'like', 'to', 'eat', 'posh', 'food'], tags=[1]), TaggedDocument(words=['as', 'senior', 'software', 'engineer', 'want', 'to', 'be', 'able', 'to', 'redesign', 'my', 'database'], tags=[2])]
	sol_cv_ttt = [TaggedDocument(words=['user', 'want', 'able', 'change', 'peudonym'], tags=['NN', 'VBP', 'JJ', 'VB', 'NN']), TaggedDocument(words=['admin', 'would', 'like', 'eat', 'posh', 'food'], tags=['NN', 'MD', 'VB', 'VB', 'JJ', 'NN']), TaggedDocument(words=['senior', 'software', 'engineer', 'want', 'able', 'redesign', 'database'], tags=['JJ', 'NN', 'NN', 'VBP', 'JJ', 'VB', 'NN'])]
	sol_cv_ttf = [TaggedDocument(words=['user', 'want', 'abl', 'chang', 'peudonym'], tags=[0]), TaggedDocument(words=['admin', 'would', 'like', 'eat', 'posh', 'food'], tags=[1]), TaggedDocument(words=['senior', 'softwar', 'engin', 'want', 'abl', 'redesign', 'databas'], tags=[2])]
	sol_cv_tff = [TaggedDocument(words=['user', 'want', 'able', 'change', 'peudonym'], tags=[0]), TaggedDocument(words=['admin', 'would', 'like', 'eat', 'posh', 'food'], tags=[1]), TaggedDocument(words=['senior', 'software', 'engineer', 'want', 'able', 'redesign', 'database'], tags=[2])]
	sol_cv_ftt = [TaggedDocument(words=['as', 'a', 'user', 'i', 'want', 'to', 'be', 'able', 'to', 'change', 'my', 'peudonym'], tags=['IN', 'DT', 'NN', 'NN', 'VBP', 'TO', 'VB', 'JJ', 'TO', 'VB', 'PRP$', 'NN']), TaggedDocument(words=['as', 'an', 'admin', 'i', 'would', 'like', 'to', 'eat', 'posh', 'food'], tags=['IN', 'DT', 'NN', 'NN', 'MD', 'VB', 'TO', 'VB', 'JJ', 'NN']), TaggedDocument(words=['as', 'a', 'senior', 'software', 'engineer', 'i', 'want', 'to', 'be', 'able', 'to', 'redesign', 'my', 'database'], tags=['IN', 'DT', 'JJ', 'NN', 'NN', 'NN', 'VBP', 'TO', 'VB', 'JJ', 'TO', 'VB', 'PRP$', 'NN'])]
	sol_cv_ftf = [TaggedDocument(words=['as', 'a', 'user', 'i', 'want', 'to', 'be', 'abl', 'to', 'chang', 'my', 'peudonym'], tags=[0]), TaggedDocument(words=['as', 'an', 'admin', 'i', 'would', 'like', 'to', 'eat', 'posh', 'food'], tags=[1]), TaggedDocument(words=['as', 'a', 'senior', 'softwar', 'engin', 'i', 'want', 'to', 'be', 'abl', 'to', 'redesign', 'my', 'databas'], tags=[2])]
	sol_cv_fff = [TaggedDocument(words=['as', 'a', 'user', 'i', 'want', 'to', 'be', 'able', 'to', 'change', 'my', 'peudonym'], tags=[0]), TaggedDocument(words=['as', 'an', 'admin', 'i', 'would', 'like', 'to', 'eat', 'posh', 'food'], tags=[1]), TaggedDocument(words=['as', 'a', 'senior', 'software', 'engineer', 'i', 'want', 'to', 'be', 'able', 'to', 'redesign', 'my', 'database'], tags=[2])]

	# Tests
	def test_simple_preprocessing(self):
		self.assertEqual(self.preprocessing.simple_tokenize(self.documents), self.sol_s, "Did not preprocess the phrases correctly for simple applications")
	
	def test_complex_preprocessing_tf(self):
		self.assertEqual(self.preprocessing.complex_tokenize(self.documents, True, False), self.sol_c_tf, "Did not preprocess the phrases correctly for complex applications (True, False)")
	def test_complex_preprocessing_tt(self):
		self.assertEqual(self.preprocessing.complex_tokenize(self.documents, True, True), self.sol_c_tt, "Did not preprocess the phrases correctly for complex applications (True, True)")
	def test_complex_preprocessing_ff(self):
		self.assertEqual(self.preprocessing.complex_tokenize(self.documents, False, False), self.sol_c_ff, "Did not preprocess the phrases correctly for complex applications (False, False)")
	def test_complex_preprocessing_ft(self):
		self.assertEqual(self.preprocessing.complex_tokenize(self.documents, False, True), self.sol_c_ft, "Did not preprocess the phrases correctly for complex applications (False, True)")
	
	def test_simple_preprocessing_vec(self):
		self.assertEqual(self.preprocessing.simple_tokenize_tag(self.documents), self.sol_sv, "Did not preprocess the phrases correctly for simple vector applications")

	def test_complex_preprocessing_vec_ttt(self):
		self.assertEqual(self.preprocessing.complex_tokenize_tag(self.documents, True, True, True), self.sol_cv_ttt, "Did not preprocess the phrases correctly for complex vector applications (True, True, True)")
	def test_complex_preprocessing_vec_ttf(self):
		self.assertEqual(self.preprocessing.complex_tokenize_tag(self.documents, True, True, False), self.sol_cv_ttf, "Did not preprocess the phrases correctly for complex vector applications (True, True, False)")
	def test_complex_preprocessing_vec_tft(self):
		self.assertEqual(self.preprocessing.complex_tokenize_tag(self.documents, True, False, True), self.sol_cv_ttt, "Did not preprocess the phrases correctly for complex vector applications (True, False, True)")
	def test_complex_preprocessing_vec_tff(self):
		self.assertEqual(self.preprocessing.complex_tokenize_tag(self.documents, True, False, False), self.sol_cv_tff, "Did not preprocess the phrases correctly for complex vector applications (True, False, False)")
	def test_complex_preprocessing_vec_ftt(self):
		self.assertEqual(self.preprocessing.complex_tokenize_tag(self.documents, False, True, True), self.sol_cv_ftt, "Did not preprocess the phrases correctly for complex vector applications (False, True, True)")
	def test_complex_preprocessing_vec_ftf(self):
		self.assertEqual(self.preprocessing.complex_tokenize_tag(self.documents, False, True, False), self.sol_cv_ftf, "Did not preprocess the phrases correctly for complex vector applications (False, True, False)")
	def test_complex_preprocessing_vec_fft(self):
		self.assertEqual(self.preprocessing.complex_tokenize_tag(self.documents, False, False, True), self.sol_cv_ftt, "Did not preprocess the phrases correctly for complex vector applications (False, False, True)")
	def test_complex_preprocessing_vec_fff(self):
		self.assertEqual(self.preprocessing.complex_tokenize_tag(self.documents, False, False, False), self.sol_cv_fff, "Did not preprocess the phrases correctly for complex vector applications (False, False, False)")
	
	def test_to_dictionary(self):
		self.preprocessing.simple_tokenize(self.documents)
		self.assertEqual(self.preprocessing.to_dictionary(self.documents), self.dictionary, "Dictionary was not generated correctly")

	def test_to_doc_term_matrix(self):
		self.preprocessing.simple_tokenize(self.documents)
		self.preprocessing.to_dictionary(self.documents)
		self.assertEqual(self.preprocessing.to_doc_term_matrix(self.documents), self.doc_term_matrix, "Doc-term matrix was not generated correctly")

	def test_remove_subject(self):
		test = self.preprocessing.remove_subject(self.documents)
		results = "to be able to change my peudonym \n\nto eat posh food \n\nto be able to redesign my database \n\n"
		self.assertEqual(str(test), results, "Does not remove subjects correctly")

	def test_preprocess_lsi_lsa(self):
		t_doc = Documents([Document("one", "first"), Document("two", "second")])
		results = "Dictionary<2 unique tokens: ['one', 'two']>[[(0, 1)], [(1, 1)]]"
		a, b = self.preprocessing.preprocess_lsi_lda(t_doc)
		test = str(a)+str(b)
		self.assertEqual(test, results, "Does not generate full preprocessing for LSI / LDA")
	def test_preprocess_word2vec(self):
		t_doc = Documents([Document("one", "first"), Document("two", "second")])
		results = "[['one'], ['two']]"
		a = self.preprocessing.preprocess_word2vec(t_doc)
		test = str(a)
		self.assertEqual(test, results, "Does not generate full preprocessing for wor2vec")
	def test_preprocess_doc2vec(self):
		t_doc = Documents([Document("one", "first"), Document("two", "second")])
		results = "[TaggedDocument(words=['one'], tags=[0]), TaggedDocument(words=['two'], tags=[1])][['one'], ['two']]"
		a, b = self.preprocessing.preprocess_doc2vec(t_doc)
		test = str(a)+str(b)
		self.assertEqual(test, results, "Does not generate full preprocessing for doc2vec")

	def test_generate_ngrams(self):
		ngram = self.preprocessing.generate_ngrams(self.documents)
		sol = "Phrases<30 vocab, min_count=5, threshold=10, max_vocab_size=40000000>"
		self.assertEqual(str(ngram), sol, "Does not generate ngram model correctly.")
	def test_get_ngrams(self):
		ngram = self.preprocessing.generate_ngrams(self.documents)
		values = self.preprocessing.get_ngrams(self.documents)
		result = ""
		for val in values:
			result += str(val)
		sol = "['user', 'want', 'able', 'change', 'peudonym']['admin', 'would', 'like', 'eat', 'posh', 'food']['senior', 'software', 'engineer', 'want', 'able', 'redesign', 'database']"
		self.assertEqual(result, sol, "Does not find correct ngrams.")

	def test_pos_en(self):
		texts = self.preprocessing.pos(["This", "is", "a", "long", "arse", "sentence", "."])
		sol = "TaggedDocument<['this', 'is', 'a', 'long', 'arse', 'sentence', '.'], ['DT', 'VBZ', 'DT', 'JJ', 'JJ', 'NN', '.']>"
		self.assertEqual(str(texts), sol, "Does not generate parts of speech for list of words correctly.")
	def test_pos_fr(self):
		texts = self.preprocessing.pos(["Ceci", "n'", "est", "pas", "assez", "long"], lang="fr")
		sol = """TaggedDocument<['ceci', "n'", 'est', 'pas', 'assez', 'long'], ['NN', 'NN', 'JJS', 'NN', 'NN', 'RB']>"""
		self.assertEqual(str(texts), sol, "Does not generate parts of speech for list of french words correctly.")
	def test_pos_rem_stopwords(self):
		texts = self.preprocessing.pos(["This", "sentence", "has", "no", "stopwords", "in", "it", "."], remove_stopwords=True)
		sol = "TaggedDocument<['sentence', 'stopwords', '.'], ['NN', 'NNS', '.']>"
		self.assertEqual(str(texts), sol, "Does not generate parts of speech without stopwords for list of words correctly.")
	def test_pos_stem(self):
		texts = self.preprocessing.pos(["All", "the", "words", "inside", "this", "word", "list", "were", "stemmed"], stem=True)
		sol = "TaggedDocument<['all', 'the', 'word', 'insid', 'this', 'word', 'list', 'were', 'stem'], ['PDT', 'DT', 'NN', 'NN', 'DT', 'NN', 'NN', 'VBD', 'NN']>"
		self.assertEqual(str(texts), sol, "Does not generate parts of speach for stemmed tokens correctly.")

	def test_to_trees(self):
		docs = Documents(texts=["This is a document that needs to be a tree."])
		trees = self.preprocessing.to_trees(docs)
		sol = "[Tree('is', ['This', Tree('document', ['a', Tree('needs', ['that', Tree('be', ['to', Tree('tree', ['a'])])])]), '.'])]"
		self.assertEqual(str(trees), sol, "Does not generate trees correctly.")
	def test_to_trees_fr(self):
		docs = Documents(texts=["Ceci est un document qui devrait être un arbre."])
		trees = self.preprocessing.to_trees(docs, lang="fr")
		sol = "[Tree('document', ['Ceci', 'est', 'un', Tree('devrait', ['qui', Tree('arbre', ['être', 'un'])]), '.'])]"
		self.assertEqual(str(trees), sol, "Does not generate french trees correctly.")

	def test_get_pos_tag(self):
		tag = self.preprocessing._get_pos_tag("i", TaggedDocument(["i", "am", "here"], ["NN", "VB", "JJ"]))
		sol = "NN"
		self.assertEqual(tag, sol, "Does not fetch correct POS tag.")
	def test_get_pos_tag_alt(self):
		tag = self.preprocessing._get_pos_tag("get", TaggedDocument(['i', 'want', 'to', 'get', 'some', 'stuff'], ['NN', 'VB', 'TO', 'NN', 'JJ', 'NN']))
		sol = "VB"
		self.assertEqual(tag, sol, "Does not fetch correct POS tag based on presence of word 'to'.")

	def test_parts_of_user_stories(self):
		parts = self.preprocessing.parts_of_user_stories(self.documents)
		sol = "[[['user'], ['want', 'change'], ['able', 'peudonym'], []], [['admin'], ['like', 'eat'], ['posh', 'food'], []], [['senior', 'software', 'engineer'], ['want', 'be', 'redesign'], ['able', 'database'], []]]"
		self.assertEqual(str(parts), sol, "Does not separate parts of user stories correctly.")



if __name__ == '__main__':
	unittest.main()