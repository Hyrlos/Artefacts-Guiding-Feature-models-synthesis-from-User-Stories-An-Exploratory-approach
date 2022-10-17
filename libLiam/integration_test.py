# author : Liam RICE

import unittest
from dataset import Dataset
from preprocessing import Preprocessing
from model import Model
from data_structures import Documents, Document

class TestIntegration(unittest.TestCase):

	seed = 0

	dataset = Dataset()
	preprocessing = Preprocessing()
	model = Model()

	test_phrase = ["as a user, i would like to process something"]
	test_word = "user"

	docs = Documents(texts=test_phrase)
	tdocs = Documents(texts=test_phrase)

	simple_token_query_phrase = preprocessing.simple_tokenize(docs)
	complex_token_query_phrase = preprocessing.complex_tokenize(tdocs, True, True)

	ddocs = dataset.read_from_file("test/user_stories_dataset_1.txt")
	tokens = preprocessing.simple_tokenize_tag(ddocs)
	d2v = model.doc2vec(ddocs, seed=seed)
	simple_d2v_result = sum(model.doc2vec_get_vector(simple_token_query_phrase[0]))

	tddocs = Documents(deep_copy=ddocs)
	tokens = preprocessing.complex_tokenize_tag(tddocs, True, True, True)
	d2vc = model.doc2vec(tddocs, 200, 20, True, True, True, seed)
	complex_d2v_result = sum(model.doc2vec_get_vector(complex_token_query_phrase[0]))

	def test_integration_simple_lda(self):
		docs = self.dataset.read_from_file("test/user_stories_dataset_1.txt")
		tokens = self.preprocessing.simple_tokenize(docs)
		dictionary = self.preprocessing.to_dictionary(docs)
		doc_term_matrix = self.preprocessing.to_doc_term_matrix(docs)
		lda = self.model.lda(docs, seed=self.seed)
		self.docs.document_dictionary = dictionary
		simple_dtm_query_phrase = self.preprocessing.to_doc_term_matrix(self.docs)
		self.assertEqual(str(self.model.get_value(simple_dtm_query_phrase[0])), "[(43, 0.8599975)]", "Did not generate correct result for simple LDA")
	def test_integration_complex_lda(self):
		data = self.dataset.read_from_file("test/user_stories_dataset_1.txt")
		tokens = self.preprocessing.complex_tokenize(data, True, True)
		dictionary = self.preprocessing.to_dictionary(data)
		doc_term_matrix = self.preprocessing.to_doc_term_matrix(data)
		lda = self.model.lda(data, 100, 20, self.seed)
		self.docs.document_dictionary = dictionary
		complex_dtm_query_phrase = self.preprocessing.to_doc_term_matrix(self.docs)
		self.assertEqual(str(self.model.get_value(complex_dtm_query_phrase[0])), "[(38, 0.80199873)]", "Did not generate correct result for complex LDA")

	def test_integration_simple_lsi(self):
		data = self.dataset.read_from_file("test/user_stories_dataset_1.txt")
		tokens = self.preprocessing.simple_tokenize(data)
		dictionary = self.preprocessing.to_dictionary(data)
		doc_term_matrix = self.preprocessing.to_doc_term_matrix(data)
		lsi = self.model.lsi(data, seed=self.seed)
		self.docs.document_dictionary = dictionary
		simple_dtm_query_phrase = self.preprocessing.to_doc_term_matrix(self.docs)
		self.assertEqual(str(self.model.get_value(simple_dtm_query_phrase[0])), "[(0, 1.292037604495913), (1, -0.6788368193184013), (2, -0.11291152007133785), (3, 0.9083922140575272), (4, 0.0869465088359997), (5, -0.4177697303731867), (6, -0.23152967804848562), (7, -0.47184236052221207), (8, 0.15687637832292953), (9, 0.15702060630485926), (10, 0.14592927975608044), (11, 0.13042584063176624), (12, -0.19010325107783374), (13, -0.10373076789962234), (14, 0.2418136768070038), (15, -0.4516315145172006), (16, 0.20901263937418724), (17, -0.20459726813137108), (18, -0.01657495373618926), (19, -0.13662412347114383), (20, 0.06789109885741043), (21, 0.11633461263590777), (22, 0.05649545672569009), (23, -0.058851506010072184), (24, 0.14541183314689834), (25, 0.19087827142674835), (26, -0.05057881963033849), (27, -0.1404180364157189), (28, 0.05190173739209606), (29, 0.16712177879423448), (30, -0.10264658153638768), (31, 0.0168899600201748), (32, -0.13343814953693667), (33, 0.03305956022543096), (34, 0.10527817696223915), (35, 0.048939115840631583), (36, 0.2208975978791551), (37, 0.002527199103016755), (38, -0.05258736450317937), (39, 0.14559271953823635), (40, -0.031801582039219914), (41, 0.09658622218310772), (42, 0.08624586393779245), (43, 0.21204506846157795), (44, -0.003992955924437259), (45, 0.16094592230952912), (46, 0.0813401207906452), (47, -0.02892214961714128), (48, 0.24000919285572758), (49, 0.24677497803531245)]", "Did not generate correct result for simple LSI")
	def test_integration_complex_lsi(self):
		data = self.dataset.read_from_file("test/user_stories_dataset_1.txt")
		tokens = self.preprocessing.complex_tokenize(data, True, True)
		dictionary = self.preprocessing.to_dictionary(data)
		doc_term_matrix = self.preprocessing.to_doc_term_matrix(data)
		lsi = self.model.lsi(data, 100, 20, self.seed)
		self.docs.document_dictionary = dictionary
		complex_dtm_query_phrase = self.preprocessing.to_doc_term_matrix(self.docs)
		result = "-1.5250292734023794"
		test = self.model.get_value(complex_dtm_query_phrase[0])
		to_count = []
		for t in test:
			to_count.append(t[1])
		self.assertEqual(str(sum(to_count)), result, "Did not generate correct result for complex LSI")

	def test_integration_simple_w2v(self):
		data = self.dataset.read_from_file("test/user_stories_dataset_1.txt")
		tokens = self.preprocessing.simple_tokenize(data)
		w2v = self.model.word2vec(data, seed=self.seed)
		result = "0.25603283087184536"
		self.assertEqual(str(sum(self.model.word2vec_get_vector(self.test_word))), result, "Did not generate correct result for simple Word2Vec")
	def test_integration_complex_w2v(self):
		data = self.dataset.read_from_file("test/user_stories_dataset_1.txt")
		tokens = self.preprocessing.complex_tokenize(data, True, True)
		w2v = self.model.word2vec(data, seed=self.seed)
		result = "0.03651245502260281"
		self.assertEqual(str(sum(self.model.word2vec_get_vector(self.test_word))), result, "Did not generate correct result for complex Word2Vec")

	def test_integration_simple_d2v(self):
		data = self.dataset.read_from_file("test/user_stories_dataset_1.txt")
		tokens = self.preprocessing.simple_tokenize_tag(data)
		d2v = self.model.doc2vec(data, seed=self.seed)
		self.assertEqual(sum(self.model.doc2vec_get_vector(self.simple_token_query_phrase[0])), self.simple_d2v_result, "Did not generate correct result for simple Doc2Vec")
	def test_integration_complex_d2v(self):
		data = self.dataset.read_from_file("test/user_stories_dataset_1.txt")
		tokens = self.preprocessing.complex_tokenize_tag(data, True, True, True)
		d2vc = self.model.doc2vec(data, 200, 20, True, True, True, self.seed)
		self.assertEqual(sum(self.model.doc2vec_get_vector(self.complex_token_query_phrase[0])), self.complex_d2v_result, "Did not generate correct result for complex Doc2Vec")











