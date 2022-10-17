# author : Liam RICE

"""
## This module provides access to preprocessing functions, formatting data for use in various models.
"""

from gensim.utils import simple_preprocess
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
from gensim.models.doc2vec import TaggedDocument
from gensim.models.phrases import Phrases, ENGLISH_CONNECTOR_WORDS
from gensim.interfaces import TransformedCorpus
import nltk
from gensim import corpora
from libLiam.data_structures import *
from libLiam.dataset import Dataset
import spacy
from nltk import Tree
from libLiam.model import Model
import nltk.corpus
from nltk.corpus import wordnet

# TODO - switch PorterStemmer to SnowballStemmer (can do more than just english)

class Preprocessing:

	"""
	This class contains the functions of this module. It also holds several data objects so they can be accessed independant of return values.
	"""


	def download_data(self):
		"""
		## This function downloads stopwords and the perceptron tagger from NLTK, necessary for complex preprocessing to function.
		"""
		nltk.download('stopwords')
		nltk.download('averaged_perceptron_tagger')
		nltk.download('tagsets')
		nltk.download('wordnet')
		nltk.download('omw-1.4')


	def simple_tokenize(self, documents, length="s"):
		"""
		## This function takes a list of documents and applies gensim's simple preprocessing.
		### Args :
		* `src.data_structures.Documents` documents : the documents to which the processing needs to be applied.
		* String length : short ("s") by default, can be set to long ("l") or full ("f"). If input is not one of those three, uses short by default.
		### Returns :
		* String[][] : returns an array of String arrays containing the words of the split documents.
		"""
		if(length == "l"):
			original_documents = documents.long_training_set()
		elif(length == "f"):
			original_documents = documents.full_training_set()
		else:
			original_documents = documents.short_training_set()
		clean_text = [simple_preprocess(str(sentence).lower(), deacc=True) for sentence in original_documents]
		documents.tokenized_documents = clean_text
		return clean_text


	def complex_tokenize(self, documents, remove_stopwords=True, stem=True, length="s", lang="en"):
		"""
		## This function takes a list of documents and applies complex preprocessing using NLTK.
		### Args :
		* `src.data_structures.Documents` documents : the documents to which the processing needs to be applied.
		* bool remove_stopwords : default True. If true, this function removes english stopwords from the list of tokens
		* bool stem : default True. If true, this function stems the words.
		* String length : short ("s") by default, can be set to long ("l") or full ("f"). If input is not one of those three, uses short by default.
		### Returns :
		* String[][] : returns an array of String arrays containing the split documents which have been tokenised.
		"""
		if(length == "l"):
			original_documents = documents.long_training_set()
		elif(length == "f"):
			original_documents = documents.full_training_set()
		else:
			original_documents = documents.short_training_set()

		if(lang == "fr"):
			nlp = spacy.load("fr_core_news_sm")
			stop = set(stopwords.words('french'))
		else:
			nlp = spacy.load("en_core_web_sm")
			stop = set(stopwords.words('english'))

		texts = []
		for doc in original_documents:
			processed = nlp(doc)
			processed_doc = []
			for tok in processed:
				if(tok.pos_ != "PUNCT"):
					if((str(tok).lower() not in stop) or (not remove_stopwords)):
						if(stem):
							processed_doc.append(tok.lemma_.lower())
						else:
							processed_doc.append(str(tok).lower())
			texts.append(processed_doc)
		documents.tokenized_documents = texts
		return texts


	def to_dictionary(self, documents):
		"""
		## This function takes in a Documents object and returns those documents' dictionary.
		### Args :
		* `src.data_structures.Documents` documents : the documents, containing the tokenized documents in it's tokenized_documents property.
		### Returns :
		* Dictionary(int, String) : returns a Dictionary(int, String) containing all the words of the input text.
		"""
		t_data = documents
		if(isinstance(documents, Documents)):
			t_data = documents.tokenized_documents
		if(isinstance(documents, TransformedCorpus)):
			t_data = documents

		dictionary = corpora.Dictionary(t_data)
		documents.document_dictionary = dictionary
		return dictionary


	def to_doc_term_matrix(self, documents):
		"""
		## This function takes a Documents object, and returns the doc-term matrix of the text according to the Documents object's dictionary.
		### Args :
		* `src.data_structures.Documents` documents : the documents, containing the dictionary in it's document_dictionary propery and tokenized documents in it's tokenized_docuuments property.
		### Returns :
		* (int, int)[][] : returns an array of (int, int) tuple arrays containing the doc-term matrix of the input text.
		"""
		dictionary = documents.document_dictionary

		t_data = documents
		if(isinstance(documents, Documents)):
			t_data = documents.tokenized_documents
		if(isinstance(documents, TransformedCorpus)):
			t_data = documents

		doc_term_matrix = [dictionary.doc2bow(doc) for doc in t_data]
		documents.document_term_matrix = doc_term_matrix
		return doc_term_matrix
	

	def tokens_to_dtm(self, dictionary, tokens):
		"""
		"""
		doc_term_matrix = dictionary.doc2bow(tokens)
		return doc_term_matrix

	
	def simple_tokenize_tag(self, documents, length="s"):
		"""
		## This function takes a list of documents and applies simple preprocessing using Gensim.
		### Args :
		* `src.data_structures.Documents` documents : the documents to which the processing needs to be applied.
		* String length : short ("s") by default, can be set to long ("l") or full ("f"). If input is not one of those three, uses short by default.
		### Returns :
		* TaggedDocument[] : returns an array of TaggedDocuments ready for word2vec or doc2vec processing.
		"""
		if(length == "l"):
			original_documents = documents.long_training_set()
		elif(length == "f"):
			original_documents = documents.full_training_set()
		else:
			original_documents = documents.short_training_set()
		texts = []
		num = 0
		for document in original_documents:
			tokens = simple_preprocess(str(document).lower(), deacc=True)
			texts.append(TaggedDocument(tokens, [num]))
			num+=1
		documents.tagged_documents = texts
		return texts

	
	def complex_tokenize_tag(self, documents, remove_stopwords=True, stem=True, pos_tag=False, length="s", lang="en"):
		"""
		## This function takes a list of documents and applies complex preprocessing using NLTK. Due to the nature of the pos-tagger, you cannot both stem and pos-tag the documents.
		### Args :
		* `src.data_structures.Documents` documents : the documents to which the processing needs to be applied.
		* bool remove_stopwords : default True. If true, this function removes english stopwords from the list of tokens
		* bool stem : default True. If true, this function stems the words.
		* bool pos_tag : default False. If true, this function assigns part-of-speach tagging on the words.
		* String length : short ("s") by default, can be set to long ("l") or full ("f"). If input is not one of those three, uses short by default.
		### Returns :
		* TaggedDocument[] returns an array of TaggedDocuments ready for word2vec or doc2vec processing.
		"""
		if(length == "l"):
			original_documents = documents.long_training_set()
		elif(length == "f"):
			original_documents = documents.full_training_set()
		else:
			original_documents = documents.short_training_set()
		if(pos_tag):
			stem = False

		if(lang == "fr"):
			en_stop = set(stopwords.words("french"))
			p_stemmer = SnowballStemmer("french")
		else:
			en_stop = set(stopwords.words('english'))
			p_stemmer = SnowballStemmer("english")

		tokenizer = RegexpTokenizer(r'\w+')
		texts = []
		num = 0
		for document in original_documents:
			raw = document.lower()
			tokens = tokenizer.tokenize(raw)
			if(stem):
				stemmed_tokens = [p_stemmer.stem(i) for i in tokens]
			else:
				stemmed_tokens = tokens

			if(pos_tag):
				tagged_documents = nltk.pos_tag(stemmed_tokens)
				final_documents = []
				final_document_tags = []
				for i in range(len(tagged_documents)):
					final_documents.append(tagged_documents[i][0])
					final_document_tags.append(tagged_documents[i][1])
				if(remove_stopwords):
					cleaned_docs = []
					cleaned_doc_tags = []
					for i in range(len(final_documents)):
						if(final_documents[i] not in en_stop):
							cleaned_docs.append(final_documents[i])
							cleaned_doc_tags.append(final_document_tags[i])
					texts.append(TaggedDocument(cleaned_docs, cleaned_doc_tags))
				else:
					texts.append(TaggedDocument(final_documents, final_document_tags))
			else:
				if(remove_stopwords):
					stemmed_tokens = [i for i in stemmed_tokens if i not in en_stop]
				texts.append(TaggedDocument(stemmed_tokens, [num]))
			num+=1
		documents.tagged_documents = texts
		return texts


	def preprocess_lsi_lda(self, documents, dictionary=None, length="s"):
		"""
		## This function takes in a training document set and classify document set (the classify document set should be contained within the training document set) and returns all the information to begin using a model for training and classification using these documents via LDA or LSI.
		### Args :
		* `src.data_structures.Documents` documents : the Documents object that will be used to train the model.
		* @Optional String length : the length of the documents used, "s" for short, "l" for long or "f" for full.
		### Returns :
		* Dictionary(int, String) : returns a Dictionary(int, String) containing all the words of the input text.
		* (int, int)[][] : returns an array of (int, int) tuple arrays containing the doc-term matrix of the input text.
		"""
		tokens = self.complex_tokenize(documents, True, False, length)
		if(dictionary == None):
			dictionary = self.to_dictionary(documents)
		else:
			documents.document_dictionary = dictionary
		doc_term_matrix = self.to_doc_term_matrix(documents)
		return dictionary, doc_term_matrix


	def steam_to_line_sentence(self, file_in, file_out, num_docs=1000001, jump=10000):
		"""
		## This function streams a set of documents, read from a formatted file, to a file in LineSentence format.
		### Args :
		* String file_in : the file where the formatted Documents data can be found.
		* String file_out : the file where the processed data will be written in LineSentence format.
		* @Optional int num_docs : the maximum number of documents to analyse.
		* @Optional int jump : the number of documents to analyse in each batch.
		### Returns :
		* bool : True if the method generates the file correctly, and False if not.
		"""
		try:
			dataset = Dataset()
			documents = Documents()
			documents.document_dictionary = corpora.Dictionary()
			r = range(-1, num_docs, jump)
			for i in r:
				print(str(((i+1)/num_docs)*100)+"%")
				new_docs = dataset.read_documents_from_file(file_in, i+1, i+jump)
				tokens = self.complex_tokenize(new_docs, True, False, "f")
				dataset.append_line_sentence(new_docs, file_out)
			return True
		except:
			return False


	def steam_to_line_sentence_dict(self, file_in, file_out, num_docs=1000001, jump=10000):
		"""
		## This function streams a set of documents, read from a formatted file, to a file in LineSentence format, and isolates the dictionary of that LineSentence file.
		### Args :
		* String file_in : the file where the formatted Documents data can be found.
		* String file_out : the file where the processed data will be written in LineSentence format.
		* @Optional int num_docs : the maximum number of documents to analyse.
		* @Optional int jump : the number of documents to analyse in each batch.
		### Returns :
		* `src.data_structures.Documents` : the Documents object containing the full Dictionary of the processed documents.
		"""
		dataset = Dataset()
		documents = Documents()
		documents.document_dictionary = corpora.Dictionary()
		r = range(-1, num_docs, jump)
		try:
			for i in r:
				print(str(((i+1)/num_docs)*100)+"%")
				new_docs = dataset.read_documents_from_file(file_in, i+1, i+jump)
				tokens = self.complex_tokenize(new_docs, True, False, "f")
				documents.document_dictionary.add_documents(tokens)
				dataset.append_line_sentence(new_docs, file_out)
			return documents
		except:
			return documents


	def preprocess_word2vec(self, training_documents, length="s"):
		"""
		## This function takes in a training document set and classify document set (the classify document set should be contained within the training document set) and returns all the information to begin using a model for training and classification using these documents via Word2Vec.
		### Args :
		* `src.data_structures.Documents` training_documents : the Documents object that will be used to train the model.
		* @Optional String length : the length of the documents used, "s" for short, "l" for long or "f" for full.
		### Returns :
		* String[][] : the array of tokenized documents.
		"""
		t_tokens = self.complex_tokenize(training_documents, True, False, length)
		return t_tokens


	def preprocess_doc2vec(self, training_documents, length="s"):
		"""
		## This function takes in a training document set and classify document set (the classify document set should be contained within the training document set) and returns all the information to begin using a model for training and classification using these documents via Doc2Vec.
		### Args :
		* `src.data_structures.Documents` training_documents : the Documents object that will be used to train the model.
		* @Optional String length : the length of the documents used, "s" for short, "l" for long or "f" for full.
		### Returns :
		* TaggedDocument[] : the array of documents, tagged and ready for training.
		* String[][] : the array of tokenized documents.
		"""
		t_tags = self.complex_tokenize_tag(training_documents, True, False, False, length)
		t_tokens = self.complex_tokenize(training_documents, True, False, length)
		return t_tags, t_tokens


	# input data can only be user stories or result will be nonsensical
	def remove_subject(self, documents):
		"""
		## This function isolates the subject from a set of user story documents and returns them. Does not function for any object that isn't a user story.
		### Args :
		* `src.data_structures.Documents` documents : the Documents object containing the raw documents.
		### Returns :
		* `src.data_structures.Documents` : a new Documents object with the same documents as before but with all the subjects removed.
		"""
		data = self.complex_tokenize_tag(documents, False, False, True)
		verb_indexes = []
		for doc in data:
			subject = []
			verb_index = -1
			for i in range(len(doc[1])):
				if(doc[1][i] in ['VBP', 'VB'] and verb_index == -1):
					verb_index = i
				elif(verb_index == -1):
					subject.append(doc[0][i])
			verb_indexes.append(verb_index)

		desubjected_documents = []
		for i in range(len(data)):
			document = data[i][0]
			for j in range(verb_indexes[i], -1, -1):
				document.pop(j)
			str_doc = ""
			for word in document:
				str_doc = str_doc + word + " "
			desubjected_documents.append(str_doc)
		return Documents(texts=desubjected_documents)


	def isolate_parts(self, documents):
		"""
		## This function uses nltk pos tagging instead of spacy to separate the action words from object and subject words.
		### Args :
		* `src.data_structures.Documents` documents : the documents object containing the documents to split.
		### Returns :
		* string[][] : an array of string arrays containing all the action word tokens.
		* string[][] : an array of string arrays containing all the object word tokens.
		* string[][] : an array of string arrays containing all the subject word tokens.
		"""
		data = self.complex_tokenize_tag(documents, True, False, True)
		action_words = []
		object_words = []
		subject_words = []
		found_verb = False
		for isolated_part in data:
			aw = []
			ow = []
			sw = []
			for i in range(len(isolated_part[0])):
				if(found_verb):
					if(isolated_part[1][i].startswith("VB")):
						aw.append(isolated_part[0][i])
					if(isolated_part[1][i].startswith("NN") or isolated_part[1][i].startswith("JJ")):
						ow.append(isolated_part[0][i])
				else:
					if(isolated_part[1][i].startswith("VB")):
						aw.append(isolated_part[0][i])
						found_verb = True
					if(isolated_part[1][i].startswith("NN") or isolated_part[1][i].startswith("JJ")):
						sw.append(isolated_part[0][i])
			action_words.append(aw)
			object_words.append(ow)
			subject_words.append(sw)
		return action_words, object_words, subject_words


	def generate_ngrams(self, documents, lang="en"):
		"""
		## This function initialises the ngram generator using the tokenised documents passed as a parameter.
		### Args :
		* `src.data_structures.Documents` documents : the documents object containing the tokenized documents used to train the ngram generator.
		### Returns :
		Phrases : a bigram transformer associated to the documents object.
		"""
		if(lang=="fr"):
			cw = []
		else:
			cw = ENGLISH_CONNECTOR_WORDS
		bigram_transformer = Phrases(documents.tokenized_documents, min_count=5, threshold=10, connector_words=cw)
		documents.ngrams = bigram_transformer
		return bigram_transformer


	def get_ngrams(self, documents):
		"""
		## This function finds the most common ngrams for the documents object. Must have initialised the generator first.
		### Args :
		* `src.data_structures.Documents` documents : the documents objects containing the tokenized documents and trained ngram generator.
		### Returns :
		TransformedCorpus : the input corpus, but containing the most common ngrams as single words.
		"""
		val = documents.ngrams[documents.tokenized_documents]
		documents.tokenized_documents = val
		return val


	def replace_docs_with_ngram_docs(self, documents, lang="en"):
		"""
		## This function replaces the documents in the documents object with equivalent documents with ngrams found.
		### Args :
		* `src.data_structures.Documents` documents : the documents object whose texts are to be replaced with ngram-containing texts.
		* @Optional string lang : the language to detect the ngrams in, the only supported languages are english "en" and french "fr".
		### Returns :
		* `src.data_structures.Documents` : the documents object with the changed text.
		"""
		self.generate_ngrams(documents, lang)
		self.get_ngrams(documents)
		for i in range(len(documents)):
			doc = ""
			for word in documents.tokenized_documents[i]:
				doc += word + " "
			documents.documents[i].set_title(doc)
		return documents


	def replace_tokens_with_key_parts(self, documents, lang="en", length="s"):
		"""
		## This function produces tokenized and tagged documents from the separated parts of the input user story documents.
		### Args :
		* `src.data_structures.Documents` documents : the documents object whose tokenized documents and tagged documents are to be replaced with relevant parts of the input user stories.
		* @Optional string lang : the language to separate the user stories in, the only supported languages are english "en" and french "fr".
		* @Optional string length : the length of the user stories to use, can be "s" for short, "l" for long and "f" for full, default and recommended is short and this should not be changed unless the document description includes part of the user story.
		### Returns :
		* `src.data_structures.Documents` : the documents object with the tags and tokens included.
		"""
		self.parts_of_user_stories(documents, lang, length)
		for i in range(len(documents)):
			words = []
			words.extend(documents.documents[i].action)
			words.extend(documents.documents[i].object)
			documents.tokenized_documents[i] = words
			documents.tagged_documents[i] = TaggedDocument(words, [i])
		return documents


	def pos(self, words, remove_stopwords=False, stem=False, lang="en"):
		"""
		## This function applies part of speach tagging to an array of words.
		### Args :
		* string[] words : an array of words to be tagged.
		* @Optional bool remove_stopwords : if True, removes stopwords from the array of words.
		* @Optional bool stem : if True, stems the input words.
		* @Optional string lang : supported languages are english "en" or french "fr"
		### Returns :
		TaggedDocument : a TaggedDocument object containing the words and their associated POS tags.
		"""
		if(lang == "fr"):
			en_stop = set(stopwords.words("french"))
			p_stemmer = SnowballStemmer("french")
		else:
			en_stop = set(stopwords.words('english'))
			p_stemmer = SnowballStemmer("english")

		tokens = [i.lower() for i in words]
		if(remove_stopwords):
			stopped_tokens = [i for i in tokens if not i in en_stop]
		else:
			stopped_tokens = tokens
		if(stem):
			stemmed_tokens = [p_stemmer.stem(i) for i in stopped_tokens]
		else:
			stemmed_tokens = stopped_tokens

		tagged_documents = nltk.pos_tag(stemmed_tokens)
		final_documents = []
		final_document_tags = []
		for i in range(len(tagged_documents)):
			final_documents.append(tagged_documents[i][0])
			final_document_tags.append(tagged_documents[i][1])
		texts = TaggedDocument(final_documents, final_document_tags)
		return texts

	
	def _to_nltk_tree(self, node):
		"""
		## This function turns a spacy Doc into an NLTK Tree from a parent node.
		### Args :
		spacy.tokens.Doc node : the spacy doc containing the processed document.
		### Returns :
		nltk.Tree : the tree reprisentation of the input spacy Doc.
		"""
		if node.n_lefts + node.n_rights > 0:
			return Tree(node.orth_, [self._to_nltk_tree(child) for child in node.children])
		else:
			return node.orth_


	def _get_children(self, node):
		"""
		## This function recursively fetches the children of a tree node.
		### Args :
		* nltk.Tree node : the node from which to recursively descend.
		### Returns :
		* string[] the list of tokens contained beneath the given tree node (original node excluded).
		"""
		node_leaves = []
		for child in node:
			if(str(type(child)) == "<class 'str'>"):
				node_leaves.append(child)
			else:
				node_leaves.append(child.label())
				node_leaves.extend(self._get_children(child))
		return node_leaves
	

	def _get_semantic_parts(self, tree, clss):
		"""
		## This function isolates the base semantic parts from a tree.
		### Args :
		* nltk.Tree tree : the tree whose parts will be isolated.
		* string clss : the class of the tree being sent (ie. the structure type of the tree as produced by the spacy algorithm).
		### Returns :
		* string[][] : each array contains an array of tokens and reprisents a semantic part of the tree (words out of order).
		"""
		semantic_parts = []
		if(clss == "A"):
			for top_node in tree:
				if(str(type(top_node)) != "<class 'str'>"):
					tokens = [top_node.label()]
					tokens.extend(self._get_children(top_node))
					semantic_parts.append(tokens)
		else:
			semantic_parts.append(tree.label())
			semantic_parts.extend(self._get_children(tree))
		
		return semantic_parts


	def to_trees(self, docs, lang="en", length="s"):
		"""
		## This function turns a document into a semantic tree describing the links between component words.
		### Args :
		* `src.data_structures.Documents` docs : the documents object containing the documents to turn into trees.
		* @Optional string lang : the language of the input documents, currently supported languages are english and french.
		* @Optional string length : the length of the input documents to use, "s" for short, "l" for long and "f" for full.
		### Returns :
		* nltk.Tree[] : the array of trees corresponding to the input documents.
		"""
		if(length == "l"):
			data = docs.long_training_set()
		if(length == "f"):
			data = docs.full_training_set()
		else:
			data = docs.short_training_set()

		if(lang == "fr"):
			nlp = spacy.load("fr_core_news_sm")
		else:
			nlp = spacy.load("en_core_web_sm")

		tree_docs = []
		for doc, document in zip(data, docs):
			n = nlp(doc)
			count = 0
			sents = []
			for sent in n.sents:
				if(count == 0):
					tree_doc = self._to_nltk_tree(sent.root)
					tree_docs.append(tree_doc)
					document.set_tree(tree_doc)
				count += 1
		return tree_docs


	def _get_pos_tag(self, word, tagged_documents):
		"""
		## This function fetches the POS tag associated to the input word according to the tagged documents.
		### Args:
		* String word : the word whose POS tag we're interested in.
		* TaggedDocument : the tagged document that contains the word.
		### Returns :
		* String : the POS tag of the input word.
		"""
		word = word.lower()
		i = 0
		num = -1
		for w in tagged_documents[0]:
			if(w == word):
				num = i
			i+=1
		if(num != -1):
			tag = tagged_documents[1][num]
			if(num > 0):
				if(tag.find("NN") == 0 and tagged_documents[1][num-1].find("TO") == 0):
					tag = "VB"
			return tag
		else:
			return "NN"


	def parts_of_user_stories(self, docs, lang="en", length="s", debug=False):
		"""
		## This function isolates the subject, actions, object and description of a set of user stories and returns them.
		### Args :
		* `src.data_structures.Documents` docs : the documents object containing the documents to be separated.
		* @Optional string lang : the language of the input documents, currently supported languages are english and french.
		* @Optional string length : the length of the input documents to use, "s" for short, "l" for long and "f" for full. Default is "s" and "s" is the recommended value as most user stories are described by the "short_training_set".
		### Returns :
		* String[][][] : an array of string arrays for each input document, each string array containing the words relating to that part of the phrase (subject, action, object or description).
		"""
		self.complex_tokenize_tag(docs, False, False, True, length, lang)
		tree_docs = self.to_trees(docs, lang, length)
		parts_all_docs = []
		for i, doc in zip(range(len(docs)), docs):
			subject = []
			actions = []
			objects = []
			other_words = []
			num = 0
			use_alt = False
			if(debug):
				print("\n")
				print(docs.short_training_set()[i])
				print(docs.tagged_documents[i])
				if(doc.tree != None):
					doc.tree.pretty_print()
				print(doc.tree.label(), ":", self._get_pos_tag(doc.tree.label(), docs.tagged_documents[i]))
			if(str(type(doc.tree)) != "<class 'str'>"):
				if(self._get_pos_tag(doc.tree.label(), docs.tagged_documents[i]).find("VB") == 0):
					actions.append(doc.tree.label())
			if(doc.tree != None and str(type(doc.tree)) != "<class 'str'>"):
				for child in doc.tree:
					if(str(type(child)) != "<class 'str'>"):
						temp = []
						temp.append(child.label())
						temp.extend(self._get_children(child))
						if(num == 0):
							for word in temp:
								tag = self._get_pos_tag(word, docs.tagged_documents[i])
								if(debug):
									print(word, ":", tag)
								if(tag.find("VB") == 0 or tag.find("NN") == 0 or tag.find("JJ") == 0):
									subject.append(word)
							c = 0
							for baby in child:
								c += 1
							if(c > 3):
								use_alt = True
						if(num == 1):
							for word in temp:
								tag = self._get_pos_tag(word, docs.tagged_documents[i])
								if(debug):
									print(word, ":", tag)
								if(tag.find("VB") == 0):
									actions.append(word)
								elif(tag.find("NN") == 0 or tag.find("JJ") == 0):
									objects.append(word)
						if(num >= 2):
							for word in temp:
								tag = self._get_pos_tag(word, docs.tagged_documents[i])
								if(debug):
									print(word, ":", tag)
								if(tag.find("VB") == 0 or tag.find("NN") == 0 or tag.find("JJ") == 0):
									other_words.append(word)
						num += 1
				if(debug):
					print("Alt :", use_alt)
					print("Subject :", subject)
					print("Actions :", actions)
					print("Objects :", objects)
			else:
				print(doc.short_training_value())
			if(use_alt):
				if(debug):
					print("USING ALT 1")
				num = 0
				subject = []
				actions = []
				objects = []
				other_words = []
				if(self._get_pos_tag(doc.tree.label(), docs.tagged_documents[i]).find("VB") == 0):
					actions.append(doc.tree.label())
				for child in doc.tree:
					if(str(type(child)) != "<class 'str'>"):
						if(num == 0):
							n = 0
							for infant in child:
								if(str(type(infant)) != "<class 'str'>"):
									temp = []
									temp.append(infant.label())
									temp.extend(self._get_children(infant))
									if(n == 0):
										for word in temp:
											tag = self._get_pos_tag(word, docs.tagged_documents[i])
											if(tag.find("VB") == 0 or tag.find("NN") == 0 or tag.find("JJ") == 0):
												subject.append(word)
									if(n == 1):
										for word in temp:
											tag = self._get_pos_tag(word, docs.tagged_documents[i])
											if(tag.find("VB") == 0):
												actions.append(word)
											elif(tag.find("NN") == 0 or tag.find("JJ") == 0):
												objects.append(word)
									if(n >= 2):
										for word in temp:
											tag = self._get_pos_tag(word, docs.tagged_documents[i])
											if(tag.find("VB") == 0 or tag.find("NN") == 0 or tag.find("JJ") == 0):
												other_words.append(word)
									n += 1
						if(num >= 1):
							a = self._get_children(child)
							for word in a:
								tag = self._get_pos_tag(word, docs.tagged_documents[i])
								if(tag.find("NN") == 0 or tag.find("VB") == 0 or tag.find("JJ") == 0):
									other_words.append(word)
						num += 1
			if((subject == [] and actions == []) or (subject == [] and objects == []) or (objects == [] and actions == [])):
				if(debug):
					print("USING ALT 2")
				act, obj, sub = self.isolate_parts(Documents(documents=[docs.documents[i]]))
				if(sub != []):
					subject = sub[0]
				if(act != []):
					actions = act[0]
				if(obj != []):
					objects = obj[0]
			# reorganise subject, actions, objects and other_words to the order they appear in.
			ordered_subject = []
			ordered_actions = []
			ordered_objects = []
			ordered_other_words = []
			subject = [word.lower() for word in subject]
			objects = [word.lower() for word in objects]
			actions = [word.lower() for word in actions]
			other_words = [word.lower() for word in other_words]
			for word in docs.tagged_documents[i][0]:
				if(word in subject):
					ordered_subject.append(word)
				if(word in actions):
					ordered_actions.append(word)
				if(word in objects):
					ordered_objects.append(word)
				if(word in other_words):
					ordered_other_words.append(word)
			parts_all_docs.append([ordered_subject, ordered_actions, ordered_objects, ordered_other_words])

		if(debug):
			print("Tree docs :", len(tree_docs))
			print("Docs :", len(docs.short_training_set()))
		return parts_all_docs

		# refactor to have trees part of Documents data structure
	

	def prepare_parts_of_user_stories(self, docs, subject=False, actions=False, objects=True, others=False, synonyms=False, remove_stopwords=True, lang="en"):
		"""
		## This function preprocesses the input documents and splits them into parts of user stories in preparation for being processed by the model.
		### Args :
		* `src.data_structures.Documents` docs : the documents to preprocess and split up.
		* @Optional bool subject : determines whether the subjects of the user stories will be used in the training documents.
		* @Optional bool actions : determines whether the actions of the user stories will be used in the training documents.
		* @Optional bool objects : determines whether the objects of the user stories will be used in the training documents.
		* @Optional bool others : determines wheter the rest of the user stories will be used in the training documents.
		* @Optional bool synonyms : determines wheter the documents will be supplemented by synonym tokens.
		* @Optional bool remove_stopwords : determines whether the documents will have their stopwords removed.
		* @Optional string lang : selection for the stopwords language, "en" for english and "fr" for french, which are the only currently supported languages.
		### Returns :
		* (string, string)[][] : an array of tuple arrays, each tuple array reprisenting a document and each tuple containing a word and its corresponding tag.
		"""
		p = Preprocessing()
		pop = p.parts_of_user_stories(docs)
		#verification(pop, docs)
		tagged_docs = docs.tagged_documents
		# remove stopwords
		if(lang=="fr"):
			stop = set(stopwords.words("french"))
		else:
			stop = set(stopwords.words('english'))
		pod = []
		if(remove_stopwords):
			for document in pop:
				doc_val = []
				for clss in document:
					temp = [i for i in clss if i not in stop]
					doc_val.append(temp)
				pod.append(doc_val)
		else:
			pod = pop
		if(synonyms):
			#pad = add_synonyms_to_parts_of_phrase(pod)
			pad = self.add_synonyms_to_parts_of_phrase_wordnet(pod, "en", 0.5, True, True, True, 5)
			for i in range(len(pad)):
				if(len(pod[i]) >= 4):
					pad[i].append(pod[i][3])
		else:
			pad = pod
		new_tok_docs = []
		new_tag_docs = []
		for i, pa in zip(range(len(pad)), pad):
			temp = []
			if(subject):
				temp.extend(pa[0])
			if(actions):
				temp.extend(pa[1])
			if(objects):
				temp.extend(pa[2])
			if(others):
				temp.extend(pa[3])
			if(temp == [] or len(temp) <= 5):
				temp.extend(pa[3])
				if(temp == [] or len(temp) <= 5):
					temp.extend(pa[1])
					if(temp == [] or len(temp) <= 5):
						temp.extend(pa[0])
			new_tok_docs.append(temp)
			new_tag_docs.append(TaggedDocument(temp, [i]))
		docs.tokenized_documents = new_tok_docs
		docs.tagged_documents = new_tag_docs
		p.to_dictionary(docs)
		p.to_doc_term_matrix(docs)
		return tagged_docs


	def add_synonyms_to_parts_of_phrase_wordnet(self, pop, lang="en", min_similarity_index=0.6, use_word2vec=True, use_wordnet=True, get_hypernyms=True, max_synonyms=5):
		"""
		"""
		m = Model()

		if(use_word2vec):
			print("Loading model...")
			model = m.word2vec_load('glove-wiki-gigaword-300')

		if(lang=="fr"):
			language="fra"
		else:
			language="eng"

		print("Processing synsets...")

		total_length = 0
		for doc in pop:
			total_length += len(doc)
		count = 0
		x = total_length/20
		i = 0
		print("[", end="", flush=True)

		result = []
		for document in pop:
			doc_array = []
			num = 0
			for words in document:
				if(i >= x * count):
					print("=", end="", flush=True)
					count += 1
				if(num > 0):
					synonyms = [w for w in words]
					for word in words:
						if(use_wordnet):
							try:
								word_synsets = wordnet.synsets(word, lang=language)
								if(len(word_synsets) > 0):
									word_synset = word_synsets[0]
									for syn in word_synsets:
										num_syns = 0
										for s in syn.lemmas():
											if(use_word2vec):
												if(model.similarity(word, s.name()) > min_similarity_index and num_syns < max_synonyms):
													synonyms.append(s.name())
											else:
												if(num_syns < max_synonyms):
													synonyms.append(s.name())
													num_syns += 1
							except:
								pass
						if(get_hypernyms):
							try:
								word_synsets = wordnet.synsets(word, lang=language)
								if(len(word_synsets) > 0):
									wordnet_word = wordnet.synset(word_synsets[0].name())
									hypernyms = wordnet_word.hypernyms()
									for hypernym in hypernyms:
										for lemma in hypernyms.lemmas():
											synonyms.append(lemma.name())
							except:
								pass
					doc_array.append(synonyms)
				i += 1
				num += 1
			result.append(doc_array)
		print("]")

		final = []
		for document in result:
			new_doc = []
			for set_words in document:
				added_words = []
				new_set = []
				for word in set_words:
					go = True
					for check in added_words:
						if(check == word):
							go = False
					if(go):
						new_set.append(word)
						added_words.append(word)
				new_doc.append(new_set)
			final.append(new_doc)
		return final


	def main(self):
		documents = Documents([Document("as a user i want to be able to change my peudonym"), Document("as an admin i would like to eat posh food"), Document("as a senior software engineer i want to be able to redesign my database")])
		results = self.remove_subject(documents)
		print(results)



if __name__ == '__main__':
	from dataset import Dataset
	prep = Preprocessing()
	d = Dataset()
	documents = d.read_from_file("data/user_story_datasets/g27-culrepo.txt")
	prep.remove_subject(documents)
	prep.main()



