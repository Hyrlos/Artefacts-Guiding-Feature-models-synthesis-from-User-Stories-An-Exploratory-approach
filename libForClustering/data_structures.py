
"""
## This module contains key data structures used in the rest of the scripts.
"""

class Issue:

	"""
	## This object describes a GitLab issue.
	"""

	def __init__(self, iid, title="", description=""):
		"""
		## This function creates a new Issue object from the input issue ID, title and description.
		### Args :
		* int iid : the issue ID, unique for each issue in a GitLab project.
		* String title : the title of the issue.
		* String description : the description of the issue.
		### Returns :
		* `src.data_structures.Issue` : a reprisentation of a GitLab Issue object (smaller than the default ProjectIssue object from python-gitlab library, containing only important information for this application).
		"""
		self.id : int = iid #: int : The ID of the GitLab issue.
		self.title : str = title #: String : The title of the GitLab issue.
		self.description : str = description #: String : The Description of the GitLab issue.


	def __str__(self):
		return str(self.id)+" : "+self.title+"\n"+self.description




class Document:

	"""
	## This object describes a natural language document and offers methods for easy manipulation of the document.
	"""

	def __init__(self, title="", description="", issue=None):
		"""
		## This function creates a new Document object from either the string reprisentation of it's title and description, or from a previously created Issue object.
		### Args :
		* @Optional String title : the title of the Document object.
		* @Optional String description : the description of the Document object.
		* @Optional `src.data_structures.Issue` issue : a GitLab issue object that the Document will take the title and description of.
		### Result :
		* `src.data_structures.Document` : a document containing the input values/
		"""
		self.title = title #: String : The title of the document. In the case of user stories, this would be the user story itself.
		self.description = description #: String : The description of the document, can be used to improve classification.
		self.subject = []
		self.object = []
		self.actions = []
		self.desc = []
		self.tree = None
		if(issue != None):
			self.title = issue.title #: String : The title of the document.
			self.description = issue.description #: String : The description of the document.


	def __str__(self):
		return self.title+"\n"+self.description


	def set_title(self, new_title):
		self.title = new_title


	def short_training_value(self):
		"""
		## This function provides a short document for NLP applications.
		### Returns :
		* String : a short string reprisentation of the document. It's title.
		"""
		return self.title


	def long_training_value(self):
		"""
		## This function provides a long document for NLP applications.
		### Returns :
		* String : a long string reprisentation of the document. It's description.
		"""
		return self.description


	def full_training_value(self):
		"""
		## This function provides the full document for NLP applications. Only marginally longer than the long document.
		### Returns :
		* String : a long string reprisentation of the entire document.
		"""
		return self.title+" "+self.description


	def to_write_format(self):
		"""
		## This function turns the document into a writable string than can be easily parsed later.
		### Returns :
		* String : the formatted data corresponding to this object.
		"""
		return self.title+"<@split_here>"+self.description


	def from_write_format(self, raw_data):
		"""
		## This function reads a formatted string and stores the contained data in this object.
		### Args :
		* String raw_data : the formatted string that contains the information to be stored in the document object.
		"""
		if(raw_data.find("<@split_here>") != -1):
			split_data = raw_data.split("<@split_here>")
			self.title = split_data[0]
			self.description = split_data[1]
		else:
			print("ERROR - raw data format not recognised")


	def set_tree(self, tree):
		self.tree = tree



class Documents:

	"""
	## This object contains an array of natural language documents, and offers methods to easily manage them.
	"""

	def __init__(self, documents=None, texts=None, deep_copy=None):
		"""
		## This function creates a new Documents object from an array of Document objects. Order of priority : deep copy > texts > documents
		### Args :
		* `src.data_structures.Document`[] documents : the array of documents from which the Documents object is created.
		* String[] texts : an array of strings to be turned into documents.
		* `src.data_structures.Documents` deep_copy : a Documents object that this object will be a deep copy of.
		### Returns :
		* `src.data_structures.Documents` : the newly created Documents object.
		"""
		self.documents = [] #: `src.data_structures.Document`[] : The documents corresponding to the corpus.
		if(documents != None):
			self.documents = documents
		self.tokenized_documents = [] #: String[][] : Contains the array of tokenized documents.
		self.tagged_documents = [] #: TaggedDocument([String], [int] | [String]) : Contains the array of tagged documents.
		self.document_dictionary = None #: Dictionary(int, String) : Contains the dictionary of the documents.
		self.document_term_matrix = [] #: (int, int)[][] : Contains the document-term matrix of the documents.
		self.predicted_values = [] #: float[][] : The document vectors corresponding to the documents contained in this object.
		self.ngrams = [] #: ngrams to do with the current document.
		if(deep_copy != None):
			self.documents = []
			for document in deep_copy.documents:
				self.documents.append(Document(document.title, document.description))
		elif(texts != None):
			self.documents = []
			for text in texts:
				self.documents.append(Document(text))


	def __str__(self):
		s = ""
		for doc in self.documents:
			s += str(doc) + "\n"
		return s


	def __len__(self):
		return len(self.documents)


	def __iter__(self):
		for doc in self.documents:
			yield doc


	def add_document(self, document):
		"""
		## This function adds a document to the array contained in this Documents object.
		### Args :
		* `src.data_structures.Document` document : the document to add to the array.
		"""
		self.documents.append(document)


	def add_documents(self, documents):
		"""
		## This function adds all the documents contained in a Documents object to this object.
		### Args :
		* `src.data_structures.Documents` documents : the Documents object whose documents you wish to add to this Documents object.
		"""
		for doc in documents.documents:
			self.documents.append(doc)


	def short_training_set(self):
		"""
		## This function returns an array of short documents as strings ready for preprocessing.
		### Returns :
		* String[] : an array of short strings reprisenting a corpus of documents.
		"""
		training_set = []
		for doc in self.documents:
			training_set.append(doc.short_training_value())
		return training_set


	def long_training_set(self):
		"""
		## This function returns an array of long documents as strings ready for preprocessing.
		### Returns :
		* String[] : an array of long strings reprisenting a corpus of documents.
		"""
		training_set = []
		for doc in self.documents:
			training_set.append(doc.long_training_value())
		return training_set


	def full_training_set(self):
		"""
		## This function returns an array of full documents as strings ready for preprocessing.
		### Returns :
		* String[] : an array of full strings reprisenting a corpus of documents.
		"""
		training_set = []
		for doc in self.documents:
			training_set.append(doc.full_training_value())
		return training_set


	def to_write_format_list(self):
		"""
		## This function translates the contained array of documents to an easily parsable array of strings, ready to be written to a file.
		### Returns :
		* String[] : an array of strings each corresponding to a formatted reprisentation of a Document object.
		"""
		write_format = []
		for doc in self.documents:
			write_format.append(doc.to_write_format()+"<@end_doc>\n")
		return write_format


	def from_write_format_list(self, raw_data):
		"""
		## This function takes properly formatted data and parses it into an array of Document objects.
		### Args :
		* String : a single string containing a formatted list of documents in a specific format, to be turned into this Documents object's array of Document objects.
		"""
		raw_data = raw_data.split("<@end_doc>")
		documents = []
		for raw in raw_data:
			if(raw.find("<@split_here>") != -1):
				d = Document()
				d.from_write_format(raw)
				documents.append(d)
		self.documents = documents



if __name__ == '__main__':
	doc1 = Documents([Document("One")])
	doc2 = Documents([Document("Three")])
	print("Normal :")
	print(doc1)
	doc1.add_document(Document("Two"))
	print("Add two :")
	print(doc1)
	doc1.add_documents(doc2)
	print("Add three :")
	print(doc1)














