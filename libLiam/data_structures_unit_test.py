# author : Liam RICE

"""
This module runs tests for the `src.data_structures` module.
"""

import unittest
from data_structures import *

class TestDataset(unittest.TestCase):

	def test_create_empty_issue(self):
		issue = Issue(0)
		check = "0 : \n"
		self.assertEqual(str(issue), check, "Empty issue not created properly")
	def test_create_issue(self):
		issue = Issue(0, "Title", "Description")
		check = "0 : Title\nDescription"
		self.assertEqual(str(issue), check, "Full issue not created properly")

	def test_create_empty_document(self):
		doc = Document()
		check = "\n"
		self.assertEqual(str(doc), check, "Empty document not created properly")
	def test_create_document(self):
		doc = Document("Title", "Description")
		check = "Title\nDescription"
		self.assertEqual(str(doc), check, "Full document not created properly")
	def test_create_document_from_issue(self):
		issue = Issue(0, "Title", "Description")
		doc = Document(issue=issue)
		check = "Title\nDescription"
		self.assertEqual(str(doc), check, "Full document not created properly from issue")
	def test_create_document_from_issue_overwrite(self):
		issue = Issue(0, "Title", "Description")
		doc = Document("Fake", "Irrelevant", issue)
		check = "Title\nDescription"
		self.assertEqual(str(doc), check, "Full document not created properly, issue doesn't overwrite new title or description")

	def test_short_tv(self):
		doc = Document("Title", "Description")
		check = "Title"
		self.assertEqual(doc.short_training_value(), check, "Does not fetch short training value (title) correctly")
	def test_long_tv(self):
		doc = Document("Title", "Description")
		check = "Description"
		self.assertEqual(doc.long_training_value(), check, "Does not fetch long training value (description) correctly")
	def test_full_tv(self):
		doc = Document("Title", "Description")
		check = "Title Description"
		self.assertEqual(doc.full_training_value(), check, "Does not fetch full training value (all data) correctly")

	def test_to_write_format(self):
		doc = Document("Title", "Description")
		check = "Title<@split_here>Description"
		self.assertEqual(doc.to_write_format(), check, "Does not translate to write format correctly")
	def test_from_write_format(self):
		doc = Document()
		write_format = "Title<@split_here>Description"
		check = "Title\nDescription"
		doc.from_write_format(write_format)
		self.assertEqual(str(doc), check, "Document does not parse write format correctly")

	def test_create_empty_documents(self):
		docs = Documents()
		check = ""
		self.assertEqual(str(docs), check, "Does not create empty Documents object correctly")
	def test_create_documents(self):
		doc = Document("Title", "Description")
		doc2 = Document("", "Two")
		docs = Documents([doc, doc2])
		check = "Title\nDescription\n\nTwo\n"
		self.assertEqual(str(docs), check, "Does not create new Documents object correctly")
	def test_deep_copy_documents(self):
		doc = Document("Title", "Description")
		doc2 = Document("", "Two")
		docs = Documents([doc, doc2])
		dc_docs = Documents(deep_copy=docs)
		check = "Title\nDescription\n\nTwo\n"
		self.assertEqual(str(dc_docs), check, "Does not create deep copy of Documents object correctly")
	def test_add_document(self):
		doc = Document("Title", "Description")
		doc2 = Document("", "Two")
		docs = Documents([doc])
		docs.add_document(doc2)
		check = "Title\nDescription\n\nTwo\n"
		self.assertEqual(str(docs), check, "Does not add new Document objects correctly")
	def test_short_training_set(self):
		doc = Document("Title", "Description")
		doc2 = Document("", "Two")
		docs = Documents([doc, doc2])
		check = "['Title', '']"
		self.assertEqual(str(docs.short_training_set()), check, "Does not generate short training set correctly")
	def test_long_training_set(self):
		doc = Document("Title", "Description")
		doc2 = Document("", "Two")
		docs = Documents([doc, doc2])
		check = "['Description', 'Two']"
		self.assertEqual(str(docs.long_training_set()), check, "Does not generate long training set correctly")
	def test_full_training_set(self):
		doc = Document("Title", "Description")
		doc2 = Document("", "Two")
		docs = Documents([doc, doc2])
		check = "['Title Description', ' Two']"
		self.assertEqual(str(docs.full_training_set()), check, "Does not generate full training set correctly")
	def test_to_write_format_list(self):
		doc = Document("Title", "Description")
		doc2 = Document("", "Two")
		docs = Documents([doc, doc2])
		check = "['Title<@split_here>Description<@end_doc>\\n', '<@split_here>Two<@end_doc>\\n']"
		self.assertEqual(str(docs.to_write_format_list()), check, "Does not generate format list correctly")
	def test_from_write_format_list(self):
		source = "Title<@split_here>Description<@end_doc><@split_here>Two<@end_doc>"
		docs = Documents()
		docs.from_write_format_list(source)
		check = "Title\nDescription\n\nTwo\n"
		self.assertEqual(str(docs), check, "Documents object does not parse write format correctly")





























