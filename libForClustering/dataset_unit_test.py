
"""
This module runs tests for the `src.dataset` module.
"""

import unittest
import os
from dataset import Dataset
from data_structures import *

#print(get_issues_from_project(12440105))
#print(get_issues_from_project(12440105, True))

class TestDataset(unittest.TestCase):

	# Creating necessary classes


	# Setting up constants and check values
	pid = 36812272
	dataset = Dataset()
	issues = dataset.get_issues_from_project(pid, True)
	list_of_issues = [Issue(0, "As a user, I want to do a thing"), Issue(1, "Problems in using the UI as admin"), Issue(2, "Research : As a user, I want to be able to run a logistic regression")]
	documents = Documents([Document("One"), Document("Two")])
	list_of_user_stories = [list_of_issues[0], list_of_issues[2]]
	list_of_rejected_issues = [list_of_issues[1]]
	list_of_issue_titles = [list_of_issues[0].title, list_of_issues[1].title, list_of_issues[2].title]

	# Tests
	def test_get_issues(self):
		test_values = self.dataset.get_issues_from_project(self.pid, False, 20, 1)
		s = ""
		for val in test_values:
			s += str(val)
		vs = ""
		for val in self.issues[:20]:
			vs += str(val)
		self.assertEqual(s, vs, "Did not fetch correct issues from GitLab project")
	def test_get_all_issues(self):
		test_values = self.dataset.get_issues_from_project(self.pid, True)
		s = ""
		for val in test_values:
			s += str(val)
		vs = ""
		for val in self.issues:
			vs += str(val)
		self.assertEqual(s, vs, "Did not fetch all the correct issues from GitLab project")

	def test_isolate_user_stories(self):
		self.assertEqual(self.dataset.isolate_user_stories_from_issues(self.list_of_issues), (self.list_of_user_stories, self.list_of_rejected_issues), "User stories not isolated correctly")

	def test_get_documents(self):
		documents = self.dataset.get_documents(self.list_of_issues)
		check = "As a user, I want to do a thing\n\nProblems in using the UI as admin\n\nResearch : As a user, I want to be able to run a logistic regression\n\n"
		self.assertEqual(str(documents), check, "Doesn't instantiate documents from issues correctly")

	def test_write_to_file(self):
		self.assertEqual(self.dataset.write_to_file(self.documents, 'test/test_file.txt'), True, "Write to file not functionning correctly")
	def test_write_to_file_fail(self):
		self.assertEqual(self.dataset.write_to_file(None, 'test/test.txt'), False, "Write to file fails correctly")
	def test_read_from_file(self):
		check = "One\n\nTwo\n\n"
		self.assertEqual(str(self.dataset.read_from_file('test/test_file_read.txt')), check, "List of issues file not properly read")
	def test_read_from_file_fail(self):
		self.assertEqual(self.dataset.read_from_file('test/this_file_does_not_exist.txt'), None, "Read list of issues from file fails correctly")

if __name__ == '__main__':
	unittest.main()




