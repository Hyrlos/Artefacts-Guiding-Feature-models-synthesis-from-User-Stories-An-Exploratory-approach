"""
## This module provides access to functions that fetch, read and write documents to and from various locations such as GitLab or local files.
"""

from libForClustering.data_structures import Issue, Document, Documents
import gitlab

class Dataset:

	"""
	## This class contains the functions of the module. It also stores data so it can be accessed independantly of return values.
	"""

	def __init__(self):
		self.pid = 12440105 #: int : The project ID of the project I'm currently pulling all of the user stories (issues) from.
		self.issues = None #: `src.data_structures.Issue`[] : The issues downloaded from a project.
		#self.documents = None #: `src.data_structures.Documents` : The documents extracted from a project's issues.


	def get_issues_from_project(self, projectid, getAll=False, per_page=20, page=1):
		"""
		## This function fetches all of the "issues" from a gitlab repository.
		### Args :
		* int projectid : this is an integer reprisenting the project id found on all gitlab projects.
		* @optionnal bool getAll : False by default, if false it will only fetch the first 20 issues, else this function will fetch all of them (warning, there may be thousands)
		### Returns :
		* `src.data_structures.Issue`[] : returns an array of issue objects.
		"""
		gl = gitlab.Gitlab()
		project = gl.projects.get(projectid)
		issues = project.issues.list(all=getAll, per_page=per_page, page=page)
		final_issues = []
		for issue in issues:
			final_issues.append(Issue(issue.iid, issue.title, issue.description))
		self.issues = final_issues
		return final_issues

	
	def isolate_user_stories_from_issues(self, issues):
		"""
		## This function isolates the user stories from the rest of the issues. A user story being a title in the format "As X I want to be able to do Y".
		### Args :
		* `src.data_structures.Issue`[] issues : this is an array of issues, in the same format as returned by the getIssuesFromProject function.
		### Returns :
		* `src.data_structures.Issue`[] : this is an array of issues corresponding to the user stories.
		* `src.data_structures.Issue`[] : this is an array of issues that aren't user stories.
		"""
		user_stories = []
		rejected = []
		for issue in issues:
			low_issue = issue.title.lower()
			find_as = low_issue.find("as ")
			find_mid_as = low_issue.find(" as ")
			find_has = low_issue.find("has")

			found_has = True
			if(find_has != -1):
				found_has = find_as != find_has+1

			if(find_as != -1 and found_has and find_as <= 12):
				user_stories.append(issue)
			else:
				rejected.append(issue)
		self.issues = user_stories
		return user_stories, rejected

	
	def get_documents(self, issues):
		"""
		## This function isolates the titles of an array of Issues.
		### Args :
		* `src.data_structures.Issue`[] issues : this is an array of issues, in the same format as returned by the getIssuesFromProject function.
		### Returns :
		* `src.data_structures.Documents` : returns Documents object containing the array of Document objects.
		"""
		document_array = []
		for issue in issues:
			document_array.append(Document(issue=issue))
		self.documents = Documents(document_array)
		return self.documents

	
	def write_to_file_readable(self, issues, filename):
		"""
		## This function writes the titles of the issues to a file in an easily human readable format.
		### Args :
		* `src.data_structures.Issue`[] issues : this is an array of issues, in the same format as returned by the getIssuesFromProject function.
		* String filename : this is the name of the file that the titles will be written to. Do not write the file extension, it is always a .txt.
		### Returns :
		* bool : returns True if the issues were written correctly, and False if not.
		"""
		f = open(filename, "w", encoding="UTF-8")
		try:
			counter = 0
			to_write = ""
			for issue in issues:
				to_write = to_write + counter.__str__()+") "+issue.title+"\n\n"
				counter+=1
			f.write(to_write)
			f.close()
			return True
		except:
			#print("ERROR - write_to_file_readable ("+filename+")")
			f.close()
			return False


	def write_to_file(self, documents, filename):
		"""
		## This function writes the titles of the issues to a file in an easily computer readable format.
		### Args :
		* `src.data_structures.Documents` documents : this is a Documents object containing the array of documents.
		* String filename : this is the name of the file that the titles will be written to. Do not write the file extension, it is always a .txt.
		### Returns :
		* bool : returns True if the documents were written correctly, and False if not.
		"""
		#print("Write to", filename)
		f = open(filename, "w", encoding="UTF-8")
		try:
			data = documents.to_write_format_list()
			to_write = ""
			for d in data:
				to_write = to_write+d
			f.write(to_write)
			f.close()
			return True
		except:
			#print("ERROR - WriteToFileError ("+filename+")")
			f.close()
			return False


	def write_strings_to_file(self, str_list, filename):
		"""
		## This function writes the input array of strings to a file.
		### Args :
		* String[] str_list : the array of strings to write to the file.
		* String filename : the path to the file to which you want to write the strings.
		### Returns :
		* bool : returns True if the issues were written correctly, and False if not.
		"""
		f = open(filename, "w", encoding="UTF-8")
		try:
			to_write = ""
			for d in str_list:
				to_write = to_write+str(d)+"\n"
			to_write.replace("\n\n", "\n")
			f.write(to_write)
			f.close()
			return True
		except:
			#print("ERROR - WriteToFileError ("+filename+")")
			f.close()
			return False


	def write_line_sentence(self, documents, filename):
		"""
		## This function writes tokenized documents to a file in LineSentence format.
		### Args :
		* `src.data_structures.Documents` documents : the Documents object containing the tokenized documents.
		* String filename : the file to which to write the tokenized documents in LineSentence format.
		### Returns :
		* bool : returns True if write is successful, or False if not.
		"""
		to_write = ""
		for doc in documents.tokenized_documents:
			for word in doc:
				to_write += word + " "
			to_write += "\n"
		f = open(filename, "w", encoding="UTF-8")
		try:
			f.write(to_write)
			f.close()
			return True
		except:
			f.close()
			return False


	def append_line_sentence(self, documents, filename):
		"""
		## This function appends tokenized documents to a file in LineSentence format.
		### Args :
		* `src.data_structures.Documents` documents : the Documents object containing the tokenized documents.
		* String filename : the file to which to append the tokenized documents in LineSentence format.
		### Returns :
		* bool : returns True if write is successful, or False if not.
		"""
		to_write = ""
		for doc in documents.tokenized_documents:
			for word in doc:
				to_write += word + " "
			to_write += "\n"
		f = open(filename, "a", encoding="UTF-8")
		try:
			f.write(to_write)
			f.close()
			return True
		except:
			f.close()
			return False



	def append_to_file(self, documents, filename):
		"""
		## This function appends the titles of the issues to a file in an easily computer readable format.
		### Args :
		* `src.data_structures.Documents` documents : this is a Documents object containing the array of documents.
		* String filename : this is the name of the file that the titles will be written to. Do not write the file extension, it is always a .txt.
		### Returns :
		* bool : returns True if the documents were written correctly, and False if not.
		"""
		#print("Write to", filename)
		f = open(filename, "a", encoding="UTF-8")
		try:
			data = documents.to_write_format_list()
			to_write = ""
			for d in data:
				to_write = to_write+d
			f.write(to_write)
			f.close()
			return True
		except:
			#print("ERROR - WriteToFileError ("+filename+")")
			f.close()
			return False

	
	def read_from_file(self, filename):
		"""
		## This function reads a file containing formatted document data and returns the associated Documents object.
		### Args :
		* String filename : the name of the file that contains the formatted data.
		### Returns :
		* `src.data_structures.Documents` : returns a Documents object containing the read documents or None if the read failed.
		"""
		#print("Reading from", filename)
		try:
			f = open(filename, "r", encoding="UTF-8")
			data = f.read()
			f.close()
			documents = Documents()
			documents.from_write_format_list(data)
			self.documents = documents
			return documents
		except:
			print("ERROR - ReadFromFileError ("+filename+")")
			return None


	def read_documents_from_file(self, fname, min_doc=0, max_doc=100000):
		"""
		## This function reads a certain number of documents from a file containing formatted documents data.
		### Args :
		* String fname : the name of the file from which to read the data.
		* @Optional int min_doc : the minimum index of the document to read (skips the first "min_doc" documents).
		* @Optional int max_doc : the maximum index of the document to read (inclusive).
		### Returns :
		* `src.data_structures.Documents` : the documents read from the file.
		"""
		try:
			data = ""
			with open(fname, "r", encoding="UTF-8") as file:
				for line in file:
					fount_EOD = False
					if(line.find("<@end_doc>") != -1):
						fount_EOD = True
					if(min_doc <= 0 and max_doc >= 0):
						data += line
					if(fount_EOD):
						min_doc -= 1
						max_doc -= 1
			documents = Documents()
			documents.from_write_format_list(data)
			self.documents = documents
			return documents
		except:
			return None



	def read_unformatted_from_file(self, filename):
		"""
		## This function reads a file containing documents and returns them in a new Documents object.
		### Args :
		* String filename : the name of the file that contains the unformatted data.
		### Returns :
		* `src.data_structures.Documents` : returns a Documents object containing the read documents or None if the read failed.
		"""
		#print("Reading from", filename)
		try:
			f = open(filename, "r", encoding="UTF-8")
			data = f.read()
			f.close()
			documents = Documents()
			data = data.split('\n')
			try:
				while(True):
					data.remove('')
			except ValueError:
				pass
			for d in data:
				documents.add_document(Document(d))
			#self.documents = documents
			return documents
		except:
			print("ERROR - ReadFromFileError ("+filename+")")
			return None



	def main(self):
		pass



if __name__ == '__main__':
	dataset = Dataset()
	dataset.main()

