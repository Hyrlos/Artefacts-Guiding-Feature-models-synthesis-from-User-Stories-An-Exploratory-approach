# author : Liam RICE

from enum import Enum

import math
from libLiam.data_structures import *
from libLiam.feature_model import *
from libLiam.preprocessing import *
from libLiam.cluster import *
from libLiam.model import *
from libLiam.dataset import *
from gensim.models.doc2vec import TaggedDocument


class FeatureProcessing:

	def __init__(self, model_name="glove-wiki-gigaword-300"):
		model = Model()
		self.word2vec_model = model.word2vec_load(model_name) #: Word2Vec word2vec_model : the model used for selection of similar words.
		self.added_relations = [] #: `src.feature_model.Relation`[] added_relatios : a fail-save check to make sure no relations are added to the model twice.
		self.existing_names = [] #: string[] a list of all names in the feature list so doubles are not generated and cause errors
		self.MAX_CLUSTERS = 7


	def construct_tree(self, tree_name, top_subtrees):
		"""
		## This method builds a new feature model from the feature model's name and its direct subtrees.
		### Args :
		* string tree_name : the name of the tree, which will end up as the root feature's name.
		* `src.feature_model.FeatureOrientedDomainAnalysisTree`|`src.feature_model.Feature`[] top_subtrees : the top feature models or features that are the direct children of the root node.
		Returns :
		* `src.feature_model.FeatureOrientedDomainAnalysisTree` : the final assembled feature model.
		"""
		tree = FeatureOrientedDomainAnalysisTree()
		root = Feature(tree_name, object_type=ObjectType.ABSTRACT)
		tree.add_node(root)
		for subtree in top_subtrees:
			print("Adding", subtree, "to", tree)
			if subtree != tree:
				if str(type(subtree)).find("feature_model.FeatureOrientedDomainAnalysisTree") != -1:
					tree.add_subtree(subtree, root)
					#dataset = Dataset()
					#dataset.write_strings_to_file([subtree.to_uml()], str(subtree)+".uml")
				elif(str(type(subtree)).find("feature_model.Feature") != -1):
					tree.add_node(subtree, root)
				else:
					pass
		return tree


	def create_feature(self, name, is_concrete=True):
		"""
		## This method creates a feature from a name and whether it is concrete or not.
		### Args :
		* string name : the name of the feature.
		* @Optional bool is_concrete : True if the feature is concrete, or False if it's abstract.
		### Returns :
		* `src.feature_model.Feature` : the assembled feature.
		"""
		if(is_concrete):
			return Feature(name)
		else:
			return Feature(name, object_type=ObjectType.ABSTRACT)
	

	def generate_name(self, cluster):
		"""
		## This function takes a cluster and infers a name for the super-feature encompassing them.
		### Args :
		* (`src.feature_model.Feature`|`src.feature_model.FeatureOrientedDomainAnalysisTree`|string), string[], (string, string)[], (int, int)[][])[] cluster : a single cluster found by clustering method including sorted into component data. 
		### Returns :
		* String : the new name of the super-feature.
		"""
		name = ""
		highest_sim_words = None
		max_sim = -1
		for f1, t1, ta1, d1 in cluster:
			for tok1 in t1:
				for f2, t2, ta2, d2 in cluster:
					if f1 != f2:
						for tok2 in t2:
							if tok1 != tok2 and not (tok1.startswith(tok2) or tok2.startswith(tok1)):
								try:
									sim = self.word2vec_model.similarity(tok1, tok2)
									if sim > max_sim or max_sim == -1:
										max_sim = sim
										highest_sim_words = (tok1, tok2)
								except:
									pass
		i = 0
		if highest_sim_words != None:
			max = len(highest_sim_words)
			for word in highest_sim_words:
				name += word
				if(i < max-1):
					name += " "
				i += 1
		if name == "" or name.isspace() or name in self.existing_names:
			print("Generating alternate name...")
			for f, t, ta, d in cluster:
				name += t[math.floor(len(t)/2)]
			return name, max_sim
		else:
			return name, max_sim


	def group(self, clusters):
		"""
		## This function takes in the generated clusters and groups the sub-features under each parent feature, and returns the stats of the parent feature.
		### Args :
		* (`src.feature_model.Feature`|`src.feature_model.FeatureOrientedDomainAnalysisTree`|string), string[], (string, string)[], (int, int)[][])[][] clusters : the data of the current features in use sorted by cluster.
		### Returns :
		* (`src.feature_model.Feature`|`src.feature_model.FeatureOrientedDomainAnalysisTree`|string), string[], float, (`src.feature_model.Feature`|`src.feature_model.FeatureOrientedDomainAnalysisTree`|string), string[], (string, string)[], (int, int)[][])[], bool)[]
		"""
		new_features = []
		max_len = 0
		for cluster in clusters:
			max_len += len(cluster)
			if len(cluster) > 1:
				# getting name of super-feature
				name, sim = self.generate_name(cluster)
				# generating new tokens
				tokens = []
				for d, t, ta, d in cluster:
					tokens.extend(t)
				no_duplicates_tokens = remove_duplicates(tokens)
				# generating new feature information
				new_feature = (name, no_duplicates_tokens, sim, cluster, False)
				new_features.append(new_feature)
			elif len(cluster) == 1:
				new_feature = (cluster[0][0], cluster[0][1], 1, [], True)
				new_features.append(new_feature)
			else:
				pass
		return new_features


	def initial_features(self, documents):
		"""
		## This method assembles the leaf features of the feature model to build.
		### Args :
		* `src.data_structures.Documents` documents : the user stories as documents, with each user story reprisenting a feature.
		### Returns :
		* `src.feature_model.Feature`[] : the leaf features in the model-to-be.
		"""
		base_features = []
		for document, keywords in zip(documents, documents.tokenized_documents):
			# make analysis to determine if mandatory or optional?
			name = document.short_training_value()
			length = 5
			n = ""
			for i, word in zip(range(1, len(keywords)), keywords):
				if(i<length):
					n += word + " "
			new_feature = Feature(n)
			base_features.append(new_feature)
		return base_features


	def get_vectors_doc2vec(self, tags, tokens):
		"""
		## This method fetches document vectors using Doc2Vec for the input tags and tokens.
		### Args :
		* TaggedDocument[] tags : the tagged documents used to train the model.
		* string[][] tokens : the tokens of the documents that are passed in to be vectorised.
		### Returns :
		* float[][] : the vectors of all the input documents.
		"""
		documents = Documents()
		documents.tokenized_documents = tokens
		documents.tagged_documents = tags

		vector_size = 20
		iterations = 20
		model = Model()

		model.doc2vec(documents, vector_size, iterations, True, True, True)
		vectors = model.doc2vec_get_vectors(documents)
		return vectors
	

	def get_vectors_lda(self, doc_term_matrices):
		"""
		## This method fetches document vectors using Latent Dirichlet Allocation for the input document-term matrices.
		### Args :
		* (int, int)[][] doc_term_matrices : the document-term matrices used to train the LDA model and that will be vectorised in the LDA model.
		### Returns :
		* float[][] : the vectors of all the input documents.
		"""
		documents = Documents()
		documents.document_term_matrix = doc_term_matrices

		vector_size = 20
		iterations = 20
		model = Model()

		model.lda(documents, vector_size, iterations)
		vectors = model.get_values(documents, vector_size)
		return vectors


	def integrate_relations(self, feature_subs):
		"""
		## This method integrates the sub-features to the parent features generated by each classification iteration.
		### Args :
		* (`src.feature_model.Feature`, `src.feature_model.Feature`[]|`src.feature_model.FeatureOrientedDomainAnalysisTree`[])[] feature_subs : the features and their assigned sub-features.
		* `src.feature_model.Feature`[]|`src.feature_model.FeatureOrientedDomainAnalysisTree`[] sub_features : the list of all the features present in the model.
		### Returns :
		* `src.feature_model.FeatureOrientedDomainAnalysisTree`[] : the new subtrees, with features as roots and their sub_features as in 'feature_subs' as children if they exist in the currently in use sub_features.
		"""
		subtrees = []
		for feature, sub_fs in feature_subs:
			if feature != None and len(sub_fs) > 1:
				tree = FeatureOrientedDomainAnalysisTree()
				tree.add_node(feature)
				for sub in sub_fs:
					if sub != feature:
						#print("Added", sub, "to", feature)
						if str(type(sub)).find("feature_model.FeatureOrientedDomainAnalysisTree") != -1:
							tree.add_subtree(sub, feature)
						elif str(type(sub)).find("feature_model.Feature") != -1:
							tree.add_node(sub, feature)
						else:
							print(sub, "is neither a tree nor a feature !")
				subtrees.append(tree)
		if len(subtrees) != len(feature_subs):
			print("Problem in generation of subtrees !")
		return subtrees


	def cluster(self, vectors, feature_list, num_clusters=30, num_repeats=200, max_iter=2000):
		"""
		## This method clusters the vectors provided to it using K-Means and clusters the documents associated to them with the information found in feature list.
		### Args :
		* float[][] vectors : the vectors associated to the documents.
		* (`src.feature_model.Feature`|`src.feature_model.FeatureOrientedDomainAnalysisTree`, string[], TaggedDocument, (int, int)[])[] feature_list : the list of all the combined data for all currently used features/subtrees.
		* @Optional int num_clusters : the number of clusters to use in the K-Means algorithm.
		* @Optional int num_repeats : the number of times the K-Means algorithm will be repeated to ensure the best clustering.
		* @Optional int max_iter : the number of maximum iterations of the model per repeat.
		### Returns:
		* (`src.feature_model.Feature`|`src.feature_model.FeatureOrientedDomainAnalysisTree`|string), string[], (string, string)[], (int, int)[][])[][] clusters : the data of the current features in use sorted by cluster.
		"""
		clusterer = Cluster()
		clusterer.skl_kmeans(num_clusters, num_repeats, max_iter)
		clusters = clusterer.skl_kmeans_fit_and_cluster_vectors(vectors)
		sorted_clusters = clusterer._sort_features(clusters, feature_list, num_clusters)
		return sorted_clusters
	

	def generate_dict_dtm(self, documents):
		"""
		## This method generates the dictionary and document-term matrices based on the initial documents input for feature modelling.
		### Args :
		* `src.data_structures.Documents` documents : the user stories as documents containing the tokens that will be used to generate the dictionary and document-term matrices.
		### Returns :
		* Dict : the dictionary containing all of the possible terms contained in the documents.
		* (int, int)[][] : the document-term matrices of all the input documents.
		"""
		preprocessing = Preprocessing()
		dict = preprocessing.to_dictionary(documents)
		dtm = preprocessing.to_doc_term_matrix(documents)
		return dict, dtm
	

	def to_doc_term_matrix(self, dictionary, tokens):
		"""
		## This method generates document-term matrices for the input tokens based on the input dictionary.
		### Args :
		* Dict dictionary : the dictionary containing the terms of the documents.
		* string[][] tokens : the tokens reprisenting the documents to be transformed into document-term matrices.
		### Returns :
		* (int, int)[][] : the document-term matrices of all the input token lists.
		"""
		preprocessing = Preprocessing()
		dtm = preprocessing.tokens_to_dtm(dictionary, tokens)
		return dtm


	def user_stories_to_feature_model(self, documents, root_name, lang="en"):
		"""
		## This function takes a set of user stories as a Documents object, extracts the features they describe and assembles them into a feature model tree.
		### Args :
		* `data_stuctures.Documents` documents : the user story containing documents to transform into a feature model.
		* String root_name : the name of the root of the feature model.
		* @Optional String lang : the language used in the user stories. Only supports 'en' for English and 'fr' for French.
		### Returns :
		* `feature_model.FeatureOrientedDomainAnalysisTree` : the feature model generated from the input user stories.
		"""
		c = Cluster()
		print("User-stories to Feature Model\n")
		preprocessing = Preprocessing()
		tree = None

		# get initial tokens
		print("Preprocessing...")
		tagged_documents = preprocessing.prepare_parts_of_user_stories(documents, False, False, True, False, True, True, lang)

		# get dictionary and doc-term matrices
		dictionary, doc_term_matrices = self.generate_dict_dtm(documents)

		# get initial features
		print("Generating initial features...")
		initial_features = self.initial_features(documents)

		# tranform into following lists (feature names, feature keywords, feature tags)
		print("Assembling feature list...")
		feature_list = []
		for feature, tokens, tags, dtm in zip(initial_features, documents.tokenized_documents, tagged_documents, doc_term_matrices):
			feature_list.append((feature, tokens, tags, dtm))
		grouped_lists = (initial_features, documents.tokenized_documents, tagged_documents, doc_term_matrices)

		done = False
		count = 0
		while not done:
			print("\n\tIteration", count, "\n")
			# generate vectors
			print("Generating document vectors...")
			vectors = self.get_vectors_doc2vec(grouped_lists[2], grouped_lists[1])
			#vectors = self.get_vectors_lda(grouped_lists[3])

			# cluster and sort features
			print("Clustering...")
			num_clusters = c.get_optimal_elbow_num(vectors, 2, len(vectors), 1)
			if num_clusters < 2:
				num_clusters = 2
			#c.graph_elbow_vec_vector(vectors, 2, len(vectors), 1)
			sorted_clusters = self.cluster(vectors, feature_list, num_clusters)

			# create new feature information
			print("Generating new features...")
			grouping_clusters = []
			for cluster in sorted_clusters:
				new_cluster = []
				for feature, tokens, tags, dtm in cluster:
					new_tags = []
					for word, tag in zip(tags[0], tags[1]):
						new_tags.append((word, tag))
					new_cluster.append((feature, tokens, new_tags, dtm))
				grouping_clusters.append(new_cluster)

			# grouping new features
			new_features = self.group(grouping_clusters)
			print("Number of new features =", len(new_features))

			# generating new features from info
			features_and_subs = []
			kept_features = []
			for name, tokens, confidence, sub_features, is_concrete in new_features:
				# name, tokens, confidence, (feature, tokens, tags)
				if str(type(name)).find("'str'") != -1:
					feature = self.create_feature(name, is_concrete)
				else:
					feature = name
				subs = []
				if len(sub_features) > 0:
					for sub_feature, _, _, _ in sub_features:
						subs.append(sub_feature)
					features_and_subs.append((feature, subs))
				else:
					kept_features.append(feature)

			# requires (feature, feature[])[]
			# create subtrees
			print("Generating new subtrees...")
			new_trees = self.integrate_relations(features_and_subs)

			# remove sub features from initial features
			print("Preparing next iteration data...")
			ungrouped_features = kept_features

			# build new grouped_lists and feature_list objects
			new_feature_list = []
			list1_features = []
			list2_tokens = []
			list3_tags = []
			list4_dtms = []
			# adding new trees to lists
			for tree in new_trees:
				for name, tokens, confidence, sub_features, is_concrete in new_features:
					if str(name) == str(tree):
						list1_features.append(tree)
						list2_tokens.append(tokens)
						added_tags = []
						t = []
						ta = []
						for _, _, tags, _ in sub_features:
							for word, tag in tags:
								if (word, tag) not in added_tags:
									added_tags.append((word, tag))
									t.append(word)
									ta.append(tag)
						tagged_docs = TaggedDocument(t, ta)
						list3_tags.append(tagged_docs)
						dtm = self.to_doc_term_matrix(dictionary, tokens)
						list4_dtms.append(dtm)
						new_feature_list.append((tree, tokens, tagged_docs, dtm))

			# adding ungrouped features to list
			for feat, toks, tags, dtm in feature_list:
				if feat in ungrouped_features:
					new_feature_list.append((feat, toks, tags, dtm))
					list1_features.append(feat)
					list2_tokens.append(toks)
					list3_tags.append(tags)
					list4_dtms.append(dtm)

			new_grouped_lists = (list1_features, list2_tokens, list3_tags, list4_dtms)

			feature_list = new_feature_list
			grouped_lists = new_grouped_lists

			# check if done
			if(len(feature_list) <= self.MAX_CLUSTERS):
				done = True
			else:
				count += 1
		
		self.added_relations.clear()
		tree = self.construct_tree(root_name, grouped_lists[0])
		if not tree.check_validity():
			tree.set_validity()
		tree_feats = tree.get_features()
		for feature in initial_features:
			if feature not in tree_feats:
				print("ERROR :", feature, "not in tree !")
		return tree


def top_subtrees(tree, max_size=10):
	"""
	## This function takes a feature model and returns all of the largest subtrees from that model (so it can be more easily viewed), and removes those subtrees (other than the root element) from the original feature model.
	### Args :
	* `src.feature_model.FeatureOrientedDomainAnalysisTree` tree : the feature model whose subtrees might be too big to be viewed easily.
	* @Optional int max_size : the maximum size (in number of features) of a subtree before it is cut (cut if strictly larger).
	### Returns :
	* `src.feature_model.FeatureOrientedDomainAnalysisTree`[] : the large subtrees (if any), whose size is larger than the input size.
	"""
	root = tree.get_root()
	features = []
	for relation in tree.relations:
		if relation.parent == root:
			features = relation.children.copy()
	subtrees = []
	features_to_delete = []
	for feature in features:
		sub = tree.get_subtree(feature)
		if len(sub.get_features()) > max_size:
			subtrees.append(sub)
			for relation in tree.relations:
				if relation.parent == feature:
					for child in relation:
						features_to_delete.append(child)
	for feature in features_to_delete:
		tree.remove_node(feature)
	return subtrees


if __name__ == '__main__':
	dataset = Dataset()
	fp = FeatureProcessing()

	# fetching dataset
	#	documents = dataset.read_unformatted_from_file("data/user_story_datasets/g03-loudoun.txt")
	documents = dataset.read_unformatted_from_file("data/user_story_datasets/ITKdatasetEN.txt")
	# building tree
	tree = fp.user_stories_to_feature_model(documents, "ITKdatasetEN")

	# writing tree data as UML
	try:
		uml = tree.to_uml()
		dataset.write_strings_to_file([uml], "outputUML/L_1_generated_model_original.uml")
	except:
		pass
	print("Number of nodes :", len(tree.get_features()))

	# separating largest subtrees for easy viewing (large ones can exceed width of generated UML image)
	subtrees = top_subtrees(tree)
	try:
		uml = tree.to_uml()
		dataset.write_strings_to_file([uml], "outputUML/L_1_generated_model_subtrees_removed.uml")
	except:
		pass
	for t in subtrees:
		try:
			uml = t.to_uml()
			dataset.write_strings_to_file([uml], "outputUML/L_generated_model"+t.get_root().name+".uml")
		except:
			tree.pretty_print()






