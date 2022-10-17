# author : Liam RICE

from enum import Enum

def remove_duplicates(arr):
	"""
	## This function takes an array and removes all duplicate objects from it.
	### Args :
	* Any[] : an array of equatable objects.
	### Returns :
	* Any[] : the input array without any duplicate objects.
	"""
	added = []
	for a in arr:
		if a not in added:
			added.append(a)
	return added


class ObjectType(Enum):
	"""
	## This object reprisents whether the Feature is 'abstract' or 'concrete'.
	"""
	ABSTRACT = 0
	CONCRETE = 1


class FeatureType(Enum):
	"""
	## This object reprisents whether the Feature is normal, 'mandatory' 'selection' or 'optional'.
	"""
	NORMAL = 0
	MANDATORY = 1
	OPTIONAL = 2


class RelationType(Enum):
	"""
	## This object reprisents wheter the relation between children is 'and', 'or' or 'xor'.
	"""
	AND = 0
	OR = 1
	XOR = 2


class FeatureRelation(Enum):
	"""
	## This object reprisents wheter the relation between features is 'none', that one 'implies' the other, or that one 'excludes' the other.
	"""
	NONE = 0
	IMPLIES = 1
	EXCLUDES = 2



class FeatureOrientedDomainAnalysisTree:
	"""
	## This class reprisents a FODA model, and contains all the methods to manipulate the model.
	* this class is iterable (iterates over all features in the model)
	* this class has a length (number of relations in the model)
	* this class is equatable (all relations must be equal, but not necessarily in the same order)
	"""
	def __init__(self):
		self.relations = [] #: `src.feature_model.Relation`[] : the array of relations between parent features and child features.
	

	def __str__(self):
		return str(self.get_root())


	def __iter__(self):
		features = self.get_features()
		for feature in features:
			yield feature
	

	def __len__(self):
		return len(self.relations)
	

	def __eq__(self, other):
		if(other != None):
			if str(type(other)).find("FeatureOrientedDomainAnalysisTree") != -1:
				res = True
				for r1 in self.relations:
					found = False
					for r2 in other.relations:
						if r1 == r2:
							found = True
					res = res and found
				for r1 in other.relations:
					found = False
					for r2 in self.relations:
						if r1 == r2:
							found = True
					res = res and found
				return res
			elif str(type(other)).find("Feature") != -1:
				root = self.get_root()
				if(root != None):
					return root.name == other.name
				else:
					return False
			else:
				return False
		else:
			return False
	

	def pretty_print(self):
		"""
		## This method prints a human readable reprisentation of the FODA model.
		"""
		print(str(self))
		for relation in self.relations:
			try:
				print(relation)
			except:
				print(relation.parent, "->", relation.children)
	

	def to_uml(self):
		"""
		## This method returns the UML code (as an object diagram) of the feature model.
		### Returns :
		* string : the UML code reprisenting this feature model.
		"""
		uml = "@startuml\n"
		for feature in self.get_features():
			try:
				uml += 'object "' + feature.name+'" as ' + feature.get_concrete_name() + "\n"
			except:
				print(feature)
		for relation in self.relations:
			uml += "\n"
			for child in relation:
				try:
					uml += relation.parent.get_concrete_name() + " --> " + child.get_concrete_name() + "\n"
				except:
					print(relation.parent, "->", relation.children)
		uml += "@enduml"
		return uml
	

	def get_features(self):
		"""
		## This method fetches the array of all Features contained within the FODA model.
		### Returns :
		* `src.feature_model.Feature`[] : the list of all features in the tree.
		"""
		list_all_features = []
		for relation in self.relations:
			if relation.parent not in list_all_features:
				list_all_features.append(relation.parent)
			for child in relation.children:
				if child not in list_all_features:
					list_all_features.append(child)
		return list_all_features


	def get_root(self):
		"""
		## This method fetches the root Feature of the FODA model.
		### Returns :
		* `src.feature_model.Feature` : the root Feature of the tree.
		"""
		root_feature = None
		for feature in self.get_features():
			is_root = True
			for relation in self.relations:
				if feature not in relation.children:
					is_root = is_root and True
				else:
					is_root = False
			if(is_root):
				root_feature = feature
		return root_feature


	def depth(self, feat=None):
		"""
		## This method finds out the depth of the FODA model (counts from 1), or the depth of the parameter Feature (counts from 0).
		### Args :
		* @Optional `src.feature_model.Feature` feat : the feature you wish to get the depth of (0 = root, ...), if empty, this function fetches the maximum depth of the tree (1 = root, ...).
		### Returns :
		* int : the depth of the tree/feature.
		"""
		root_feature = self.get_root()
		if(feat == None):
			max_depth = 0
			for feature in self.get_features():
				depth = 0
				current_parent = feature
				max_iter = 100
				count = 0
				override = False
				while(current_parent != root_feature and not override):
					print(current_parent)
					for relation in self.relations:
						if current_parent in relation.children:
							current_parent = relation.parent
							depth += 1
					count += 1
					if(count >= max_iter):
						override = True
				if depth > max_depth:
					max_depth = depth
			return max_depth+1
		else:
			depth = 0
			current_parent = feat
			max_iter = 100
			count = 0
			override = False
			while(current_parent != root_feature and not override):
				#print(current_parent)
				for relation in self.relations:
					if current_parent in relation.children:
						current_parent = relation.parent
						depth += 1
				count += 1
				if count >= max_iter:
					override = True
			return depth


	def add_node(self, new_node, parent_node=None, relation_type=RelationType.AND):
		"""
		## This method adds a node to the FODA model, either under the parent given, or in a new Relation object with the provided relation type.
		### Args :
		* `src.feature_model.Feature` new_node : the node to add to the tree.
		* @Optional `src.feature_model.Feature` parent_node : the node under which to place the new node (leave as None if the node to add is the root).
		* @Optional `src.feature_model.RelationType` relation_type : the relation type, only used if this is a node placed under a child and not under a parent (ie. not added as a child to an object that already has children).
		### Returns :
		* bool : True if the node was correctly added, False if the node could not be added (for instance, parent doesn't exist).
		"""
		done = False
		if(parent_node != None):
			for relation in self.relations:
				if parent_node == relation.parent and not done:
					relation.add_child(new_node)
					done = True
			if not done:
				for relation in self.relations:
					if parent_node in relation.children:
						if(not done):
							i = relation.children.index(parent_node)
							new_relation = Relation(relation.children[i], [new_node], relation_type)
							self.relations.append(new_relation)
							done = True
		else:
			if(len(self.relations) == 0):
				new_relation = Relation(new_node, [], relation_type)
				self.relations.append(new_relation)
				done = True
			else:
				done = False
		return done
	

	def get_subtree(self, feature):
		"""
		## This method fetches the subtree whose root is the input feature.
		### Args :
		* `src.feature_model.Feature` feature : the feature which will be the root of the new subtree.
		### Returns :
		* `src.feature_model.FeatureOrientedDomainAnalysisTree` : the subtree found under the child feature.
		"""
		if feature in self.get_features():
			tree = FeatureOrientedDomainAnalysisTree()
			tree.add_node(feature)
			done = False
			parents = [feature]
			while not done:
				for relation in self.relations:
					if relation.parent == parents[0]:
						for child in relation:
							tree.add_node(child, parents[0])
							parents.append(child)
				parents.pop(0)
				if len(parents) <= 0:
					done = True
			return tree
		else:
			return None
	

	def add_subtree(self, subtree, parent_node, relation_type=RelationType.AND):
		"""
		## This function adds all relations of a subtree to the given parent node.
		### Args :
		* `src.feature_model.FeatureOrientedDomainAnalysisTree` subtree : the subtree to add to this tree.
		* `src.feature_model.Feature` parent_node : the feature under which the tree is enscribed.
		* @Optional `src.feature_model.RelationType` : the type of relation to make if the parent node is a leaf.
		### Returns :
		* bool : True if the subtree was correctly added and False if not.
		"""
		done = False
		for relation in self.relations:
			if parent_node == relation.parent:
				relation.add_child(subtree.get_root())
				done = True
		if not done:
			for feature in self.get_features():
				if parent_node == feature:
					relation = Relation(feature, [subtree.get_root()], relation_type)
					self.relations.append(relation)
					done = True
		if done:
			for relation in subtree.relations:
				self.relations.append(relation)
		return done


	def remove_node(self, node):
		"""
		## This function removes the given node and all subnodes from the FODA model.
		### Args :
		* `src.feature_model.Feature` node : the node to remove from the tree.
		"""
		relations_to_remove = []
		nodes_to_remove = []
		for relation in self.relations:
			if relation.parent == node:
				relations_to_remove.append(relation)
				nodes_to_remove.extend(relation.children)
			deletable_nodes = []
			if node in relation.children:
				deletable_nodes.append(node)
			for n in deletable_nodes:
				relation.children.remove(n)
			if len(relation.children) == 0:
				relations_to_remove.append(relation)
		for relation in relations_to_remove:
			self.relations.remove(relation)
		for n in nodes_to_remove:
			self.remove_node(n)
		

	def similarity(self, other, _check=0):
		"""
		## This function calculates the similarity index between this FODA model and another FODA model, and isolates the differences between this model and the other.
		### Args :
		* `src.feature_model.FeatureOrientedDomainAnalysisTree` other : the other model to which you want to compare this model.
		### Returns :
		* float : the similarity index between the two models.
		* `src.feature_model.Relation`[] : the relations present in this model that are not present in the other model.
		* `src.feature_model.Feature`[] : the features present in this model (that aren't in the Relation objects above) that are not present in the other model.
		"""
		if other != None:
			matches = []
			diff = []
			maximum_absolute_similarity = 0
			for rel1 in self.relations:
				maximum_absolute_similarity += len(rel1)*5+1
				found = False
				for rel2 in other.relations:
					if rel1.parent == rel2.parent:
						matches.append((rel1, rel2))
						found = True
				if not found:
					diff.append(rel1)
			absolute_similarity = 0
			diff_features = []
			for rel1, rel2 in matches:
				if(rel1.relation_type == rel2.relation_type):
					absolute_similarity += 1
				for c1 in rel1:
					found = False
					for c2 in rel2:
						if(c1 == c2):
							absolute_similarity += 3
							if(c1.feature_type == c2.feature_type):
								absolute_similarity += 1
							if(c1.object_type == c2.object_type):
								absolute_similarity += 1
							found = True
					if not found:
						diff_features.append(c1)
			if(_check == 0):
				sim, diff_y, diff_features_y = other.similarity(self, 1)
				diff_features.extend(diff_features_y)
				diff.extend(diff_y)
				avg = (sim + absolute_similarity/maximum_absolute_similarity)/2
				return (avg, remove_duplicates(diff), remove_duplicates(diff_features))
			else:
				return (absolute_similarity/maximum_absolute_similarity, diff, diff_features)
		else:
			return (0, self.relations, [])
	
	
	def check_validity(self):
		"""
		## This method checks if the links are valid.
		### returns :
		* bool : True if the tree is internally valid, False if not.
		"""
		valid = True
		for feature in self.get_features():
			parents = []
			num_relations = 0
			for relation in self.relations:
				count = 0
				if relation.parent == feature:
					num_relations += 1
				for check_feat in relation:
					if check_feat == feature:
						parents.append(relation.parent)
						count += 1
				if(count > 1):
					valid = False
			if len(parents) > 1 or num_relations > 1:
				valid = False
		return valid
	

	def set_validity(self):
		"""
		## This method checks all features and relations and corrects any linking errors between features.
		"""
		print("Removing duplicates...")
		for r in self.relations:
			r.children = remove_duplicates(r.children)
		print("Analysing features...\n[", end="", flush=True)
		max_len = len(self.get_features())
		count = 0
		num = 1
		relations_to_merge = []
		parents_to_remove = []
		for feature in self.get_features():
			if(count > num*(max_len/20)):
				print("=", end="", flush=True)
				num += 1
			count += 1
			parents = []
			count_times_as_parent = 0
			relations = []
			for relation in self.relations:
				if relation.parent == feature:
					count_times_as_parent += 1
					relations.append(relation)
				for check_feat in relation:
					if check_feat == feature:
						parents.append(relation.parent)
			if len(relations) > 1:
				relations_to_merge.append(relations)
			if len(parents) > 1:
				min_depth = -1
				min_parent = None
				for parent in parents:
					depth = self.depth(parent)
					if depth > min_depth or min_depth == -1:
						min_depth = depth
						min_parent = parent
				remove_parents = []
				for parent in parents:
					if parent != min_parent:
						remove_parents.append((parent, feature))
				parents_to_remove.append(remove_parents)
		print("]") 
		for remove_parents in parents_to_remove:
			for parent, feature in remove_parents:
				self.remove_parent(parent, feature)
		for group in relations_to_merge:
			r = []
			for rel in group:
				par = rel.parent
				for c in rel:
					if c not in r:
						r.append(c)
				if rel in self.relations:
					self.relations.remove(rel)
			new_rel = Relation(par, r)
			self.relations.append(new_rel)
	

	def remove_parent(self, parent, feature):
		"""
		## This method removes the relation between a parent and a feature.
		### Args :
		* `src.feature_model.Feature` parent : the parent from which to remove the feature.
		* `src.feature_model.Feature` feature : the feature from which to remove the parent.
		"""
		for relation in self.relations:
			if relation.parent == parent:
				try:
					relation.children.remove(feature)
				except:
					pass
			if len(relation) == 0:
				self.relations.remove(relation)




class Feature:
	"""
	## This class reprisents a feature, and contains the name and type of that feature.
	* this class is equatable (only the name is considered, if names are equal, the features are equal)
	"""
	def __init__(self, name, feature_type=FeatureType.NORMAL, object_type=ObjectType.CONCRETE):
		self.name = name #: String : the name of the feature.
		self.feature_type = feature_type #: `src.feature_model.FeatureType` : the type of feature, can be MANDATORY, OPTIONAL or NORMAL.
		self.object_type = object_type #: `src.feature_model.ObjectType` : the type of object, can be ABSTRACT or CONCRETE.
		#print("Created feature with name :", name)


	def __str__(self):
		return self.name

	
	def __eq__(self, other):
		if(other != None):
			if str(type(other)).find("FeatureOrientedDomainAnalysisTree") != -1:
				root = other.get_root()
				if(root != None):
					return self.name == root.name
				else:
					return False
			elif str(type(other)).find("Feature") != -1:
				return self.name == other.name
			else:
				print("Two objects not equatable :", type(self), "/", type(other))
				return False
		else:
			return False
	

	def get_concrete_name(self):
		"""
		"""
		new_name = self.name.replace(" ", "_")
		new_name = new_name.replace(",", "")
		new_name = new_name.replace(".", "")
		new_name = new_name.replace("-", "_")
		new_name = new_name.replace("'", "")
		new_name = new_name.replace("/", "")
		return new_name
	


class Relation:
	"""
	## This class reprisents the relation between a parent feature and an array of child features, including the type of relation that links them.
	* this class is iterable (iterates over the parent node's children)
	* this class has a length (number of children)
	* this class is equatable (checks equal parent, children and relation type)
	"""
	def __init__(self, parent, children=None, relation_type=RelationType.AND):
		self.parent = parent #: `src.feature_model.Feature` : the parent that has the child.
		self.children = [] #: `src.feature_model.Feature` : the child this relation links to.
		if children != None:
			self.children = children
		self.relation_type = relation_type #: `src.feature_model.RelationType` : the type of relation between the parent and the children, can be AND, OR or XOR.


	def __str__(self):
		string = ""
		for child in self.children:
			string += str(self.parent) + " --> " + str(child.feature_type) + " : " + str(child) + "\n" + str(self.relation_type) + "\n"
		return string
	

	def __iter__(self):
		for child in self.children:
			yield child
	

	def __len__(self):
		return len(self.children)
	

	def __eq__(self, other):
		if(other != None):
			ret = True
			ret = ret and (self.parent == other.parent) and (self.relation_type == other.relation_type)
			for x in self.children:
				found = False
				for y in other.children:
					if x == y:
						found = True
				ret = ret and found
			for x in other.children:
				found = False
				for y in self.children:
					if x == y:
						found = True
				ret = ret and found
			return ret
		else:
			return False


	def add_child(self, new_node):
		"""
		## This method adds a new child feature to this relation.
		### Args :
		* `src.feature_model.Feature` new_node : the node to add as a child to this relation.
		"""
		self.children.append(new_node)






if __name__ == '__main__':
	root = Feature("Recycling", feature_type=FeatureType.MANDATORY, object_type=ObjectType.ABSTRACT)
	data = Feature("Data", feature_type=FeatureType.MANDATORY, object_type=ObjectType.ABSTRACT)
	company_data = Feature("Company Data", feature_type=FeatureType.MANDATORY)
	encrypted_data = Feature("Encrypted Data", feature_type=FeatureType.MANDATORY)
	activity_fees = Feature("Online Activity Fees", feature_type=FeatureType.MANDATORY)
	transaction_history = Feature("Transaction History", feature_type=FeatureType.MANDATORY)
	doc = Feature("Doc", feature_type=FeatureType.MANDATORY, object_type=ObjectType.ABSTRACT)
	api = Feature("Site API", feature_type=FeatureType.MANDATORY)
	user_doc = Feature("View User Documentation", feature_type=FeatureType.MANDATORY)
	time = Feature("Time", feature_type=FeatureType.MANDATORY, object_type=ObjectType.ABSTRACT)
	facility_hours = Feature("Get Facility Hours", feature_type=FeatureType.MANDATORY)
	pick_up_time = Feature("Flexible Pick Up Time", feature_type=FeatureType.MANDATORY)
	upload_schedule = Feature("Upload Schedule", feature_type=FeatureType.MANDATORY)
	choose_time = Feature("Choose Dropoff Time", feature_type=FeatureType.MANDATORY)
	location = Feature("Location", feature_type=FeatureType.MANDATORY)
	nearby_facilities = Feature("Get Nearby Facilities", feature_type=FeatureType.MANDATORY)
	feedback = Feature("Invalid ZIP Feedback", feature_type=FeatureType.MANDATORY)
	view_map = Feature("View Map", feature_type=FeatureType.MANDATORY)
	see_centres = Feature("See All Centres on Map", feature_type=FeatureType.MANDATORY)
	map_recycling = Feature("View Map of Recycling Centres", feature_type=FeatureType.MANDATORY)
	route_planning = Feature("Route Planning", feature_type=FeatureType.MANDATORY)
	special_waste = Feature("View Map of Special Waste", feature_type=FeatureType.MANDATORY)
	create_account = Feature("Create account", feature_type=FeatureType.MANDATORY)
	link_email = Feature("Link Email", feature_type=FeatureType.MANDATORY)
	email_notifications = Feature("Email Notifications", feature_type=FeatureType.MANDATORY)
	rewards = Feature("Tempting Rewards", feature_type=FeatureType.MANDATORY)
	access = Feature("Access", feature_type=FeatureType.MANDATORY)
	multidevice_support = Feature("Multi-device Support", feature_type=FeatureType.MANDATORY)
	ui = Feature("UI", feature_type=FeatureType.MANDATORY)
	view_public_info = Feature("View Public Information", feature_type=FeatureType.MANDATORY)
	browse_facilities = Feature("Browse Facilities", feature_type=FeatureType.MANDATORY)
	great_uiux = Feature("Great UI/UX", feature_type=FeatureType.MANDATORY)
	easy_to_use = Feature("Easy to Use", feature_type=FeatureType.MANDATORY)
	uiux_lessons = Feature("UI/UX lessons", feature_type=FeatureType.MANDATORY)
	bootstrap = Feature("Bootstrap", feature_type=FeatureType.MANDATORY)
	selectable_waste = Feature("Selectable Types of Waste", feature_type=FeatureType.MANDATORY)
	change_info = Feature("Change Information", feature_type=FeatureType.MANDATORY)
	address_to_map = Feature("Click on Address to Go to Map", feature_type=FeatureType.MANDATORY)
	disposal_events = Feature("View Disposal Events", feature_type=FeatureType.MANDATORY)
	favourite_facilities = Feature("Favourite Facilities", feature_type=FeatureType.MANDATORY)

	extra_node = Feature("Extra Node")

	tree = FeatureOrientedDomainAnalysisTree()
	tree.add_node(root)
	tree.add_node(data, root)
	tree.add_node(company_data, data)
	tree.add_node(encrypted_data, company_data)
	tree.add_node(activity_fees, company_data)
	tree.add_node(transaction_history, activity_fees)
	tree.add_node(doc, root)
	tree.add_node(api, doc)
	tree.add_node(user_doc, doc)
	tree.add_node(time, root)
	tree.add_node(facility_hours, time)
	tree.add_node(pick_up_time, time)
	tree.add_node(upload_schedule, pick_up_time)
	tree.add_node(choose_time, pick_up_time)
	tree.add_node(location, root)
	tree.add_node(nearby_facilities, location)
	tree.add_node(feedback, nearby_facilities)
	tree.add_node(view_map, location)
	tree.add_node(see_centres, view_map)
	tree.add_node(map_recycling, view_map)
	tree.add_node(route_planning, view_map)
	tree.add_node(special_waste, view_map)
	tree.add_node(create_account, root)
	tree.add_node(link_email, create_account)
	tree.add_node(email_notifications, link_email)
	tree.add_node(rewards, email_notifications)
	tree.add_node(access, root)
	tree.add_node(multidevice_support, access)
	tree.add_node(multidevice_support, root)
	tree.add_node(ui, access)
	tree.add_node(great_uiux, ui)
	tree.add_node(easy_to_use, ui)
	tree.add_node(uiux_lessons, great_uiux)
	tree.add_node(bootstrap, great_uiux)
	tree.add_node(view_public_info, access)
	tree.add_node(selectable_waste, view_public_info)
	tree.add_node(change_info, view_public_info)
	tree.add_node(address_to_map, view_public_info)
	tree.add_node(disposal_events, view_public_info)
	tree.add_node(browse_facilities, view_public_info)
	tree.add_node(favourite_facilities, browse_facilities)

	# eliminate superfluous user stories
		# find out what separates user stories that describe the same feature VS user stories that describe similar features
		# use separation actions and object?
	# 1 user story = 1 feature
	# generate feature names
		# use separation object
	# generate feature keywords
		# use object words and synsets
	# begin clustering between features using keywords
		# cluster two by two?
		# generate new features using common words, hypernyms and their synsets when two features are clustered?
			# make generated features not based on user-stories abstract?
		# determine how to decide if features are peers or if one is a sub-feature of another

	tree.set_validity()
	print(tree.check_validity())
	print(tree.to_uml())
	tree.remove_node(location)
	print(tree.check_validity())

	


















