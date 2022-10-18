
"""
This module runs tests for the `src.feature_model` module.
"""

import unittest
from feature_model import *

class TestModel(unittest.TestCase):

	root = Feature("E-Shop")
	catalogue = Feature("Catalogue", feature_type=FeatureType.MANDATORY)
	payment = Feature("Payment", feature_type=FeatureType.MANDATORY)
	bank = Feature("Bank Transfer")
	card = Feature("Credit Card")
	security = Feature("Security", feature_type=FeatureType.MANDATORY)
	high = Feature("High Security")
	physical = Feature("Physical Security", feature_type=FeatureType.OPTIONAL)
	standard = Feature("Standard Security")
	search = Feature("Search", feature_type=FeatureType.OPTIONAL)
	delivery = Feature("Delivery", feature_type=FeatureType.OPTIONAL)
	express = Feature("Express Delivery", feature_type=FeatureType.OPTIONAL)
	normal = Feature("Normal Delivery", feature_type=FeatureType.MANDATORY)

	tree = FeatureOrientedDomainAnalysisTree()
	tree.add_node(root)
	tree.add_node(catalogue, root)
	tree.add_node(payment, root)
	tree.add_node(security, root)
	tree.add_node(search, root)
	tree.add_node(delivery, root)
	tree.add_node(bank, payment, relation_type=RelationType.OR)
	tree.add_node(card, payment)
	tree.add_node(high, security, relation_type=RelationType.XOR)
	tree.add_node(standard, security)
	tree.add_node(physical, high)
	tree.add_node(express, delivery, relation_type=RelationType.OR)
	tree.add_node(normal, delivery)
	
	guard = Feature("Guard")
	
	def test_new_foda(self):
		foda = FeatureOrientedDomainAnalysisTree()
		test_foda = FeatureOrientedDomainAnalysisTree()
		res = (foda == test_foda) and len(foda) == 0
		self.assertEqual(res, True, "Does not generate new FODA model correctly.")

	def test_len_foda(self):
		self.assertEqual(len(self.tree), 5, "Does not fetch correct length of FODA. Either error in adding new nodes or in reading the length.")

	def test_len_relations(self):
		arr = []
		for rel in self.tree.relations:
			arr.append(len(rel))
		res = "[5, 2, 2, 1, 2]"
		self.assertEqual(str(arr), res, "Does not fetch correct length of Relations. Either error in adding new nodes or in reading the length.")

	def test_get_features(self):
		l = self.tree.get_features()
		test = [self.root, self.catalogue, self.payment, self.security, self.search, self.delivery, self.bank, self.card, self.high, self.standard, self.physical, self.express, self.normal]
		self.assertEqual(str(l), str(test), "Does not fetch correct array of features. Either error in adding new nodes or in producing the feature list.")
	
	def test_iterate(self):
		arr = []
		for obj in self.tree:
			arr.append(obj)
		test = [self.root, self.catalogue, self.payment, self.security, self.search, self.delivery, self.bank, self.card, self.high, self.standard, self.physical, self.express, self.normal]
		self.assertEqual(str(arr), str(test), "Does not iterate over features correctly. Either error in adding new nodes or in producing the feature list.")

	def test_depth(self):
		d = self.tree.depth()
		test = 4
		self.assertEqual(d, test, "Does not get proper depth of FODA model. Either error in adding new nodes, searching for the root or searching for the depth.")

	def test_feature_depth(self):
		d = self.tree.depth(self.payment)
		test = 1
		self.assertEqual(d, test, "Does not get proper depth of Feature in FODA model. Either error in adding new nodes, searching for the root or searching for the depth.")

	def test_get_root(self):
		rt = self.tree.get_root()
		test = self.root
		self.assertEqual(rt, test, "Does not get proper root of FODA model. Either error in adding new nodes or searching for the root.")

	def test_add_node(self):
		self.tree.add_node(self.guard, self.physical)
		checks = (self.tree.depth() == 5) and (len(self.tree.relations) == 6) and (len(self.tree.get_features()) == 14) and (self.guard in self.tree.get_features())
		self.tree.remove_node(self.guard)
		self.assertEqual(checks, True, "Does not add node to FODA model correctly.")
	
	def test_remove_node(self):
		self.tree.add_node(self.guard, self.physical)
		self.tree.remove_node(self.guard)
		checks = (self.tree.depth() == 4) and (len(self.tree.relations) == 5) and (len(self.tree.get_features()) == 13) and (self.guard not in self.tree.get_features())
		self.assertEqual(checks, True, "Does not remove node from FODA model correctly.")
	
	def test_remove_node(self):
		t = FeatureOrientedDomainAnalysisTree()
		n1 = Feature("One")
		n2 = Feature("Two")
		n3 = Feature("Three")
		n4 = Feature("Four")
		t.add_node(n1)
		t.add_node(n2, n1)
		t.add_node(n3, n1)
		t.add_node(n4, n2)
		checks = (t.depth() == 3) and (len(t.relations) == 2) and (len(t.get_features()) == 4)
		t.remove_node(n2)
		checks = (t.depth() == 2) and (len(t.relations) == 1) and (len(t.get_features()) == 2) and (n2 not in t.get_features()) and (n4 not in t.get_features())
		self.assertEqual(checks, True, "Does not remove multiple nodes from FODA model correctly.")
	
	def test_remove_duplicates(self):
		arr = [2, 3, 7, 3, 1, 2]
		result = "[2, 3, 7, 1]"
		self.assertEqual(str(remove_duplicates(arr)), result, "Does not remove duplicates from array correctly.")
	
	def test_similarity(self):
		a = Feature("A")
		b = Feature("B")
		c = Feature("C")
		d = Feature("D")
		e = Feature("E")
		f = Feature("F")

		t1 = FeatureOrientedDomainAnalysisTree()
		t2 = FeatureOrientedDomainAnalysisTree()

		t1.add_node(a)
		t1.add_node(b, a)
		t1.add_node(c, a)
		t2.add_node(d)
		t2.add_node(e, d)
		t2.add_node(f, d)
		sim, _, _ = t1.similarity(t2)
		self.assertEqual(sim, 0, "Does not generate zero similarity between two trees correctly.")
	
	def test_similarity_nonzero(self):
		a = Feature("A")
		b = Feature("B")
		c = Feature("C")
		d = Feature("D")
		e = Feature("E")
		f = Feature("F")

		t1 = FeatureOrientedDomainAnalysisTree()
		t2 = FeatureOrientedDomainAnalysisTree()

		t1.add_node(a)
		t1.add_node(b, a)
		t1.add_node(c, a)
		t1.add_node(d, c)
		t2.add_node(a)
		t2.add_node(b, a)
		t2.add_node(e, a)
		t2.add_node(f, a)
		sim, _, _ = t1.similarity(t2)
		self.assertEqual(sim, 0.36397058823529416, "Does not generate similarity between two trees correctly.")
	
	def test_add_subtree(self):
		a = Feature("A")
		b = Feature("B")
		c = Feature("C")
		d = Feature("D")
		e = Feature("E")
		f = Feature("F")

		tree = FeatureOrientedDomainAnalysisTree()
		subtree = FeatureOrientedDomainAnalysisTree()

		tree.add_node(a)
		tree.add_node(b, a)
		tree.add_node(c, a)
		subtree.add_node(d)
		subtree.add_node(e, d)
		subtree.add_node(f, d)
		tree.add_subtree(subtree, c)

		check = False
		for relation in tree.relations:
			if relation.parent == c:
				if relation.children[0] == d:
					check = True
		self.assertEqual(check, True, "Does not add subtree to tree correctly.")






