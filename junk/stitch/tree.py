class ImageTree():
	def __init__(self, elem=None):
		self.children = []
		self.data = elem

	def __len__(self):
		if self.data == None:
			count = 0
		else:
			count = 1

		for c in self.children:
			count += len(c)

		return count

	def elements(self):
		seq = [self.data]
		for c in self.children:
			seq += c.elements()

		return seq

	def insert(self, elem):
		self.data = elem

	def makeChildOf(self, child, parent=None):
		if parent == None:
			node = self
		else:
			node = self.find(parent)

		if node != None:
			tree = ImageTree(child)
			node.children.append(tree)
			return tree
		else:
			print("Element not yet present in tree.")
			return None

	def find(self, elem):
		if self.data == elem:
			return self

		for c in self.children:
			if c.find(elem) != None:
				return c

		return None