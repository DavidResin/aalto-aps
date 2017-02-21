class TupleTree():
	def __init__(self, index=0, elem=None):
		self.children = []
		self.data = elem
		self.index = index

	def __len__(self):
		if self.data == None:
			count = 0
		else:
			count = 1

		for c in self.children:
			count += len(c)

		return count

	def __str__(self):
		return self.__str_rec(0)

	def __str_rec(self, tab=0):
		if self.data == None:
			value = "NONE"
		elif self.index == -1:
			value = self.data
		else:
			value = str(self.data[self.index])

		text = ('\t' * tab) + value + '\n'

		for c in self.children:
			text += c.__str_rec(tab + 1)

		return text

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
			tree = TupleTree(self.index, child)
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

	def flatten(self):
		seq = []
		self.__flatten_rec(seq)
		return seq

	def __flatten_rec(self, seq):
		seq.append(self.data)
		for c in self.children:
			c.__flatten_rec(seq)