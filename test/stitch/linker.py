import cv2
import numpy as np
from tree import TupleTree

class Linker():
	def __init__(self):
		pass

	def solve(self, array, index=0):
		shape = array.shape
		edge_array = np.full(shape, False)
		cost_array = np.empty(shape)

		for (x, y), elem in np.ndenumerate(array):
			if elem is None or elem[index] is None:
				cost_array[x][y] = 0
			else:
				cost_array[x][y] = len(elem[index])
		
		sets = []

		for i in range(len(array)):
			sets.append({i})

		while len(sets[0]) != len(sets) and np.sum(cost_array) > 0:
			pos = self.__find_biggest(edge_array, cost_array)
			(x, y) = pos

			if sets[x] != sets[y]:
				self.__edge(pos, edge_array, cost_array)
				temp = sets[x] | sets[y]

				for e in temp:
					sets[e] = temp
			else:
				self.__update(cost_array, pos, 0)

		return edge_array

	def __update(self, array, pos, value):
		(x, y) = pos
		array[x][y] = value
		array[y][x] = value

	def __edge(self, pos, edge_array, cost_array):
		self.__update(cost_array, pos, 0)
		self.__update(edge_array, pos, True)

	def __find_biggest(self, edge_array, cost_array):
		pos = np.unravel_index(cost_array.argmax(), cost_array.shape)
		return pos

	def edgesToTree(self, firstelem, firstdata, array, edge_table, data_table):
		tree = TupleTree(0, firstelem + (None, firstdata, None,))
		self.__edge_rec(tree, array.index(firstelem), array, edge_table, data_table)
		return tree

	def __edge_rec(self, tree, index, array, edge_table, data_table):
		line = edge_table[index].tolist()

		while any(line):
			temp = line.index(True)
			pos = (index, temp)
			self.__update(edge_table, pos, False)
			(matches, matrix, status) = data_table[index][temp]
			newTree = tree.makeChildOf(array[temp] + (matches, np.dot(matrix, tree.data[4]), status),)
			line = edge_table[index].tolist()
			self.__edge_rec(newTree, temp, array, edge_table, data_table)